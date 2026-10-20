"""X6: 9×9 の詰みの「関連領域」を証明つきで求め、他の局面にどれだけ流用できるかを測る

1. floodgate 強豪AI同士の対局（X7 と同じ 9:1 の分割）の終盤から、手番側に 9手以内の詰み（王手の連続）がある局面を集める
2. 学習側の各局面で証明木を作る。詰ませる手は cshogi の詰み探索から受け取り、
   全ノードで Z.py（読んだマスを記録する規則）を通して「王手であること」「受け方の応手の全列挙」を確かめる。
   → 読んだマス = 関連領域。そこの中身が同じなら、他のマスに何があっても同じ手順で詰む（証明つき）
   持ち駒: 攻め方は「元の局面以上」、受け方は「元の局面以下」なら成り立つ（多いほど攻め方が得・少ないほど受け方が損）
3. 評価側の対局の全局面（手番側が攻め方として）とパターンを照合し、
   当てはまった局面が本当に詰むかを cshogi で全件確かめる（1件でも外れたら不具合）
   あわせて「評価側の詰み局面のうち、パターンで見つかる割合（流用率）」を測る

局面は「攻め方 = 先手」に揃える（後手番なら盤を180度回して先後を入れ替える）。照合は左右反転も試す。

使い方: python x6_mate_zones.py --plies 9
"""

import argparse
import collections
import json
import os
import random
import time

import cshogi
import numpy as np

import zone
import zone_exact

MODE = "cons"
Z = zone  # --mode で切り替え（cons: 許される駒の集合 / exact: 読んだマスの中身を固定）

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/mainline")


def normalized(board):
    """手番側を先手にそろえた (盤81, 持ち駒[2][7])"""
    pcs = list(board.pieces)
    hb, hw = board.pieces_in_hand
    if board.turn == 0:
        return pcs, [list(hb), list(hw)]
    out = [0] * 81
    for sq, p in enumerate(pcs):
        if p:
            out[80 - sq] = p ^ 16
    return out, [list(hw), list(hb)]


def mirror(pcs):
    out = [0] * 81
    for sq, p in enumerate(pcs):
        f, r = divmod(sq, 9)
        out[(8 - f) * 9 + r] = p
    return out


def board_from(pcs, hands):
    """正規化した盤から cshogi の盤を作る（先手番）"""
    b = cshogi.Board()
    b.set_pieces(pcs, (hands[0], hands[1]))
    return b


def shortest_mate(b, maxp):
    m = b.mate_move_in_1ply()
    if m:
        return m, 1
    for p in range(3, maxp + 1, 2):
        m = b.mate_move(p)
        if m:
            return m, p
    return 0, 0


class Fail(Exception):
    pass


def prove(ctx, b, left, my_k, op_k, stats):
    """攻め方の手番。left 手以内で詰むことを、読み取りを記録しながら確かめる"""
    stats["nodes"] += 1
    if stats["nodes"] > 20000:
        raise Fail("too many nodes")
    m, p = shortest_mate(b, left)
    if not m:
        raise Fail("no mate")
    mv = Z.from_cshogi(m, cshogi)
    try:
        c = ctx.verify_attack_move(mv, my_k, op_k)
    except AssertionError as e:
        raise Fail(f"verify: {e}")
    nk = mv[1] if mv[0] == my_k else my_k
    b.push(m)
    ev = c.evasions(op_k)
    if ev is None:
        b.pop()
        raise Fail("not check")
    ref = {cshogi.move_to_usi(x) for x in b.legal_moves}
    mine = {Z.to_usi(e) for e in ev}
    if not ref <= mine:
        b.pop()
        raise Fail("evasion mismatch")
    if not ev and mv[0] is None and mv[2] == Z.PAWN:
        b.pop()
        raise Fail("pawn drop mate")
    for e in ev:
        if left - 2 < 1:
            b.pop()
            raise Fail("depth")
        c2 = c.apply(e)
        u = Z.to_usi(e)
        b.push_usi(u)
        try:
            prove(c2, b, left - 2, nk, e[1] if e[0] == op_k else op_k, stats)
        finally:
            b.pop()
    b.pop()


def make_pattern(pcs, hands, plies, maxp):
    b = board_from(pcs, hands)
    hreads = set()
    cons = {} if MODE == "cons" else set()
    ctx = Z.Ctx(pcs, hands, 0, cons, hreads)
    my_k = pcs.index(Z.KING) if Z.KING in pcs else None
    op_k = pcs.index(Z.KING + 16)
    ctx.get(op_k)
    if my_k is not None:
        ctx.get(my_k)
    stats = {"nodes": 0}
    prove(ctx, b, maxp, my_k, op_k, stats)
    sq = np.array(sorted(cons), dtype=np.int16)
    allow = np.array([cons[x] if MODE == "cons" else (1 << pcs[x]) for x in sq], dtype=np.uint32)
    att = {pt: hands[0][Z.HAND_INDEX[pt]] for (col, pt) in hreads if col == 0}
    dfn = {pt: hands[1][Z.HAND_INDEX[pt]] for (col, pt) in hreads if col == 1}
    return {"sq": sq, "val": np.array([pcs[s] for s in sq], dtype=np.int16), "allow": allow, "exact": int(sum(bin(int(x)).count("1") == 1 for x in allow)), "att_min": att, "def_max": dfn, "plies": plies, "nodes": stats["nodes"], "op_k": op_k}


class Library:
    def __init__(self, pats):
        self.by_k = collections.defaultdict(list)
        for i, p in enumerate(pats):
            self.by_k[p["op_k"]].append(i)
        self.pats = pats
        self.mask = {}
        for k, idx in self.by_k.items():
            A = np.full((len(idx), 81), 0xFFFFFFFF, dtype=np.uint32)  # 各マスに許される駒コードの集合
            for j, i in enumerate(idx):
                A[j, pats[i]["sq"]] = pats[i]["allow"]
            self.mask[k] = A

    def match(self, pcs, hands):
        if Z.KING + 16 not in pcs:
            return None
        k = pcs.index(Z.KING + 16)
        if k not in self.mask:
            return None
        A = self.mask[k]
        x = np.array(pcs, dtype=np.uint32)
        ok = ((A >> x[None, :]) & 1).all(axis=1)
        for j in np.where(ok)[0]:
            p = self.pats[self.by_k[k][j]]
            if all(hands[0][Z.HAND_INDEX[pt]] >= n for pt, n in p["att_min"].items()) and all(hands[1][Z.HAND_INDEX[pt]] <= n for pt, n in p["def_max"].items()):
                return p
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plies", type=int, default=9)
    ap.add_argument("--tail", type=int, default=24, help="各対局の最後の何手を詰み局面の候補にするか")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--mode", default="cons", choices=["cons", "exact"])
    a = ap.parse_args()
    global Z, MODE
    MODE = a.mode
    Z = zone if a.mode == "cons" else zone_exact
    t0 = time.time()
    with open(os.path.join(DATA, "games.jsonl")) as f:
        games = [json.loads(l) for l in f]
    random.Random(0).shuffle(games)
    if a.max_games:
        games = games[: a.max_games]
    n_test = len(games) // 10
    test, train = games[:n_test], games[n_test:]

    # 1. 学習側の詰み局面からパターンを作る
    pats, fails = [], collections.Counter()
    seen = set()
    for g in train:
        b = cshogi.Board()
        n = len(g["moves"])
        for i, mv in enumerate(g["moves"]):
            if i >= n - a.tail and not b.is_check():
                m, p = shortest_mate(b, a.plies)
                if m:
                    pcs, hands = normalized(b)
                    key = (tuple(pcs), tuple(hands[0]), tuple(hands[1]))
                    if key not in seen:
                        seen.add(key)
                        try:
                            pats.append(make_pattern(pcs, hands, p, a.plies))
                        except Fail as e:
                            fails[str(e).split(":")[0]] += 1
            b.push_usi(mv)
    print(f"patterns {len(pats)}, fails {dict(fails)}, {time.time() - t0:.0f}s", flush=True)
    lib = Library(pats)

    # 2. 評価側の全局面と照合
    res = collections.Counter()
    by_plies = collections.defaultdict(collections.Counter)
    tm_match = tm_mate = 0.0
    for g in test:
        b = cshogi.Board()
        for mv in g["moves"]:
            if not b.is_check():
                pcs, hands = normalized(b)
                t = time.perf_counter()
                hit = lib.match(pcs, hands) or lib.match(mirror(pcs), hands)
                tm_match += time.perf_counter() - t
                t = time.perf_counter()
                m, p = shortest_mate(b, a.plies)
                tm_mate += time.perf_counter() - t
                res["positions"] += 1
                if m:
                    res["mates"] += 1
                    by_plies[p]["mates"] += 1
                    if hit:
                        res["mates_found"] += 1
                        by_plies[p]["found"] += 1
                if hit:
                    res["hits"] += 1
                    # 証明つきのはずなので、パターンの手数以内で必ず詰むはず
                    if not m or p > hit["plies"] + 0:
                        mm, pp = shortest_mate(b, hit["plies"]) if not m else (m, p)
                        if not mm:
                            res["false_positive"] += 1
            b.push_usi(mv)

    sizes = collections.defaultdict(list)
    for p in pats:
        sizes[p["plies"]].append(len(p["sq"]))
    lines = [f"# X6 9×9 の詰みの関連領域（証明つき）と流用率（mode={a.mode}）", "", f"学習側 {len(train)}局 → パターン {len(pats)}件（作れなかった: {dict(fails)}） / 評価側 {len(test)}局 / 計算 {time.time() - t0:.0f}s", ""]
    exact = collections.defaultdict(list)
    for p in pats:
        exact[p["plies"]].append(p["exact"])
    lines += ["## 関連領域の大きさ（81マス中。制約のあるマス / うち中身を固定したマス）", "| 詰みの手数 | 件数 | 制約のあるマス(平均) | 中身を固定したマス(平均) | 最小 | 最大 |", "|---|---|---|---|---|---|"]
    for k in sorted(sizes):
        s = sizes[k]
        lines.append(f"| {k} | {len(s)} | {np.mean(s):.1f} | {np.mean(exact[k]):.1f} | {min(s)} | {max(s)} |")
    lines += ["", "## 評価側の局面との照合", f"- 照合した局面: {res['positions']:,}（王手されていない局面すべて）", f"- パターンに当てはまった局面: {res['hits']:,}", f"- **うち詰まなかった局面（不具合の検出）: {res['false_positive']}**", f"- 評価側の詰み局面（{a.plies}手以内）: {res['mates']:,} → パターンで見つかった: {res['mates_found']:,}（{res['mates_found'] / max(res['mates'], 1):.1%}）", ""]
    lines += ["| 詰みの手数 | 詰み局面 | パターンで見つかった割合 |", "|---|---|---|"]
    for k in sorted(by_plies):
        c = by_plies[k]
        lines.append(f"| {k} | {c['mates']} | {c['found'] / max(c['mates'], 1):.1%} |")
    lines += ["", f"1局面あたりの時間: パターン照合 {1e6 * tm_match / max(res['positions'], 1):.0f}µs（Python）/ cshogi の詰み探索 {1e6 * tm_mate / max(res['positions'], 1):.0f}µs（C++）"]
    out = os.path.join(HERE, f"../results/x6_mate_zones_{a.mode}.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(DATA, f"mate_patterns_{a.mode}.jsonl"), "w") as f:
        for p in pats:
            f.write(json.dumps({"sq": p["sq"].tolist(), "val": p["val"].tolist(), "allow": p["allow"].tolist(), "att_min": p["att_min"], "def_max": p["def_max"], "plies": p["plies"], "op_k": p["op_k"]}) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
