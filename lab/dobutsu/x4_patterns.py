"""X4: 証明済みの局所パターン（勝ちを保証する形）を真値DBから抜き出し、評価関数の誤りを減らせるか

パターン = 一部のマス（＋持ち駒）だけを指定した部分局面。
「DBにある到達可能局面のうち、そのパターンに当てはまるものは全て X（勝ち or 負け）」が確認できたものだけを採用する。
→ DB全体を調べて確かめているので、到達可能局面の範囲では証明済みの命題になる。

作り方（貪欲な一般化。囲碁の Relevance Zone の考え方を、打ち駒のあるゲームに持ち込む）:
  1. 局面の全マス・全持ち駒を固定した状態から始める（両ライオンのマスは常に固定）
  2. ライオンから遠いマスから順に「固定を外す（何があってもよい）」を試す
  3. 外しても当てはまる局面が全て同じ結論なら外したままにする
  4. 最後に持ち駒の欄も同様に外せるか試す

実験:
  - 学習側の局面から、負けパターン（＝そこへ指せば勝てる形）の辞書を作る
  - 評価側（X1/X1b と同じ局面）で「辞書に当てはまる手があればそれを指し、なければ KPP の手」を指したときの誤り率を測る
  - パターンは証明済みなので、当てはまった手が間違うことはない（誤りが増えることはない）
  - 勝つまでの手数ごとの、パターンの大きさ（固定マス数）と流用率も記録する

使い方: python x4_patterns.py --lib-parents 20000 --train 3000000 --test 300000
"""

import argparse
import collections
import json
import os
import time

import numpy as np

from x1_expressivity import HERE, Model, cls, run_sampler

FULL_SQ = [np.uint64(15) << np.uint64(4 * i) for i in range(12)]
FULL_HAND = [np.uint64(3) << np.uint64(48 + 2 * j) for j in range(6)]


def mirror_sq(i):
    r, c = divmod(i, 3)
    return r * 3 + (2 - c)


def mirror_key(k):
    k = int(k)
    r = k & ~((1 << 48) - 1)
    for i in range(12):
        r |= ((k >> (4 * i)) & 15) << (4 * mirror_sq(i))
    return r


def lions(k):
    k = int(k)
    my = op = 0
    for i in range(12):
        v = (k >> (4 * i)) & 15
        if v == 1:
            my = i
        elif v == 9:
            op = i
    return my, op


class DB:
    def __init__(self, data):
        self.keys = np.memmap(os.path.join(data, "plain_keys.bin"), dtype=np.uint64, mode="r")
        self.vals = np.memmap(os.path.join(data, "plain_vals.bin"), dtype=np.uint8, mode="r")
        self.off = np.fromfile(os.path.join(data, "groups.bin"), dtype=np.uint64).astype(np.int64)
        self.cache = {}

    def group(self, my, op):
        g = my * 12 + op
        if g not in self.cache:
            if len(self.cache) > 40:  # 使うグループだけを手元に置く（メモリ節約）
                self.cache.pop(next(iter(self.cache)))
            s, e = self.off[g], self.off[g + 1]
            self.cache[g] = (np.asarray(self.keys[s:e]), np.asarray(self.vals[s:e]))
        return self.cache[g]

    def all_in_class(self, mask, pat, my, op, target):
        """パターン（と左右反転）に当てはまる局面が全て target クラスか。当てはまる件数も返す"""
        n = 0
        for (m, p, a, b) in ((mask, pat, my, op), (mirror_key(mask), mirror_key(pat), mirror_sq(my), mirror_sq(op))):
            keys, vals = self.group(a, b)
            hit = (keys & np.uint64(m)) == np.uint64(p)
            if not hit.any():
                continue
            if (cls(vals[hit]) != target).any():
                return False, 0
            n += int(hit.sum())
        return True, n


def generalize(db, key, target):
    key = int(key)
    my, op = lions(key)
    mask = (1 << 60) - 1
    order = sorted((i for i in range(12) if i not in (my, op)), key=lambda i: -min(abs(i // 3 - op // 3) + abs(i % 3 - op % 3), abs(i // 3 - my // 3) + abs(i % 3 - my % 3)))
    fields = [int(FULL_SQ[i]) for i in order] + [int(h) for h in FULL_HAND]
    cover = 1
    for f in fields:
        m2 = mask & ~f
        ok, n = db.all_in_class(m2, key & m2, my, op, target)
        if ok:
            mask, cover = m2, n
    fixed_sq = sum(1 for i in range(12) if mask & int(FULL_SQ[i]))
    fixed_hand = sum(1 for j in range(6) if mask & int(FULL_HAND[j]))
    return {"mask": mask, "pat": key & mask, "my": my, "op": op, "fixed_sq": fixed_sq, "fixed_hand": fixed_hand, "cover": cover}


class Library:
    def __init__(self, pats):
        self.by_group = collections.defaultdict(list)
        for p in pats:
            for (m, q, a, b) in ((p["mask"], p["pat"], p["my"], p["op"]), (mirror_key(p["mask"]), mirror_key(p["pat"]), mirror_sq(p["my"]), mirror_sq(p["op"]))):
                self.by_group[a * 12 + b].append((m, q))
        self.arr = {g: (np.array([m for m, _ in v], dtype=np.uint64), np.array([q for _, q in v], dtype=np.uint64)) for g, v in self.by_group.items()}

    def match(self, key):
        my, op = lions(key)
        a = self.arr.get(my * 12 + op)
        if a is None:
            return False
        return bool(((np.uint64(int(key)) & a[0]) == a[1]).any())


def bucket_d(d):
    for hi in (5, 15, 25, 45, 80):
        if d <= hi:
            return f"<={hi}"
    return ">80"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--lib-parents", type=int, default=20000)
    ap.add_argument("--train", type=int, default=3000000)
    ap.add_argument("--test", type=int, default=300000)
    a = ap.parse_args()
    db = DB(a.data)
    t0 = time.time()

    # 1. 辞書づくり（評価側とは別の乱数で抜いた局面から）
    lkeys, lvals, lkids = run_sampler(a.data, a.lib_parents, 3, True)
    pats, stats = [], collections.defaultdict(list)
    lc = cls(lvals)
    for i, kd in enumerate(lkids):
        if lc[i] != 1:
            continue
        cc = cls(kd["val"])
        good = cc == -1
        if good.all():
            continue
        for j in np.where(good)[0]:
            p = generalize(db, kd["key"][j], -1)
            p["d"] = int(kd["val"][j]) - 1
            pats.append(p)
            stats[bucket_d(p["d"])].append(p)
    print(f"library: {len(pats)} patterns from {a.lib_parents} parents, {time.time() - t0:.0f}s", flush=True)
    lib = Library(pats)
    with open(os.path.join(a.data, "patterns.jsonl"), "w") as f:
        for p in pats:
            f.write(json.dumps(p) + "\n")

    # 2. 評価（X1/X1b と同じ評価局面）
    keys, vals, _ = run_sampler(a.data, a.train, 1, False)
    keep = vals != 2
    kpp = Model("KPP").fit(keys[keep], vals[keep])
    tkeys, tvals, tkids = run_sampler(a.data, a.test, 2, True)
    flat = np.concatenate([k["key"] for k in tkids])
    flat_v = np.concatenate([k["val"] for k in tkids])
    off = np.cumsum([0] + [len(k) for k in tkids])
    cp = kpp.p_win(flat)
    cp[flat_v == 2] = 1.0
    tc = cls(tvals)
    res = collections.Counter()
    by_d = collections.defaultdict(collections.Counter)
    for i, kd in enumerate(tkids):
        if tc[i] != 1:
            continue
        good = cls(kd["val"]) == -1
        if good.all():
            continue
        b = bucket_d(int(tvals[i]) - 1)
        hit = [j for j in range(len(kd)) if kd["val"][j] != 2 and lib.match(kd["key"][j])]
        kpp_ok = bool(good[int(np.argmin(cp[off[i] : off[i + 1]]))])
        lib_ok = bool(good[hit[0]]) if hit else None
        combo_ok = lib_ok if hit else kpp_ok
        for c in (res, by_d[b]):
            c["n"] += 1
            c["kpp_err"] += not kpp_ok
            c["combo_err"] += not combo_ok
            c["lib_hit"] += bool(hit)
            c["lib_wrong"] += bool(hit) and not lib_ok  # 証明済みなので 0 のはず
            c["rescued"] += (not kpp_ok) and bool(hit)

    lines = ["# X4 証明済みパターン辞書の効果", "", f"辞書: {len(pats)} パターン（{a.lib_parents} 局面から） / 計算 {time.time() - t0:.0f}s", ""]
    lines += ["## パターンの大きさ（勝つまでの手数ごと）", "| 手数 | 件数 | 固定マス数(平均) | 固定持ち駒欄(平均) | 当てはまる局面数(中央値) |", "|---|---|---|---|---|"]
    for b in ("<=5", "<=15", "<=25", "<=45", "<=80", ">80"):
        ps = stats.get(b, [])
        if ps:
            lines.append(f"| {b} | {len(ps)} | {np.mean([p['fixed_sq'] for p in ps]):.1f} | {np.mean([p['fixed_hand'] for p in ps]):.1f} | {int(np.median([p['cover'] for p in ps]))} |")
    lines += ["", "## 評価局面での効果（勝ち局面・どの手でもよいわけではないもの）", "| 手数 | 件数 | KPP誤り | 辞書+KPP誤り | 辞書が当たった率 | KPPの誤りを救った数 | 辞書の誤り |", "|---|---|---|---|---|---|---|"]
    for name, c in [("全体", res)] + [(b, by_d[b]) for b in ("<=5", "<=15", "<=25", "<=45", "<=80", ">80") if b in by_d]:
        n = max(c["n"], 1)
        lines.append(f"| {name} | {c['n']} | {c['kpp_err'] / n:.1%} | {c['combo_err'] / n:.1%} | {c['lib_hit'] / n:.1%} | {c['rescued']} | {c['lib_wrong']} |")
    out = os.path.join(HERE, "../results/x4_patterns.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
