"""X15: 9×9 の将棋の格言を NNUE で採点する（どうぶつしょうぎの X12③ を 9×9 へ）

物差し: 格言どおりの手を指した後の局面を NNUE（水匠5、--nodes ノード）で評価し、
        元の局面の評価値（解析済み、30万ノード）との差（損した点数）を測る。
比較: 無作為に選んだ手 / NNUE 自身の最善手（＝測り方そのものの誤差の目安）
対象: NNUE 解析済みの局面のうち |評価値| < 1500 のもの（勝負がほぼ決まった局面は除く）から --n 局面

格言（当てはまらない局面では数えない。当てはまる手が複数あれば無作為に1つ）:
  高い駒から取る / 王手できるならする（王手は追う手、の検証） / 成れるなら成る /
  浮き駒を作らない（自分の取られそうな駒が最も少ない手） / 相手の浮き駒を増やす /
  玉の周りの金銀を減らさない / 桂馬を跳ねる（桂の高跳び歩の餌食、の検証） / 持ち駒の歩を打つ

使い方: python x15_proverbs9.py --n 1500 --nodes 100000
"""

import argparse
import collections
import json
import os
import random
import sys
import time

import cshogi

import zone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../corpus"))
from analyze_positions import Usi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/mainline")
VAL = [0, 1, 3, 4, 5, 8, 10, 6, 0, 6, 6, 6, 6, 10, 12]


def enprise(board, color):
    """color 側の駒（玉以外）のうち、相手の利きの数 > 自分の守りの数 のもの"""
    c = zone.Ctx(board.pieces, board.pieces_in_hand, board.turn, {}, set())
    n = 0
    for sq, p in enumerate(board.pieces):
        if p and ((p >= 16) == (color == 1)) and (p & 15) != 8:
            if len(c.attackers(sq, 1 - color)) > len(c.attackers(sq, color, exclude_king=True)):
                n += 1
    return n


def guards(board, color):
    k = board.king_square(color)
    kf, kr = divmod(k, 9)
    n = 0
    for sq, p in enumerate(board.pieces):
        if p and ((p >= 16) == (color == 1)) and (p & 15) in (4, 7, 9, 10, 11, 12):
            f, r = divmod(sq, 9)
            if max(abs(f - kf), abs(r - kr)) <= 1:
                n += 1
    return n


def candidates(board):
    me = board.turn
    moves = list(board.legal_moves)
    info = []
    g0 = guards(board, me)
    for m in moves:
        to = cshogi.move_to(m)
        drop = cshogi.move_is_drop(m)
        cap = 0 if drop else (board.piece(to) & 15)
        pt = cshogi.move_drop_hand_piece(m) if drop else (board.piece(cshogi.move_from(m)) & 15)
        board.push(m)
        chk = board.is_check()
        mine_hang = enprise(board, me)
        their_hang = enprise(board, 1 - me)
        g1 = guards(board, me)
        board.pop()
        info.append({"m": m, "cap": VAL[cap], "check": chk, "promo": cshogi.move_is_promotion(m), "mine_hang": mine_hang, "their_hang": their_hang, "guard_kept": g1 >= g0, "knight_jump": (not drop) and pt == 3, "pawn_drop": drop and pt == 0})
    rules = {}
    caps = [x for x in info if x["cap"] > 0]
    if caps:
        hv = max(x["cap"] for x in caps)
        rules["高い駒から取る"] = [x["m"] for x in caps if x["cap"] == hv]
    if any(x["check"] for x in info):
        rules["王手できるならする"] = [x["m"] for x in info if x["check"]]
    if any(x["promo"] for x in info):
        rules["成れるなら成る"] = [x["m"] for x in info if x["promo"]]
    lo = min(x["mine_hang"] for x in info)
    rules["浮き駒を作らない"] = [x["m"] for x in info if x["mine_hang"] == lo]
    hi = max(x["their_hang"] for x in info)
    if hi > 0:
        rules["相手の浮き駒を増やす"] = [x["m"] for x in info if x["their_hang"] == hi]
    kept = [x["m"] for x in info if x["guard_kept"]]
    if kept and len(kept) < len(info):
        rules["玉の周りの金銀を減らさない"] = kept
    if any(x["knight_jump"] for x in info):
        rules["桂馬を跳ねる"] = [x["m"] for x in info if x["knight_jump"]]
    if any(x["pawn_drop"] for x in info):
        rules["持ち駒の歩を打つ"] = [x["m"] for x in info if x["pawn_drop"]]
    return moves, rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--nodes", type=int, default=100000)
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x15_proverbs9.jsonl"))
    a = ap.parse_args()
    nn = []
    for k in [""] + [f".{i}" for i in range(3)]:
        p = os.path.join(DATA, "nnue_300k.jsonl" + k)
        if os.path.exists(p):
            with open(p) as f:
                nn += [json.loads(l) for l in f if l.strip()]
    nn = [r for r in nn if "score_cp" in r and abs(r["score_cp"]) < 1500 and r.get("bestmove") not in ("resign", "win")]
    rng = random.Random(3)
    rng.shuffle(nn)
    eng = Usi(1, 64)
    cache = {}

    def value_after(b, m):
        """m を指した後の局面の評価（指した側から見た点数）"""
        b.push(m)
        key = b.sfen().rsplit(" ", 1)[0]
        if key not in cache:
            if b.is_game_over():
                cache[key] = 30000
            else:
                r = eng.go(key, a.nodes)
                cache[key] = -(r["score_cp"] if "score_cp" in r else (30000 if not str(r.get("mate", "")).startswith("-") else -30000))
        b.pop()
        return cache[key]

    loss = collections.defaultdict(list)
    t0 = time.time()
    for i, r in enumerate(nn[: a.n]):
        b = cshogi.Board(r["sfen"] + " 1")
        moves, rules = candidates(b)
        best = b.move_from_usi(r["bestmove"])
        root = r["score_cp"]  # 元の局面の評価（手番側から見た値、30万ノード）
        loss["NNUEの最善手（測り方の誤差）"].append(max(0, root - value_after(b, best)))
        for name, ms in list(rules.items()) + [("無作為な手", moves)]:
            m = rng.choice(ms)
            loss[name].append(max(0, root - value_after(b, m)))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{a.n} {time.time() - t0:.0f}s", flush=True)
    res = {}
    for name, ls in loss.items():
        if name.startswith("_"):
            continue
        ls = [min(x, 3000) for x in ls]
        res[name] = {"n": len(ls), "mean_loss": round(sum(ls) / len(ls), 1), "blunder300": round(sum(x >= 300 for x in ls) / len(ls), 4), "best_or_close50": round(sum(x <= 50 for x in ls) / len(ls), 4)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    with open(a.out, "w") as f:
        json.dump({"n": a.n, "nodes": a.nodes, "results": res}, f, ensure_ascii=False, indent=1)
    eng.send("quit")


if __name__ == "__main__":
    main()
