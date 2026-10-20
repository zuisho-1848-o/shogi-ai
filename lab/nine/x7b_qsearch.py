"""X7b: 静止探索（駒の取り合いだけを読む）を入れて、評価関数ごとの指し手の質を測る

X7 の 1手読みでは、取り返される駒取りを正しく評価できず「駒数だけ」が最善手の一致で一番になった。
ここでは各候補手の後に、駒を取る手（王手されていれば全ての応手）だけを深さ --qdepth まで読んでから評価する。

指標: NNUE（水匠5、30万ノード）の最善手と一致する率。X7 と同じ解析済み局面（|評価値| < 1500）から --n 件。

使い方: python x7b_qsearch.py --n 2000 --qdepth 4
"""

import argparse
import json
import os
import random
import time

import cshogi
import numpy as np

import x7_expressivity as x7

PIECE_VALUE = [0, 1, 3, 4, 5, 8, 10, 6, 100, 6, 6, 6, 6, 10, 12]  # 駒の種類(1..14)ごとの取る順の目安


class Evaluator:
    def __init__(self, kind, model):
        self.kind, self.m = kind, model
        if kind != "GBDT":
            self.w = model.coef_.astype(np.float64)
            self.b = float(np.ravel(model.intercept_)[0])
        self.calls = 0

    def __call__(self, board):
        self.calls += 1
        p, kp, t, d = x7.features(board)
        if self.kind == "GBDT":
            return float(self.m.predict(d[None, :])[0])
        kind, w, v = self.kind, self.w, self.b
        if kind.startswith("M+"):  # 先頭42次元は駒数/4
            v += float(d[:42] @ w[:42]) / 4
            kind, w = kind[2:], w[42:]
        if kind == "material":
            return v + float(d[:42] @ w)
        idx = {"P": p, "T": t}.get(kind)
        if idx is None:  # KP / KP+T
            idx = p + [x7.P_DIM + i for i in kp]
            if kind == "KP+T":
                idx += [x7.P_DIM + x7.KP_DIM + i for i in t]
        return v + float(w[idx].sum())


def qsearch(board, ev, alpha, beta, depth):
    in_check = board.is_check()
    if in_check:
        moves = list(board.legal_moves)
        if not moves:
            return -1e6  # 詰み（評価値の範囲より十分大きく）
    else:
        stand = ev(board)
        if depth == 0 or stand >= beta:
            return stand
        alpha = max(alpha, stand)
        moves = [m for m in board.legal_moves if not cshogi.move_is_drop(m) and board.piece(cshogi.move_to(m))]
        moves.sort(key=lambda m: -PIECE_VALUE[board.piece(cshogi.move_to(m)) & 15])
    if depth == 0:
        return ev(board)
    for m in moves:
        board.push(m)
        s = -qsearch(board, ev, -beta, -alpha, depth - 1)
        board.pop()
        if s >= beta:
            return s
        alpha = max(alpha, s)
    return alpha


def choose(board, ev, qdepth):
    best, best_m = -2.0, None
    for m in board.legal_moves:
        board.push(m)
        s = 1.0 if board.is_game_over() else -qsearch(board, ev, -2.0, -best, qdepth)
        board.pop()
        if s > best:
            best, best_m = s, m
    return best_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--qdepth", type=int, default=4)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--kinds", default="material,P,KP,KP+T,GBDT")
    ap.add_argument("--out", default=os.path.join(x7.HERE, "../results/x7b_qsearch.jsonl"))
    a = ap.parse_args()
    rng = random.Random(0)
    games = x7.load_games()
    rng.shuffle(games)
    n_test = len(games) // 10
    Ftr, ytr, wtr = x7.collect(games[n_test:], a.every, rng, "score")
    nn = []
    for k in [""] + [f".{i}" for i in range(3)]:
        path = os.path.join(x7.DATA, "nnue_300k.jsonl" + k)
        if os.path.exists(path):
            with open(path) as f:
                nn += [json.loads(l) for l in f if l.strip()]
    nn = [r for r in nn if "score_cp" in r and abs(r["score_cp"]) < 1500 and r.get("bestmove") not in ("resign", "win")]
    random.Random(1).shuffle(nn)
    nn = nn[: a.n]
    for kind in a.kinds.split(","):
        t = time.time()
        ev = Evaluator(kind, x7.fit(kind, Ftr, ytr, wtr))
        ok = {"cap": [0, 0], "quiet": [0, 0]}
        for r in nn:
            b = cshogi.Board(r["sfen"] + " 1")
            bm = b.move_from_usi(r["bestmove"])
            kindm = "cap" if not cshogi.move_is_drop(bm) and b.piece(cshogi.move_to(bm)) else "quiet"
            m = choose(b, ev, a.qdepth)
            ok[kindm][0] += 1
            ok[kindm][1] += m is not None and cshogi.move_to_usi(m) == r["bestmove"]
        total = (ok["cap"][1] + ok["quiet"][1]) / len(nn)
        res = {"kind": kind, "qdepth": a.qdepth, "n": len(nn), "agree_nnue": round(total, 4), "agree_capture": round(ok["cap"][1] / max(ok["cap"][0], 1), 4), "agree_quiet": round(ok["quiet"][1] / max(ok["quiet"][0], 1), 4), "evals_per_pos": round(ev.calls / len(nn)), "sec": round(time.time() - t)}
        print(json.dumps(res), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
