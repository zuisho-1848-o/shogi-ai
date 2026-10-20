"""X13: 深く読むと学習した評価が駒数だけに負ける（X11）原因を調べる

最初の版（駒数だけの学習評価との凸結合）で分かったこと: 学習した評価は駒の価値をほとんど知らない
（強豪AI同士の対局では駒の損得がまれで、教師の評価値が互角付近に集中するため。飛車1枚で −0.11 程度、歩は符号が逆）。
そこで基準を「固定の駒の価値」（歩1 香3 桂4 銀5 金6 角8 飛10、と6 成香6 成桂6 成銀6 馬10 龍12、1点=100）に替える。

評価 = tanh(固定の駒の価値の差×100/1200) + λ・学習した評価（M+KP+T、線形）
λ を振って、固定の駒の価値だけ（λ=0）と深さ --depth で対局させる。
あわせて対局の終わり方（詰み/千日手/上限手数での判定）・手数・終局時の駒の差（λ側から見て）を記録する。

使い方: python x13_mix.py --lams 0.25,0.5,0.75,1.0 --n 30 --depth 2
"""

import argparse
import collections
import json
import os
import random
import sys
import time

import cshogi
import numpy as np

import x7_expressivity as x7
from x7b_qsearch import Evaluator
from x11_match import choose

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../corpus"))
from analyze_positions import Usi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PV = [0, 1, 3, 4, 5, 8, 10, 6, 0, 6, 6, 6, 6, 10, 12]


class Classic:
    """固定の駒の価値（手番側から見た値）"""

    def __init__(self):
        self.calls = 0

    def __call__(self, board):
        self.calls += 1
        return float(np.tanh(material_balance(board, board.turn) * 100 / 1200))


class Mix:
    def __init__(self, a, b, lam):
        self.a, self.b, self.lam = a, b, lam
        self.calls = 0

    def __call__(self, board):
        self.calls += 1
        if self.lam == 0:
            return self.a(board)
        return self.a(board) + self.lam * self.b(board)


def material_balance(board, color):
    """color から見た駒の価値の差（盤上＋持ち駒、玉は除く）"""
    s = 0
    for p in board.pieces:
        if p and (p & 15) != 8:
            v = PV[p & 15]
            s += v if ((p >= 16) == (color == 1)) else -v
    hb, hw = board.pieces_in_hand
    hv = [1, 3, 4, 5, 6, 8, 10]
    s += sum(c * v for c, v in zip(hb, hv)) * (1 if color == 0 else -1)
    s += sum(c * v for c, v in zip(hw, hv)) * (1 if color == 1 else -1)
    return s


def play(sfen, evs, depth, max_plies, judge):
    """evs[0] が先手番側。戻り値: (結果 1/-1/0, 終わり方, 手数, 終局時の evs[0] から見た駒の差)"""
    b = cshogi.Board(sfen + " 1")
    first = b.turn
    for ply in range(max_plies):
        if b.is_game_over():
            return (1 if b.turn != first else -1), "詰み", ply, material_balance(b, first)
        if b.is_draw() == cshogi.REPETITION_DRAW:
            return 0, "千日手", ply, material_balance(b, first)
        side = 0 if b.turn == first else 1
        m = choose(b, evs[side], depth)
        if m is None:
            return (-1 if side == 0 else 1), "指し手なし", ply, material_balance(b, first)
        b.push(m)
    r = judge.go(b.sfen().rsplit(" ", 1)[0], 100000)
    sc = r.get("score_cp", 0) if "mate" not in r else (30000 if not str(r["mate"]).startswith("-") else -30000)
    mb = material_balance(b, first)
    if abs(sc) < 300:
        return 0, "判定（互角）", max_plies, mb
    return (1 if ((sc > 0) == (b.turn == first)) else -1), "判定", max_plies, mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", default="0.5,1,2,4")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--learned", default="M+KP+T")
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x13_mix.jsonl"))
    a = ap.parse_args()
    rng = random.Random(0)
    games = x7.load_games()
    rng.shuffle(games)
    n_test = len(games) // 10
    Ftr, ytr, wtr = x7.collect(games[n_test:], a.every, rng, "score")
    mat = Classic()
    learned = Evaluator(a.learned, x7.fit(a.learned, Ftr, ytr, wtr))
    del Ftr
    test_files = {g["file"] for g in games[:n_test]}
    with open(os.path.join(x7.DATA, "exits.jsonl")) as f:
        starts = [json.loads(l) for l in f]
    starts = [s for s in starts if s["file"] in test_files and 20 <= s["ply"] <= 60]
    random.Random(2).shuffle(starts)
    starts = starts[: a.n]
    judge = Usi(1, 64)
    for lam in [float(x) for x in a.lams.split(",")]:
        ev = Mix(mat, learned, lam)
        score = collections.Counter()
        ends = collections.Counter()
        plies, mb_win, mb_loss = [], [], []
        t0 = time.time()
        for s in starts:
            for swap in (False, True):
                pair = [mat, ev] if swap else [ev, mat]
                r, how, ply, mb = play(s["sfen"], pair, a.depth, a.max_plies, judge)
                if swap:
                    r, mb = -r, -mb  # λ 側から見た値に直す
                score["win" if r > 0 else "loss" if r < 0 else "draw"] += 1
                ends[how] += 1
                plies.append(ply)
                (mb_win if r > 0 else mb_loss if r < 0 else []).append(mb)
        n = sum(score.values())
        res = {
            "lam": lam, "learned": a.learned, "depth": a.depth, "games": n, **score,
            "score_rate": round((score["win"] + 0.5 * score["draw"]) / max(n, 1), 4),
            "ends": dict(ends), "mean_plies": round(float(np.mean(plies)), 1),
            "material_at_end_when_lam_loses": round(float(np.mean(mb_loss)), 1) if mb_loss else None,
            "material_at_end_when_lam_wins": round(float(np.mean(mb_win)), 1) if mb_win else None,
            "sec": round(time.time() - t0),
        }
        print(json.dumps(res, ensure_ascii=False), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    judge.send("quit")


if __name__ == "__main__":
    main()
