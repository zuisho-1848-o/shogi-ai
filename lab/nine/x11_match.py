"""X11: 9×9 で評価関数どうしを対局させる（同じ探索・同じ深さで、評価関数だけを差し替える）

探索: αβ（深さ --depth、駒を取る手→王手→その他の順に並べる）＋ 静止探索（駒を取る手だけ、深さ4）
評価関数: X7 と同じ学習（教師=棋譜のエンジン評価値）
  material : 駒数だけ
  M+KP+T   : 駒数＋玉×駒×位置＋脅威（線形）
  GBDT     : 駒数＋脅威の約60個の数値の決定木
開始局面: 強豪AI同士の本譜で定跡が尽きた局面（exits.jsonl。手数20〜60）から --n 局面。先後を入れ替えて2局ずつ
判定: 詰み・千日手（cshogi）。上限 --max-plies 手で終わらなければ、やねうら王 NNUE（10万ノード）の評価値で ±300 を超えた側の勝ち、それ以外は引き分け

使い方: python x11_match.py --pair material,GBDT --n 50 --depth 2
"""

import argparse
import json
import os
import random
import sys
import time

import cshogi

import x7_expressivity as x7
from x7b_qsearch import PIECE_VALUE, Evaluator, qsearch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../corpus"))
from analyze_positions import Usi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INF = 2.0


def ordered(board):
    caps, checks, rest = [], [], []
    for m in board.legal_moves:
        to = cshogi.move_to(m)
        if not cshogi.move_is_drop(m) and board.piece(to):
            caps.append((-PIECE_VALUE[board.piece(to) & 15], m))
        else:
            rest.append(m)
    caps.sort(key=lambda x: x[0])
    return [m for _, m in caps] + rest


def search(board, ev, depth, alpha, beta):
    if board.is_draw() in (cshogi.REPETITION_DRAW,):
        return 0.0
    moves = ordered(board)
    if not moves:
        return -1.0
    if depth == 0:
        return qsearch(board, ev, alpha, beta, 4)
    best = -INF
    for m in moves:
        board.push(m)
        s = -search(board, ev, depth - 1, -beta, -alpha)
        board.pop()
        if s > best:
            best = s
        if s > alpha:
            alpha = s
        if alpha >= beta:
            break
    return best


def choose(board, ev, depth):
    m1 = board.mate_move_in_1ply()
    if m1:
        return m1
    best, best_m = -INF, None
    alpha = -INF
    for m in ordered(board):
        board.push(m)
        s = -search(board, ev, depth - 1, -INF, -alpha)
        board.pop()
        if s > best:
            best, best_m = s, m
        alpha = max(alpha, s)
    return best_m


def play(sfen, evs, depth, max_plies, judge):
    """evs[0] が先手番側。戻り値: 1 = evs[0] の勝ち, -1 = 負け, 0 = 引き分け"""
    b = cshogi.Board(sfen + " 1")
    first = b.turn
    for ply in range(max_plies):
        if b.is_game_over():
            loser = b.turn
            return 1 if loser != first else -1
        if b.is_draw() == cshogi.REPETITION_DRAW:
            return 0
        side = 0 if b.turn == first else 1
        m = choose(b, evs[side], depth)
        if m is None:
            return -1 if side == 0 else 1
        b.push(m)
    r = judge.go(b.sfen().rsplit(" ", 1)[0], 100000)
    sc = r.get("score_cp", 0) if "mate" not in r else (30000 if not str(r["mate"]).startswith("-") else -30000)
    if abs(sc) < 300:
        return 0
    stm_wins = sc > 0
    return 1 if (stm_wins == (b.turn == first)) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="material,GBDT")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x11_match.jsonl"))
    a = ap.parse_args()
    kinds = a.pair.split(",")
    rng = random.Random(0)
    games = x7.load_games()
    rng.shuffle(games)
    n_test = len(games) // 10
    Ftr, ytr, wtr = x7.collect(games[n_test:], a.every, rng, "score")
    evs = [Evaluator(k, x7.fit(k, Ftr, ytr, wtr)) for k in kinds]
    # 開始局面: 評価側の対局の「定跡を外れた局面」から（学習に使っていない対局）
    test_files = {g["file"] for g in games[:n_test]}
    with open(os.path.join(x7.DATA, "exits.jsonl")) as f:
        starts = [json.loads(l) for l in f]
    starts = [s for s in starts if s["file"] in test_files and 20 <= s["ply"] <= 60]
    random.Random(2).shuffle(starts)
    starts = starts[: a.n]
    judge = Usi(1, 64)
    score = {"win": 0, "loss": 0, "draw": 0}
    t0 = time.time()
    for i, s in enumerate(starts):
        for swap in (False, True):
            pair = [evs[1], evs[0]] if swap else evs
            r = play(s["sfen"], pair, a.depth, a.max_plies, judge)
            r = -r if swap else r  # kinds[0] から見た結果
            score["win" if r > 0 else "loss" if r < 0 else "draw"] += 1
        print(f"{i + 1}/{len(starts)} {kinds[0]} vs {kinds[1]}: {score} {time.time() - t0:.0f}s", flush=True)
    n = sum(score.values())
    res = {"pair": kinds, "depth": a.depth, "games": n, **score, "score_rate": round((score["win"] + 0.5 * score["draw"]) / max(n, 1), 4), "sec": round(time.time() - t0)}
    print(json.dumps(res), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    judge.send("quit")


if __name__ == "__main__":
    main()
