"""X21: 探索中に実際に評価される局面を集めて NNUE で評価し、教師に足すための局面を作る

X19/X19b で、低ランク評価は「教師にない見慣れない局面」で外れて探索に突かれると分かった。
X17 は無作為な手で局面を作ったが、ここでは自作の探索（深さ2＋静止探索、評価は「固定の駒の価値＋0.5×線形」）が
実際に評価した局面を、確率 --p で抜き出して記録し、NNUE（--nodes ノード）で評価する。中断しても続きから再開できる。

出力: ../data/mainline/searchpos.jsonl（sfen, score_cp）
使い方: python x21_searchpos.py --n 30000 --nodes 20000 --p 0.02
"""

import argparse
import json
import os
import random
import sys
import time

import cshogi

import x7_expressivity as x7
from x7b_qsearch import Evaluator
from x11_match import choose
from x13_mix import Classic, Mix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../corpus"))
from analyze_positions import Usi  # noqa: E402

OUT = os.path.join(x7.DATA, "searchpos.jsonl")


class Logger:
    """評価関数を包んで、評価した局面を確率 p で記録する"""

    def __init__(self, ev, p, rng):
        self.ev, self.p, self.rng = ev, p, rng
        self.seen = []
        self.calls = 0

    def __call__(self, board):
        self.calls += 1
        if self.rng.random() < self.p and not board.is_check():
            self.seen.append(board.sfen().rsplit(" ", 1)[0])
        return self.ev(board)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--p", type=float, default=0.02)
    a = ap.parse_args()
    rng = random.Random(0)
    games = x7.load_games()
    rng.shuffle(games)
    n_test = len(games) // 10
    train = games[n_test:]
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {json.loads(l)["sfen"] for l in f if l.strip()}
    F, y, w = x7.collect(train, 3, random.Random(0), "score")
    lin = Evaluator("M+KP+T", x7.fit("M+KP+T", F, y, w))
    del F
    log = Logger(Mix(Classic(), lin, 0.5), a.p, random.Random(len(done) + 7))
    eng = Usi(1, 64)
    r2 = random.Random(len(done) + 11)
    t0 = time.time()
    with open(OUT, "a") as f:
        while len(done) < a.n:
            g = r2.choice(train)
            if len(g["moves"]) < 30:
                continue
            b = cshogi.Board()
            for mv in g["moves"][: r2.randrange(10, len(g["moves"]) - 10)]:
                b.push_usi(mv)
            if b.is_game_over():
                continue
            log.seen = []
            choose(b, log, 2)  # 探索させて、評価した局面を集める
            for sfen in log.seen:
                if sfen in done or len(done) >= a.n:
                    continue
                r = eng.go(sfen, a.nodes)
                if "score_cp" in r:
                    sc = r["score_cp"]
                elif "mate" in r:
                    sc = -30000 if str(r["mate"]).startswith("-") else 30000
                else:
                    continue
                done.add(sfen)
                f.write(json.dumps({"sfen": sfen, "score_cp": sc}) + "\n")
                f.flush()
                if len(done) % 1000 == 0:
                    print(f"{len(done)}/{a.n} {time.time() - t0:.0f}s", flush=True)
    eng.send("quit")


if __name__ == "__main__":
    main()
