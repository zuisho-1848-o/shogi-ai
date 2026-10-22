"""USI エンジンどうしを対局させる（研究用の C++ エンジンと、やねうら王など）

開始局面: 強豪AI同士の本譜で定跡を外れた局面（exits.jsonl、手数20〜60、評価側の対局）から --n 局面。先後を入れ替えて2局ずつ。
終局: 詰み・千日手（cshogi で判定）、投了。上限 --max-plies 手で終わらなければ NNUE（10万ノード）で ±300 を超えた側の勝ち。

エンジンの指定（--a / --b）:
  nonnn:LAMBDA        研究用エンジン（固定の駒の価値 + LAMBDA×線形。LAMBDA=0 なら固定の駒の価値だけ）
  nonnn2:LAMBDA       同じ評価で、探索に枝刈り（null move / LMR / futility）と評価値の使い回しを足した版
  material            やねうら王 Material 版
  nnue                やねうら王 NNUE 版（水匠5）
時間: --a-ms / --b-ms（1手あたりのミリ秒。go movetime）

使い方: python match.py --a nonnn:0.5 --b nonnn:0 --a-ms 200 --b-ms 200 --n 30
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

import cshogi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(HERE, "../nine"))
import x7_expressivity as x7  # noqa: E402


class Engine:
    def __init__(self, spec):
        self.spec = spec
        if spec.startswith("nonnn"):
            lam = spec.split(":")[1] if ":" in spec else "0.5"
            cmd = [os.path.join(HERE, "build/" + spec.split(":")[0])]  # nonnn / nonnn2（改良版）
            opts = {"EvalFile": os.path.join(HERE, "../data/engine/weights_mkpt.bin"), "Lambda": lam, "UseLinear": "true" if float(lam) != 0 else "false"}
            cwd = HERE
        elif spec == "material":
            cmd = [os.path.join(ROOT, "engines/yaneuraou/YaneuraOu-Material-ARM64")]
            opts = {"Threads": 1, "USI_Hash": 64, "BookFile": "no_book", "NetworkDelay": 0, "NetworkDelay2": 0, "MinimumThinkingTime": 0}
            cwd = os.path.dirname(cmd[0])
        elif spec == "nnue":
            cmd = [os.path.join(ROOT, "engines/yaneuraou-std/yaneuraou")]
            opts = {"Threads": 1, "USI_Hash": 64, "BookFile": "no_book", "NetworkDelay": 0, "NetworkDelay2": 0, "MinimumThinkingTime": 0, "EvalDir": os.path.join(ROOT, "engines/yaneuraou-std/eval"), "FV_SCALE": 24}
            cwd = os.path.dirname(cmd[0])
        else:
            raise ValueError(spec)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=cwd)
        self.send("usi")
        self.wait("usiok")
        for k, v in opts.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, token):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.spec} died")
            lines.append(line.strip())
            if line.startswith(token):
                return lines

    def new_game(self):
        self.send("usinewgame")

    def go(self, start, moves, ms, nodes=None):
        self.send(f"position sfen {start} moves {' '.join(moves)}" if moves else f"position sfen {start}")
        self.send(f"go nodes {nodes}" if nodes else f"go movetime {ms}")
        lines = self.wait("bestmove")
        depth = 0
        for l in lines:
            t = l.split()
            if t[:1] == ["info"] and "depth" in t:
                depth = int(t[t.index("depth") + 1])
        return lines[-1].split()[1], depth

    def quit(self):
        try:
            self.send("quit")
        except Exception:
            pass


def judge_score(judge, sfen):
    judge.send(f"position sfen {sfen}")
    judge.send("go nodes 100000")
    lines = judge.wait("bestmove")
    for l in reversed(lines):
        t = l.split()
        if "score" in t:
            i = t.index("score")
            if t[i + 1] == "cp":
                return int(t[i + 2])
            return 30000 if not t[i + 2].startswith("-") else -30000
    return 0


def play(start, eng, ms, max_plies, judge, depths):
    """eng[0] が開始局面の手番側。戻り値: 1 = eng[0] の勝ち, -1 = 負け, 0 = 引き分け、終わり方、手数"""
    b = cshogi.Board(start)
    first = b.turn
    moves = []
    for ply in range(max_plies):
        if b.is_game_over():
            return (1 if b.turn != first else -1), "詰み", ply
        if b.is_draw() == cshogi.REPETITION_DRAW:
            return 0, "千日手", ply
        side = 0 if b.turn == first else 1
        mv, d = eng[side].go(start, moves, ms[side])
        depths[side].append(d)
        if mv in ("resign",):
            return (-1 if side == 0 else 1), "投了", ply
        if mv == "win":
            return (1 if side == 0 else -1), "宣言勝ち", ply
        try:
            b.push_usi(mv)
        except Exception:
            return (-1 if side == 0 else 1), "不正な手", ply
        moves.append(mv)
    sc = judge_score(judge, b.sfen())
    if abs(sc) < 300:
        return 0, "判定（互角）", max_plies
    return (1 if ((sc > 0) == (b.turn == first)) else -1), "判定", max_plies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-ms", type=int, default=200)
    ap.add_argument("--b-ms", type=int, default=200)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(HERE, "../results/engine_match.jsonl"))
    a = ap.parse_args()
    rng = random.Random(0)
    games = x7.load_games()
    rng.shuffle(games)
    n_test = len(games) // 10
    test_files = {g["file"] for g in games[:n_test]}
    with open(os.path.join(x7.DATA, "exits.jsonl")) as f:
        starts = [json.loads(l) for l in f]
    starts = [s for s in starts if s["file"] in test_files and 20 <= s["ply"] <= 60]
    random.Random(2).shuffle(starts)
    starts = starts[: a.n]
    A, B = Engine(a.a), Engine(a.b)
    judge = Engine("nnue")
    score = {"win": 0, "loss": 0, "draw": 0}
    ends = {}
    depths = {0: [], 1: []}
    t0 = time.time()
    for i, s in enumerate(starts):
        start = s["sfen"] + " 1"
        for swap in (False, True):
            A.new_game()
            B.new_game()
            eng = [B, A] if swap else [A, B]
            ms = [a.b_ms, a.a_ms] if swap else [a.a_ms, a.b_ms]
            dd = {0: [], 1: []}
            r, how, ply = play(start, eng, ms, a.max_plies, judge, dd)
            # A から見た結果と探索深さに直す
            r = -r if swap else r
            depths[0] += dd[1] if swap else dd[0]
            depths[1] += dd[0] if swap else dd[1]
            score["win" if r > 0 else "loss" if r < 0 else "draw"] += 1
            ends[how] = ends.get(how, 0) + 1
        print(f"{i + 1}/{len(starts)} {a.a} vs {a.b}: {score} {time.time() - t0:.0f}s", flush=True)
    n = sum(score.values())
    res = {
        "a": a.a, "b": a.b, "a_ms": a.a_ms, "b_ms": a.b_ms, "games": n, **score,
        "score_rate_a": round((score["win"] + 0.5 * score["draw"]) / max(n, 1), 4), "ends": ends,
        "mean_depth_a": round(sum(depths[0]) / max(len(depths[0]), 1), 1), "mean_depth_b": round(sum(depths[1]) / max(len(depths[1]), 1), 1),
        "sec": round(time.time() - t0),
    }
    print(json.dumps(res, ensure_ascii=False), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    for e in (A, B, judge):
        e.quit()


if __name__ == "__main__":
    main()
