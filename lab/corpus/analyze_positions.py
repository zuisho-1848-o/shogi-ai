"""局面リストをやねうら王（NNUE）で固定ノード数ずつ解析して貯める（中断・再開可）

裏で長時間回す前提なので 1スレッド・小さいハッシュで動かす。
入力: mainline.jsonl / exits.jsonl（sfen を持つ jsonl）
出力: <out>.jsonl に sfen ごとに score_cp / mate / depth / bestmove / pv を追記。既にある sfen は飛ばす。

使い方:
  nice -n 19 taskpolicy -b python analyze_positions.py --inputs ../data/mainline/mainline.jsonl ../data/mainline/exits.jsonl \
      --out ../data/mainline/nnue_1m.jsonl --nodes 1000000
"""

import argparse
import json
import os
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ENGINE = os.path.join(ROOT, "engines/yaneuraou-std/yaneuraou")
EVAL_DIR = os.path.join(ROOT, "engines/yaneuraou-std/eval")


class Usi:
    def __init__(self, threads, hash_mb):
        self.p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=os.path.dirname(ENGINE))
        self.send("usi")
        self.wait("usiok")
        for k, v in {"Threads": threads, "USI_Hash": hash_mb, "EvalDir": EVAL_DIR, "FV_SCALE": 24, "BookFile": "no_book", "NetworkDelay": 0, "NetworkDelay2": 0}.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait("readyok")
        self.send("usinewgame")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, token):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            lines.append(line.strip())
            if line.startswith(token):
                return lines

    def go(self, sfen, nodes):
        self.send(f"position sfen {sfen} 1")
        self.send(f"go nodes {nodes}")
        lines = self.wait("bestmove")
        res = {"bestmove": lines[-1].split()[1]}
        for line in reversed(lines):
            t = line.split()
            if t[:1] == ["info"] and "score" in t and "pv" in t:
                i = t.index("score")
                if t[i + 1] == "cp":
                    res["score_cp"] = int(t[i + 2])
                else:
                    res["mate"] = t[i + 2]
                for key in ("depth", "seldepth", "nodes"):
                    if key in t:
                        res[key] = int(t[t.index(key) + 1])
                res["pv"] = t[t.index("pv") + 1 :]
                break
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nodes", type=int, default=1000000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=128)
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["sfen"] for l in f if l.strip()}
    todo = []
    for path in a.inputs:
        with open(path) as f:
            for l in f:
                s = json.loads(l)["sfen"]
                if s not in done:
                    done.add(s)
                    todo.append(s)
    print(f"todo {len(todo)}", flush=True)
    eng = Usi(a.threads, a.hash)
    t0 = time.time()
    with open(a.out, "a") as f:
        for i, sfen in enumerate(todo):
            r = eng.go(sfen, a.nodes)
            f.write(json.dumps({"sfen": sfen, **r}) + "\n")
            f.flush()
            if i % 200 == 0:
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    eng.send("quit")


if __name__ == "__main__":
    main()
