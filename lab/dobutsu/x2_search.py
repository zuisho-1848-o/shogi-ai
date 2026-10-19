"""X2: 知識と読みの交換比 — 評価関数ごとに、何手読めば真値に追いつくか

X1 と同じ評価局面（勝ち局面のうち、どの手でもよいわけではないもの）から --roots 件を使い、
深さ D の全幅 negamax（葉で評価関数の勝率、即勝ちは1手読みで分かるので確定値）で手を選んだときの誤り率を測る。

  V(局面, 0) = 評価関数の勝率（手番側から見た値）
  V(局面, r) = 即勝ちがあれば 1、なければ max(1 - V(子, r-1))
  根では argmax(1 - V(子, D-1)) の手を指す

使い方: python x2_search.py --roots 5000 --depths 1,2,3 --kinds material,KPP,KPP+T,MLP
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from x1_expressivity import HERE, Model, cls, run_sampler
from x5_threats import ThreatModel

REC = np.dtype([("key", "<u8"), ("val", "u1"), ("mv", "<u2")])
IMM_LEAF = set()  # 即勝ちのある局面（葉で見つかったら確定値 1）


def expand(data, keys):
    """局面の配列を展開して {key: (即勝ちか, 子局面の配列)} を返す"""
    out = subprocess.run([os.path.join(HERE, "sample"), data, "expand"], input=np.asarray(keys, dtype="<u8").tobytes(), check=True, capture_output=True).stdout
    res, p = {}, 0
    for k in keys:
        imm, nc = out[p + 9], out[p + 10]
        p += 11
        rec = np.frombuffer(out, dtype=REC, count=nc, offset=p) if nc else np.zeros(0, dtype=REC)
        p += 11 * nc
        res[int(k)] = (bool(imm), rec["key"])
        # 子が「即勝ち」（ライオンを取れる/トライ成立）かはルールだけで分かるので記録しておく（葉の確定値に使う）
        IMM_LEAF.update(int(c) for c in rec["key"][rec["val"] == 2])
    return res


def build_tree(data, roots, depth):
    """根から depth-1 段目までを展開（葉は展開しない）"""
    tree = {}
    frontier = list({int(k) for k in roots})
    for _ in range(depth):
        frontier = [k for k in frontier if k not in tree]
        if not frontier:
            break
        for s in range(0, len(frontier), 200000):
            tree.update(expand(data, frontier[s : s + 200000]))
        nxt = set()
        for k in frontier:
            imm, ch = tree[k]
            if not imm:
                nxt.update(int(c) for c in ch)
        frontier = list(nxt)
    return tree, frontier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--train", type=int, default=3000000)
    ap.add_argument("--test", type=int, default=300000)
    ap.add_argument("--roots", type=int, default=5000)
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--kinds", default="material,KPP,KPP+T,MLP")
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x2_search.jsonl"))
    a = ap.parse_args()
    sys.setrecursionlimit(10000)
    keys, vals, _ = run_sampler(a.data, a.train, 1, False)
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    tkeys, tvals, tkids = run_sampler(a.data, a.test, 2, True)
    tc = cls(tvals)
    roots = []  # (根の子局面, 勝つ手か)
    for i, kd in enumerate(tkids):
        if tc[i] != 1:
            continue
        good = cls(kd["val"]) == -1
        if good.all():
            continue
        roots.append((kd["key"], good, int(tvals[i]) - 1))
        if len(roots) >= a.roots:
            break
    depths = [int(d) for d in a.depths.split(",")]
    t0 = time.time()
    # 根の子局面を起点に max(D)-1 段を展開。葉はその次の段
    tree, _ = build_tree(a.data, np.concatenate([r[0] for r in roots]), max(depths) - 1)
    print(f"tree nodes {len(tree)}, {time.time() - t0:.0f}s", flush=True)

    for kind in a.kinds.split(","):
        m = (ThreatModel(kind) if "T" in kind.split("+") else Model(kind)).fit(keys, vals)
        for D in depths:
            t = time.time()
            # 葉（残り深さ0で評価する局面）を集めて一括評価
            leaves, memo = set(), {}

            def collect(k, r):
                if (k, r) in memo:
                    return
                memo[(k, r)] = None
                if r == 0:
                    leaves.add(k)
                    return
                imm, ch = tree[k]
                if not imm:
                    for c in ch:
                        collect(int(c), r - 1)

            for ch, _, _ in roots:
                for c in ch:
                    collect(int(c), D - 1)
            lk = np.array(sorted(leaves), dtype=np.uint64)
            pw = dict(zip(lk.tolist(), m.p_win(lk).tolist())) if len(lk) else {}
            memo = {}

            def V(k, r):
                hit = memo.get((k, r))
                if hit is not None:
                    return hit
                if r == 0:
                    v = 1.0 if k in IMM_LEAF else pw[k]
                else:
                    imm, ch = tree[k]
                    v = 1.0 if imm else (max(1.0 - V(int(c), r - 1) for c in ch) if len(ch) else 0.0)
                memo[(k, r)] = v
                return v

            err = 0
            by_d = {}
            for ch, good, dist in roots:
                scores = []
                for c in ch:
                    c = int(c)
                    imm, _ = tree[c]
                    scores.append(0.0 if imm else 1.0 - V(c, D - 1))
                ok = bool(good[int(np.argmax(scores))])
                err += not ok
                b = "<=15" if dist <= 15 else ("<=45" if dist <= 45 else ">45")
                e = by_d.setdefault(b, [0, 0])
                e[0] += 1
                e[1] += not ok
            res = {"kind": kind, "depth": D, "roots": len(roots), "err": err / len(roots), "leaves": len(lk), "by_d": {b: round(e[1] / e[0], 4) for b, e in by_d.items()}, "sec": round(time.time() - t)}
            print(json.dumps(res), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
