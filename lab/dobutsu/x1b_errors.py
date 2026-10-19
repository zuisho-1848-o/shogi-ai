"""X1b: 評価関数が1手読みで間違える「難所」の分析

X1 と同じ条件で KPP（線形3駒関係）と MLP（小NN、参考）を学習し、
勝ち局面（どの手でもよいわけではないもの）で勝ちを逃した局面の性質を集計する。

出力:
  ../results/x1b_errors.md      : 属性ごとの誤り率の表
  ../data/dobutsu/hard.jsonl    : 両方が間違えた局面（X4 の材料）

使い方: python x1b_errors.py --train 3000000 --test 300000
"""

import argparse
import collections
import json
import os

import numpy as np

from x1_expressivity import HERE, Model, cls, run_sampler

DIRS = {
    1: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    2: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    3: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    4: [(-1, 0)],
    5: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
}
NAME = {1: "L", 2: "G", 3: "E", 4: "C", 5: "H"}


def decode(key):
    sq = [(key >> (4 * i)) & 15 for i in range(12)]
    hm = [(key >> (48 + 2 * j)) & 3 for j in range(3)]
    ho = [(key >> (54 + 2 * j)) & 3 for j in range(3)]
    return sq, hm, ho


def attacked_by_opp(sq, s):
    sr, sc = divmod(s, 3)
    for i, v in enumerate(sq):
        if not v & 8:
            continue
        r, c = divmod(i, 3)
        for dr, dc in DIRS[v & 7]:
            if r - dr == sr and c + dc == sc:
                return True
    return False


def attrs(key, kids):
    sq, hm, ho = decode(int(key))
    my_lion = sq.index(1)
    opp_lion = sq.index(9)
    mine = sum(1 for v in sq if 2 <= v <= 5) + sum(hm)
    theirs = sum(1 for v in sq if 10 <= v <= 13) + sum(ho)
    good = cls(kids["val"]) == -1
    mv = kids["mv"].astype(int)
    frm, cap, piece = mv & 15, (mv >> 12) & 7, (mv >> 8) & 7
    g_cap = bool((good & (cap > 0)).any())
    g_drop = bool((good & (frm == 15)).any())
    g_lion = bool((good & (piece == 1) & (frm != 15)).any())
    return {
        "d": None,  # 呼び出し側で手数の区分を入れる
        "my_lion_row": my_lion // 3,  # 0 = 相手陣（トライの段）
        "opp_lion_row": opp_lion // 3,  # 3 = 自陣（相手のトライの段）
        "in_check": attacked_by_opp(sq, my_lion),
        "material": int(np.clip(mine - theirs, -3, 3)),
        "hand": int(sum(hm)),
        "good_moves": int(good.sum()),
        "frac_good": round(float(good.mean()), 1),
        "good_has_capture": g_cap,
        "good_only_quiet": not g_cap,
        "good_has_drop": g_drop,
        "good_has_lion": g_lion,
    }


def bucket_d(d):
    for hi in (3, 5, 9, 15, 25, 45, 80):
        if d <= hi:
            return f"<= {hi}"
    return "> 80"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--train", type=int, default=3000000)
    ap.add_argument("--test", type=int, default=300000)
    a = ap.parse_args()
    keys, vals, _ = run_sampler(a.data, a.train, 1, False)
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    tkeys, tvals, tkids = run_sampler(a.data, a.test, 2, True)
    flat = np.concatenate([k["key"] for k in tkids])
    flat_v = np.concatenate([k["val"] for k in tkids])
    off = np.cumsum([0] + [len(k) for k in tkids])
    choice = {}
    for kind in ("KPP", "MLP"):
        m = Model(kind).fit(keys, vals)
        cp = m.p_win(flat)
        cp[flat_v == 2] = 1.0
        choice[kind] = [int(np.argmin(cp[off[i] : off[i + 1]])) for i in range(len(tkids))]

    c = cls(tvals)
    rows, hard = [], []
    for i, kd in enumerate(tkids):
        if c[i] != 1:
            continue
        good = cls(kd["val"]) == -1
        if good.all():
            continue
        at = attrs(tkeys[i], kd)
        at["d"] = bucket_d(int(tvals[i]) - 1)
        ok = {k: bool(good[choice[k][i]]) for k in choice}
        rows.append((at, ok))
        if not ok["KPP"] and not ok["MLP"]:
            gm = [int(x) for x in kd["mv"][good]]
            hard.append({"key": int(tkeys[i]), "d": int(tvals[i]) - 1, "good_mv": gm, "kpp_mv": int(kd["mv"][choice["KPP"][i]]), **{k: v for k, v in at.items() if k != "d"}})

    n = len(rows)
    lines = [f"# X1b 誤り分析（勝ち局面 {n} 件、学習 {a.train} / 評価 {a.test}）", ""]
    tot = {k: sum(not ok[k] for _, ok in rows) / n for k in ("KPP", "MLP")}
    both = sum(not ok["KPP"] and not ok["MLP"] for _, ok in rows) / n
    lines.append(f"全体: KPP 誤り {tot['KPP']:.1%} / MLP 誤り {tot['MLP']:.1%} / 両方とも誤り {both:.1%}")
    lines.append("")
    for attr in rows[0][0]:
        g = collections.defaultdict(list)
        for at, ok in rows:
            g[at[attr]].append(ok)
        lines.append(f"## {attr}")
        lines.append("| 値 | 件数 | 構成比 | KPP誤り | MLP誤り | 両方誤り |")
        lines.append("|---|---|---|---|---|---|")
        for val in sorted(g, key=lambda x: str(x)):
            oks = g[val]
            m = len(oks)
            lines.append(
                f"| {val} | {m} | {m / n:.1%} | {sum(not o['KPP'] for o in oks) / m:.1%} | {sum(not o['MLP'] for o in oks) / m:.1%} | {sum(not o['KPP'] and not o['MLP'] for o in oks) / m:.1%} |"
            )
        lines.append("")
    out_md = os.path.join(HERE, "../results/x1b_errors.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(a.data, "hard.jsonl"), "w") as f:
        for h in hard:
            f.write(json.dumps(h) + "\n")
    print("\n".join(lines[:3]))
    print(f"hard positions: {len(hard)} -> {a.data}/hard.jsonl, table -> {out_md}")


if __name__ == "__main__":
    main()
