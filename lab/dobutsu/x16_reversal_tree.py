"""X16: 駒損からの逆転の規則を、浅い決定木の枝として取り出す

X10/X12/X14 のビームサーチ（条件3つまでの AND）では、駒損からの逆転は最良でも 6割台だった。
決定木なら「条件の組み合わせ方」を枝分かれで表せるので、駒損の局面だけで浅い木（深さ 4〜6）を学習し、
勝率の高い葉までの道のりを規則として書き出す。別の局面で精度を確かめ、上位の規則は全局面でも確かめる。

出力: ../results/x16_reversal_tree.md
使い方: python x16_reversal_tree.py --n 3000000 --full
"""

import argparse
import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier

os.environ.setdefault("X5_LION", "1")
from x1_expressivity import HERE, cls, run_sampler  # noqa: E402
from x10_theory import fmt, mask_of  # noqa: E402
import x10_theory  # noqa: E402
from x14_rules import JA2, feats  # noqa: E402

x10_theory.JA = JA2


def leaf_rules(tree, names):
    """各葉までの条件を (特徴量, 向き, しきい値) の並びにする。しきい値は整数に丸める（特徴量はすべて整数）"""
    t = tree.tree_
    out = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            v = t.value[node][0]
            out.append((tuple(conds), v[1] / v.sum(), int(t.n_node_samples[node])))
            return
        f, th = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [(f, "<=", int(np.floor(th)))])
        walk(t.children_right[node], conds + [(f, ">=", int(np.floor(th)) + 1)])

    walk(0, [])
    return out


def simplify(conds):
    """同じ特徴量の条件をまとめる（最も厳しいものだけ残す）"""
    lo, hi = {}, {}
    for f, op, t in conds:
        if op == "<=":
            hi[f] = min(hi.get(f, 99), t)
        else:
            lo[f] = max(lo.get(f, -99), t)
    out = [(f, ">=", t) for f, t in lo.items()] + [(f, "<=", t) for f, t in hi.items()]
    return tuple(sorted(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--n", type=int, default=3000000)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    keys, vals, _ = run_sampler(a.data, a.n, 13, False)
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    F = feats(keys)
    c = cls(vals)
    hk, hv, _ = run_sampler(a.data, a.n, 14, False)
    keep = hv != 2
    hk, hv = hk[keep], hv[keep]
    H = feats(hk)
    hc = cls(hv)
    names = sorted(F)
    L = ["# X16 駒損からの逆転の規則（浅い決定木の枝）", ""]
    allrules = []
    for title, base_f, base_h, want in (
        ("駒損なのに勝ち（駒の枚数の差 ≤ -1）", F["mat"] <= -1, H["mat"] <= -1, 1),
        ("駒得なのに負け（駒の枚数の差 ≥ +1）", F["mat"] >= 1, H["mat"] >= 1, -1),
    ):
        X = np.stack([F[k][base_f] for k in names], axis=1)
        y = (c[base_f] == want).astype(int)
        Xh = np.stack([H[k][base_h] for k in names], axis=1)
        yh = (hc[base_h] == want).astype(int)
        L += [f"## {title}（基準 {y.mean():.1%}）", "", "| 木の深さ | 確認用での当たり率（勝ち/それ以外） |", "|---|---|"]
        trees = {}
        for depth in (3, 4, 5, 6):
            t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(500, len(y) // 400), class_weight="balanced").fit(X, y)
            trees[depth] = t
            L.append(f"| {depth} | {(t.predict(Xh) == yh).mean():.1%} |")
        L += ["", "### 深さ5の木で、結論どおりの割合が高い枝（規則）", "| 規則 | 精度（探索） | 精度（確認用） | 当てはまる割合（前提の中で） |", "|---|---|---|---|"]
        # 葉の値は件数を釣り合わせた重みつきなので、探索用データで実際の割合を数え直す
        rules = []
        for cd, _, _ in leaf_rules(trees[5], names):
            cd = simplify(cd)
            m = mask_of(F, cd, base_f)
            if m.any():
                rules.append((cd, float((c[m] == want).mean()), int(m.sum())))
        rules.sort(key=lambda r: -(r[1] - y.mean()) * np.sqrt(r[2]))
        shown = 0
        for cd, p, n in rules:
            if p < max(0.5, 2 * y.mean()):
                continue
            m = mask_of(H, cd, base_h)
            hp = float((hc[m] == want).mean()) if m.any() else float("nan")
            L.append(f"| {fmt(cd)} | {p:.1%} | {hp:.1%} | {n / len(y):.1%} |")
            allrules.append((title, cd, want))
            shown += 1
            if shown >= 10:
                break
        L.append("")

    if a.full:
        pk = np.memmap(os.path.join(a.data, "plain_keys.bin"), dtype=np.uint64, mode="r")
        pv = np.memmap(os.path.join(a.data, "plain_vals.bin"), dtype=np.uint8, mode="r")
        st = [[0, 0] for _ in allrules]
        for s in range(0, len(pk), 400000):
            k = np.asarray(pk[s : s + 400000])
            v = np.asarray(pv[s : s + 400000])
            keep = v != 2
            k, v = k[keep], v[keep]
            G = feats(k)
            gc = cls(v)
            for i, (title, cd, want) in enumerate(allrules):
                base = (G["mat"] <= -1) if want == 1 else (G["mat"] >= 1)
                m = mask_of(G, cd, base)
                st[i][0] += int(m.sum())
                st[i][1] += int((gc[m] == want).sum())
        L += ["## 全数検証（真値DBの全局面、即勝ちを除く）", "| 種類 | 規則 | 当てはまる局面 | 精度 |", "|---|---|---|---|"]
        for (title, cd, want), (m, ok) in zip(allrules, st):
            L.append(f"| {title.split('（')[0]} | {fmt(cd)} | {m:,} | {ok / max(m, 1):.2%} |")
    out = os.path.join(HERE, "../results/x16_reversal_tree.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
