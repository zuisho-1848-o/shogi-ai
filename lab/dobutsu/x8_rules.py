"""X8: 「脅威の構造」から人が読める規則（中盤理論の候補）を取り出す

どうぶつしょうぎの真値DBを教師に、X5 の脅威の特徴量（13種）＋駒数だけを入力にした
浅い決定木（深さ 3〜6）を学習し、規則として書き出す。
規則だけで1手読みしたときの誤り率も測り、「何本の規則でどこまで真値に近づくか」を見る。
あわせて、勾配ブースティング（深い木の集まり）の特徴量の重要度も出す。

出力: ../results/x8_rules.md

使い方: python x8_rules.py --train 3000000 --test 300000
"""

import argparse
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

from x1_expressivity import HERE, cls, evaluate, labels, material, run_sampler
from x5_threats import NAMES, threat_values

JA = {
    "my_safe": "自ライオンの安全な逃げ場",
    "op_safe": "相手ライオンの安全な逃げ場",
    "my_try_d": "自ライオンの奥の段までの距離",
    "my_try_n": "奥の段で自ライオンが安全に入れるマス",
    "op_try_d": "相手ライオンの自陣奥までの距離",
    "op_try_n": "自陣奥で相手ライオンが安全に入れるマス",
    "my_press": "相手ライオンの周りへの自分の利き",
    "op_press": "自ライオンの周りへの相手の利き",
    "my_hang": "自分の浮き駒",
    "op_hang": "相手の浮き駒",
    "my_mob": "自分の自由度",
    "op_mob": "相手の自由度",
    "check": "王手されている",
    "material": "駒の枚数の差（盤上+持ち駒）",
}
FEATS = NAMES + ["material"]


def dense(keys):
    v, _ = threat_values(keys)
    m = material(keys).toarray()
    diff = m[:, 0:4].sum(1) + m[:, 8:11].sum(1) - m[:, 4:8].sum(1) - m[:, 11:14].sum(1)
    return np.stack([v[n] for n in NAMES] + [diff], axis=1).astype(np.float32)


class TreeModel:
    def __init__(self, m):
        self.m = m

    def p_win(self, keys):
        return self.m.predict_proba(dense(keys))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--train", type=int, default=3000000)
    ap.add_argument("--test", type=int, default=300000)
    a = ap.parse_args()
    keys, vals, _ = run_sampler(a.data, a.train, 1, False)
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    rows, y, w = labels(vals)
    X = dense(keys)[rows]
    tkeys, tvals, tkids = run_sampler(a.data, a.test, 2, True)

    lines = ["# X8 脅威の構造から取り出した規則", "", "入力: 脅威の特徴量13種＋駒の枚数の差（すべて手番側から見た値）。教師: どうぶつしょうぎの真値。", ""]
    lines += ["| 木の深さ | 葉（規則）の数 | 勝敗の当たり率 | 1手読み誤り率（勝ち局面） | 引き分けで負けに落ちる率 |", "|---|---|---|---|---|"]
    trees = {}
    for depth in (3, 4, 5, 6, 8):
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=2000).fit(X, y, sample_weight=w)
        r = evaluate(TreeModel(t), tkeys, tvals, tkids)
        trees[depth] = t
        lines.append(f"| {depth} | {t.get_n_leaves()} | {r['acc']:.1%} | {1 - r['move_win']:.1%} | {1 - r['move_draw']:.1%} |")
    g = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63).fit(X[:600000], y[:600000], sample_weight=w[:600000])
    rg = evaluate(TreeModel(g), tkeys, tvals, tkids)
    lines.append(f"| 勾配ブースティング（参考） | 300本×63葉 | {rg['acc']:.1%} | {1 - rg['move_win']:.1%} | {1 - rg['move_draw']:.1%} |")
    lines += ["", "比較（X1/X5）: 駒数だけ 13.9% / KPP 6.0% / KPP＋脅威 5.3% / 小NN 4.5%", ""]

    td = dense(tkeys)
    tc = cls(tvals)
    dec = tc != 0
    imp = permutation_importance(g, td[dec][:20000], (tc[dec][:20000] == 1).astype(int), n_repeats=3, random_state=0)
    order = np.argsort(-imp.importances_mean)
    lines += ["## 特徴量の重要度（勾配ブースティング、並べ替え重要度）", "| 特徴量 | 重要度 |", "|---|---|"]
    lines += [f"| {JA[FEATS[i]]} | {imp.importances_mean[i]:.4f} |" for i in order]
    lines += ["", "## 深さ4の木（規則として読む。class 1 = 手番側の勝ち）", "```", export_text(trees[4], feature_names=[JA[f] for f in FEATS], show_weights=False), "```"]
    out = os.path.join(HERE, "../results/x8_rules.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
