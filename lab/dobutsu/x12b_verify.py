"""X12b: X12 ①で見つかった定理候補・逆転の規則を、真値DBの全局面（即勝ちを除く）で確かめる

使い方: python x12b_verify.py
"""

import os

import numpy as np

os.environ.setdefault("X5_LION", "1")
from x1_expressivity import HERE, cls  # noqa: E402
from x10_theory import fmt, mask_of  # noqa: E402
import x10_theory  # noqa: E402
from x12_rules import JA2, feats  # noqa: E402

x10_theory.JA = JA2
RULES = [
    ("勝ち", (("my_press", ">=", 6), ("op_press", "<=", 0)), None),
    ("勝ち", (("my_def", ">=", 4), ("op_mob", "<=", 3)), None),
    ("負け", (("my_mob", "<=", 3), ("op_hand", ">=", 3)), None),
    ("負け", (("my_mob", "<=", 3), ("op_def", ">=", 5)), None),
    ("負け", (("op_fwd", ">=", 2), ("op_row", ">=", 2)), None),
    ("勝ち（駒損でも）", (("mat", ">=", -3), ("my_mob", ">=", 8), ("op_5", "<=", 0)), ("mat", "<=", -1)),
    ("勝ち（駒損でも）", (("my_mob", ">=", 9), ("op_5", "<=", 0)), ("mat", "<=", -1)),
    ("負け（駒得でも）", (("check", ">=", 1), ("op_fwd", ">=", 1), ("op_row", ">=", 2)), ("mat", ">=", 1)),
]


def main():
    data = os.path.join(HERE, "../data/dobutsu")
    pk = np.memmap(os.path.join(data, "plain_keys.bin"), dtype=np.uint64, mode="r")
    pv = np.memmap(os.path.join(data, "plain_vals.bin"), dtype=np.uint8, mode="r")
    st = [[0, 0] for _ in RULES]
    for s in range(0, len(pk), 400000):
        k = np.asarray(pk[s : s + 400000])
        v = np.asarray(pv[s : s + 400000])
        keep = v != 2
        k, v = k[keep], v[keep]
        G = feats(k)
        gc = cls(v)
        for i, (kind, conds, pre) in enumerate(RULES):
            base = np.ones(len(gc), bool)
            if pre:
                base = mask_of(G, (pre,), base)
            m = mask_of(G, conds, base)
            want = 1 if kind.startswith("勝ち") else -1
            st[i][0] += int(m.sum())
            st[i][1] += int((gc[m] == want).sum())
    L = ["# X12b 規則の全数検証（真値DBの全局面、即勝ちを除く）", "", "| 結論 | 規則 | 当てはまる局面 | 結論どおり | 判定 |", "|---|---|---|---|---|"]
    for (kind, conds, pre), (m, ok) in zip(RULES, st):
        cond = fmt(conds) + (f"（前提: {fmt((pre,))}）" if pre else "")
        L.append(f"| {kind} | {cond} | {m:,} | {ok:,} | {'**定理（例外なし）**' if ok == m and m else f'{ok / max(m, 1):.3%}'} |")
    out = os.path.join(HERE, "../results/x12b_verify.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
