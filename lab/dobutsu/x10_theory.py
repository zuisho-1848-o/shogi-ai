"""X10: どうぶつしょうぎの中盤理論 — 人が読める規則を探し、真値で確かめる

X9 の仮説「長期の勝ちは、駒損をライオンの活動力で補う」を、条件3つまでの規則として取り出す。

  1. 探索（学習局面 --n 件）: 人が読める特徴量の条件（f <= v / f >= v）を最大3つ AND でつなぎ、
     次の3種類の規則をビームサーチで探す（精度 >= --prec を満たすもののうち、当てはまる局面が多い順）
       逆転の勝ち: 駒損（駒の枚数の差 <= -1）なのに「手番側の勝ち」
       逆転の負け: 駒得（駒の枚数の差 >= +1）なのに「手番側の負け」
       確実な勝ち: 全局面で「手番側の勝ち」（精度 99% 以上）
  2. 検証: 別の乱数で抜いた局面で精度が保たれるか
  3. 全数検証（--full）: 真値DBの全局面（2.1億）で例外の件数を数える。例外 0 なら、到達可能な局面についての定理

出力: ../results/x10_theory.md

使い方: X5_LION=1 python x10_theory.py --n 3000000 --full
"""

import argparse
import os
import time

import numpy as np

os.environ.setdefault("X5_LION", "1")
from x1_expressivity import HERE, cls, run_sampler  # noqa: E402
from x5_threats import threat_values  # noqa: E402

JA = {
    "mat": "駒の枚数の差",
    "my_row": "自ライオンの段（0=相手陣の奥）",
    "op_row": "相手ライオンの段（3=自陣の奥）",
    "my_safe": "自ライオンの安全な逃げ場",
    "op_safe": "相手ライオンの安全な逃げ場",
    "my_try_n": "相手陣の奥で自ライオンが安全に入れるマス",
    "op_try_n": "自陣の奥で相手ライオンが安全に入れるマス",
    "my_press": "相手ライオンの周りへの自分の利き",
    "op_press": "自ライオンの周りへの相手の利き",
    "my_hang": "自分の浮き駒",
    "op_hang": "相手の浮き駒",
    "my_fwd": "自ライオンが前に安全に動けるマス",
    "op_fwd": "相手ライオンが前に安全に動けるマス",
    "my_hand": "自分の持ち駒の枚数",
    "op_hand": "相手の持ち駒の枚数",
    "check": "王手されている",
}


def feats(keys):
    keys = keys.astype(np.uint64)
    v, _ = threat_values(keys)
    code = np.stack([((keys >> np.uint64(4 * s)) & np.uint64(15)).astype(np.int64) for s in range(12)], axis=1)
    hm = sum(((keys >> np.uint64(48 + 2 * j)) & np.uint64(3)).astype(np.int64) for j in range(3))
    ho = sum(((keys >> np.uint64(54 + 2 * j)) & np.uint64(3)).astype(np.int64) for j in range(3))
    mine = ((code >= 2) & (code <= 5)).sum(1) + hm
    theirs = ((code >= 10) & (code <= 13)).sum(1) + ho
    f = {k: v[k] for k in ("my_safe", "op_safe", "my_try_n", "op_try_n", "my_press", "op_press", "my_hang", "op_hang", "my_fwd", "op_fwd", "check")}
    f["mat"] = mine - theirs
    f["my_row"] = np.argmax(code == 1, axis=1) // 3
    f["op_row"] = np.argmax(code == 9, axis=1) // 3
    f["my_hand"] = hm
    f["op_hand"] = ho
    return {k: x.astype(np.int8) for k, x in f.items()}


def conditions(F):
    out = []
    for k, x in F.items():
        lo, hi = int(x.min()), int(x.max())
        for t in range(lo, hi + 1):
            if t < hi:
                out.append((k, "<=", t))
            if t > lo:
                out.append((k, ">=", t))
    return out


def mask_of(F, conds, base):
    m = base.copy()
    for k, op, t in conds:
        m &= (F[k] <= t) if op == "<=" else (F[k] >= t)
    return m


def beam(F, target, base, prec, min_sup, width=30, depth=3):
    """target を高精度で当てる条件の組を探す。戻り値: [(条件, 精度, 件数)]"""
    cands = conditions(F)
    beam_ = [((), base)]
    found = {}
    for _ in range(depth):
        nxt = []
        for conds, m in beam_:
            used = {c[0] for c in conds}
            for c in cands:
                if c[0] in used and c[0] != "mat":
                    continue
                m2 = m & ((F[c[0]] <= c[2]) if c[1] == "<=" else (F[c[0]] >= c[2]))
                sup = int(m2.sum())
                if sup < min_sup:
                    continue
                p = float(target[m2].mean())
                key = tuple(sorted(conds + (c,)))
                if p >= prec:
                    found[key] = (p, sup)
                nxt.append((p * np.log(sup), key, m2))
        nxt.sort(key=lambda x: -x[0])
        seen, beam_ = set(), []
        for _, key, m2 in nxt:
            if key in seen:
                continue
            seen.add(key)
            beam_.append((key, m2))
            if len(beam_) >= width:
                break
    # 当てはまる件数の多い順。より一般的な規則に含まれる規則は除く
    rules = sorted(found.items(), key=lambda x: -x[1][1])
    out = []
    for key, (p, sup) in rules:
        # より一般的な規則に含まれるもの、同じ局面集合を指すもの（精度と件数が完全に同じ）は除く
        if any(set(k2) <= set(key) or (p2 == p and s2 == sup) for k2, p2, s2 in out):
            continue
        out.append((key, p, sup))
        if len(out) >= 8:
            break
    return out


def fmt(conds):
    return " かつ ".join(f"{JA[k]} {'≤' if op == '<=' else '≥'} {t}" for k, op, t in sorted(conds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--n", type=int, default=3000000)
    ap.add_argument("--prec", type=float, default=0.95)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    keys, vals, _ = run_sampler(a.data, a.n, 7, False)
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    F = feats(keys)
    c = cls(vals)
    hk, hv, _ = run_sampler(a.data, a.n, 8, False)
    keep = hv != 2
    hk, hv = hk[keep], hv[keep]
    H = feats(hk)
    hc = cls(hv)
    n = len(c)
    min_sup = max(200, n // 2000)
    one, hone = np.ones(n, bool), np.ones(len(hc), bool)
    tasks = [
        ("逆転の勝ち（駒損なのに勝ち）", c == 1, F["mat"] <= -1, 0.50, hc == 1, H["mat"] <= -1),
        ("逆転の負け（駒得なのに負け）", c == -1, F["mat"] >= 1, 0.70, hc == -1, H["mat"] >= 1),
        ("確実な勝ち", c == 1, one, 0.99, hc == 1, hone),
        ("定理候補・勝ち（例外なし）", c == 1, one, 1.0, hc == 1, hone),
        ("定理候補・負け（例外なし）", c == -1, one, 1.0, hc == -1, hone),
    ]
    lines = ["# X10 どうぶつしょうぎの中盤理論（規則と真値による確認）", "", f"探索 {n} 局面 / 確認用 {len(hc)} 局面（即勝ちの局面は除く）。精度 = 規則に当てはまる局面のうち結論どおりの割合", ""]
    allrules = []
    for name, target, base, prec, htarget, hbase in tasks:
        lines += [f"## {name}", f"前提に当てはまる局面のうち結論どおりの割合（基準）: {target[base].mean():.1%}", "", "| 規則 | 精度（探索） | 精度（確認用） | 当てはまる割合（前提の中で） |", "|---|---|---|---|"]
        for conds, p, sup in beam(F, target, base, prec, min_sup):
            hm = mask_of(H, conds, hbase)
            hp = float(htarget[hm].mean()) if hm.any() else float("nan")
            lines.append(f"| {fmt(conds)} | {p:.2%} | {hp:.2%} | {sup / base.sum():.1%} |")
            allrules.append((name, conds))
        lines.append("")
        print(f"{name}: done {time.time() - t0:.0f}s", flush=True)

    if a.full:
        # 全数検証: plain_keys/vals を少しずつ読み、各規則の例外を数える
        pk = np.memmap(os.path.join(a.data, "plain_keys.bin"), dtype=np.uint64, mode="r")
        pv = np.memmap(os.path.join(a.data, "plain_vals.bin"), dtype=np.uint8, mode="r")
        stats = [[0, 0] for _ in allrules]  # 当てはまる件数, 例外
        step = 400000
        for s in range(0, len(pk), step):
            k = np.asarray(pk[s : s + step])
            v = np.asarray(pv[s : s + step])
            keep = v != 2
            k, v = k[keep], v[keep]
            G = feats(k)
            gc = cls(v)
            for i, (name, conds) in enumerate(allrules):
                base = (G["mat"] <= -1) if name.startswith("逆転の勝ち") else (G["mat"] >= 1) if name.startswith("逆転の負け") else np.ones(len(gc), bool)
                m = mask_of(G, conds, base)
                want = -1 if ("負け" in name.split("（")[0]) else 1
                stats[i][0] += int(m.sum())
                stats[i][1] += int((gc[m] != want).sum())
        lines += ["## 全数検証（真値DBの全局面、即勝ちを除く）", "| 種類 | 規則 | 当てはまる局面 | 例外 | 判定 |", "|---|---|---|---|---|"]
        for (name, conds), (m, e) in zip(allrules, stats):
            lines.append(f"| {name.split('（')[0]} | {fmt(conds)} | {m:,} | {e:,} | {'**定理（例外なし）**' if e == 0 else f'{1 - e / max(m, 1):.3%}'} |")
        print(f"full check done {time.time() - t0:.0f}s", flush=True)
    out = os.path.join(HERE, "../results/x10_theory.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
