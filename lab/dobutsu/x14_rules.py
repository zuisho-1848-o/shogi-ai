"""X14: どうぶつしょうぎの規則をさらに別の方向から

  (a) 駒の構成表: 「どちらがどの駒を何枚（盤上・持ち駒）持っているか」だけで勝敗が決まる組み合わせ
      （チェスの終盤理論の「キング＋クイーン対キングは必勝」に相当）。真値DBの全局面で集計
  (b) 規則の優先順位リスト: 上から順に当てはめる「もし〜なら勝ち／負け」の表を貪欲に作り、何本でどこまで当たるか
  (c) 取り合いの規則: 取られそうな駒（相手の利き > 自分の守り）の数などを加えた条件
  (d) 短手数の勝ちの見分け方: 5手以内に勝てる局面の条件
  (e) 格言の優先順位: X12③の格言を、上から順に当てはめる順番を貪欲に決める

出力: ../results/x14_rules.md
使い方: python x14_rules.py --n 2000000
"""

import argparse
import collections
import os
import time

import numpy as np

os.environ.setdefault("X5_LION", "1")
from x1_expressivity import HERE, cls, run_sampler  # noqa: E402
import x10_theory  # noqa: E402
from x10_theory import beam, fmt, mask_of  # noqa: E402
from x12_rules import JA2, PV, feats as feats12  # noqa: E402
from x5_threats import threat_values  # noqa: E402

JA2.update({"my_enprise": "自分の取られそうな駒", "op_enprise": "相手の取られそうな駒", "my_lion_att": "自ライオンに利いている相手の駒"})
x10_theory.JA = JA2
NAMES = ["キリン", "ぞう", "ひよこ", "にわとり"]


def feats(keys):
    F = feats12(keys)
    tv, _ = threat_values(keys.astype(np.uint64))
    for k in ("my_enprise", "op_enprise", "my_lion_att"):
        F[k] = tv[k].astype(np.int8)
    return F


def signature(keys):
    """駒の構成（手番側/相手 × 盤上のキリン・ぞう・ひよこ・にわとり ＋ 持ち駒のキリン・ぞう・ひよこ）を整数に"""
    keys = keys.astype(np.uint64)
    code = np.stack([((keys >> np.uint64(4 * s)) & np.uint64(15)).astype(np.int64) for s in range(12)], axis=1)
    digs = []
    for c in (2, 3, 4, 5):
        digs.append((code == c).sum(1))
    for j in range(3):
        digs.append(((keys >> np.uint64(48 + 2 * j)) & np.uint64(3)).astype(np.int64))
    for c in (10, 11, 12, 13):
        digs.append((code == c).sum(1))
    for j in range(3):
        digs.append(((keys >> np.uint64(54 + 2 * j)) & np.uint64(3)).astype(np.int64))
    sig = np.zeros(len(keys), dtype=np.int64)
    for d in digs:
        sig = sig * 3 + d
    return sig


def show_sig(sig):
    d = []
    for _ in range(14):
        d.append(sig % 3)
        sig //= 3
    d = d[::-1]

    def side(x):
        board = "".join(f"{n}{x[i]}" for i, n in enumerate(NAMES) if x[i])
        hand = "".join(f"{n}{x[4 + i]}" for i, n in enumerate(NAMES[:3]) if x[4 + i])
        return f"盤[{board or 'なし'}] 持[{hand or 'なし'}]"

    return f"自: {side(d[:7])} / 相: {side(d[7:])}"


def rule_list(F, c, H, hc, max_rules=25):
    """「勝ち」「負け」を高い精度で当てる規則を、まだ当たっていない局面が最も多く減る順に積む"""
    remain = np.ones(len(c), bool)
    hremain = np.ones(len(hc), bool)
    out = []
    for _ in range(max_rules):
        best = None
        for target, want in ((c == 1, 1), (c == -1, -1)):
            cand = beam(F, target, remain, 0.97, max(300, len(c) // 3000), width=15, depth=2)
            for conds, p, sup in cand[:3]:
                if best is None or sup > best[2]:
                    best = (conds, p, sup, want)
        if best is None:
            break
        conds, p, sup, want = best
        m = mask_of(F, conds, remain)
        hm = mask_of(H, conds, hremain)
        hp = float((hc[hm] == want).mean()) if hm.any() else float("nan")
        remain &= ~m
        hremain &= ~hm
        out.append((conds, want, p, hp, 1 - remain.mean(), 1 - hremain.mean()))
    # 残り（どの規則にも当たらない局面）は「勝ち」と見なす（多数派）
    return out, remain, hremain


def proverb_order(tkeys, tvals, tkids, max_len=6):
    """X12 ③の格言を「当てはまる最初の格言で指す」並べ方にしたとき、勝ち局面で正しい手を選ぶ率"""
    c = cls(tvals)
    data = []
    for i, kd in enumerate(tkids):
        if c[i] != 1:
            continue
        good = cls(kd["val"]) == -1
        safe = kd["val"] != 2
        if good.all() or not safe.any():
            continue
        mv = kd["mv"].astype(int)
        frm, to, piece, cap = mv & 15, (mv >> 4) & 15, (mv >> 8) & 7, (mv >> 12) & 7
        cf = feats(kd["key"])
        rules = {
            "高い駒を取る": np.where(cap > 0, np.vectorize(lambda x: PV.get(x, 0))(cap), -1),
            "ひよこを成る": np.where((piece == 4) & (to // 3 == 0) & (frm != 15), 1, -1),
            "相手の指せる手を減らす": 20 - cf["my_mob"].astype(int),
            "ライオンを安全に前へ": np.where((piece == 1) & (frm != 15) & (to // 3 < frm // 3), 1, -1),
            "自玉の逃げ場を増やす": cf["op_safe"].astype(int),
            "取られそうな駒を減らす": 10 - cf["op_enprise"].astype(int),
            "相手の取られそうな駒を増やす": cf["my_enprise"].astype(int),
        }
        picks = {}
        for name, sc in rules.items():
            sc = np.where(safe, sc, -999)
            if sc.max() < 0:
                picks[name] = None
                continue
            picks[name] = np.where(sc == sc.max())[0]
        data.append((good, safe, picks))
    names = list(data[0][2])
    order = []
    base = np.mean([g[s].mean() for g, s, _ in data])

    def score(seq):
        tot = 0.0
        for good, safe, picks in data:
            cand = np.where(safe)[0]
            for name in seq:
                p = picks[name]
                if p is not None:
                    cand = np.intersect1d(cand, p) if len(np.intersect1d(cand, p)) else cand
            tot += good[cand].mean()
        return tot / len(data)

    hist = []
    for _ in range(max_len):
        best = max((n for n in names if n not in order), key=lambda n: score(order + [n]), default=None)
        if best is None:
            break
        order.append(best)
        hist.append((list(order), score(order)))
    return base, hist, len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--n", type=int, default=2000000)
    a = ap.parse_args()
    t0 = time.time()
    L = ["# X14 どうぶつしょうぎの規則をさらに別の方向から", ""]

    # (a) 駒の構成表（全局面）
    pk = np.memmap(os.path.join(a.data, "plain_keys.bin"), dtype=np.uint64, mode="r")
    pv = np.memmap(os.path.join(a.data, "plain_vals.bin"), dtype=np.uint8, mode="r")
    size = 3 ** 14
    cnt = np.zeros((3, size), dtype=np.int64)  # 0 勝ち 1 負け 2 引き分け
    for s in range(0, len(pk), 1000000):
        k = np.asarray(pk[s : s + 1000000])
        v = np.asarray(pv[s : s + 1000000])
        keep = v != 2
        k, v = k[keep], v[keep]
        sig = signature(k)
        cc = cls(v)
        for i, want in enumerate((1, -1, 0)):
            cnt[i] += np.bincount(sig[cc == want], minlength=size)
    tot = cnt.sum(0)
    used = np.where(tot > 0)[0]
    L += [f"## (a) 駒の構成表（全局面 / 構成の種類 {len(used):,}）", ""]
    pure_w = [(tot[i], i) for i in used if cnt[0, i] == tot[i] and tot[i] >= 1000]
    pure_l = [(tot[i], i) for i in used if cnt[1, i] == tot[i] and tot[i] >= 1000]
    L += [f"- 駒の配置によらず**必ず勝ち**の構成（1,000局面以上）: {len(pure_w)} 種類 / **必ず負け**: {len(pure_l)} 種類", ""]
    for title, lst in (("必ず勝ちの構成（多い順）", pure_w), ("必ず負けの構成（多い順）", pure_l)):
        L += [f"### {title}", "| 構成 | 局面数 |", "|---|---|"]
        for n_, i in sorted(lst, reverse=True)[:10]:
            L.append(f"| {show_sig(int(i))} | {n_:,} |")
        L.append("")
    big = sorted(((tot[i], i) for i in used), reverse=True)[:12]
    L += ["### よく出る構成の勝率（多い順）", "| 構成 | 局面数 | 勝ち | 負け | 引き分け |", "|---|---|---|---|---|"]
    for n_, i in big:
        L.append(f"| {show_sig(int(i))} | {n_:,} | {cnt[0, i] / n_:.1%} | {cnt[1, i] / n_:.1%} | {cnt[2, i] / n_:.1%} |")
    L.append("")
    print(f"(a) done {time.time() - t0:.0f}s", flush=True)

    keys, vals, _ = run_sampler(a.data, a.n, 11, False)
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    F = feats(keys)
    c = cls(vals)
    hk, hv, _ = run_sampler(a.data, a.n, 12, False)
    keep = hv != 2
    hk, hv = hk[keep], hv[keep]
    H = feats(hk)
    hc = cls(hv)

    # (b) 規則の優先順位リスト
    rl, rem, hrem = rule_list(F, c, H, hc)
    L += ["## (b) 規則の優先順位リスト（上から順に当てはめる。精度 97% 以上の規則を、当たる局面が多い順に積む）", "", "| 順 | もし | なら | 精度（探索） | 精度（確認用） | ここまでで当てはまった割合（確認用） |", "|---|---|---|---|---|---|"]
    for i, (conds, want, p, hp, cov, hcov) in enumerate(rl):
        L.append(f"| {i + 1} | {fmt(conds)} | {'勝ち' if want == 1 else '負け'} | {p:.1%} | {hp:.1%} | {hcov:.1%} |")
    # 表全体の精度（当てはまらない局面は「勝ち」と答える）
    pred = np.ones(len(hc), dtype=int)
    hremain = np.ones(len(hc), bool)
    for conds, want, *_ in rl:
        m = mask_of(H, conds, hremain)
        pred[m] = want
        hremain &= ~m
    L += ["", f"表全体の当たり率（確認用、引き分けは外れ扱い）: {(pred == hc).mean():.1%}（表に当てはまる局面だけなら {(pred == hc)[~hremain].mean():.1%}）", ""]
    print(f"(b) done {time.time() - t0:.0f}s", flush=True)

    # (c) 取り合いの規則
    L += ["## (c) 取り合いの規則（取られそうな駒 = 相手の利き > 自分の守り）", ""]
    for name, target, base, prec in (
        ("逆転の勝ち（駒損なのに勝ち）", c == 1, F["mat"] <= -1, 0.6),
        ("逆転の負け（駒得なのに負け）", c == -1, F["mat"] >= 1, 0.8),
    ):
        Fx = {k: F[k] for k in ("mat", "my_enprise", "op_enprise", "my_lion_att", "my_hang", "op_hang", "my_mob", "op_mob", "check", "op_row", "op_fwd")}
        L += [f"### {name}（基準 {target[base].mean():.1%}）", "| 規則 | 精度 | 当てはまる割合 |", "|---|---|---|"]
        for conds, p, sup in beam(Fx, target, base, prec, max(200, len(c) // 2000)):
            L.append(f"| {fmt(conds)} | {p:.1%} | {sup / base.sum():.1%} |")
        L.append("")
    print(f"(c) done {time.time() - t0:.0f}s", flush=True)

    # (d) 短手数の勝ち（5手以内）
    d = vals.astype(int) - 1
    short = (c == 1) & (d <= 5)
    L += [f"## (d) 5手以内に勝てる局面の見分け方（勝ち局面の中で。基準 {short[c == 1].mean():.1%}）", "| 規則 | 精度 | 当てはまる割合 |", "|---|---|---|"]
    for conds, p, sup in beam(F, short, c == 1, 0.9, max(200, len(c) // 2000)):
        L.append(f"| {fmt(conds)} | {p:.1%} | {sup / (c == 1).sum():.1%} |")
    L.append("")
    print(f"(d) done {time.time() - t0:.0f}s", flush=True)

    # (e) 格言の優先順位
    tkeys, tvals, tkids = run_sampler(a.data, 150000, 2, True)
    base, hist, nn = proverb_order(tkeys, tvals, tkids)
    L += [f"## (e) 格言の優先順位（勝ち局面 {nn:,}。当てはまる格言で候補を絞っていき、残った手から無作為に選ぶ）", f"- 無作為（安全な手から）: {base:.1%}", ""]
    for seq, sc in hist:
        L.append(f"- {' → '.join(seq)}: **{sc:.1%}**")
    L.append("")
    print(f"(e) done {time.time() - t0:.0f}s", flush=True)

    out = os.path.join(HERE, "../results/x14_rules.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
