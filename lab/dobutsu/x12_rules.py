"""X12: どうぶつしょうぎの規則を、いろいろな方向から探して真値で確かめる

  ① 特徴の条件（X10 の拡張）: 駒の種類ごとの枚数・ライオン同士の距離・玉の周りの守り駒などを足して、条件3つまでの規則を探す
  ② 形の規則: 自ライオン（または相手ライオン）を中心にした 3×3 の形と持ち駒だけで勝敗が決まるもの。
     盤上の位置によらない形の定理を、真値DBの全局面で確かめる（--full）
  ③ 手の規則（格言の採点）: 「取れる駒は取る」「ライオンを前に出す」などの指し方の規則が、勝ち局面で正しい手を選ぶ率
  ④ 手番の規則: 手番が損になる局面（指すと負けるが、パスできれば負けない）の割合と、それが起きやすい局面
  ⑤ 手数の上限: 「この条件なら最長でも何手で勝てる」を全局面で求める（--full）
  ⑥ 点数表: 人が暗算できる小さな整数の点数表で、どこまで真値に近づくか

出力: ../results/x12_rules.md
使い方: python x12_rules.py --n 3000000 --full
"""

import argparse
import collections
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

os.environ.setdefault("X5_LION", "1")
from x1_expressivity import HERE, cls, evaluate, labels, run_sampler  # noqa: E402
from x10_theory import JA, beam, fmt, mask_of  # noqa: E402
from x10_theory import feats as base_feats  # noqa: E402
from x4_patterns import DB, mirror_key  # noqa: E402
from x5_threats import ATT, POP, threat_values  # noqa: E402

PIECE = {2: "キリン", 3: "ぞう", 4: "ひよこ", 5: "にわとり"}
JA2 = dict(JA)
for c, n in PIECE.items():
    JA2[f"my_{c}"] = f"盤上の自分の{n}"
    JA2[f"op_{c}"] = f"盤上の相手の{n}"
for j, n in enumerate(["キリン", "ぞう", "ひよこ"]):
    JA2[f"my_h{j}"] = f"持ち駒の{n}（自分）"
    JA2[f"op_h{j}"] = f"持ち駒の{n}（相手）"
JA2.update({"my_mob": "自分の自由度", "op_mob": "相手の自由度", "dist": "ライオン同士の距離", "my_def": "自ライオンの隣の自分の駒", "op_def": "相手ライオンの隣の相手の駒", "my_col": "自ライオンが中央の筋", "op_col": "相手ライオンが中央の筋"})


def decode_codes(keys):
    keys = keys.astype(np.uint64)
    return np.stack([((keys >> np.uint64(4 * s)) & np.uint64(15)).astype(np.int64) for s in range(12)], axis=1)


def feats(keys):
    F = base_feats(keys)
    tv, _ = threat_values(keys.astype(np.uint64))
    F["my_mob"] = tv["my_mob"].astype(np.int8)
    F["op_mob"] = tv["op_mob"].astype(np.int8)
    code = decode_codes(keys)
    keys = keys.astype(np.uint64)
    for c in (2, 3, 4, 5):
        F[f"my_{c}"] = (code == c).sum(1).astype(np.int8)
        F[f"op_{c}"] = (code == c + 8).sum(1).astype(np.int8)
    for j in range(3):
        F[f"my_h{j}"] = ((keys >> np.uint64(48 + 2 * j)) & np.uint64(3)).astype(np.int8)
        F[f"op_h{j}"] = ((keys >> np.uint64(54 + 2 * j)) & np.uint64(3)).astype(np.int8)
    ml = np.argmax(code == 1, axis=1)
    ol = np.argmax(code == 9, axis=1)
    F["dist"] = np.maximum(abs(ml // 3 - ol // 3), abs(ml % 3 - ol % 3)).astype(np.int8)
    mine = (code >= 2) & (code <= 5)
    theirs = code >= 10
    sq = np.arange(12)
    F["my_def"] = (mine & (((ATT[1][ml][:, None] >> sq[None, :]) & 1) == 1)).sum(1).astype(np.int8)
    F["op_def"] = (theirs & (((ATT[1][ol][:, None] >> sq[None, :]) & 1) == 1)).sum(1).astype(np.int8)
    F["my_col"] = (ml % 3 == 1).astype(np.int8)
    F["op_col"] = (ol % 3 == 1).astype(np.int8)
    return F


def window_key(keys, around_opp=False):
    """ライオンを中心にした3×3（盤の外=15）と持ち駒（12bit）を1つの整数にする。左右反転で小さい方に正規化"""
    code = decode_codes(keys)
    keys = keys.astype(np.uint64)
    lion = np.argmax(code == (9 if around_opp else 1), axis=1)
    lr, lc = lion // 3, lion % 3
    cells = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = lr + dr, lc + dc
            ok = (r >= 0) & (r < 4) & (c >= 0) & (c < 3)
            cells.append(np.where(ok, code[np.arange(len(keys)), np.clip(r, 0, 3) * 3 + np.clip(c, 0, 2)], 15))
    rows = lr  # 段の情報（トライ/成り）を含める
    hands = (keys >> np.uint64(48)).astype(np.int64) & 0xFFF

    def pack(cs):
        k = rows.astype(np.int64)
        for x in cs:
            k = k * 16 + x
        return k * 4096 + hands

    a = pack(cells)
    mirrored = [cells[i * 3 + (2 - j)] for i in range(3) for j in range(3)]
    return np.minimum(a, pack(mirrored))


def show_window(k):
    hands = k % 4096
    k //= 4096
    cells = []
    for _ in range(9):
        cells.append(k % 16)
        k //= 16
    cells = cells[::-1]
    row = k
    sym = {0: "・", 1: "獅", 2: "麒", 3: "象", 4: "雛", 5: "鶏", 9: "v獅", 10: "v麒", 11: "v象", 12: "v雛", 13: "v鶏", 15: "＃"}
    grid = " / ".join("".join(sym.get(int(cells[i * 3 + j]), "?") for j in range(3)) for i in range(3))
    hm = [(hands >> (2 * j)) & 3 for j in range(3)]
    ho = [(hands >> (6 + 2 * j)) & 3 for j in range(3)]
    return f"段{row}: {grid}（持ち駒 自 麒{hm[0]}象{hm[1]}雛{hm[2]} / 相 麒{ho[0]}象{ho[1]}雛{ho[2]}）"


# ---- ③ 手の規則 ----
PV = {2: 3, 3: 3, 4: 1, 5: 4}  # 取る駒の価値の目安（キリン・ぞう3、ひよこ1、にわとり4）


def move_rules(tkeys, tvals, tkids):
    """勝ち局面で、規則が選ぶ手が勝ちを保つ率。規則が当てはまらない局面は数えない"""
    c = cls(tvals)
    res = collections.defaultdict(lambda: [0, 0, 0.0])
    for i, kd in enumerate(tkids):
        if c[i] != 1:
            continue
        good = cls(kd["val"]) == -1
        safe = kd["val"] != 2  # 相手に即勝ちを許さない手
        if good.all() or not safe.any():
            continue
        mv = kd["mv"].astype(int)
        frm, to, piece, cap = mv & 15, (mv >> 4) & 15, (mv >> 8) & 7, (mv >> 12) & 7
        cf = feats(kd["key"])  # 子局面（相手から見た値）
        base_rate = good[safe].mean()
        cands = {
            "取れる駒は一番高い駒を取る": np.where(cap > 0, np.vectorize(lambda x: PV.get(x, 0))(cap), -1),
            "相手の自由度が一番小さくなる手": -cf["my_mob"].astype(int),
            "相手の浮き駒を作らず自分の浮き駒が一番少ない手": -cf["op_hang"].astype(int) * 10 + 0,
            "ライオンを前に出す（安全な前進）": np.where((piece == 1) & (frm != 15) & (to // 3 < frm // 3), 1, -1),
            "相手ライオンの周りへの利きが一番多くなる手": cf["op_press"].astype(int),
            "自ライオンの逃げ場が一番多くなる手": cf["op_safe"].astype(int),
            "持ち駒を相手ライオンの近くに打つ": np.where(frm == 15, -cf["dist"].astype(int), -99),
            "ひよこを成る": np.where((piece == 4) & (to // 3 == 0) & (frm != 15), 1, -1),
            "駒を取らずにライオンを相手陣へ近づける": np.where((piece == 1) & (cap == 0) & (frm != 15), 3 - cf["op_row"].astype(int), -9),
        }
        for name, score in cands.items():
            score = np.where(safe, score, -999)
            best = score.max()
            if best < 0 and name not in ("相手の自由度が一番小さくなる手", "相手の浮き駒を作らず自分の浮き駒が一番少ない手", "持ち駒を相手ライオンの近くに打つ"):
                continue  # 当てはまらない
            if name == "持ち駒を相手ライオンの近くに打つ" and best <= -99:
                continue
            pick = np.where(score == best)[0]
            r = res[name]
            r[0] += 1
            r[1] += float(good[pick].mean())  # 同点はその中で無作為に選ぶ期待値
            r[2] += float(base_rate)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--n", type=int, default=3000000)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    keys, vals, _ = run_sampler(a.data, a.n, 9, False)
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    c = cls(vals)
    F = feats(keys)
    hk, hv, _ = run_sampler(a.data, a.n, 10, False)
    keep = hv != 2
    hk, hv = hk[keep], hv[keep]
    H = feats(hk)
    hc = cls(hv)
    n = len(c)
    L = ["# X12 どうぶつしょうぎの規則を、いろいろな方向から", "", f"探索 {n:,} 局面 / 確認用 {len(hc):,} 局面（即勝ちの局面は除く）", ""]

    # ① 特徴の条件（拡張）
    one, hone = np.ones(n, bool), np.ones(len(hc), bool)
    tasks = [
        ("逆転の勝ち（駒損なのに勝ち）", c == 1, F["mat"] <= -1, 0.60, hc == 1, H["mat"] <= -1),
        ("逆転の負け（駒得なのに負け）", c == -1, F["mat"] >= 1, 0.80, hc == -1, H["mat"] >= 1),
        ("定理候補・勝ち", c == 1, one, 1.0, hc == 1, hone),
        ("定理候補・負け", c == -1, one, 1.0, hc == -1, hone),
        ("引き分け（千日手）になりやすい", c == 0, one, 0.30, hc == 0, hone),
    ]
    import x10_theory

    x10_theory.JA = JA2
    L += ["## ① 特徴の条件（駒の種類・ライオン同士の距離・守り駒などを追加）", ""]
    for name, target, base, prec, ht, hb in tasks:
        L += [f"### {name}（前提の中の基準: {target[base].mean():.1%}）", "| 規則 | 精度（探索） | 精度（確認用） | 当てはまる割合 |", "|---|---|---|---|"]
        for conds, p, sup in beam(F, target, base, prec, max(200, n // 2000)):
            m = mask_of(H, conds, hb)
            L.append(f"| {fmt(conds)} | {p:.2%} | {ht[m].mean() if m.any() else float('nan'):.2%} | {sup / base.sum():.2%} |")
        L.append("")
    print(f"① done {time.time() - t0:.0f}s", flush=True)

    # ② 形の規則（ライオン中心の3×3＋持ち駒＋段）
    L += ["## ② 形の規則（ライオンを中心にした3×3の形・段・持ち駒だけで決まるもの）", ""]
    shape_cands = {}
    for around_opp, title in ((False, "自ライオンの周り"), (True, "相手ライオンの周り")):
        wk = window_key(keys, around_opp)
        u, inv, cnt = np.unique(wk, return_inverse=True, return_counts=True)
        win = np.bincount(inv, weights=(c == 1)).astype(int)
        loss = np.bincount(inv, weights=(c == -1)).astype(int)
        L += [f"### {title}: 形の種類 {len(u):,}", "| 形 | 出現 | 勝ち | 負け |", "|---|---|---|---|"]
        pure = [(cnt[i], u[i], "勝ち" if win[i] == cnt[i] else "負け") for i in range(len(u)) if cnt[i] >= 300 and (win[i] == cnt[i] or loss[i] == cnt[i])]
        pure.sort(key=lambda x: -x[0])
        for k_, w_, kind in pure[:10]:
            L.append(f"| {show_window(int(w_))} | {k_} | {kind} | |")
        shape_cands[around_opp] = {int(w_): kind for _, w_, kind in pure[:200]}
        L += [f"（全て勝ち/全て負けで、出現300回以上の形: {len(pure)} 種類。上位10を表示）", ""]
    print(f"② done {time.time() - t0:.0f}s", flush=True)

    # ③ 手の規則
    tkeys, tvals, tkids = run_sampler(a.data, 300000, 2, True)
    mr = move_rules(tkeys, tvals, tkids)
    L += ["## ③ 手の規則（格言の採点）: 勝ち局面（どの手でもよいわけではないもの）で、規則が選ぶ手が勝ちを保つ率", "", "| 規則 | 当てはまった局面 | 規則の手が正しい率 | 無作為（安全な手から）| 差 |", "|---|---|---|---|---|"]
    for name, (m, ok, base) in sorted(mr.items(), key=lambda x: -(x[1][1] / max(x[1][0], 1) - x[1][2] / max(x[1][0], 1))):
        L.append(f"| {name} | {m:,} | {ok / m:.1%} | {base / m:.1%} | {100 * (ok - base) / m:+.1f}pt |")
    L.append("")
    print(f"③ done {time.time() - t0:.0f}s", flush=True)

    # ④ 手番の規則（パスした局面を DB で引く）
    db = DB(a.data)
    order_cache = {}

    def lookup(kk):
        out = np.zeros(len(kk), dtype=np.int16)
        code = decode_codes(kk)
        ml = np.argmax(code == 1, axis=1)
        ol = np.argmax(code == 9, axis=1)
        for g in np.unique(ml * 12 + ol):
            sel = np.where(ml * 12 + ol == g)[0]
            if g not in order_cache:
                ks, vs = db.group(g // 12, g % 12)
                o = np.argsort(ks)
                order_cache[g] = (ks[o], vs[o])
            ks, vs = order_cache[g]
            q = kk[sel]
            pos = np.clip(np.searchsorted(ks, q), 0, len(ks) - 1)
            hit = ks[pos] == q
            out[sel] = np.where(hit, vs[pos].astype(np.int16), -1)
        return out

    def pass_key(kk):
        """同じ盤で相手の手番にした局面（相手から見た向き）に直し、左右反転の小さい方に"""
        code = decode_codes(kk)
        kk = kk.astype(np.uint64)
        out = np.zeros(len(kk), dtype=np.uint64)
        for s in range(12):
            v = code[:, s]
            v = np.where(v > 0, v ^ 8, 0).astype(np.uint64)
            out |= v << np.uint64(4 * (11 - s))
        hm = (kk >> np.uint64(48)) & np.uint64(0x3F)
        ho = (kk >> np.uint64(54)) & np.uint64(0x3F)
        out |= ho << np.uint64(48)
        out |= hm << np.uint64(54)
        mir = np.array([mirror_key(int(x)) for x in out], dtype=np.uint64)
        return np.minimum(out, mir)

    sub = np.random.default_rng(0).choice(len(keys), size=min(300000, len(keys)), replace=False)
    pk = pass_key(keys[sub])
    pv = lookup(pk)
    found = pv >= 0
    pc = np.where(found, cls(np.maximum(pv, 0).astype(np.uint8)), 9)
    # 手番側が負け（指すと負け）なのに、パスすれば相手が負け or 引き分け（= 手番が損）
    me = c[sub]
    zug = (me == -1) & found & (pc != 1)
    mutual = (me == -1) & found & (pc == -1)
    L += ["## ④ 手番の規則（ツークツワンク）", "", f"手番側の負け局面のうち、パスした局面がDBにあるもの: {(me == -1)[found].sum():,} / 負け局面 {(me == -1).sum():,}（パスした局面は到達できないことがある）", f"- 指すと負けるが、パスできれば負けない（手番が損）: **{zug.sum() / max(((me == -1) & found).sum(), 1):.1%}**", f"- そのうち「どちらが指しても負け」（相互ツークツワンク）: {mutual.sum() / max(((me == -1) & found).sum(), 1):.1%}", ""]
    Fs = {k: v[sub] for k, v in F.items()}
    base_l = (me == -1) & found
    zt = beam(Fs, zug, base_l, 0.5, 200, depth=2)
    L += ["手番が損になりやすい条件（負け局面の中で）:", "| 条件 | 手番が損な割合 | 当てはまる割合 |", "|---|---|---|"]
    for conds, p, sup in zt:
        L.append(f"| {fmt(conds)} | {p:.1%} | {sup / base_l.sum():.1%} |")
    L.append("")
    print(f"④ done {time.time() - t0:.0f}s", flush=True)

    # ⑥ 点数表
    names = ["mat", "my_2", "my_3", "my_4", "my_5", "op_2", "op_3", "op_4", "op_5", "my_h0", "my_h1", "my_h2", "op_h0", "op_h1", "op_h2", "my_row", "op_row", "my_fwd", "op_fwd", "my_press", "op_press", "my_hang", "op_hang", "my_safe", "op_safe", "op_try_n", "check"]
    X = np.stack([F[k].astype(np.float32) for k in names], axis=1)
    rows, y, w = labels(vals)
    lr = LogisticRegression(max_iter=300).fit(X[rows][:600000], y[:600000], sample_weight=w[:600000])
    coef = lr.coef_[0]
    L += ["## ⑥ 点数表（人が暗算できる整数の重み）", ""]
    for scale_name, scale in (("連続値（参考）", None), ("整数 -3〜+3", 3), ("整数 -1〜+1", 1)):
        if scale is None:
            wv, b = coef, lr.intercept_[0]
        else:
            s = scale / np.abs(coef).max()
            wv = np.round(coef * s)
            b = np.round(lr.intercept_[0] * s)

        class PT:
            def p_win(self, kk, wv=wv, b=b):
                Fk = feats(kk)
                return 1 / (1 + np.exp(-(np.stack([Fk[k].astype(np.float32) for k in names], 1) @ wv + b)))

        r = evaluate(PT(), tkeys, tvals, tkids)
        L.append(f"- {scale_name}: 勝敗の当たり率 {r['acc']:.1%} / 1手読み誤り率 {1 - r['move_win']:.1%} / 引き分けで負けに落ちる率 {1 - r['move_draw']:.1%}")
        if scale == 3:
            L.append("  - 点数表: " + "、".join(f"{JA2.get(k, k)} {int(v):+d}" for k, v in zip(names, wv) if v) + f"、定数 {int(b):+d}")
    L += ["- 比較: 駒数だけ 13.9% / KPP 6.0% / 決定木 4.35%", ""]
    print(f"⑥ done {time.time() - t0:.0f}s", flush=True)

    # ②の全数検証と ⑤ 手数の上限（全局面を少しずつ読む）
    if a.full:
        pk_ = np.memmap(os.path.join(a.data, "plain_keys.bin"), dtype=np.uint64, mode="r")
        pv_ = np.memmap(os.path.join(a.data, "plain_vals.bin"), dtype=np.uint8, mode="r")
        shape_stat = {side: {k: [0, 0] for k in d} for side, d in shape_cands.items()}
        cand_arr = {side: np.array(sorted(d), dtype=np.int64) for side, d in shape_cands.items()}
        maxd = collections.defaultdict(int)
        for s in range(0, len(pk_), 400000):
            k = np.asarray(pk_[s : s + 400000])
            v = np.asarray(pv_[s : s + 400000])
            keep = v != 2
            k, v = k[keep], v[keep]
            gc = cls(v)
            for side in (False, True):
                wk = window_key(k, side)
                hit = np.isin(wk, cand_arr[side])
                for kk_, cc_ in zip(wk[hit], gc[hit]):
                    st = shape_stat[side][int(kk_)]
                    st[0] += 1
                    st[1] += (cc_ == 1) if shape_cands[side][int(kk_)] == "勝ち" else (cc_ == -1)
            G = base_feats(k)
            d = v.astype(int) - 1
            winm = (v > 0) & (d % 2 == 1)
            for mt in range(-6, 7):
                sel = winm & (G["mat"] == mt)
                if sel.any():
                    maxd[mt] = max(maxd[mt], int(d[sel].max()))
        L += ["## ② の全数検証（真値DBの全局面）", "| 形 | 当てはまる局面 | 結論どおり | 判定 |", "|---|---|---|---|"]
        for side, d in shape_stat.items():
            items = sorted(d.items(), key=lambda x: -x[1][0])[:15]
            for k_, (m, ok) in items:
                L.append(f"| {'相手' if side else '自'}ライオン中心 {show_window(k_)} → {shape_cands[side][k_]} | {m:,} | {ok:,} | {'**定理**' if ok == m else f'{ok / m:.3%}'} |")
        L += ["", "## ⑤ 手数の上限（勝ち局面で、勝つまでの最長手数。真値DBの全局面）", "| 駒の枚数の差 | 最長の勝ち手数 |", "|---|---|"]
        for mt in sorted(maxd):
            L.append(f"| {mt:+d} | {maxd[mt]} |")
        print(f"full done {time.time() - t0:.0f}s", flush=True)

    out = os.path.join(HERE, "../results/x12_rules.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
