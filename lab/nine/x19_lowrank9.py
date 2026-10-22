"""X19: 低ランクの3駒関係（X18）を 9×9 に持ち込む

評価（手番側から見た値）:
  定数 + Σ_i w[p_i] + Σ_r (a[自玉, r] + c[相手玉, r]) · Σ_{i<j} b[p_i, r] b[p_j, r] + 脅威の特徴量の線形和
  p_i = 駒×マス（手番側/相手 × 14種 × 81マス）と持ち駒の枚数（X7 の P 特徴量）
多重線形（掛け算はするが非線形の変換はない）で NN ではない。重みの数は 約2,500 + (81+81+2,500)·r + 脅威 約1,400

教師: X7 と同じ（棋譜のエンジン評価値を手番側に直して tanh(評価値/1200) を回帰）。対局単位で 9:1 に分割。
比べる相手（X7）: KP 0.576 / KP＋脅威 0.599 / 決定木 0.608（評価値との相関）
--match: 一番良いランクを「固定の駒の価値 + λ×低ランク評価」にして、固定の駒の価値だけと深さ2で対局（X13b の線形版は λ=0.5 で 76%）

使い方: python x19_lowrank9.py --ranks 8,32,64 --match
"""

import argparse
import json
import os
import random
import sys
import time

import cshogi
import numpy as np

import x7_expressivity as x7

HERE = os.path.dirname(os.path.abspath(__file__))
MAXP = 64  # 1局面の駒×マス特徴の最大数（盤上40枚＋持ち駒の温度計表現）
MAXT = 2 * len(x7.T_NAMES)
RESIDUAL = False  # True: 教師の値から固定の駒の価値（X13 の Classic）を引いた残りを学習する（--residual）


def target(sc, board):
    y = float(np.tanh(np.clip(sc, -3000, 3000) / 1200))
    if RESIDUAL:
        from x13_mix import material_balance

        y -= float(np.tanh(material_balance(board, board.turn) * 100 / 1200))
    return y


def encode(board):
    pcs, hm, ho, ks, ko = x7.persp(board)
    p_idx, _, t_idx, _ = x7.features(board)
    p = np.full(MAXP, -1, dtype=np.int32)
    p[: min(len(p_idx), MAXP)] = p_idx[:MAXP]
    t = np.array(t_idx, dtype=np.int32)
    return p, t, ks, ko


def collect(games, every, rng):
    P, T, K, O, Y = [], [], [], [], []
    from cshogi import CSA

    for g in games:
        try:
            comments = CSA.Parser.parse_file(os.path.join(HERE, "../data/floodgate/csa", g["file"]))[0].comments
        except Exception:
            continue
        b = cshogi.Board()
        n = len(g["moves"])
        for ply, mv in enumerate(g["moves"]):
            b.push_usi(mv)
            if ply < 2 or ply >= n - 1 or rng.randrange(every):
                continue
            c = comments[ply] if ply < len(comments) else ""
            if not c:
                continue
            sc = int(c.split()[0])
            sc = sc if b.turn == 0 else -sc
            p, t, ks, ko = encode(b)
            P.append(p)
            T.append(t)
            K.append(ks)
            O.append(ko)
            Y.append(target(sc, b))
    return np.array(P), np.array(T), np.array(K), np.array(O), np.array(Y)


class LowRank9:
    def __init__(self, r, seed=0):
        rng = np.random.default_rng(seed)
        self.r = r
        self.w = np.zeros(x7.P_DIM)
        self.a = rng.normal(0, 0.05, (81, max(r, 1))) * (r > 0)
        self.c = rng.normal(0, 0.05, (81, max(r, 1))) * (r > 0)
        self.b = rng.normal(0, 0.05, (x7.P_DIM, max(r, 1))) * (r > 0)
        self.wt = np.zeros(x7.T_DIM)
        self.bias = 0.0

    def nparams(self):
        return self.w.size + self.a.size + self.c.size + self.b.size + self.wt.size + 1

    def forward(self, P, T, K, O):
        mask = P >= 0
        ii = np.where(mask, P, 0)
        B = self.b[ii] * mask[..., None]
        s = B.sum(1)
        Q = 0.5 * (s * s - (B * B).sum(1))
        coef = self.a[K] + self.c[O]
        out = self.bias + (self.w[ii] * mask).sum(1) + (coef * Q).sum(1) + self.wt[T].sum(1)
        return out, ii, mask, B, s, Q, coef

    def fit(self, P, T, K, O, Y, epochs=3, batch=2048, lr=0.003, l2=1e-6):
        params = ["w", "a", "c", "b", "wt", "bias"]
        m = {p: np.zeros_like(np.asarray(getattr(self, p), dtype=float)) for p in params}
        v = {p: np.zeros_like(m[p]) for p in params}
        rng = np.random.default_rng(1)
        step = 0
        for ep in range(epochs):
            perm = rng.permutation(len(Y))
            for s0 in range(0, len(Y), batch):
                sel = perm[s0 : s0 + batch]
                out, ii, mask, B, s, Q, coef = self.forward(P[sel], T[sel], K[sel], O[sel])
                g = 2 * (out - Y[sel]) / len(sel)
                gr = {"bias": g.sum()}
                gw = np.zeros_like(self.w)
                np.add.at(gw, ii[mask], np.repeat(g, mask.sum(1)))
                gr["w"] = gw
                gQ = g[:, None] * Q
                ga = np.zeros_like(self.a)
                np.add.at(ga, K[sel], gQ)
                gc = np.zeros_like(self.c)
                np.add.at(gc, O[sel], gQ)
                gr["a"], gr["c"] = ga, gc
                cb = (g[:, None] * coef)[:, None, :] * (s[:, None, :] - B)
                gb = np.zeros_like(self.b)
                np.add.at(gb, ii[mask], cb[mask])
                gr["b"] = gb
                gwt = np.zeros_like(self.wt)
                np.add.at(gwt, T[sel].ravel(), np.repeat(g, T.shape[1]))
                gr["wt"] = gwt
                step += 1
                for p in params:
                    x = np.asarray(getattr(self, p))
                    d = gr[p] + l2 * x
                    m[p] = 0.9 * m[p] + 0.1 * d
                    v[p] = 0.999 * v[p] + 0.001 * d * d
                    setattr(self, p, x - lr * (m[p] / (1 - 0.9 ** step)) / (np.sqrt(v[p] / (1 - 0.999 ** step)) + 1e-8))
            lr *= 0.5
        return self

    def predict(self, P, T, K, O):
        return np.concatenate([self.forward(P[s : s + 20000], T[s : s + 20000], K[s : s + 20000], O[s : s + 20000])[0] for s in range(0, len(P), 20000)])

    def __call__(self, board):
        """1局面の評価（探索から呼ぶ）"""
        p, t, ks, ko = encode(board)
        return float(self.forward(p[None], t[None], np.array([ks]), np.array([ko]))[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="8,32,64")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--residual", action="store_true", help="教師から固定の駒の価値を引いた残りを学習（使うときは 固定の駒の価値 + λ×評価）")
    ap.add_argument("--match-ranks", default="", help="対局させるランク（カンマ区切り。省略時は相関が最良のランク）")
    ap.add_argument("--extra", default="", help="教師に足す局面のファイル（data/mainline 内、カンマ区切り）")
    ap.add_argument("--with-imb", action="store_true", help="見慣れない局面（駒の損得のある局面）の半分を学習に足す")
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x19_lowrank9.jsonl"))
    a = ap.parse_args()
    global RESIDUAL
    RESIDUAL = a.residual
    t0 = time.time()
    rng = random.Random(0)
    games = x7.load_games()
    rng.shuffle(games)
    n_test = len(games) // 10
    tr = collect(games[n_test:], a.every, rng)
    te = collect(games[:n_test], a.every, rng)
    # 見慣れない局面（X17 で作った駒の損得のある局面、NNUE 評価つき）: 前半を学習に足す（--with-imb）、後半で外れ方を測る
    with open(os.path.join(x7.DATA, "imbalanced.jsonl")) as f:
        imb = [json.loads(l) for l in f]
    half = len(imb) // 2

    def enc_rows(rows):
        P, T, K, O, Y = [], [], [], [], []
        for r in rows:
            bd = cshogi.Board(r["sfen"] + " 1")
            p, t, ks, ko = encode(bd)
            P.append(p)
            T.append(t)
            K.append(ks)
            O.append(ko)
            Y.append(target(r["score_cp"], bd))
        return np.array(P), np.array(T), np.array(K), np.array(O), np.array(Y)

    off = enc_rows(imb[half:])
    if a.with_imb:
        add = enc_rows(imb[:half])
        tr = tuple(np.concatenate([x, y]) for x, y in zip(tr, add))
    for path in [p for p in a.extra.split(",") if p]:  # 追加の教師（例: X21 の探索中の局面）
        with open(os.path.join(x7.DATA, path)) as f:
            add = enc_rows([json.loads(l) for l in f if l.strip()])
        tr = tuple(np.concatenate([x, y]) for x, y in zip(tr, add))
        print(f"extra {path}: {len(add[4])}", flush=True)
    print(f"train {len(tr[4])} / test {len(te[4])}, {time.time() - t0:.0f}s", flush=True)
    best = None
    models = {}
    for r in [int(x) for x in a.ranks.split(",")]:
        t = time.time()
        m = LowRank9(r).fit(*tr)
        p = m.predict(*te[:4])
        corr = float(np.corrcoef(p, te[4])[0, 1])
        po = m.predict(*off[:4])
        models[r] = m
        res = {"rank": r, "residual": a.residual, "with_imb": a.with_imb, "extra": a.extra, "params": int(m.nparams()), "corr": round(corr, 4), "corr_offdist": round(float(np.corrcoef(po, off[4])[0, 1]), 4), "mse": round(float(np.mean((p - te[4]) ** 2)), 4), "sec": round(time.time() - t)}
        print(json.dumps(res), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        if best is None or corr > best[0]:
            best = (corr, r, m)
    if a.match:
        sys.path.insert(0, os.path.join(HERE, "../corpus"))
        from analyze_positions import Usi
        from x13_mix import Classic, Mix, play

        classic = Classic()
        test_files = {g["file"] for g in games[:n_test]}
        with open(os.path.join(x7.DATA, "exits.jsonl")) as f:
            starts = [json.loads(l) for l in f]
        starts = [s for s in starts if s["file"] in test_files and 20 <= s["ply"] <= 60]
        random.Random(2).shuffle(starts)
        starts = starts[: a.games]
        judge = Usi(1, 64)
        for mr in ([int(x) for x in a.match_ranks.split(",")] if a.match_ranks else [best[1]]):
            ev = Mix(classic, models[mr], a.lam)
            score = {"win": 0, "loss": 0, "draw": 0}
            for s in starts:
                for swap in (False, True):
                    pair = [classic, ev] if swap else [ev, classic]
                    r_, *_ = play(s["sfen"], pair, 2, 160, judge)
                    r_ = -r_ if swap else r_
                    score["win" if r_ > 0 else "loss" if r_ < 0 else "draw"] += 1
            n = sum(score.values())
            tag = ("（残差学習）" if a.residual else "") + ("（見慣れない局面を学習に追加）" if a.with_imb else "") + (f"（追加: {a.extra}）" if a.extra else "")
            res = {"match": f"固定の駒の価値+{a.lam}×{'線形（駒×位置＋脅威）' if mr == 0 else f'低ランクr={mr}'}" + tag, "vs": "固定の駒の価値だけ", "depth": 2, **score, "score_rate": round((score["win"] + 0.5 * score["draw"]) / n, 4)}
            print(json.dumps(res, ensure_ascii=False), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        judge.send("quit")


if __name__ == "__main__":
    main()
