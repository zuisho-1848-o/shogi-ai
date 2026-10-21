"""X18（A5）: 3駒関係（KPP）を低ランクの形にしても精度が保てるか

KPP の重み W[k][p][q]（k=自ライオンのマス、p,q=駒×マス）を
    W[k][p][q] ≈ Σ_r a[k, r] · b[p, r] · b[q, r]
の形（ランク r）に絞って、直接学習する（NN ではない: 掛け算はするが非線形の変換はない多重線形）。
評価 = 定数 + Σ_i w[p_i] + Σ_r a[k, r] · Σ_{i<j} b[p_i, r] b[p_j, r]（＋ 脅威の特徴量の線形和）
重みの数は 186 + 12r + 186r（r=8 で約1,800。完全な KPP は約41万）。

比べる相手（X1/X5）: P 10.5% / PP 7.1% / KPP 6.0% / KPP＋脅威 5.3% / 小NN 4.5%（1手読み誤り率）

使い方: python x18_lowrank.py --ranks 1,2,4,8,16,32
"""

import argparse
import json
import os
import time

import numpy as np

from x1_expressivity import HERE, active_features, evaluate, labels, lion_square, run_sampler
from x5_threats import threat_sparse

P_DIM = 186


class LowRank:
    def __init__(self, r, with_t, seed=0):
        rng = np.random.default_rng(seed)
        self.r, self.with_t = r, with_t
        self.w = np.zeros(P_DIM)
        self.a = rng.normal(0, 0.1, (12, r))
        self.b = rng.normal(0, 0.1, (P_DIM, r))
        self.bias = 0.0
        self.wt = None

    def nparams(self):
        return P_DIM + 12 * self.r + P_DIM * self.r + (len(self.wt) if self.wt is not None else 0)

    def _parts(self, keys):
        idx = active_features(keys)
        mask = idx >= 0
        ii = np.where(mask, idx, 0)
        k = lion_square(keys)
        T = threat_sparse(keys) if self.with_t else None
        return ii, mask, k, T

    def _forward(self, ii, mask, k, T):
        B = self.b[ii] * mask[..., None]  # n×18×r
        s = B.sum(1)
        sq = (B * B).sum(1)
        Q = 0.5 * (s * s - sq)  # 駒の組ごとの相性の和（ランクごと）
        out = self.bias + (self.w[ii] * mask).sum(1) + (self.a[k] * Q).sum(1)
        if T is not None:
            out = out + T @ self.wt
        return out, B, s, Q

    def fit(self, keys, vals, epochs=3, batch=4096, lr=0.01, l2=1e-6):
        rows, y, wts = labels(vals)
        keys = keys[rows]
        ii, mask, k, T = self._parts(keys)
        if self.with_t:
            self.wt = np.zeros(T.shape[1])
        params = ["w", "a", "b", "bias"] + (["wt"] if self.with_t else [])
        m = {p: np.zeros_like(np.asarray(getattr(self, p), dtype=float)) for p in params}
        v = {p: np.zeros_like(m[p]) for p in params}
        step = 0
        rng = np.random.default_rng(1)
        n = len(y)
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s0 in range(0, n, batch):
                sel = perm[s0 : s0 + batch]
                Ts = T[sel] if T is not None else None
                out, B, s, Q = self._forward(ii[sel], mask[sel], k[sel], Ts)
                g = (1 / (1 + np.exp(-out)) - y[sel]) * wts[sel] / len(sel)
                grads = {"bias": g.sum()}
                gw = np.zeros(P_DIM)
                np.add.at(gw, ii[sel][mask[sel]], np.repeat(g, mask[sel].sum(1)))
                grads["w"] = gw
                ga = np.zeros_like(self.a)
                np.add.at(ga, k[sel], g[:, None] * Q)
                grads["a"] = ga
                # d/d b[p_i] = g · a[k] · (s - b[p_i])
                coef = (g[:, None] * self.a[k[sel]])[:, None, :] * (s[:, None, :] - B)  # n×18×r
                gb = np.zeros_like(self.b)
                np.add.at(gb, ii[sel][mask[sel]], coef[mask[sel]])
                grads["b"] = gb
                if Ts is not None:
                    grads["wt"] = Ts.T @ g
                step += 1
                for p in params:
                    gr = grads[p] + l2 * np.asarray(getattr(self, p))
                    m[p] = 0.9 * m[p] + 0.1 * gr
                    v[p] = 0.999 * v[p] + 0.001 * gr * gr
                    upd = lr * (m[p] / (1 - 0.9 ** step)) / (np.sqrt(v[p] / (1 - 0.999 ** step)) + 1e-8)
                    setattr(self, p, np.asarray(getattr(self, p)) - upd)
            lr *= 0.5
        return self

    def p_win(self, keys):
        out = []
        for s in range(0, len(keys), 100000):
            o, *_ = self._forward(*self._parts(keys[s : s + 100000]))
            out.append(1 / (1 + np.exp(-o)))
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--train", type=int, default=3000000)
    ap.add_argument("--test", type=int, default=300000)
    ap.add_argument("--ranks", default="1,2,4,8,16,32")
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x18_lowrank.jsonl"))
    a = ap.parse_args()
    keys, vals, _ = run_sampler(a.data, a.train, 1, False)
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    tkeys, tvals, tkids = run_sampler(a.data, a.test, 2, True)
    ap_t = os.environ.get("X18_THREAT", "both")  # both / only（脅威の特徴量ありだけ）
    for with_t in ((True,) if ap_t == "only" else (False, True)):
        for r in [int(x) for x in a.ranks.split(",")]:
            t = time.time()
            m = LowRank(r, with_t).fit(keys, vals)
            res = {"rank": r, "threat": with_t, "params": int(m.nparams()), **evaluate(m, tkeys, tvals, tkids), "sec": round(time.time() - t)}
            print(json.dumps({k: res[k] for k in ("rank", "threat", "params", "acc", "move_win", "move_draw", "sec")}), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
