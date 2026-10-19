"""X1: 評価関数の「形」ごとに、どうぶつしょうぎの真値をどこまで表現できるか

真値DB（solve.c の出力）から局面を抜き出し、次の評価関数の族を学習して比べる。
  material : 駒の枚数だけ
  P        : 駒×マス（＋持ち駒の枚数）の線形和（いわゆる駒割＋駒位置）
  PP       : P ＋ 2駒関係
  KPP      : PP ＋ 手番側ライオン位置ごとの2駒関係（Bonanza系の3駒関係に相当）
  PPP      : PP ＋ 任意の3駒関係（ハッシュ）
  GBDT     : P の決定木ブースティング（非線形・非NN）
  MLP      : P の小さいNN（参考。NNUEに相当する立ち位置）

指標:
  acc      : 勝ち/負け確定局面での勝敗の当たり率
  move_win : 勝ち局面で、評価関数が1手読みで選んだ手が勝ちを保つ率
  move_draw: 引き分け局面で、選んだ手が負けに落ちない率
  random   : 同じ局面で手をランダムに選んだときの期待値（下限の基準）

使い方:
  python x1_expressivity.py --data ../data/dobutsu --train 1000000 --test 100000
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
P_DIM = 12 * 14 + 18  # 盤(マス×駒コード) + 持ち駒(6種×枚数0..2)
PP_DIM = P_DIM * P_DIM
HASH_DIM = 1 << 22
DENSE_MAX = 300000


def run_sampler(data, n, seed, with_children):
    out = subprocess.run(
        [os.path.join(HERE, "sample"), data, str(n), str(seed), str(int(with_children))],
        check=True,
        capture_output=True,
    ).stdout
    if not with_children:
        rec = np.frombuffer(out, dtype=[("key", "<u8"), ("val", "u1"), ("imm", "u1"), ("nc", "u1")])
        return rec["key"].copy(), rec["val"].copy(), None
    keys, vals, kids = [], [], []
    p = 0
    while p < len(out):
        k = int.from_bytes(out[p : p + 8], "little")
        v, imm, nc = out[p + 8], out[p + 9], out[p + 10]
        p += 11
        ch = np.frombuffer(out[p : p + 11 * nc], dtype=[("key", "<u8"), ("val", "u1"), ("mv", "<u2")]).copy()
        p += 11 * nc
        if imm or nc == 0:
            continue
        keys.append(k)
        vals.append(v)
        kids.append(ch)
    return np.array(keys, dtype=np.uint64), np.array(vals, dtype=np.uint8), kids


def active_features(keys):
    """各局面の P 特徴のインデックス（N×18、無いところは -1）"""
    keys = keys.astype(np.uint64)
    n = len(keys)
    idx = np.full((n, 18), -1, dtype=np.int64)
    for s in range(12):
        code = ((keys >> np.uint64(4 * s)) & np.uint64(15)).astype(np.int64)
        idx[:, s] = np.where(code > 0, s * 14 + code, -1)
    for j in range(6):
        cnt = ((keys >> np.uint64(48 + 2 * j)) & np.uint64(3)).astype(np.int64)
        idx[:, 12 + j] = 168 + j * 3 + cnt
    # 盤上の駒を前に詰めて並べる（ソートすれば a<b の組が作りやすい）
    idx = np.sort(np.where(idx < 0, np.iinfo(np.int64).max, idx), axis=1)
    idx[idx == np.iinfo(np.int64).max] = -1
    return idx


def lion_square(keys):
    sq = np.zeros(len(keys), dtype=np.int64)
    for s in range(12):
        code = ((keys >> np.uint64(4 * s)) & np.uint64(15)).astype(np.int64)
        sq = np.where(code == 1, s, sq)
    return sq


def material(keys):
    feats = []
    for code in (2, 3, 4, 5, 10, 11, 12, 13):
        c = np.zeros(len(keys))
        for s in range(12):
            c += ((keys >> np.uint64(4 * s)) & np.uint64(15)) == code
        feats.append(c)
    for j in range(6):
        feats.append(((keys >> np.uint64(48 + 2 * j)) & np.uint64(3)).astype(float))
    return sp.csr_matrix(np.stack(feats, axis=1))


def build(keys, kind):
    if kind == "material":
        return material(keys)
    idx = active_features(keys)
    n, w = idx.shape
    cols = [idx]
    offset = P_DIM
    if kind in ("PP", "KPP", "PPP"):
        a, b = np.triu_indices(w, 1)
        ia, ib = idx[:, a], idx[:, b]
        pp = np.where((ia >= 0) & (ib >= 0), offset + ia * P_DIM + ib, -1)
        cols.append(pp)
        offset += PP_DIM
        if kind == "KPP":
            ls = lion_square(keys)[:, None]
            cols.append(np.where(pp >= 0, offset + ls * PP_DIM + (pp - P_DIM), -1))
            offset += 12 * PP_DIM
        if kind == "PPP":
            t = np.array([(x, y, z) for x in range(w) for y in range(x + 1, w) for z in range(y + 1, w)])
            ia, ib, ic = idx[:, t[:, 0]], idx[:, t[:, 1]], idx[:, t[:, 2]]
            h = ((ia * P_DIM + ib) * P_DIM + ic) % HASH_DIM
            cols.append(np.where((ia >= 0) & (ib >= 0) & (ic >= 0), offset + h, -1))
            offset += HASH_DIM
    allc = np.concatenate(cols, axis=1)
    rows = np.repeat(np.arange(n), allc.shape[1])
    flat = allc.ravel()
    m = flat >= 0
    return sp.csr_matrix((np.ones(m.sum(), dtype=np.float32), (rows[m], flat[m])), shape=(n, offset))


def labels(vals):
    """勝ち=1 負け=0、引き分けは 0 と 1 を重み0.5ずつ"""
    win = (vals > 0) & ((vals.astype(int) - 1) % 2 == 1)
    draw = vals == 0
    y = np.concatenate([win.astype(int), np.ones(draw.sum(), dtype=int)])
    w = np.concatenate([np.where(draw, 0.5, 1.0), np.full(draw.sum(), 0.5)])
    rows = np.concatenate([np.arange(len(vals)), np.where(draw)[0]])
    return rows, y, w


def cls(v):
    v = v.astype(int)
    return np.where(v == 0, 0, np.where((v - 1) % 2 == 1, 1, -1))  # 1勝ち 0引分 -1負け


class Model:
    def __init__(self, kind):
        self.kind = kind
        self.dense = kind in ("GBDT", "MLP")

    def feats(self, keys):
        if self.dense:
            return build(keys, "P").toarray().astype(np.float32)
        return build(keys, self.kind)

    def fit(self, keys, vals):
        if self.dense:  # 密行列はメモリを食うので件数を絞る
            keys, vals = keys[:DENSE_MAX], vals[:DENSE_MAX]
        rows, y, w = labels(vals)
        if self.kind == "GBDT":
            self.m = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63)
            self.m.fit(self.feats(keys)[rows], y, sample_weight=w)
        elif self.kind == "MLP":
            self.m = MLPClassifier(hidden_layer_sizes=(256, 32), max_iter=int(os.environ.get("X1_MLP_ITER", "20")), batch_size=1024)
            self.m.fit(self.feats(keys)[rows], y)  # MLPは重み非対応: 引分は両ラベル1回ずつ
        else:
            self.m = SGDClassifier(loss="log_loss", alpha=1e-7, learning_rate="adaptive", eta0=0.05)
            bs = 50000
            perm = np.random.default_rng(0).permutation(len(rows))
            for ep in range(4):
                for s in range(0, len(perm), bs):
                    sel = perm[s : s + bs]
                    X = self.feats(keys[rows[sel]])
                    self.m.partial_fit(X, y[sel], classes=[0, 1], sample_weight=w[sel])
                perm = np.random.default_rng(ep + 1).permutation(len(rows))
        return self

    def p_win(self, keys):
        out = []
        for s in range(0, len(keys), 100000):
            out.append(self.m.predict_proba(self.feats(keys[s : s + 100000]))[:, 1])
        return np.concatenate(out)


def evaluate(model, tkeys, tvals, tkids):
    c = cls(tvals)
    dec = c != 0
    pw = model.p_win(tkeys)
    acc = float(((pw > 0.5) == (c == 1))[dec].mean())
    flat = np.concatenate([k["key"] for k in tkids])
    off = np.cumsum([0] + [len(k) for k in tkids])
    cp = model.p_win(flat)
    # 相手に即勝ちを許す手は1手読みで分かるので、評価関数に関係なく「最悪」とみなす
    cp[np.concatenate([k["val"] for k in tkids]) == 2] = 1.0
    stats = {"win": [0, 0, 0.0], "draw": [0, 0, 0.0]}
    for i, kd in enumerate(tkids):
        cc = cls(kd["val"])
        good = (cc == -1) if c[i] == 1 else (cc == 0) if c[i] == 0 else None
        if good is None or good.all():
            continue  # 負け局面・どれを指しても同じ局面は除外（「難所」だけを見る）
        name = "win" if c[i] == 1 else "draw"
        choice = np.argmin(cp[off[i] : off[i + 1]])
        stats[name][0] += 1
        stats[name][1] += int(good[choice])
        stats[name][2] += float(good.mean())
    res = {"acc": acc}
    for name, (n, ok, rnd) in stats.items():
        res[f"move_{name}"] = ok / max(n, 1)
        res[f"random_{name}"] = rnd / max(n, 1)
        res[f"n_{name}"] = n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--train", type=int, default=1000000)
    ap.add_argument("--test", type=int, default=100000)
    ap.add_argument("--kinds", default="material,P,PP,KPP,PPP,GBDT,MLP")
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x1_expressivity.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    keys, vals, _ = run_sampler(a.data, a.train, 1, False)
    # 即勝ち局面（d=1: ライオンを取れる/トライ成立）は1手読みで分かるので学習・評価から除く
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    tkeys, tvals, tkids = run_sampler(a.data, a.test, 2, True)
    for kind in a.kinds.split(","):
        t = time.time()
        m = Model(kind).fit(keys, vals)
        res = {"kind": kind, "train": a.train, **evaluate(m, tkeys, tvals, tkids), "sec": round(time.time() - t)}
        print(json.dumps(res, ensure_ascii=False), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
