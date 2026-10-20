"""X5: 「脅威の構造」を表す少数の特徴量を KPP に足すと、小NNとの差（と真値との差）が埋まるか

X1b で、評価関数の誤りは長期の勝ち筋（静かな手・駒損の代償・相手ライオンの前進）に集中していた。
NN が学んでいるのが「利きの関係から生まれる脅威」なら、それを明示した特徴量で線形モデルでも届くはず。

足す特徴量（すべて手番側から見た値。整数を区分して one-hot にする）:
  my_safe / op_safe       : ライオンが動ける安全なマス数（相手の利きがなく、自駒もない）
  my_try / op_try         : トライの段（奥の段）までの距離と、そこに安全に入れるマスの数
  my_press / op_press     : 相手ライオンの周り8マスに利いている自分の駒の数（攻めの圧力）
  my_hang / op_hang       : 相手に利かれていて、自分の駒で守られていない駒の数（浮き駒）
  my_mob / op_mob         : 盤上の駒の移動先の数＋打てるマス×持ち駒の種類数（自由度）
  check                   : 自ライオンに相手の利きがあるか
さらに各特徴を (自ライオンの段, 相手ライオンの段) と掛け合わせた版も入れる（線形のまま、局面の型ごとに重みを変える）

使い方: python x5_threats.py --train 3000000 --test 300000
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

import x1_expressivity as x1
from x1_expressivity import HERE, Model, evaluate, run_sampler

DIRS = {
    1: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    2: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    3: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    4: [(-1, 0)],
    5: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)],
}

# ATT[code, s] = その駒が s にいるとき利いているマスのビット集合（相手駒は向きが逆）
ATT = np.zeros((16, 12), dtype=np.int64)
for code in list(range(1, 6)) + list(range(9, 14)):
    t, opp = code & 7, code >= 8
    for s in range(12):
        r, c = divmod(s, 3)
        m = 0
        for dr, dc in DIRS[t]:
            nr, nc = (r - dr, c + dc) if opp else (r + dr, c + dc)
            if 0 <= nr < 4 and 0 <= nc < 3:
                m |= 1 << (nr * 3 + nc)
        ATT[code, s] = m
NEIGH = ATT[1]  # ライオンの周り
ROW0 = 0b000000000111
ROW3 = 0b111000000000
POP = np.array([bin(i).count("1") for i in range(1 << 12)], dtype=np.int64)

BUCKETS = 10  # 各特徴量は 0..9 にクリップ
NAMES = ["my_safe", "op_safe", "my_try_d", "my_try_n", "op_try_d", "op_try_n", "my_press", "op_press", "my_hang", "op_hang", "my_mob", "op_mob", "check"]
# X9 の仮説（長期の勝ちは「ライオンの活動力」で駒損を補う）を試すための追加特徴量。LION=1 のときだけ使う
LION_NAMES = ["my_fwd", "op_fwd", "my_center", "op_center", "mat_x_myrow", "mat_x_oprow"]
LION = int(os.environ.get("X5_LION", "0"))


def threat_values(keys):
    keys = keys.astype(np.uint64)
    n = len(keys)
    code = np.stack([((keys >> np.uint64(4 * s)) & np.uint64(15)).astype(np.int64) for s in range(12)], axis=1)
    hm = np.stack([((keys >> np.uint64(48 + 2 * j)) & np.uint64(3)).astype(np.int64) for j in range(3)], axis=1)
    ho = np.stack([((keys >> np.uint64(54 + 2 * j)) & np.uint64(3)).astype(np.int64) for j in range(3)], axis=1)
    mine = (code >= 1) & (code <= 5)
    theirs = code >= 9
    sq = np.arange(12)
    att = ATT[code, sq[None, :]]  # n×12 のビット集合
    my_att = np.bitwise_or.reduce(np.where(mine, att, 0), axis=1)
    op_att = np.bitwise_or.reduce(np.where(theirs, att, 0), axis=1)
    # マスごとの利きの数
    bits = (att[:, :, None] >> sq[None, None, :]) & 1  # n×12(駒)×12(マス)
    my_cnt = (bits * mine[:, :, None]).sum(axis=1)
    op_cnt = (bits * theirs[:, :, None]).sum(axis=1)
    occ_me = np.bitwise_or.reduce(np.where(mine, 1 << sq[None, :], 0), axis=1)
    occ_op = np.bitwise_or.reduce(np.where(theirs, 1 << sq[None, :], 0), axis=1)
    occ = occ_me | occ_op
    my_l = np.argmax(code == 1, axis=1)
    op_l = np.argmax(code == 9, axis=1)
    my_n, op_n = NEIGH[my_l], NEIGH[op_l]
    v = {}
    v["my_safe"] = POP[my_n & ~occ_me & ~op_att & 0xFFF]
    v["op_safe"] = POP[op_n & ~occ_op & ~my_att & 0xFFF]
    v["my_try_d"] = my_l // 3
    v["my_try_n"] = POP[ROW0 & ~occ_me & ~op_att & 0xFFF]
    v["op_try_d"] = 3 - op_l // 3
    v["op_try_n"] = POP[ROW3 & ~occ_op & ~my_att & 0xFFF]
    v["my_press"] = (my_cnt * ((op_n[:, None] >> sq[None, :]) & 1)).sum(axis=1)
    v["op_press"] = (op_cnt * ((my_n[:, None] >> sq[None, :]) & 1)).sum(axis=1)
    v["my_hang"] = (mine & (code != 1) & (op_cnt > 0) & (my_cnt == 0)).sum(axis=1)
    v["op_hang"] = (theirs & (code != 9) & (my_cnt > 0) & (op_cnt == 0)).sum(axis=1)
    empty = 12 - POP[occ]
    v["my_mob"] = POP[0] + sum(POP[np.where(mine[:, s], att[:, s] & ~occ_me & 0xFFF, 0)] for s in range(12)) + empty * (hm > 0).sum(axis=1) // 3
    v["op_mob"] = sum(POP[np.where(theirs[:, s], att[:, s] & ~occ_op & 0xFFF, 0)] for s in range(12)) + empty * (ho > 0).sum(axis=1) // 3
    v["check"] = ((op_att >> my_l) & 1).astype(np.int64)
    # 取り合いの判定（X14 用）: 相手の利きの数が自分の守りの数より多い駒（取られそうな駒）
    v["my_enprise"] = (mine & (code != 1) & (op_cnt > my_cnt)).sum(axis=1)
    v["op_enprise"] = (theirs & (code != 9) & (my_cnt > op_cnt)).sum(axis=1)
    v["my_lion_att"] = op_cnt[np.arange(n), my_l]  # 自ライオンに利いている相手の駒の数
    rows = (my_l // 3) * 4 + op_l // 3  # 16 通りの局面の型
    if LION:
        # 前方（相手陣の方向）へ安全に動けるマスの数
        my_fwd_mask = (1 << (3 * (my_l // 3))) - 1
        op_fwd_mask = 0xFFF ^ ((1 << (3 * (op_l // 3 + 1))) - 1)
        v["my_fwd"] = POP[my_n & ~occ_me & ~op_att & my_fwd_mask & 0xFFF]
        v["op_fwd"] = POP[op_n & ~occ_op & ~my_att & op_fwd_mask & 0xFFF]
        v["my_center"] = (my_l % 3 == 1).astype(np.int64) + (my_l // 3 <= 1)
        v["op_center"] = (op_l % 3 == 1).astype(np.int64) + (op_l // 3 >= 2)
        mine_n = mine.sum(1) - 1 + hm.sum(1)
        theirs_n = theirs.sum(1) - 1 + ho.sum(1)
        mat = np.clip(mine_n - theirs_n + 2, 0, 4)  # 駒の枚数の差 -2..+2
        v["mat_x_myrow"] = mat * 4 + my_l // 3
        v["mat_x_oprow"] = mat * 4 + op_l // 3
    return {k: np.clip(x, 0, (BUCKETS if not k.startswith("mat_x") else 20) - 1) for k, x in v.items()}, rows


def threat_sparse(keys):
    v, rows = threat_values(keys)
    n = len(keys)
    cols = []
    for i, name in enumerate(NAMES):
        cols.append(i * BUCKETS + v[name])  # 単独
        cols.append(len(NAMES) * BUCKETS + (rows * len(NAMES) + i) * BUCKETS + v[name])  # 局面の型 × 特徴
    extra = 0
    if LION:
        for i, name in enumerate(LION_NAMES):
            cols.append(len(NAMES) * BUCKETS * 17 + 400 + i * 20 + v[name])
        extra = len(LION_NAMES) * 20
    # 主要な組み合わせ（攻めの圧力 × 守りの安全度、トライの脅威の打ち合い）
    base = len(NAMES) * BUCKETS * 17
    cols.append(base + v["my_press"] * BUCKETS + v["op_safe"])
    cols.append(base + 100 + v["op_press"] * BUCKETS + v["my_safe"])
    cols.append(base + 200 + v["my_try_n"] * BUCKETS + v["op_try_n"])
    cols.append(base + 300 + v["my_hang"] * BUCKETS + v["op_hang"])
    c = np.stack(cols, axis=1)
    dim = base + 400 + extra
    r = np.repeat(np.arange(n), c.shape[1])
    return sp.csr_matrix((np.ones(c.size, dtype=np.float32), (r, c.ravel())), shape=(n, dim))


class ThreatModel(Model):
    """kind: "T"（脅威の特徴量だけ） / "KPP+T" / "MLP+T"（参考） / "GBDT+T" """

    def __init__(self, kind):
        super().__init__(kind)
        self.base = kind.split("+")[0]
        self.dense = self.base in ("GBDT", "MLP")

    def feats(self, keys):
        t = threat_sparse(keys)
        if self.base == "T":
            return t
        if self.dense:
            return np.hstack([x1.build(keys, "P").toarray(), t.toarray()]).astype(np.float32)
        return sp.hstack([x1.build(keys, self.base), t]).tocsr()

    def fit(self, keys, vals):
        k = self.kind
        self.kind = "KPP" if not self.dense else self.base  # 親クラスの学習分岐に合わせる
        super().fit(keys, vals)
        self.kind = k
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--train", type=int, default=3000000)
    ap.add_argument("--test", type=int, default=300000)
    ap.add_argument("--kinds", default="T,KPP+T,GBDT+T,MLP+T")
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x5_threats.jsonl"))
    ap.add_argument("--seed", type=int, default=1, help="学習局面を抜き出す乱数（評価局面は固定）")
    a = ap.parse_args()
    keys, vals, _ = run_sampler(a.data, a.train, a.seed, False)
    keep = vals != 2
    keys, vals = keys[keep], vals[keep]
    tkeys, tvals, tkids = run_sampler(a.data, a.test, 2, True)
    for kind in a.kinds.split(","):
        t = time.time()
        m = ThreatModel(kind).fit(keys, vals)
        res = {"kind": kind, "train": a.train, "seed": a.seed, "lion": LION, "mlp_iter": int(os.environ.get("X1_MLP_ITER", "20")), **evaluate(m, tkeys, tvals, tkids), "sec": round(time.time() - t)}
        print(json.dumps(res), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
