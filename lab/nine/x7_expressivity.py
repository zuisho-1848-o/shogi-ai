"""X7: 9×9 将棋での評価関数の「形」ごとの表現力（どうぶつしょうぎの X1/X5 を 9×9 に持ち込む）

教師: floodgate 2026 の強豪AI同士の対局（両者レート4000以上、15,168局）。対局単位で学習用と評価用に分ける。
  --teacher score（既定）: 棋譜に残るエンジン自身の評価値（先手から見た値。手番側に直して tanh(評価値/1200) を回帰）
  --teacher result       : 対局の勝敗（雑音が大きく、KP 系は過学習した。2026-10-20 の初回結果）
評価:
  1. 評価用対局の局面で、手番側の勝敗をどれだけ当てるか（対数損失・正解率）
  2. NNUE（水匠5、30万ノード）で解析済みの局面で、1手読みで選ぶ手が NNUE の最善手と一致する率
     ＝「強いAIが選ぶ手」をどれだけ再現できるか（真値ではないが、強さの代わりの指標）

評価関数の族:
  material : 駒の枚数（盤上・成り・持ち駒）
  P        : 駒×マス ＋ 持ち駒の枚数
  KP       : P ＋ 自玉の位置×駒×マス ＋ 相手玉の位置×駒×マス（NNUE の入力層だけを線形で使う形）
  T        : 脅威の特徴量だけ（指し手の数・王手・1手詰・詰めろ・玉の逃げ道・玉の周りへの利き・取れる駒・守りの金銀）
  KP+T     : KP ＋ T
  GBDT(material+T) : 少数の特徴量の非線形（決定木）

局面はすべて手番側から見た向きに揃える（後手番なら盤を180度回して先後を入れ替える）。

使い方: python x7_expressivity.py --every 3 --agree 4000
"""

import argparse
from array import array
import json
import os
import random
import time

import cshogi
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from cshogi import CSA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/mainline")

NP = 28  # 手番側の駒14種 + 相手の駒14種（玉を含む）
P_DIM = NP * 81 + 2 * 7 * 19  # 盤 + 持ち駒（手番/相手 × 7種 × 枚数を温度計表現）
KP_DIM = 2 * 81 * P_DIM
T_NAMES = ["mob", "op_mob", "check", "mate1", "op_mate1", "k_esc", "op_k_esc", "press", "op_press", "caps", "op_caps", "guard", "op_guard", "k_rank", "op_k_rank", "hand", "op_hand"]
T_B = 16
PHASES = 4
T_DIM = len(T_NAMES) * T_B * (1 + PHASES)
NEIGH = {}


def neigh(sq):
    if sq not in NEIGH:
        f, r = divmod(sq, 9)
        NEIGH[sq] = {(f + df) * 9 + (r + dr) for df in (-1, 0, 1) for dr in (-1, 0, 1) if (df or dr) and 0 <= f + df < 9 and 0 <= r + dr < 9}
    return NEIGH[sq]


def persp(board):
    """手番側から見た (駒コード0..27 or -1, マス) の一覧と持ち駒、両玉の位置"""
    turn = board.turn
    pcs = []
    for sq, p in enumerate(board.pieces):
        if not p:
            continue
        own = (p < 16) == (turn == 0)
        code = (p & 15) - 1 + (0 if own else 14)
        s = sq if turn == 0 else 80 - sq
        pcs.append((code, s))
    hb, hw = board.pieces_in_hand
    hm, ho = (hb, hw) if turn == 0 else (hw, hb)
    ks = board.king_square(turn)
    ko = board.king_square(1 - turn)
    if turn == 1:
        ks, ko = 80 - ks, 80 - ko
    return pcs, hm, ho, ks, ko


def moves_info(board, king_target):
    """指し手の数、玉の周りへ利く手の数、取る手の数"""
    n = press = caps = 0
    tgt = neigh(king_target)
    for m in board.pseudo_legal_moves if hasattr(board, "pseudo_legal_moves") else board.legal_moves:
        n += 1
        to = cshogi.move_to(m)
        if to in tgt:
            press += 1
        if not cshogi.move_is_drop(m) and board.piece(to):
            caps += 1
    return n, press, caps


def threat(board, pcs, hm, ho, ks, ko):
    turn = board.turn
    t = {}
    king_real = board.king_square(turn)
    opk_real = board.king_square(1 - turn)
    t["mob"], t["press"], t["caps"] = moves_info(board, opk_real)
    t["check"] = int(board.is_check())
    t["mate1"] = int(bool(board.mate_move_in_1ply())) if not t["check"] else 0
    t["k_esc"] = sum(1 for m in board.legal_moves if cshogi.move_from(m) == king_real and not cshogi.move_is_drop(m))
    if not t["check"]:
        board.push_pass()
        t["op_mob"], t["op_press"], t["op_caps"] = moves_info(board, king_real)
        t["op_mate1"] = int(bool(board.mate_move_in_1ply()))  # 詰めろがかかっているか
        t["op_k_esc"] = sum(1 for m in board.legal_moves if cshogi.move_from(m) == opk_real and not cshogi.move_is_drop(m))
        board.pop_pass()
    else:
        t["op_mob"] = t["op_press"] = t["op_caps"] = t["op_mate1"] = t["op_k_esc"] = 0
    gold_like = {3, 6, 8, 9, 10, 11}  # 銀・金・と・成香・成桂・成銀（コード 0始まり: 歩0 香1 桂2 銀3 角4 飛5 金6 玉7 と8 …）
    t["guard"] = sum(1 for c, s in pcs if c in gold_like and s in neigh(ks) | {ks})
    t["op_guard"] = sum(1 for c, s in pcs if c - 14 in gold_like and s in neigh(ko) | {ko})
    t["k_rank"] = 8 - ks % 9  # 自陣の一番下からの段数
    t["op_k_rank"] = ko % 9
    t["hand"] = sum(hm)
    t["op_hand"] = sum(ho)
    return t


def phase(pcs, hm, ho):
    # 持ち駒と成り駒の合計で、序盤〜終盤を4段階に
    x = sum(hm) + sum(ho) + sum(1 for c, _ in pcs if c % 14 >= 8)
    return min(x // 4, PHASES - 1)


def features(board):
    """1局面の特徴量（族ごとのインデックス列）と、密な少数特徴（material+T の値）"""
    pcs, hm, ho, ks, ko = persp(board)
    p_idx = [c * 81 + s for c, s in pcs]
    base = NP * 81
    for side, hand in ((0, hm), (1, ho)):
        for k, cnt in enumerate(hand):
            for j in range(min(cnt, 19)):
                p_idx.append(base + (side * 7 + k) * 19 + j)
    kp_idx = [ks * P_DIM + i for i in p_idx] + [81 * P_DIM + ko * P_DIM + i for i in p_idx]
    t = threat(board, pcs, hm, ho, ks, ko)
    ph = phase(pcs, hm, ho)
    t_idx = []
    for i, name in enumerate(T_NAMES):
        v = min(t[name], T_B - 1)
        t_idx.append(i * T_B + v)
        t_idx.append(len(T_NAMES) * T_B * (1 + ph) + i * T_B + v)
    mat = np.zeros(28 + 14, dtype=np.float32)
    for c, _ in pcs:
        mat[c] += 1
    mat[28:35] = hm
    mat[35:42] = ho
    dense = np.concatenate([mat, np.array([t[n] for n in T_NAMES] + [ph], dtype=np.float32)])
    return p_idx, kp_idx, t_idx, dense


class Feats:
    """特徴量のインデックスを族ごとに1本の整数配列で持つ（Python のリストだと数GBになるため）"""

    def __init__(self):
        self.idx = {k: array("i") for k in ("P", "KP", "T")}
        self.len = {k: array("i") for k in ("P", "KP", "T")}
        self.dense = array("f")
        self.n = 0

    def add(self, board):
        p, kp, t, d = features(board)
        for k, v in (("P", p), ("KP", kp), ("T", t)):
            self.idx[k].extend(v)
            self.len[k].append(len(v))
        self.dense.extend(d.tolist())
        self.n += 1

    def matrix(self, fam, dim):
        indptr = np.concatenate([[0], np.cumsum(np.frombuffer(self.len[fam], dtype=np.int32), dtype=np.int64)])
        idx = np.frombuffer(self.idx[fam], dtype=np.int32)
        return sp.csr_matrix((np.ones(len(idx), dtype=np.float32), idx, indptr), shape=(self.n, dim))

    def D(self):
        return np.frombuffer(self.dense, dtype=np.float32).reshape(self.n, -1)

    def X(self, kind):
        if kind == "material":
            return sp.csr_matrix(self.D()[:, :42])
        if kind == "P":
            return self.matrix("P", P_DIM)
        if kind == "T":
            return self.matrix("T", T_DIM)
        kp = sp.hstack([self.matrix("P", P_DIM), self.matrix("KP", KP_DIM)])
        if kind == "KP+T":
            kp = sp.hstack([kp, self.matrix("T", T_DIM)])
        return kp.tocsr()


def load_games():
    with open(os.path.join(DATA, "games.jsonl")) as f:
        return [json.loads(l) for l in f]


def collect(games, every, rng, teacher):
    F, y, w = Feats(), [], []
    for g in games:
        b = cshogi.Board()
        n = len(g["moves"])
        comments = None
        if teacher == "score":
            try:
                comments = CSA.Parser.parse_file(os.path.join(HERE, "../data/floodgate/csa", g["file"]))[0].comments
            except Exception:
                continue
        for ply, mv in enumerate(g["moves"]):
            b.push_usi(mv)
            if ply < 2 or ply >= n - 1 or rng.randrange(every):
                continue
            if teacher == "score":
                c = comments[ply] if ply < len(comments) else ""
                if not c:
                    continue
                sc = int(c.split()[0])  # 先手から見た値。指した手の後の局面の評価
                sc = sc if b.turn == 0 else -sc
                F.add(b)
                y.append(float(np.tanh(np.clip(sc, -3000, 3000) / 1200)))
                w.append(1.0)
            else:
                stm = b.turn
                F.add(b)
                if g["win"] == 0:
                    y.append(0.0)
                else:
                    y.append(1.0 if (g["win"] == 1) == (stm == 0) else -1.0)
                w.append(1.0)
    return F, np.array(y), np.array(w)


def fit(kind, F, y, w):
    if kind == "GBDT":
        m = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63)
        m.fit(F.D(), y)
        return m
    m = SGDRegressor(alpha=1e-6, learning_rate="invscaling", eta0=0.002, max_iter=10, tol=None)
    m.fit(F.X(kind), y)
    return m


def predict(kind, m, F):
    return m.predict(F.D() if kind == "GBDT" else F.X(kind))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3, help="何手に1局面を使うか")
    ap.add_argument("--agree", type=int, default=4000, help="NNUE 最善手との一致を見る局面数")
    ap.add_argument("--kinds", default="material,P,KP,T,KP+T,GBDT")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--teacher", default="score", choices=["score", "result"])
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x7_expressivity.jsonl"))
    a = ap.parse_args()
    rng = random.Random(0)
    t0 = time.time()
    games = load_games()
    rng.shuffle(games)
    if a.max_games:
        games = games[: a.max_games]
    n_test = len(games) // 10
    Ftr, ytr, wtr = collect(games[n_test:], a.every, rng, a.teacher)
    Fte, yte, wte = collect(games[:n_test], a.every, rng, a.teacher)
    print(f"train {len(ytr)} / test {len(yte)} positions, {time.time() - t0:.0f}s", flush=True)

    # NNUE 解析済みの局面（勝負がほぼ決まった局面は除く）
    nn = []
    for path in [os.path.join(DATA, "nnue_300k.jsonl")] + [os.path.join(DATA, f"nnue_300k.jsonl.{k}") for k in range(3)]:
        if os.path.exists(path):
            with open(path) as f:
                nn += [json.loads(l) for l in f if l.strip()]
    nn = [r for r in nn if "score_cp" in r and abs(r["score_cp"]) < 1500 and r.get("bestmove") not in ("resign", "win")]
    rng.shuffle(nn)
    nn = nn[: a.agree]
    Fch, groups, best = Feats(), [], []
    for r in nn:
        b = cshogi.Board(r["sfen"] + " 1")
        ms = list(b.legal_moves)
        if len(ms) < 2:
            continue
        start = Fch.n
        for m in ms:
            b.push(m)
            Fch.add(b)
            b.pop()
        groups.append((start, len(ms)))
        best.append([cshogi.move_to_usi(m) for m in ms].index(r["bestmove"]) if r["bestmove"] in [cshogi.move_to_usi(m) for m in ms] else -1)
    rand = float(np.mean([1 / n for _, n in groups]))
    print(f"agreement set {len(groups)} positions ({Fch.n} children), random {rand:.3f}, {time.time() - t0:.0f}s", flush=True)

    for kind in a.kinds.split(","):
        t = time.time()
        m = fit(kind, Ftr, ytr, wtr)
        p = predict(kind, m, Fte)
        mse = float(np.mean((p - yte) ** 2))
        corr = float(np.corrcoef(p, yte)[0, 1])
        sign = float(np.mean(np.sign(p) == np.sign(yte)))
        pc = predict(kind, m, Fch)
        agree = float(np.mean([int(np.argmin(pc[s : s + n]) == b) for (s, n), b in zip(groups, best)]))
        res = {"kind": kind, "teacher": a.teacher, "mse": round(mse, 4), "corr": round(corr, 4), "sign": round(sign, 4), "agree_nnue": round(agree, 4), "random": round(rand, 4), "sec": round(time.time() - t)}
        print(json.dumps(res), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
