"""X23: 強いAIが選んだ手を、自分の探索で選べるように評価関数を学習する（MMTO 型の比較学習）

X19〜X22 で「評価値との相関は高いのに対局で弱い」ことが続いた（教師=強豪AIの評価値と、探索での使われ方が食い違う）。
そこで、評価値を当てるのではなく「強豪AIが実際に指した手を、自分の探索が一番良いと判断する」ことを学習の目的にする
（保木の Bonanza / MMTO と同じ考え方）。

評価 = 固定の駒の価値（X13 の Classic）+ λ・線形（駒数＋KP＋脅威、X7 の M+KP+T）。学習するのは線形の重み。
1回の繰り返し:
  学習側の対局から局面と「実際に指された手」を --n 件。各合法手について、指した後の局面を静止探索（駒取り、深さ4）し、
  読み筋の末端の局面の特徴量を記録。損失 = Σ_{他の手} softplus((v(他の手) − v(指された手)) / T)。
  末端は評価関数に依存するので、重みを更新したら次の繰り返しで求め直す。
初期値: X7 と同じ（棋譜のエンジン評価値への回帰）。最後に深さ2で「固定の駒の価値だけ」と対局（現時点の最良 = 76%）。

使い方: python x23_comparison.py --n 3000 --iters 4 --games 30
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
from x13_mix import Classic, play
from x7b_qsearch import PIECE_VALUE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../corpus"))
from analyze_positions import Usi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OFF_P = 42
OFF_KP = OFF_P + x7.P_DIM
OFF_T = OFF_KP + x7.KP_DIM
DIM = OFF_T + x7.T_DIM


def sparse(board):
    """線形（M+KP+T）の特徴量: (インデックス, 値)。M は駒数/4 の密な42次元"""
    p, kp, t, d = x7.features(board)
    idx = np.concatenate([np.arange(42), OFF_P + np.array(p, dtype=np.int64), OFF_KP + np.array(kp, dtype=np.int64), OFF_T + np.array(t, dtype=np.int64)])
    val = np.concatenate([d[:42] / 4, np.ones(len(p) + len(kp) + len(t))])
    return idx, val


class Model:
    """評価 = Classic + lam・(b + w·x)。探索から呼ぶと手番側から見た値を返し、最後に評価した末端の特徴量を残す"""

    def __init__(self, w, b, lam):
        self.w, self.b, self.lam = w, b, lam
        self.classic = Classic()

    def value(self, board):
        idx, val = sparse(board)
        c = self.classic(board)
        return c + self.lam * (self.b + float(self.w[idx] @ val)), (idx, val, c)

    def __call__(self, board):
        return self.value(board)[0]


def qsearch_leaf(board, m, alpha, beta, depth):
    """静止探索（駒取り）。(値, 末端の特徴量, 末端が手番側から見て何手先か(偶奇)) を返す"""
    if board.is_check():
        moves = list(board.legal_moves)
        if not moves:
            return -1e6, None, 0
        v, leaf = m.value(board)
        if depth == 0:
            return v, leaf, 0
    else:
        v, leaf = m.value(board)
        if depth == 0 or v >= beta:
            return v, leaf, 0
        moves = [mv for mv in board.legal_moves if not cshogi.move_is_drop(mv) and board.piece(cshogi.move_to(mv))]
        moves.sort(key=lambda mv: -PIECE_VALUE[board.piece(cshogi.move_to(mv)) & 15])
    best, best_leaf, best_par = v, leaf, 0
    alpha = max(alpha, v) if not board.is_check() else alpha
    if board.is_check():
        best, best_leaf = -1e9, None
    for mv in moves:
        board.push(mv)
        s, lf, par = qsearch_leaf(board, m, -beta, -alpha, depth - 1)
        board.pop()
        s = -s
        if s > best:
            best, best_leaf, best_par = s, lf, par + 1
        if s > alpha:
            alpha = s
        if alpha >= beta:
            break
    return best, best_leaf, best_par


def build_examples(positions, model):
    """各局面: 合法手ごとの末端特徴量と、根から見た符号"""
    ex = []
    for b, played in positions:
        rows = []
        tgt = None
        for mv in b.legal_moves:
            b.push(mv)
            if b.is_game_over():
                b.pop()
                continue
            _, leaf, par = qsearch_leaf(b, model, -1e9, 1e9, 4)
            b.pop()
            if leaf is None:
                continue
            sign = -1.0 if par % 2 == 0 else 1.0  # 末端の手番側の値 → 根の手番側から見た値
            if cshogi.move_to_usi(mv) == played:
                tgt = len(rows)
            rows.append((leaf, sign))
        if tgt is not None and len(rows) >= 2:
            ex.append((rows, tgt))
    return ex


ADA = {}  # 特徴ごとの勾配の二乗和（Adagrad）


def train_step(ex, w, b, lam, lr, temp, w0, reg):
    """比較の損失で w を更新（Adagrad: 特徴ごとに歩幅を自動調整して発散を防ぐ）"""
    g = {}
    loss = 0.0
    for rows, tgt in ex:
        # 根の手番側から見た値 = 符号 × (末端の固定の駒の価値 + λ(b + w·x))。固定の駒の価値は w に依存しないが、比べる値には含める
        vals = np.array([sign * (c + lam * (b + float(w[idx] @ val))) for (idx, val, c), sign in rows])
        d = (vals - vals[tgt]) / temp
        d[tgt] = -np.inf
        sp = np.logaddexp(0, d)
        loss += float(np.sum(sp[np.isfinite(d)]))
        sig = 1 / (1 + np.exp(-np.clip(d, -50, 50)))
        sig[tgt] = 0
        coef = sig / temp
        tot = coef.sum()
        for k, ((idx, val, _), sign) in enumerate(rows):
            c = (coef[k] if k != tgt else -tot) * sign * lam
            if c == 0:
                continue
            for i, v in zip(idx, val):
                g[i] = g.get(i, 0.0) + c * v
    for i, gi in g.items():
        gi = gi / len(ex) + reg * (w[i] - w0[i])
        ADA[i] = ADA.get(i, 0.0) + gi * gi
        w[i] -= lr * gi / (np.sqrt(ADA[i]) + 1e-8)
    return loss / max(len(ex), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "../results/x23_comparison.jsonl"))
    a = ap.parse_args()
    t0 = time.time()
    rng = random.Random(0)
    games = x7.load_games()
    rng.shuffle(games)
    n_test = len(games) // 10
    train = games[n_test:]
    F, y, wt = x7.collect(train, 3, random.Random(0), "score")
    lin = x7.fit("M+KP+T", F, y, wt)
    del F
    w = lin.coef_.astype(np.float64).copy()
    b = float(np.ravel(lin.intercept_)[0])
    w0 = w.copy()
    print(f"init done {time.time() - t0:.0f}s", flush=True)
    r2 = random.Random(1)
    for it in range(a.iters):
        positions = []
        while len(positions) < a.n:
            g = r2.choice(train)
            if len(g["moves"]) < 30:
                continue
            ply = r2.randrange(10, len(g["moves"]) - 5)
            bd = cshogi.Board()
            for mv in g["moves"][:ply]:
                bd.push_usi(mv)
            if not bd.is_game_over():
                positions.append((bd, g["moves"][ply]))
        model = Model(w, b, a.lam)
        ex = build_examples(positions, model)
        for ep in range(a.epochs):
            loss = train_step(ex, w, b, a.lam, a.lr, a.temp, w0, a.reg)
        # 学習局面での一致率（自分の静止探索つき1手読みが、指された手を選ぶ率）
        agree = 0
        for rows, tgt in ex:
            vals = [sign * (c + a.lam * (b + float(w[idx] @ val))) for (idx, val, c), sign in rows]
            agree += int(np.argmax(vals) == tgt)
        print(json.dumps({"iter": it, "examples": len(ex), "loss": round(loss, 4), "agree_after": round(agree / max(len(ex), 1), 4), "sec": round(time.time() - t0)}), flush=True)

    classic = Classic()
    ev = Model(w, b, a.lam)
    test_files = {g["file"] for g in games[:n_test]}
    with open(os.path.join(x7.DATA, "exits.jsonl")) as f:
        starts = [json.loads(l) for l in f]
    starts = [s for s in starts if s["file"] in test_files and 20 <= s["ply"] <= 60]
    random.Random(2).shuffle(starts)
    starts = starts[: a.games]
    judge = Usi(1, 64)
    score = {"win": 0, "loss": 0, "draw": 0}
    for s in starts:
        for swap in (False, True):
            pair = [classic, ev] if swap else [ev, classic]
            r_, *_ = play(s["sfen"], pair, 2, 160, judge)
            r_ = -r_ if swap else r_
            score["win" if r_ > 0 else "loss" if r_ < 0 else "draw"] += 1
    n = sum(score.values())
    res = {"method": "比較学習（MMTO型）", "n": a.n, "iters": a.iters, "lam": a.lam, "vs": "固定の駒の価値だけ", "depth": 2, **score, "score_rate": round((score["win"] + 0.5 * score["draw"]) / n, 4), "sec": round(time.time() - t0)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    judge.send("quit")


if __name__ == "__main__":
    main()
