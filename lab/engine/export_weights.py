"""Python で学習した線形評価（駒数＋KP＋脅威、X7 の M+KP+T）の重みを、C++ エンジン用のファイルに書き出す

形式: float32 λ, float32 定数, int32 次元数, float32 × 次元数
使い方: python export_weights.py --out ../data/engine/weights_mkpt.bin --lam 0.5 [--check 2000]
  --check N: 評価側の対局から N 局面を取り、C++ の evaltest と Python の値が一致するかを確かめる
"""

import argparse
import os
import random
import subprocess
import sys

import cshogi
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../nine"))
import x7_expressivity as x7  # noqa: E402
from x7b_qsearch import Evaluator  # noqa: E402
from x13_mix import Classic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "../data/engine/weights_mkpt.bin"))
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--check", type=int, default=2000)
    ap.add_argument("--kind", default="M+KP+T", help="M+KP+T / M+KP（脅威なし。C++ 側は UseThreat false で使う）")
    a = ap.parse_args()
    rng = random.Random(0)
    games = x7.load_games()
    rng.shuffle(games)
    n_test = len(games) // 10
    F, y, w = x7.collect(games[n_test:], a.every, rng, "score")
    m = x7.fit(a.kind, F, y, w)
    del F
    coef = m.coef_.astype(np.float32)
    full = 42 + x7.P_DIM + x7.KP_DIM + x7.T_DIM
    if len(coef) < full:  # 脅威なしの版は、脅威の部分を 0 で埋めて同じ形式にする
        coef = np.concatenate([coef, np.zeros(full - len(coef), dtype=np.float32)])
    b = np.float32(np.ravel(m.intercept_)[0])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        np.array([a.lam, b], dtype=np.float32).tofile(f)
        np.array([len(coef)], dtype=np.int32).tofile(f)
        coef.tofile(f)
    print(f"wrote {a.out}: dim {len(coef)}, bias {b:.4f}, lam {a.lam}")
    if a.check:
        ev = Evaluator(a.kind, m)
        classic = Classic()
        sfens = []
        r2 = random.Random(5)
        for g in games[:n_test]:
            bd = cshogi.Board()
            for mv in g["moves"]:
                bd.push_usi(mv)
                if r2.random() < 0.05 and not bd.is_game_over():
                    sfens.append(bd.sfen())
            if len(sfens) >= a.check:
                break
        exe = os.path.join(HERE, "build/nonnn2")
        inp = f"setoption name EvalFile value {a.out}\nsetoption name UseThreat value {'true' if a.kind.endswith('T') else 'false'}\nevaltest\n" + "\n".join(sfens) + "\n\nquit\n"
        out = subprocess.run([exe], input=inp, capture_output=True, text=True).stdout.split("\n")
        rows = [l.split() for l in out if l.strip() and l[0] in "-0123456789"]
        c_cls = np.array([float(r[0]) for r in rows])
        c_lin = np.array([float(r[1]) for r in rows])
        p_cls = np.array([classic(cshogi.Board(s)) for s in sfens[: len(rows)]])
        p_lin = np.array([ev(cshogi.Board(s)) for s in sfens[: len(rows)]])
        print(f"checked {len(rows)} positions: max |classic diff| {np.abs(c_cls - p_cls).max():.2e}, max |linear diff| {np.abs(c_lin - p_lin).max():.2e}")


if __name__ == "__main__":
    main()
