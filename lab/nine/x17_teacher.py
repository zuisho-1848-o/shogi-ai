"""X17: 駒の損得がある局面を教師に足して、学習した評価が駒の価値を身につけるか

X13 で、強豪AI同士の対局だけを教師にすると駒の価値が学べない（飛車1枚で −0.11）と分かった。
  --gen  : 学習側の対局の途中局面から無作為な手を 1〜3手指し（半分は続けて駒を取る手も指す）、
           駒の損得がある局面を作って NNUE（--nodes ノード）で評価する。中断しても続きから再開できる
  --train: 元の教師（棋譜のエンジン評価値）＋ 作った局面 で学習し直し、
           (1) 駒の価値の確かめ（平手から駒を抜いたときの評価の下がり方）
           (2) 固定の駒の価値に頼らない学習評価だけで、固定の駒の価値（X13 の Classic）と深さ2で対局
出力: ../data/mainline/imbalanced.jsonl（作った局面）/ ../results/x17_teacher.jsonl

使い方:
  python x17_teacher.py --gen --n 30000 --nodes 20000
  python x17_teacher.py --train --games 30
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
from x7b_qsearch import Evaluator
from x13_mix import Classic, Mix, play

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../corpus"))
from analyze_positions import Usi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(x7.DATA, "imbalanced.jsonl")
PROBES = {
    "平手": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "先手の飛車なし": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B7/LNSGKGSNL b - 1",
    "先手の角なし": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/7R1/LNSGKGSNL b - 1",
    "先手の歩1枚なし": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPP1/1B5R1/LNSGKGSNL b - 1",
    "先手の金1枚なし": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNS1KGSNL b - 1",
}


def split_games():
    rng = random.Random(0)
    games = x7.load_games()
    rng.shuffle(games)
    n_test = len(games) // 10
    return games, n_test


def gen(n, nodes):
    games, n_test = split_games()
    train = games[n_test:]
    done = 0
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = sum(1 for _ in f)
    rng = random.Random(100 + done)
    eng = Usi(1, 64)
    t0 = time.time()
    with open(OUT, "a") as f:
        while done < n:
            g = rng.choice(train)
            if len(g["moves"]) < 30:
                continue
            ply = rng.randrange(10, len(g["moves"]) - 10)
            b = cshogi.Board()
            for mv in g["moves"][:ply]:
                b.push_usi(mv)
            for _ in range(rng.randint(1, 3)):
                ms = list(b.legal_moves)
                if not ms:
                    break
                b.push(rng.choice(ms))
            if rng.random() < 0.5 and not b.is_game_over():
                caps = [m for m in b.legal_moves if not cshogi.move_is_drop(m) and b.piece(cshogi.move_to(m))]
                if caps:
                    b.push(rng.choice(caps))
            if b.is_game_over():
                continue
            sfen = b.sfen().rsplit(" ", 1)[0]
            r = eng.go(sfen, nodes)
            if "score_cp" in r:
                sc = r["score_cp"]
            elif "mate" in r:
                sc = -30000 if str(r["mate"]).startswith("-") else 30000
            else:
                continue
            f.write(json.dumps({"sfen": sfen, "score_cp": sc}) + "\n")
            f.flush()
            done += 1
            if done % 500 == 0:
                print(f"{done}/{n} {time.time() - t0:.0f}s", flush=True)
    eng.send("quit")


def add_imbalanced(F, y, w, limit=None, weight=1.0):
    with open(OUT) as f:
        rows = [json.loads(l) for l in f]
    if limit:
        rows = rows[:limit]
    ys = list(y)
    ws = list(w)
    for r in rows:
        b = cshogi.Board(r["sfen"] + " 1")
        F.add(b)
        ys.append(float(np.tanh(np.clip(r["score_cp"], -3000, 3000) / 1200)))
        ws.append(weight)
    return F, np.array(ys), np.array(ws), len(rows)


def fit_weighted(kind, F, y, w):
    """SGDRegressor に重みを渡して学習（x7.fit は重みを使わないため）"""
    from sklearn.linear_model import SGDRegressor

    m = SGDRegressor(alpha=1e-6, learning_rate="invscaling", eta0=0.002, max_iter=10, tol=None)
    m.fit(F.X(kind), y, sample_weight=w)
    return m


def probe(ev):
    return {name: round(ev(cshogi.Board(s)), 3) for name, s in PROBES.items()}


def train_and_match(n_games, depth, kind, imb_weight=1.0, players="new,mix"):
    games, n_test = split_games()
    rng = random.Random(0)
    F, y, w = x7.collect(games[n_test:], 3, rng, "score")
    base = Evaluator(kind, x7.fit(kind, F, y, w))
    F, y, w, n_imb = add_imbalanced(F, y, w, weight=imb_weight)
    new = Evaluator(kind, fit_weighted(kind, F, y, w))
    del F
    res = {"kind": kind, "imbalanced_positions": n_imb, "imb_weight": imb_weight, "probe_before": probe(base), "probe_after": probe(new)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    test_files = {g["file"] for g in games[:n_test]}
    with open(os.path.join(x7.DATA, "exits.jsonl")) as f:
        starts = [json.loads(l) for l in f]
    starts = [s for s in starts if s["file"] in test_files and 20 <= s["ply"] <= 60]
    random.Random(2).shuffle(starts)
    starts = starts[:n_games]
    judge = Usi(1, 64)
    classic = Classic()
    cand = {"base": ("学習評価だけ（元の教師）", base), "new": ("学習評価だけ（作り直した教師）", new), "mix": ("固定の駒の価値+0.5×学習評価（作り直した教師）", Mix(classic, new, 0.5))}
    for name, ev in (cand[p] for p in players.split(",")):
        score = {"win": 0, "loss": 0, "draw": 0}
        for s in starts:
            for swap in (False, True):
                pair = [classic, ev] if swap else [ev, classic]
                r, how, ply, mb = play(s["sfen"], pair, depth, 160, judge)
                r = -r if swap else r
                score["win" if r > 0 else "loss" if r < 0 else "draw"] += 1
        n = sum(score.values())
        m = {"vs": "固定の駒の価値だけ", "player": name, "depth": depth, **score, "score_rate": round((score["win"] + 0.5 * score["draw"]) / n, 4)}
        print(json.dumps(m, ensure_ascii=False), flush=True)
        res.setdefault("matches", []).append(m)
    judge.send("quit")
    with open(os.path.join(HERE, "../results/x17_teacher.jsonl"), "a") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--kind", default="M+KP+T")
    ap.add_argument("--imb-weight", type=float, default=1.0, help="作った局面の重み")
    ap.add_argument("--players", default="new,mix", help="対局させる評価: base（元の教師）/ new / mix")
    a = ap.parse_args()
    if a.gen:
        gen(a.n, a.nodes)
    if a.train:
        train_and_match(a.games, a.depth, a.kind, a.imb_weight, a.players)


if __name__ == "__main__":
    main()
