"""強いAI同士の floodgate 棋譜から「本譜の木」（多くの対局で共有された序盤〜中盤の局面）を作る

出力（--out 以下）:
  mainline.jsonl : 共有回数 >= K の局面。sfen, 手数, 出現数, 結果(先手勝/後手勝/引分), 次の手の分布
  exits.jsonl    : 各対局の「木から外れた局面」（＝定跡が尽きた後の最初の局面。中盤研究の出発点）
  games.jsonl    : 採用した対局の一覧（USI手順・結果・レート）
  summary.json   : 件数や離脱手数の分布

使い方: python build_mainline.py --csa ../data/floodgate/csa --min-rate 4000 --k 5
"""

import argparse
import collections
import glob
import json
import os

import cshogi
from cshogi import CSA

START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def strip_ply(sfen):
    return sfen.rsplit(" ", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csa", required=True)
    ap.add_argument("--min-rate", type=float, default=4000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../data/mainline"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    games = []
    dup = 0
    seen_games = set()
    for path in sorted(glob.glob(os.path.join(a.csa, "**/*.csa"), recursive=True)):
        try:
            k = CSA.Parser.parse_file(path)[0]
        except Exception:
            continue
        if k.sfen != START or len(k.ratings) < 2 or min(k.ratings) < a.min_rate or len(k.moves) < 30:
            continue
        sig = tuple(k.moves)
        if sig in seen_games:  # 同じエンジン同士が全く同じ対局を繰り返すことがあるので除く
            dup += 1
            continue
        seen_games.add(sig)
        games.append(
            {
                "file": os.path.relpath(path, a.csa),
                "names": k.names,
                "ratings": k.ratings,
                "win": k.win,  # 1先手勝 2後手勝 0引分
                "endgame": k.endgame,
                "moves": [cshogi.move_to_usi(m) for m in k.moves],
            }
        )

    # 局面ごとの出現数・結果・次の手
    stat = {}
    for gi, g in enumerate(games):
        b = cshogi.Board()
        seen = set()
        for ply, mv in enumerate(g["moves"]):
            key = strip_ply(b.sfen())
            if key not in seen:  # 同一対局内の繰り返しは1回と数える
                seen.add(key)
                s = stat.setdefault(key, {"ply": ply, "n": 0, "res": [0, 0, 0], "next": collections.Counter()})
                s["n"] += 1
                s["res"][g["win"]] += 1
                s["next"][mv] += 1
            b.push_usi(mv)

    with open(os.path.join(a.out, "mainline.jsonl"), "w") as f:
        for key, s in sorted(stat.items(), key=lambda x: (-x[1]["n"], x[1]["ply"])):
            if s["n"] < a.k:
                continue
            draw, bw, ww = s["res"]
            f.write(json.dumps({"sfen": key, "ply": s["ply"], "n": s["n"], "black_win": bw, "white_win": ww, "draw": draw, "next": dict(s["next"].most_common())}) + "\n")

    exit_hist = collections.Counter()
    with open(os.path.join(a.out, "exits.jsonl"), "w") as fe, open(os.path.join(a.out, "games.jsonl"), "w") as fg:
        for g in games:
            b = cshogi.Board()
            exit_ply = len(g["moves"])
            for ply, mv in enumerate(g["moves"]):
                if stat[strip_ply(b.sfen())]["n"] < a.k:
                    exit_ply = ply
                    break
                b.push_usi(mv)
            g["exit_ply"] = exit_ply
            exit_hist[exit_ply // 10 * 10] += 1
            fe.write(json.dumps({"sfen": strip_ply(b.sfen()), "ply": exit_ply, "file": g["file"], "win": g["win"], "ratings": g["ratings"]}) + "\n")
            fg.write(json.dumps(g, ensure_ascii=False) + "\n")

    summary = {
        "games": len(games),
        "duplicates_removed": dup,
        "min_rate": a.min_rate,
        "k": a.k,
        "positions_total": len(stat),
        "positions_shared_ge_k": sum(1 for s in stat.values() if s["n"] >= a.k),
        "exit_ply_hist(10手刻み)": dict(sorted(exit_hist.items())),
        "result(引分,先手勝,後手勝)": [sum(1 for g in games if g["win"] == w) for w in (0, 1, 2)],
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
