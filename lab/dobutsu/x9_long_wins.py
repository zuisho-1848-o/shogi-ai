"""X9: 長期の勝ち（終局まで46手以上）は、どういう手順で勝っているのか

真値DBから勝ち局面を抜き出し、最善の手順（勝つ側は最短、負ける側は最長で粘る）を終局までたどって、
手数の区分ごとに次を集計する:
  - 終わり方: ライオン捕獲 / トライ
  - 手順中の駒取りの回数、打ち駒の回数、ライオンの手の回数（勝つ側・負ける側）
  - 駒の枚数の差（勝つ側から見た値）の推移: 開始時 → 中間 → 終局直前
  - 勝つ側が最初に駒を取るまでの手数（「まず駒得してから勝つ」のか「駒損のまま押し切る」のか）

使い方: python x9_long_wins.py --n 6000
"""

import argparse
import collections
import os
import subprocess

import numpy as np

from x1_expressivity import HERE, run_sampler
from x2_search import REC


class Expander:
    """sample の expand モードを起動したままにして、1局面ずつ問い合わせる（起動のたびに索引を作るのを避ける）"""

    def __init__(self, data):
        self.p = subprocess.Popen([os.path.join(HERE, "sample"), data, "expand"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def __call__(self, key):
        self.p.stdin.write(np.array([key], dtype="<u8").tobytes())
        self.p.stdin.flush()
        head = self.p.stdout.read(11)
        imm, nc = head[9], head[10]
        body = self.p.stdout.read(11 * nc) if nc else b""
        rec = np.frombuffer(body, dtype=REC, count=nc) if nc else np.zeros(0, dtype=REC)
        return bool(imm), rec


def material_diff(key):
    key = int(key)
    sq = [(key >> (4 * i)) & 15 for i in range(12)]
    hm = sum((key >> (48 + 2 * j)) & 3 for j in range(3))
    ho = sum((key >> (54 + 2 * j)) & 3 for j in range(3))
    return sum(1 for v in sq if 2 <= v <= 5) + hm - sum(1 for v in sq if 10 <= v <= 13) - ho


def lion_row(key, opp=False):
    key = int(key)
    for i in range(12):
        if ((key >> (4 * i)) & 15) == (9 if opp else 1):
            return i // 3
    return -1


def pv(ex, key, d):
    """最善手順をたどる。戻り値: 各手の (手番が勝つ側か, 手の情報, 指す前の局面)、終わり方"""
    line = []
    winner_to_move = True
    cur, cd = int(key), d
    while True:
        imm, rec = ex(cur)
        if imm:
            # 即勝ち: ライオン捕獲かトライかは、ライオンが奥の段の隣にいてトライ可能かで判定（どちらも可ならトライ扱い）
            end = "try" if lion_row(cur) == 1 or lion_row(cur) == 0 else "capture"
            line.append((winner_to_move, None, cur))
            return line, end
        vals = rec["val"].astype(int)
        if winner_to_move:
            # 勝つ側: 子が「負け（d-1）」のもののうち最短
            cand = np.where((vals > 0) & ((vals - 1) % 2 == 0) & (vals - 1 == cd - 1))[0]
        else:
            # 負ける側: 子が「勝ち」のうち最長（= cd-1）
            cand = np.where((vals > 0) & ((vals - 1) % 2 == 1) & (vals - 1 == cd - 1))[0]
        if len(cand) == 0:
            return line, "broken"
        j = cand[0]
        line.append((winner_to_move, int(rec["mv"][j]), cur))
        cur, cd = int(rec["key"][j]), cd - 1
        winner_to_move = not winner_to_move


def bucket(d):
    for hi in (15, 45, 80):
        if d <= hi:
            return f"<={hi}"
    return ">80"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "../data/dobutsu"))
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    keys, vals, _ = run_sampler(a.data, 3000000, 5, False)
    d = vals.astype(int) - 1
    win = (vals > 0) & (d % 2 == 1) & (d >= 3)
    rng = np.random.default_rng(0)
    picks = []
    for lo, hi in ((3, 15), (16, 45), (46, 80), (81, 300)):
        idx = np.where(win & (d >= lo) & (d <= hi))[0]
        picks += list(rng.choice(idx, size=min(len(idx), a.n // 4), replace=False))
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    ex = Expander(a.data)
    for i in picks:
        line, end = pv(ex, keys[i], int(d[i]))
        s = stats[bucket(int(d[i]))]
        s["end_" + end].append(1)
        caps = {True: 0, False: 0}
        drops = {True: 0, False: 0}
        lions = {True: 0, False: 0}
        first_cap = None
        mats = []
        for t, (wmove, mv, k) in enumerate(line):
            m = material_diff(k) * (1 if wmove else -1)  # 勝つ側から見た駒の枚数の差
            mats.append(m)
            if mv is None:
                continue
            if (mv >> 12) & 7:
                caps[wmove] += 1
                if wmove and first_cap is None:
                    first_cap = t
            if (mv & 15) == 15:
                drops[wmove] += 1
            elif (mv >> 8) & 7 == 1:
                lions[wmove] += 1
        n = max(len(line), 1)
        s["len"].append(len(line))
        s["win_caps_per10"].append(10 * caps[True] / n)
        s["lose_caps_per10"].append(10 * caps[False] / n)
        s["win_drops_per10"].append(10 * drops[True] / n)
        s["win_lion_per10"].append(10 * lions[True] / n)
        s["lose_lion_per10"].append(10 * lions[False] / n)
        s["mat_start"].append(mats[0])
        s["mat_mid"].append(mats[len(mats) // 2])
        s["mat_end"].append(mats[-1])
        s["first_cap_ply"].append(first_cap if first_cap is not None else len(line))
        s["start_behind"].append(int(mats[0] < 0))
    order = ["<=15", "<=45", "<=80", ">80"]
    lines = ["# X9 長期の勝ちの手順の中身（最善手順を終局までたどった集計）", "", "| 指標 | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    names = [
        ("件数", lambda s: len(s["len"])),
        ("トライで終わる率", lambda s: len(s["end_try"]) / max(len(s["len"]), 1)),
        ("手順の長さ(平均)", lambda s: np.mean(s["len"])),
        ("10手あたりの駒取り（勝つ側）", lambda s: np.mean(s["win_caps_per10"])),
        ("10手あたりの駒取り（負ける側）", lambda s: np.mean(s["lose_caps_per10"])),
        ("10手あたりの打ち駒（勝つ側）", lambda s: np.mean(s["win_drops_per10"])),
        ("10手あたりのライオンの手（勝つ側）", lambda s: np.mean(s["win_lion_per10"])),
        ("10手あたりのライオンの手（負ける側）", lambda s: np.mean(s["lose_lion_per10"])),
        ("駒の枚数の差: 開始時", lambda s: np.mean(s["mat_start"])),
        ("駒の枚数の差: 中間", lambda s: np.mean(s["mat_mid"])),
        ("駒の枚数の差: 終局直前", lambda s: np.mean(s["mat_end"])),
        ("開始時に駒損している率", lambda s: np.mean(s["start_behind"])),
        ("勝つ側が最初に駒を取るまでの手数", lambda s: np.mean(s["first_cap_ply"])),
        ("手順が途切れた件数（不具合検出用）", lambda s: len(s["end_broken"])),
    ]
    for name, f in names:
        row = []
        for b in order:
            v = f(stats[b]) if b in stats else float("nan")
            row.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    out = os.path.join(HERE, "../results/x9_long_wins.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
