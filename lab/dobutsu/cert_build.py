"""どうぶつしょうぎ「後手必勝」の証明書を、真値DBから作る（Lean で検査するため）

証明書 = 局面の並び。各局面は手番側から見た 64bit の key（solve.c と同じ形式。左右反転の正規化はしない）と
  - 勝ち局面（手番側＝勝つ側）: 即勝ち（ライオンを取れる・トライ成立・自ライオンが既に奥の段）なら印だけ、
    そうでなければ「その手を指した後の局面」の番号を1つ
  - 負け局面（手番側＝負ける側）: 即勝ちがないこと、と全ての手の後の局面の番号（Lean 側で合法手を生成して照合する）
  - 順位（終局までの手数）: 子は必ず親より小さい（これで無限ループがないことが保証される）
初期局面（先手番）は負け局面。後手側の手は「勝ちを保ったまま、なるべく既に証明書にある局面へ行く手」を選んで小さくする。

出力: ../data/dobutsu/cert.bin
  u32 局面数 n
  n × [u64 key, u8 種類(0=負け,1=勝ち,2=即勝ち), u16 順位, u32 子の数 c, c × u32 子の番号]
使い方: python cert_build.py
"""

import collections
import os
import struct
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/dobutsu")
REC = np.dtype([("key", "<u8"), ("val", "u1"), ("mv", "<u2")])


class Expander:
    def __init__(self):
        self.p = subprocess.Popen([os.path.join(HERE, "sample"), DATA, "expand"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def __call__(self, key):
        """正規化された key を渡すと (DB の値, 即勝ちか, 子の [(key, val)])"""
        self.p.stdin.write(np.array([key], dtype="<u8").tobytes())
        self.p.stdin.flush()
        h = self.p.stdout.read(11)
        val, imm, nc = h[8], h[9], h[10]
        body = self.p.stdout.read(11 * nc) if nc else b""
        rec = np.frombuffer(body, dtype=REC, count=nc) if nc else np.zeros(0, dtype=REC)
        return val, bool(imm), rec


def mirror(k):
    r = k & ~((1 << 48) - 1)
    for row in range(4):
        for c in range(3):
            r |= ((k >> (4 * (row * 3 + c))) & 15) << (4 * (row * 3 + (2 - c)))
    return r


def canon(k):
    return min(k, mirror(k))


def children_raw(key):
    """正規化しない子局面の key（solve.c の expand と同じ規則を Python で）。Lean 側の合法手と一致させるため"""
    sq = [(key >> (4 * i)) & 15 for i in range(12)]
    hm = [(key >> (48 + 2 * j)) & 3 for j in range(3)]
    ho = [(key >> (54 + 2 * j)) & 3 for j in range(3)]
    DIRS = {1: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], 2: [(-1, 0), (1, 0), (0, -1), (0, 1)], 3: [(-1, -1), (-1, 1), (1, -1), (1, 1)], 4: [(-1, 0)], 5: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]}

    def opp_attacks(b, s):
        sr, sc = divmod(s, 3)
        for i, v in enumerate(b):
            if v & 8:
                r, c = divmod(i, 3)
                for dr, dc in DIRS[v & 7]:
                    if r - dr == sr and c + dc == sc:
                        return True
        return False

    def flip(b, a, o):
        q = [0] * 12
        for i, v in enumerate(b):
            q[11 - i] = (v ^ 8) if v else 0
        k = 0
        for i, v in enumerate(q):
            k |= v << (4 * i)
        for j in range(3):
            k |= o[j] << (48 + 2 * j)
            k |= a[j] << (54 + 2 * j)
        return k

    if 1 in sq[:3]:
        return True, []
    out = []
    win = False
    for i, v in enumerate(sq):
        if v == 0 or v & 8:
            continue
        r, c = divmod(i, 3)
        for dr, dc in DIRS[v]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 4 and 0 <= nc < 3):
                continue
            s = nr * 3 + nc
            cap = sq[s]
            if cap and not cap & 8:
                continue
            if cap == 9:
                win = True
                continue
            b = list(sq)
            a = list(hm)
            b[i] = 0
            b[s] = 5 if (v == 4 and nr == 0) else v
            if cap:
                t = cap & 7
                a[(4 if t == 5 else t) - 2] += 1
            if v == 1 and nr == 0 and not opp_attacks(b, s):
                win = True
                continue
            out.append(flip(b, a, ho))
    for j in range(3):
        if not hm[j]:
            continue
        for s in range(12):
            if sq[s]:
                continue
            b = list(sq)
            a = list(hm)
            b[s] = 2 + j
            a[j] -= 1
            out.append(flip(b, a, ho))
    return win, out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=int, default=0, help="起点の局面（省略時は初期局面）。仕組みの試験用")
    ap.add_argument("--out", default=os.path.join(DATA, "cert.bin"))
    a = ap.parse_args()
    ex = Expander()
    init = 0
    for s, v in ((9, 2), (10, 1), (11, 3), (7, 4), (0, 11), (1, 9), (2, 10), (4, 12)):
        init |= v << (4 * s)
    if a.root:
        init = a.root
    index = {}
    order = []
    info = {}

    cache = {}

    def value(k):
        c = canon(k)
        if c not in cache:
            cache[c] = ex(c)[0] - 1  # d（終局までの手数）。偶数=手番側の負け
        return cache[c]

    q = collections.deque([init])
    index[init] = 0
    order.append(init)
    while q:
        k = q.popleft()
        d = value(k)
        win, kids = children_raw(k)
        if d % 2 == 1:  # 手番側の勝ち
            if win:
                info[k] = (2, d, [])
                continue
            cands = [c for c in kids if value(c) >= 0 and value(c) % 2 == 0 and value(c) < d]
            pick = next((c for c in cands if c in index), cands[0])
            info[k] = (1, d, [pick])
            nxt = [pick]
        else:  # 手番側の負け: 全ての手
            assert not win
            info[k] = (0, d, kids)
            nxt = kids
        for c in nxt:
            if c not in index:
                index[c] = len(order)
                order.append(c)
                q.append(c)
        if len(order) % 100000 == 0:
            print(f"{len(order)} positions", flush=True)
    path = a.out
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(order)))
        for k in order:
            kind, d, kids = info[k]
            f.write(struct.pack("<QBHI", k, kind, d, len(kids)))
            f.write(struct.pack(f"<{len(kids)}I", *[index[c] for c in kids]))
    cnt = collections.Counter(info[k][0] for k in order)
    print(f"certificate: {len(order)} positions (負け {cnt[0]}, 勝ち {cnt[1]}, 即勝ち {cnt[2]}), initial d={info[init][1]}, file {os.path.getsize(path):,} bytes")


if __name__ == "__main__":
    main()
