"""京都将棋の証明DAG（proof.py の状態）から、Lean で検査する証明書を作る

  先手番の局面（勝ち、kind 1）: 子は Fairy-Stockfish が選んだ手の後の局面1つ
  後手番の局面（負け、kind 0）: 子は全ての合法手の後の局面（rules.py の順＝Lean の moves の順）
  rank = 終局までの手数（後手が詰んでいる局面が 0）。手順が輪になっていたら証明にならないので報告する
初期局面（先手番）を 0番にする。

形式（Lean の Kyoto/Main.lean が読む）: u32 n、n × [key 18バイト, u8 kind, u16 rank, u32 子の数 c, c × u32 子の番号]
使い方: python cert_kyoto.py --state ../data/kyoto/proof_state.pkl --out ../data/kyoto/cert.bin
"""

import argparse
import collections
import os
import pickle
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rules  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(HERE, "../data/kyoto/proof_state.pkl"))
    ap.add_argument("--out", default=os.path.join(HERE, "../data/kyoto/cert.bin"))
    a = ap.parse_args()
    st = pickle.load(open(a.state, "rb"))
    fen = st["fen"]
    enc = {}

    def K(fk):
        if fk not in enc:
            b, h, s = rules.parse_fen(fen[fk])
            enc[fk] = rules.encode(b, h, s)
        return enc[fk]

    # 証明に必要な局面を根からたどる（未展開の局面があれば未完成）
    root = st["root"]
    order, idx = [], {}
    kids = {}
    missing = 0
    q = collections.deque([root])
    idx[root] = 0
    order.append(root)
    while q:
        k = q.popleft()
        side = fen[k].split()[1]
        if side == "w":
            if k not in st["win"]:
                missing += 1
                kids[k] = None
                continue
            ch = [st["win"][k]]
        else:
            if k not in st["lose"]:
                missing += 1
                kids[k] = None
                continue
            # 子の並びは rules.py（＝Lean）の順に並べ直す
            b, h, s = rules.parse_fen(fen[k])
            want = rules.children_keys(b, h, s)
            bykey = {K(c): c for c in st["lose"][k]}
            assert set(bykey) == set(want), "合法手の集合が Fairy-Stockfish と一致しない"
            ch = [bykey[w] for w in want]
        kids[k] = ch
        for c in ch:
            if c not in idx:
                idx[c] = len(order)
                order.append(c)
                q.append(c)
    print(f"reachable {len(order):,} positions, unexpanded {missing:,}", flush=True)
    if missing:
        print("未完成（未展開の局面がある）")
        return
    # rank（終局までの手数）を後ろから計算。輪の検出も行う
    rank = {}
    state = {}
    cyc = 0
    for s0 in order:
        if s0 in rank:
            continue
        stack = [(s0, 0)]
        while stack:
            k, i = stack.pop()
            if i == 0:
                if state.get(k) == 1:
                    cyc += 1
                    continue
                state[k] = 1
            ch = kids[k]
            if i < len(ch):
                stack.append((k, i + 1))
                c = ch[i]
                if c not in rank:
                    if state.get(c) == 1:
                        cyc += 1
                    else:
                        stack.append((c, 0))
                continue
            state[k] = 2
            rank[k] = 0 if not ch else max(rank.get(c, 10 ** 6) for c in ch) + 1
    if cyc:
        print(f"手順の輪が {cyc} 個ある（証明にならない。該当する先手の手を選び直す必要がある）")
        return
    with open(a.out, "wb") as f:
        f.write(struct.pack("<I", len(order)))
        for k in order:
            side = fen[k].split()[1]
            f.write(K(k).to_bytes(18, "little"))
            f.write(struct.pack("<BHI", 1 if side == "w" else 0, rank[k], len(kids[k])))
            f.write(struct.pack(f"<{len(kids[k])}I", *[idx[c] for c in kids[k]]))
    print(f"certificate: {len(order):,} positions, root rank {rank[root]}, file {os.path.getsize(a.out):,} bytes")


if __name__ == "__main__":
    main()
