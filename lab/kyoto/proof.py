"""X26: 京都将棋の先手必勝の証明DAGを作る（塩田・伊藤 2021 の方法を、独立に実装）

根 = 初期局面（先手番）。
  後手の手番の局面: 後手の合法手を全て展開（pyffish で列挙）
  先手の手番の局面: Fairy-Stockfish に最善手を1つ選ばせる（--nodes ノード）。詰みが見つかればその手順を優先
同じ局面（盤＋持ち駒＋手番）は1つにまとめる（DAG）。後手に合法手がなければ、その局面は先手の勝ち（詰み）。
  先手番の局面で先手に合法手がない／後手が詰ませた → 失敗として記録（その局面の先手の手を選び直す必要あり）
途中経過は --state に保存し、続きから再開できる。

出力: --state（pickle）。cert_kyoto.py が Lean 用の証明書に変換する
使い方: python proof.py --nodes 50000 --hours 20
"""

import argparse
import collections
import os
import pickle
import subprocess
import sys
import time

import pyffish as sf

HERE = os.path.dirname(os.path.abspath(__file__))
FSF = os.path.join(HERE, "../engine/third_party/Fairy-Stockfish/src/stockfish")
V = "kyotoshogi"


def key(fen):
    return " ".join(fen.split()[:2])  # 盤＋持ち駒、手番（手数は無視）


class Engine:
    def __init__(self, hash_mb=64):
        self.p = subprocess.Popen([FSF], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        for k, v in {"UCI_Variant": V, "Threads": 1, "Hash": hash_mb}.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, tok):
        out = []
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError("engine died")
            out.append(l.strip())
            if l.startswith(tok):
                return out

    def best(self, fen, nodes):
        self.send(f"position fen {fen}")
        self.send(f"go nodes {nodes}")
        lines = self.wait("bestmove")
        score = None
        for l in reversed(lines):
            t = l.split()
            if "score" in t:
                i = t.index("score")
                score = (t[i + 1], int(t[i + 2]))
                break
        return lines[-1].split()[1], score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=50000)
    ap.add_argument("--hours", type=float, default=20)
    ap.add_argument("--state", default=os.path.join(HERE, "../data/kyoto/proof_state.pkl"))
    ap.add_argument("--dfs", action="store_true", help="深さ優先（詰みまで早くたどり着き、部分ごとに閉じる）")
    ap.add_argument("--save-sec", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.state), exist_ok=True)
    if os.path.exists(a.state):
        st = pickle.load(open(a.state, "rb"))
    else:
        root = sf.start_fen(V)
        st = {"root": key(root), "fen": {key(root): root}, "win": {}, "lose": {}, "queue": collections.deque([key(root)]), "fail": {}, "decisions": 0}
    st["dfs"] = a.dfs
    eng = Engine()
    t0 = time.time()
    last_save = t0
    fen = st["fen"]
    while st["queue"] and time.time() - t0 < a.hours * 3600:
        k = st["queue"].pop() if st.get("dfs") else st["queue"].popleft()
        f = fen[k]
        side = f.split()[1]
        moves = sf.legal_moves(V, f, [])
        if side == "w":  # 先手の手番（Fairy-Stockfish では先手が w）
            if k in st["win"]:
                continue
            if not moves:
                st["fail"][k] = "先手に合法手がない"
                continue
            mv, score = eng.best(f, a.nodes)
            st["decisions"] += 1
            if score and score[0] == "mate" and score[1] < 0:
                st["fail"][k] = f"後手の詰みを読まれた {score}"
            nf = sf.get_fen(V, f, [mv])
            nk = key(nf)
            st["win"][k] = nk
            if nk not in fen:
                fen[nk] = nf
                st["queue"].append(nk)
        else:  # 後手の手番: 全ての応手
            if k in st["lose"]:
                continue
            kids = []
            for mv in moves:
                nf = sf.get_fen(V, f, [mv])
                nk = key(nf)
                kids.append(nk)
                if nk not in fen:
                    fen[nk] = nf
                    st["queue"].append(nk)
            st["lose"][k] = kids  # 合法手がなければ空 = 後手は詰み（先手の勝ち）
        if time.time() - last_save > a.save_sec:
            pickle.dump(st, open(a.state, "wb"))
            last_save = time.time()
            mates = sum(1 for v in st["lose"].values() if not v)
            print(f"{time.time() - t0:.0f}s: 先手局面 {len(st['win']):,} / 後手局面 {len(st['lose']):,}（うち詰み {mates:,}）/ 未処理 {len(st['queue']):,} / 失敗 {len(st['fail'])} / 判断 {st['decisions']:,}", flush=True)
    pickle.dump(st, open(a.state, "wb"))
    print(f"done: 先手局面 {len(st['win']):,} / 後手局面 {len(st['lose']):,} / 未処理 {len(st['queue']):,} / 失敗 {len(st['fail'])}", flush=True)
    eng.send("quit")


if __name__ == "__main__":
    main()
