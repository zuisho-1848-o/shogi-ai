"""証明DAGのうち、どこまで証明が閉じているかを数える（未完成の DAG の部分的な成果をまとめる）

  後手番の局面: 合法手がない（詰み）か、全ての応手の後の先手番の局面が証明済み → 証明済み
  先手番の局面: 選んだ手の後の後手番の局面が証明済み → 証明済み
根（初期局面）から: 先手の初手、その後の後手の応手ごとに証明済みかどうか、を出す。
使い方: python closed.py --state ../data/kyoto/proof_state.pkl
"""

import argparse
import collections
import json
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(HERE, "../data/kyoto/proof_state.pkl"))
    ap.add_argument("--out", default=os.path.join(HERE, "../results/kyoto_partial.json"))
    a = ap.parse_args()
    st = pickle.load(open(a.state, "rb"))
    win, lose, fen = st["win"], st["lose"], st["fen"]
    # 親を引けるようにして、詰みから逆向きに広げる
    parents = collections.defaultdict(list)
    for k, c in win.items():
        parents[c].append(k)
    remaining = {}
    for k, cs in lose.items():
        for c in cs:
            parents[c].append(k)
        remaining[k] = len(set(cs))
    proven = set()
    q = collections.deque(k for k, cs in lose.items() if not cs)
    proven.update(q)
    while q:
        k = q.popleft()
        for p in parents[k]:
            if p in proven:
                continue
            if p in win:  # 先手番: 選んだ手の行き先が証明済み
                if win[p] == k:
                    proven.add(p)
                    q.append(p)
            else:  # 後手番: 全ての応手の行き先が証明済みになったら
                remaining[p] -= 1
                if remaining[p] == 0:
                    proven.add(p)
                    q.append(p)
    root = st["root"]
    first = win.get(root)
    replies = lose.get(first, []) if first else []
    res = {
        "positions": len(win) + len(lose),
        "proven": len(proven),
        "proven_sente": sum(1 for k in proven if k in win),
        "proven_gote": sum(1 for k in proven if k in lose),
        "root_proven": root in proven,
        "first_move_position": fen.get(first),
        "replies": len(set(replies)),
        "replies_proven": sum(1 for r in set(replies) if r in proven),
        "unexpanded": len(st["queue"]),
        "fail": len(st["fail"]),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
