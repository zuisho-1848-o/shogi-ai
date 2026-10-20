"""zone.py の王手回避手が cshogi の合法手と一致するかを、実戦の王手局面で確かめる

cshogi の合法手は「歩・角・飛の不成」などを含まないことがあるので、
  zone の回避手 ⊇ cshogi の合法手、かつ 差分は不成の手だけ
であれば一致とみなす。
使い方: python test_zone.py --games 2000
"""

import argparse
import json
import os

import cshogi

import zone

HERE = os.path.dirname(os.path.abspath(__file__))


def ctx_of(b):
    hb, hw = b.pieces_in_hand
    return zone.Ctx(b.pieces, (hb, hw), b.turn, set(), set())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    a = ap.parse_args()
    n = bad = extra_np = 0
    with open(os.path.join(HERE, "../data/mainline/games.jsonl")) as f:
        for gi, line in enumerate(f):
            if gi >= a.games:
                break
            g = json.loads(line)
            b = cshogi.Board()
            for mv in g["moves"]:
                b.push_usi(mv)
                if not b.is_check():
                    continue
                c = ctx_of(b)
                ev = c.evasions(b.king_square(b.turn))
                mine = {zone.to_usi(m) for m in ev}
                ref = {cshogi.move_to_usi(m) for m in b.legal_moves}
                n += 1
                if not ref <= mine:
                    bad += 1
                    if bad <= 5:
                        print("MISSING", b.sfen(), sorted(ref - mine))
                    continue
                diff = mine - ref
                for u in diff:
                    if u.endswith("+"):
                        bad += 1
                        print("EXTRA(promo)", b.sfen(), u)
                        break
                extra_np += len(diff)
    print(f"checked {n} check positions, mismatches {bad}, extra non-promotion moves {extra_np}")


if __name__ == "__main__":
    main()
