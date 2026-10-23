"""京都将棋のルール（自前の実装。Lean へ移すための見本。Fairy-Stockfish の kyotoshogi と同じ定義）

  盤 5×5。マス番号 = (段-1)×5 + 筋（a..e → 0..4）。段1 が先手の陣。先手の前方は段が増える向き
  駒: 玉 / 香⇄金(と) / 銀⇄角 / 桂⇄金 / 歩⇄飛（表⇄裏）。玉以外は動かすたびに必ず裏返る
  取った駒は「組」（香・銀・桂・歩）として持ち駒になり、表でも裏でも空いているマスに打てる（どこでもよい）
  打ち歩詰め・二歩・行き所のない駒の禁止はない。自玉に王手がかかる手は指せない。合法手がなければ負け

駒コード: 種類 0..8（0 玉, 1 香, 2 銀, 3 桂, 4 歩, 5 と(香の裏), 6 角(銀の裏), 7 金(桂の裏), 8 飛(歩の裏)）、
          持ち主 0 先手 / 1 後手。盤の値 = 0（空き）または 1 + 種類 + 9×持ち主
持ち駒: hand[持ち主][組 0..3]（組: 香・銀・桂・歩）
"""

KING, LANCE, SILVER, KNIGHT, PAWN, TOKIN, BISHOP, GOLD, ROOK = range(9)
FLIP = {LANCE: TOKIN, TOKIN: LANCE, SILVER: BISHOP, BISHOP: SILVER, KNIGHT: GOLD, GOLD: KNIGHT, PAWN: ROOK, ROOK: PAWN}
PAIR = {LANCE: 0, TOKIN: 0, SILVER: 1, BISHOP: 1, KNIGHT: 2, GOLD: 2, PAWN: 3, ROOK: 3}
PAIR_FACES = [(LANCE, TOKIN), (SILVER, BISHOP), (KNIGHT, GOLD), (PAWN, ROOK)]
FSF = {KING: "K", LANCE: "L", SILVER: "S", KNIGHT: "N", PAWN: "P", TOKIN: "+L", BISHOP: "+S", GOLD: "+N", ROOK: "+P"}
FSF_REV = {v: k for k, v in FSF.items()}

GOLD_STEPS = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, 0)]  # (段の差, 筋の差)。先手の前方は段+1
STEPS = {
    KING: [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)],
    SILVER: [(1, -1), (1, 0), (1, 1), (-1, -1), (-1, 1)],
    PAWN: [(1, 0)],
    TOKIN: GOLD_STEPS,
    GOLD: GOLD_STEPS,
    KNIGHT: [(2, -1), (2, 1)],
}
SLIDES = {LANCE: [(1, 0)], BISHOP: [(1, 1), (1, -1), (-1, 1), (-1, -1)], ROOK: [(1, 0), (-1, 0), (0, 1), (0, -1)]}


def orient(d, owner):
    return d if owner == 0 else (-d[0], -d[1])


def sq_name(i):
    return "abcde"[i % 5] + str(i // 5 + 1)


def code(t, owner):
    return 1 + t + 9 * owner


def decode_piece(v):
    return (v - 1) % 9, (v - 1) // 9


def parse_fen(fen):
    board_part, side = fen.split()[:2]
    rows, pocket = board_part.split("[")[0], board_part.split("[")[1].rstrip("]")
    b = [0] * 25
    for ri, row in enumerate(rows.split("/")):
        rank = 4 - ri
        f = 0
        i = 0
        while i < len(row):
            ch = row[i]
            if ch.isdigit():
                f += int(ch)
                i += 1
                continue
            promo = ch == "+"
            if promo:
                i += 1
                ch = row[i]
            owner = 0 if ch.isupper() else 1
            t = FSF_REV[("+" if promo else "") + ch.upper()]
            b[rank * 5 + f] = code(t, owner)
            f += 1
            i += 1
    hand = [[0] * 4, [0] * 4]
    for ch in pocket.replace("-", ""):
        owner = 0 if ch.isupper() else 1
        hand[owner][PAIR[FSF_REV[ch.upper()]]] += 1
    return b, hand, 0 if side == "w" else 1


def attacks(b, sq, by):
    """by 側の駒が sq に利いているか"""
    r0, f0 = divmod(sq, 5)
    for i, v in enumerate(b):
        if not v:
            continue
        t, o = decode_piece(v)
        if o != by:
            continue
        r, f = divmod(i, 5)
        for d in STEPS.get(t, []):
            dr, df = orient(d, o)
            if r + dr == r0 and f + df == f0:
                return True
        for d in SLIDES.get(t, []):
            dr, df = orient(d, o)
            rr, ff = r + dr, f + df
            while 0 <= rr < 5 and 0 <= ff < 5:
                if rr == r0 and ff == f0:
                    return True
                if b[rr * 5 + ff]:
                    break
                rr, ff = rr + dr, ff + df
    return False


def king_sq(b, owner):
    return b.index(code(KING, owner))


def pseudo_moves(b, hand, me):
    """(新しい盤, 新しい持ち駒, FSF 形式の指し手) の並び（自玉の安全は未確認）"""
    out = []
    for i, v in enumerate(b):
        if not v:
            continue
        t, o = decode_piece(v)
        if o != me:
            continue
        r, f = divmod(i, 5)
        targets = []
        for d in STEPS.get(t, []):
            dr, df = orient(d, o)
            rr, ff = r + dr, f + df
            if 0 <= rr < 5 and 0 <= ff < 5:
                targets.append(rr * 5 + ff)
        for d in SLIDES.get(t, []):
            dr, df = orient(d, o)
            rr, ff = r + dr, f + df
            while 0 <= rr < 5 and 0 <= ff < 5:
                targets.append(rr * 5 + ff)
                if b[rr * 5 + ff]:
                    break
                rr, ff = rr + dr, ff + df
        for s in targets:
            cap = b[s]
            if cap and decode_piece(cap)[1] == me:
                continue
            nb = list(b)
            nh = [list(hand[0]), list(hand[1])]
            if cap:
                ct, _ = decode_piece(cap)
                if ct == KING:
                    continue  # 玉を取る手は扱わない（合法な局面では自玉の安全確認で除かれる手順）
                nh[me][PAIR[ct]] += 1
            nt = t if t == KING else FLIP[t]
            nb[i] = 0
            nb[s] = code(nt, me)
            suffix = "" if t == KING else ("+" if nt in (TOKIN, BISHOP, GOLD, ROOK) else "-")
            out.append((nb, nh, sq_name(i) + sq_name(s) + suffix))
    for p in range(4):
        if not hand[me][p]:
            continue
        for face in PAIR_FACES[p]:
            for s in range(25):
                if b[s]:
                    continue
                nb = list(b)
                nh = [list(hand[0]), list(hand[1])]
                nh[me][p] -= 1
                nb[s] = code(face, me)
                out.append((nb, nh, FSF[face] + "@" + sq_name(s)))
    return out


def legal_moves(b, hand, me):
    res = []
    for nb, nh, mv in pseudo_moves(b, hand, me):
        if not attacks(nb, king_sq(nb, me), 1 - me):
            res.append((nb, nh, mv))
    return res


def encode(b, hand, side):
    """Lean の Kyoto.encode と同じ key（盤 25×5bit、持ち駒 8×2bit を bit 125 から、手番を bit 141）"""
    k = 0
    for i, v in enumerate(b):
        k |= v << (5 * i)
    for o in range(2):
        for p in range(4):
            k |= hand[o][p] << (125 + 2 * (4 * o + p))
    return k | (side << 141)


def children_keys(b, hand, side):
    return [encode(nb, nh, 1 - side) for nb, nh, _ in legal_moves(b, hand, side)]
