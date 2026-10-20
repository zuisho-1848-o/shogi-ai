"""盤の読み取りを記録する将棋の規則（詰みの関連領域を証明つきで求めるための部品）

考え方:
  詰みの証明木を検証するあいだに「どのマスの中身を読んだか」「どの持ち駒の有無を読んだか」を全て記録する。
  計算は読んだ情報だけで決まるので、読まなかったマスの中身が何であっても同じ結論になる。
  → 読んだマスの集合が、その詰みの「関連領域」になる（証明つき）。

  そのために、受け方の応手は盤全体を走査して作らず、
    - 玉の8近傍への移動
    - 王手している駒を取る手（王手駒のマスから逆向きに走査して、そこに利いている受け方の駒を探す）
    - 合駒（間のマスに利いている受け方の駒を逆向きに探す、または持ち駒を打つ）
  だけを、必要なマスだけ読んで作る。大駒の長い利きも、逆向きの走査で自然に読み取りの記録に入る。

マス: sq = (筋-1)*9 + (段-1)。段0が一段目（後手の陣の奥）。先手の前方は段が減る向き。
駒: cshogi の番号（1..14 が先手、+16 が後手）。持ち駒の並び: 歩 香 桂 銀 金 角 飛
"""

PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK, GOLD, KING = 1, 2, 3, 4, 5, 6, 7, 8
PPAWN, PLANCE, PKNIGHT, PSILVER, HORSE, DRAGON = 9, 10, 11, 12, 13, 14
HAND_INDEX = {PAWN: 0, LANCE: 1, KNIGHT: 2, SILVER: 3, GOLD: 4, BISHOP: 5, ROOK: 6}
HAND_TYPES = [PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK]
USI_PIECE = {PAWN: "P", LANCE: "L", KNIGHT: "N", SILVER: "S", GOLD: "G", BISHOP: "B", ROOK: "R"}
UNPROMOTE = {PPAWN: PAWN, PLANCE: LANCE, PKNIGHT: KNIGHT, PSILVER: SILVER, HORSE: BISHOP, DRAGON: ROOK}
PROMOTE = {PAWN: PPAWN, LANCE: PLANCE, KNIGHT: PKNIGHT, SILVER: PSILVER, BISHOP: HORSE, ROOK: DRAGON}

ORTH = [(0, -1), (0, 1), (-1, 0), (1, 0)]
DIAG = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
GOLD_STEPS = [(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1)]
STEPS = {
    PAWN: [(0, -1)],
    SILVER: [(0, -1), (-1, -1), (1, -1), (-1, 1), (1, 1)],
    GOLD: GOLD_STEPS, PPAWN: GOLD_STEPS, PLANCE: GOLD_STEPS, PKNIGHT: GOLD_STEPS, PSILVER: GOLD_STEPS,
    KING: ORTH + DIAG,
    HORSE: ORTH,
    DRAGON: DIAG,
}
JUMPS = {KNIGHT: [(-1, -2), (1, -2)]}
SLIDES = {LANCE: [(0, -1)], BISHOP: DIAG, HORSE: DIAG, ROOK: ORTH, DRAGON: ORTH}


def ptype(p):
    return p & 15


def pcolor(p):
    return 1 if p & 16 else 0


def orient(dirs, color):
    # 後手は前後が逆（左右は対称なので段の向きだけ反転）
    return dirs if color == 0 else [(df, -dr) for df, dr in dirs]


def sq_of(f, r):
    return f * 9 + r


def fr(sq):
    return divmod(sq, 9)


def usi_sq(sq):
    f, r = fr(sq)
    return f"{f + 1}{chr(ord('a') + r)}"


def in_zone(color, r):
    # 成れる段（相手の陣）
    return r <= 2 if color == 0 else r >= 6


def dead_square(pt, color, r):
    # 行き所のない駒になる段
    rr = r if color == 0 else 8 - r
    return (pt in (PAWN, LANCE) and rr == 0) or (pt == KNIGHT and rr <= 1)


ALL = (1 << 32) - 1
EMPTY_ONLY = 1


def code_bits(codes):
    m = 0
    for c in codes:
        m |= 1 << c
    return m


def color_bits(color):
    return code_bits(t + 16 * color for t in range(1, 15))


class Ctx:
    """盤面と持ち駒。読み取りは「そのマスに許される駒の集合（32bitの集合）」として記録される。

    cons[sq] = 元の局面の sq に置いてよい駒コードの集合（0 = 空き）。証明木全体で共有。
    計算は記録した制約を満たす限り同じ結果になるので、制約を満たす局面は全て同じ手順で詰む。
    touched = 手順の途中で動いたマス（元の局面の中身とは違うので、制約は元の局面に対しては付けない。
    そのマスは動かした時点で厳密に読んでいる）
    """

    def __init__(self, board, hands, turn, cons, hreads, touched=frozenset()):
        self.b = list(board)
        self.h = [list(hands[0]), list(hands[1])]
        self.turn = turn
        self.cons = cons
        self.hreads = hreads
        self.touched = touched

    @property
    def reads(self):
        return set(self.cons)

    def copy(self):
        return Ctx(self.b, self.h, self.turn, self.cons, self.hreads, self.touched)

    def need(self, sq, mask):
        if sq in self.touched:
            return
        self.cons[sq] = self.cons.get(sq, ALL) & mask
        assert self.cons[sq] & (1 << self.b[sq]), "制約が元の局面と矛盾"

    def get(self, sq):
        """中身をそのまま使う読み取り（厳密）"""
        self.need(sq, 1 << self.b[sq])
        return self.b[sq]

    def hand(self, color, pt):
        self.hreads.add((color, pt))
        return self.h[color][HAND_INDEX[pt]]

    # ---- 利き ----
    @staticmethod
    def _forbid(by, d, k, exclude_king):
        """s から向き d・距離 k で目標に利く by 側の駒コードの集合"""
        m = 0
        for t in range(1, 15):
            if exclude_king and t == KING:
                continue
            if (k == 1 and d in orient(STEPS.get(t, []), by)) or d in orient(SLIDES.get(t, []), by):
                m |= 1 << (t + 16 * by)
        return m

    def attackers(self, sq, by, first_only=False, exclude_king=False):
        """sq に利いている by 側の駒のマスの一覧（逆向きに走査。利く駒が無いことを確かめたマスは
        「利く駒以外なら何でもよい」、遮っている駒は「何かの駒で、利かないもの」という制約になる）"""
        f0, r0 = fr(sq)
        out = []
        for df, dr in ORTH + DIAG:
            k = 1
            d = (-df, -dr)
            while True:
                f, r = f0 + df * k, r0 + dr * k
                if not (0 <= f < 9 and 0 <= r < 9):
                    break
                s = sq_of(f, r)
                p = self.b[s]
                forbid = self._forbid(by, d, k, exclude_king)
                if p and (1 << p) & forbid:
                    self.get(s)  # 利いている駒（取る手などに使うので厳密に）
                    out.append(s)
                    if first_only:
                        return out
                    break
                if p:
                    self.need(s, ALL & ~EMPTY_ONLY & ~forbid)  # 遮る駒（利かない駒なら何でもよい）
                    break
                self.need(s, ALL & ~forbid)  # 空き（利く駒でなければ何でもよい）
                k += 1
        kn = 1 << (KNIGHT + 16 * by)
        for jf, jr in orient(JUMPS[KNIGHT], by):
            f, r = f0 - jf, r0 - jr
            if 0 <= f < 9 and 0 <= r < 9:
                s = sq_of(f, r)
                if self.b[s] == KNIGHT + 16 * by:
                    self.get(s)
                    out.append(s)
                    if first_only:
                        return out
                else:
                    self.need(s, ALL & ~kn)
        return out

    def attacked(self, sq, by):
        return bool(self.attackers(sq, by, first_only=True))

    def king_sq(self, color, known):
        # 玉の位置は元の局面から分かっている。そこに玉があることを読んで確かめる
        assert self.get(known) == KING + 16 * color
        return known

    # ---- 指し手 ----
    def apply(self, mv):
        """mv = (from or None, to, 駒の種類（打ち）or 成るか)。新しい Ctx を返す"""
        c = self.copy()
        frm, to, x = mv
        me = self.turn
        c.touched = self.touched | {to} | ({frm} if frm is not None else set())
        if frm is None:
            c.h[me][HAND_INDEX[x]] -= 1
            c.b[to] = x + 16 * me
        else:
            p = c.b[frm]
            cap = c.b[to]
            if cap:
                t = UNPROMOTE.get(ptype(cap), ptype(cap))
                c.h[me][HAND_INDEX[t]] += 1
            c.b[frm] = 0
            c.b[to] = (PROMOTE[ptype(p)] + 16 * me) if x else p
        c.turn = 1 - me
        return c

    def promo_options(self, pt, color, fr_r, to_r):
        can = pt in PROMOTE and (in_zone(color, fr_r) or in_zone(color, to_r))
        must = dead_square(pt, color, to_r)
        if must:
            return [True]
        return [False, True] if can else [False]

    def evasions(self, ksq):
        """手番側（受け方）の王手回避手。王手されていなければ None"""
        me, op = self.turn, 1 - self.turn
        checkers = self.attackers(ksq, op)
        if not checkers:
            return None
        moves = []
        kf, kr = fr(ksq)
        # 1. 玉の移動（元のマスを空けて利きを調べる）
        for df, dr in ORTH + DIAG:
            f, r = kf + df, kr + dr
            if not (0 <= f < 9 and 0 <= r < 9):
                continue
            t = sq_of(f, r)
            p = self.b[t]
            if p and pcolor(p) == me:
                self.need(t, color_bits(me))
                continue
            self.get(t)
            c = self.copy()
            c.touched = self.touched | {ksq, t}
            c.b[ksq] = 0
            c.b[t] = KING + 16 * me
            if not c.attacked(t, op):
                moves.append((ksq, t, False))
        if len(checkers) >= 2:
            return moves
        chk = checkers[0]
        targets = [chk]
        cf, cr = fr(chk)
        if max(abs(cf - kf), abs(cr - kr)) > 1 and ptype(self.b[chk]) != KNIGHT:
            df = (kf > cf) - (kf < cf)
            dr = (kr > cr) - (kr < cr)
            f, r = cf + df, cr + dr
            while (f, r) != (kf, kr):
                targets.append(sq_of(f, r))
                f, r = f + df, r + dr
        for t in targets:
            # 2. 駒を動かして取る/合駒する（玉以外）
            for s in self.attackers(t, me, exclude_king=True):
                p = self.b[s]
                pt = ptype(p)
                for promo in self.promo_options(pt, me, fr(s)[1], fr(t)[1]):
                    mv = (s, t, promo)
                    c = self.apply(mv)
                    c.turn = me
                    if not c.attacked(ksq, op):
                        moves.append(mv)
            # 3. 合駒を打つ（取る対象のマスには打てない）
            if t != chk:
                for pt in HAND_TYPES:
                    if not self.hand(me, pt):
                        continue
                    if dead_square(pt, me, fr(t)[1]):
                        continue
                    if pt == PAWN and self.nifu(me, fr(t)[0]):
                        continue
                    moves.append((None, t, pt))
        return moves

    def nifu(self, color, f):
        pawn = PAWN + 16 * color
        for r in range(9):
            s = sq_of(f, r)
            if self.b[s] == pawn:
                self.get(s)
                return True
            self.need(s, ALL & ~(1 << pawn))
        return False

    def verify_attack_move(self, mv, my_k, op_k):
        """攻め方の手 mv が合法で、王手になっていることを確かめる（読み取りは記録される）"""
        me, op = self.turn, 1 - self.turn
        frm, to, x = mv
        if frm is None:
            assert self.hand(me, x) > 0, "持ち駒がない"
            assert self.get(to) == 0, "打つマスが空いていない"
            assert not dead_square(x, me, fr(to)[1])
            if x == PAWN:
                assert not self.nifu(me, fr(to)[0]), "二歩"
        else:
            p = self.get(frm)
            assert p and pcolor(p) == me, "動かす駒がない"
            q = self.get(to)
            assert not (q and pcolor(q) == me)
            assert frm in self.attackers_from_piece(frm, to), "その駒では行けない"
        c = self.apply(mv)
        # 自玉が取られる手でないこと
        c.turn = me
        if my_k is not None:
            mk = to if (frm == my_k) else my_k
            assert not c.attacked(mk, op), "自玉に王手がかかる"
        assert c.attacked(op_k, me), "王手になっていない"
        c.turn = op
        return c

    def attackers_from_piece(self, frm, to):
        """frm の駒が to に行けるか（経路のマスは読む）。行けるなら [frm]"""
        p = self.b[frm]
        pt, col = ptype(p), pcolor(p)
        f0, r0 = fr(frm)
        f1, r1 = fr(to)
        d = (f1 - f0, r1 - r0)
        if d in orient(STEPS.get(pt, []), col) or d in orient(JUMPS.get(pt, []), col):
            return [frm]
        for df, dr in orient(SLIDES.get(pt, []), col):
            f, r = f0 + df, r0 + dr
            while 0 <= f < 9 and 0 <= r < 9:
                s = sq_of(f, r)
                if s == to:
                    return [frm]
                if self.get(s):
                    break
                f, r = f + df, r + dr
        return []


def to_usi(mv):
    frm, to, x = mv
    if frm is None:
        return f"{USI_PIECE[x]}*{usi_sq(to)}"
    return f"{usi_sq(frm)}{usi_sq(to)}{'+' if x else ''}"


def from_cshogi(m, cshogi):
    if cshogi.move_is_drop(m):
        return (None, cshogi.move_to(m), HAND_TYPES[cshogi.move_drop_hand_piece(m)])
    return (cshogi.move_from(m), cshogi.move_to(m), bool(cshogi.move_is_promotion(m)))
