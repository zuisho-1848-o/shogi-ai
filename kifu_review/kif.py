"""KIF形式（棋桜・ShogiGUI・将棋ウォーズ・ShogiHome等の出力）のパーサ。

KIFの指し手 → USI手 に変換し、python-shogi で合法性を検証する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import shogi

HANDICAP_SFEN = {
    "平手": shogi.STARTING_SFEN,
    "香落ち": "lnsgkgsn1/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "右香落ち": "1nsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "角落ち": "lnsgkgsnl/1r7/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "飛車落ち": "lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "飛香落ち": "lnsgkgsn1/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "二枚落ち": "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "三枚落ち": "lnsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "四枚落ち": "1nsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "五枚落ち": "2sgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "六枚落ち": "2sgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "八枚落ち": "3gkg3/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "十枚落ち": "4k4/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
}

_ZEN = str.maketrans("１２３４５６７８９", "123456789")
_KAN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_PIECE_USI = {
    "歩": "P", "香": "L", "桂": "N", "銀": "S", "金": "G", "角": "B", "飛": "R",
    "玉": "K", "王": "K", "と": "+P", "成香": "+L", "成桂": "+N", "成銀": "+S",
    "馬": "+B", "龍": "+R", "竜": "+R",
}  # fmt: skip
_END_WORDS = (
    "投了", "中断", "千日手", "持将棋", "詰み", "切れ負け", "反則勝ち", "反則負け",
    "不戦勝", "不戦敗", "入玉勝ち", "封じ手",
)  # fmt: skip
_MOVE_RE = re.compile(
    r"^\s*(?P<no>\d+)\s+(?P<mv>\S.*?)\s*"
    r"(?:\(\s*(?P<t1>\d+:\d+)\s*/\s*(?P<t2>[\d:]+)\s*\))?\s*\+?\s*$"
)
_BODY_RE = re.compile(
    r"^(?P<dest>同\s*|[1-9１-９][一二三四五六七八九1-9])"
    r"(?P<piece>成香|成桂|成銀|[歩香桂銀金角飛玉王とと馬龍竜])"
    r"(?P<flag>不成|成|打)?(?:\((?P<src>[1-9]{2})\))?$"
)


class KifError(ValueError):
    pass


@dataclass
class KifMove:
    no: int
    usi: str
    seconds: int | None = None  # この手の消費秒数
    comment: str = ""


@dataclass
class KifGame:
    headers: dict[str, str] = field(default_factory=dict)
    start_sfen: str = shogi.STARTING_SFEN
    moves: list[KifMove] = field(default_factory=list)
    end_word: str = ""  # 投了 / 詰み / 中断 など

    @property
    def black_name(self) -> str:
        return self.headers.get("先手") or self.headers.get("下手") or "先手"

    @property
    def white_name(self) -> str:
        return self.headers.get("後手") or self.headers.get("上手") or "後手"


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise KifError("文字コードを判別できません（UTF-8 / Shift_JIS 以外）")


def _hms_to_sec(s: str) -> int:
    parts = [int(p) for p in s.split(":")]
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    return sec


def _to_usi(body: str, prev_dest: str | None, lineno: int) -> str:
    body = body.replace("　", "").replace(" ", "")
    m = _BODY_RE.match(body)
    if not m:
        raise KifError(f"{lineno}行目: 指し手を解釈できません: {body!r}")
    dest = m["dest"]
    if dest.startswith("同"):
        if not prev_dest:
            raise KifError(f"{lineno}行目: 『同』の前に指し手がありません")
        dest_usi = prev_dest
    else:
        f = int(dest[0].translate(_ZEN))
        r = _KAN[dest[1]] if dest[1] in _KAN else int(dest[1])
        dest_usi = f"{f}{chr(ord('a') + r - 1)}"
    piece, flag, src = m["piece"], m["flag"], m["src"]
    if flag == "打":
        return f"{_PIECE_USI[piece]}*{dest_usi}"
    if not src:
        raise KifError(f"{lineno}行目: 移動元(NN)がありません: {body!r}")
    src_usi = f"{src[0]}{chr(ord('a') + int(src[1]) - 1)}"
    return f"{src_usi}{dest_usi}{'+' if flag == '成' else ''}"


def parse_kif(text: str) -> KifGame:
    game = KifGame()
    prev_dest: str | None = None
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("*"):
            if game.moves:
                c = line[1:].strip()
                game.moves[-1].comment += (("\n" if game.moves[-1].comment else "") + c)
            continue
        if line.startswith("変化"):
            break  # 変化手順は無視（本譜のみ）
        if line.startswith("手数") or line.startswith("&"):
            continue
        mm = _MOVE_RE.match(line)
        if mm and not game.end_word:
            body = mm["mv"]
            if any(body.startswith(w) for w in _END_WORDS):
                game.end_word = body
                continue
            usi = _to_usi(body, prev_dest, lineno)
            prev_dest = usi[-2:] if not usi.endswith("+") else usi[-3:-1]
            sec = _hms_to_sec(mm["t1"]) if mm["t1"] else None
            # 消費時間は『分:秒』形式。累計ではなく1手分
            game.moves.append(KifMove(no=int(mm["no"]), usi=usi, seconds=sec))
            continue
        for sep in ("：", ":"):
            if sep in line:
                k, v = line.split(sep, 1)
                game.headers[k.strip()] = v.strip()
                break
    handicap = game.headers.get("手合割", "平手")
    if handicap not in HANDICAP_SFEN:
        raise KifError(f"未対応の手合割です: {handicap}（局面図付きKIFは未対応）")
    game.start_sfen = HANDICAP_SFEN[handicap]
    if not game.moves:
        raise KifError("指し手が1手も見つかりません。KIF形式のファイルですか？")
    validate(game)
    return game


def validate(game: KifGame) -> None:
    board = shogi.Board(game.start_sfen)
    for mv in game.moves:
        move = shogi.Move.from_usi(mv.usi)
        if move not in board.legal_moves:
            raise KifError(f"{mv.no}手目 {mv.usi} が不正な手です（局面と合っていません）")
        board.push(move)


def load_kif(path: str | Path) -> KifGame:
    return parse_kif(_decode(Path(path).read_bytes()))
