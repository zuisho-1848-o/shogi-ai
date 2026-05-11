"""KPP学習用データセット。

CSA形式の棋譜ファイルを読み込み、(SFEN文字列, 勝敗スコア) のペアを生成する。
勝敗スコア: 先手勝ち=1.0, 後手勝ち=-1.0, 引き分け=0.0 (先手視点)

KP特徴量エンコーディング:
  盤上の駒 (王以外): color×13types×81sq = 2106次元
  持ち駒: 2色×38カウント = 76次元
  合計: PIECE_FEAT_SIZE = 2182

KPテーブル shape: (81, PIECE_FEAT_SIZE) — 1エントリ = (先手玉マス, 駒特徴) の評価値
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

import shogi

# ---- 特徴量定数 -------------------------------------------------------

_PIECE_TYPE_TO_KP_IDX: dict[int, int] = {
    shogi.PAWN: 0, shogi.LANCE: 1, shogi.KNIGHT: 2,
    shogi.SILVER: 3, shogi.GOLD: 4, shogi.BISHOP: 5, shogi.ROOK: 6,
    shogi.PROM_PAWN: 7, shogi.PROM_LANCE: 8, shogi.PROM_KNIGHT: 9,
    shogi.PROM_SILVER: 10, shogi.PROM_BISHOP: 11, shogi.PROM_ROOK: 12,
}
N_PIECE_TYPES = 13
N_SQUARES = 81
BOARD_FEAT = 2 * N_PIECE_TYPES * N_SQUARES  # 2106

_HAND_TYPES: tuple[int, ...] = (
    shogi.PAWN, shogi.LANCE, shogi.KNIGHT,
    shogi.SILVER, shogi.GOLD, shogi.BISHOP, shogi.ROOK,
)
_HAND_MAX: dict[int, int] = {
    shogi.PAWN: 18, shogi.LANCE: 4, shogi.KNIGHT: 4,
    shogi.SILVER: 4, shogi.GOLD: 4, shogi.BISHOP: 2, shogi.ROOK: 2,
}
# 各駒種の持ち駒特徴量オフセット (PAWN=0, LANCE=18, ..., ROOK=36)
_HAND_OFFSETS: dict[int, int] = {}
_off = 0
for _pt in _HAND_TYPES:
    _HAND_OFFSETS[_pt] = _off
    _off += _HAND_MAX[_pt]
HAND_FEAT_PER_COLOR = _off  # 38

PIECE_FEAT_SIZE = BOARD_FEAT + 2 * HAND_FEAT_PER_COLOR  # 2182


# ---- KP特徴量インデックス計算 -----------------------------------------

def compute_kp_indices(board: shogi.Board, king_color: int) -> tuple[int, list[int]]:
    """
    指定した色の玉から見たKP特徴量インデックスを返す。

    Returns:
        (king_sq, [piece_feat_indices])
        king_sq: 0-80 (後手玉は鏡像化)
        piece_feat_indices: KPテーブルのセカンドインデックス
    """
    # 玉のマスを探す
    king_sq_raw = -1
    for sq in range(N_SQUARES):
        piece = board.piece_at(sq)
        if piece is not None and piece.piece_type == shogi.KING and piece.color == king_color:
            king_sq_raw = sq
            break

    if king_sq_raw == -1:
        return 0, []

    # 後手視点: 盤面を180度反転
    mirror = king_color == shogi.WHITE
    king_sq = (N_SQUARES - 1 - king_sq_raw) if mirror else king_sq_raw

    indices: list[int] = []

    # 盤上の駒 (王以外)
    for sq in range(N_SQUARES):
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == shogi.KING:
            continue
        pt_idx = _PIECE_TYPE_TO_KP_IDX.get(piece.piece_type)
        if pt_idx is None:
            continue

        if mirror:
            actual_sq = N_SQUARES - 1 - sq
            # 後手視点: 「自分の駒」がcolor=0, 「相手の駒」がcolor=1
            actual_color = 0 if piece.color == king_color else 1
        else:
            actual_sq = sq
            actual_color = piece.color  # BLACK=0, WHITE=1

        feat_idx = actual_color * N_PIECE_TYPES * N_SQUARES + pt_idx * N_SQUARES + actual_sq
        indices.append(feat_idx)

    # 持ち駒
    for hand_idx, hand_color in enumerate([king_color, 1 - king_color]):
        color_offset = BOARD_FEAT + hand_idx * HAND_FEAT_PER_COLOR
        for pt in _HAND_TYPES:
            count = board.pieces_in_hand[hand_color].get(pt, 0)
            base = color_offset + _HAND_OFFSETS[pt]
            for c in range(count):
                indices.append(base + c)

    return king_sq, indices


# ---- CSA棋譜パーサー --------------------------------------------------

_RESULT_PATTERN = re.compile(r"%(?:TORYO|RESIGN|ILLEGAL_MOVE|TIME_UP|JISHOGI|KACHI|HIKIWAKE)")
_CSA_MOVE_PATTERN = re.compile(r"^[+-](\d{4}[A-Z]{2})$")
_CSA_DROP_PIECES = {"FU": "P", "KY": "L", "KE": "N", "GI": "S", "KI": "G", "KA": "B", "HI": "R"}
_CSA_PROM_PIECES = {"TO", "NY", "NK", "NG", "UM", "RY"}


def _csa_move_to_usi(csa_body: str) -> str | None:
    """
    CSA手の本体 (色記号なし) をUSI文字列に変換する。
    例: "7776FU" → "7g7f", "0076FU" → "P*7f", "3322UM" → "3c2b+"
    """
    if len(csa_body) != 6:
        return None
    from_file = csa_body[0]
    from_rank = csa_body[1]
    to_file = csa_body[2]
    to_rank = csa_body[3]
    piece = csa_body[4:6]

    to_usi = f"{to_file}{chr(ord('a') + int(to_rank) - 1)}"

    if from_file == "0":  # 打ち駒
        usi_piece = _CSA_DROP_PIECES.get(piece)
        if usi_piece is None:
            return None
        return f"{usi_piece}*{to_usi}"

    from_usi = f"{from_file}{chr(ord('a') + int(from_rank) - 1)}"
    promote = "+" if piece in _CSA_PROM_PIECES else ""
    return f"{from_usi}{to_usi}{promote}"


def _parse_csa_outcome(lines: list[str]) -> float | None:
    """
    CSAファイルの結果行から先手視点の勝敗スコアを返す。
    先手勝=1.0 / 後手勝=-1.0 / 引き分け=0.0 / 不明=None
    """
    winner: int | None = None  # 0=先手, 1=後手
    last_mover: int | None = None

    for line in lines:
        line = line.strip()
        if len(line) >= 7 and line[0] in "+-" and _CSA_MOVE_PATTERN.match(line):
            last_mover = shogi.BLACK if line[0] == "+" else shogi.WHITE
        elif _RESULT_PATTERN.match(line):
            if "HIKIWAKE" in line or "JISHOGI" in line:
                return 0.0
            if "KACHI" in line:
                # 入玉宣言勝ち: 次に指すはずだった側が勝つ
                if last_mover is not None:
                    winner = 1 - last_mover
            else:
                # 投了・反則負け: 最後に指した側が勝つ (相手が投了)
                if last_mover is not None:
                    winner = last_mover
            break

    if winner is None:
        return None
    return 1.0 if winner == shogi.BLACK else -1.0


def load_csa_file(path: Path) -> list[tuple[str, float]]:
    """
    CSAファイルを読み込み、(SFEN, outcome) のリストを返す。
    outcomeは先手視点 (1.0=先手勝, -1.0=後手勝, 0.0=引き分け)。
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = text.splitlines()
    outcome = _parse_csa_outcome(lines)
    if outcome is None:
        return []

    board = shogi.Board()
    results: list[tuple[str, float]] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("'") or line.startswith("$") or line.startswith("P"):
            continue
        if line.startswith("V") or line.startswith("N") or line == "+":
            continue

        if _CSA_MOVE_PATTERN.match(line):
            usi_str = _csa_move_to_usi(line[1:])
            if usi_str is None:
                break
            try:
                move = shogi.Move.from_usi(usi_str)
                if move not in board.legal_moves:
                    break
                results.append((board.sfen(), outcome))
                board.push(move)
            except Exception:
                break

    return results


def load_csa_dir(
    dir_path: Path,
    max_files: int | None = None,
) -> Generator[tuple[str, float], None, None]:
    """
    ディレクトリ内の全CSAファイルから (SFEN, outcome) を逐次生成する。
    """
    files = sorted(dir_path.glob("**/*.csa"))
    if max_files is not None:
        files = files[:max_files]

    for path in files:
        yield from load_csa_file(path)
