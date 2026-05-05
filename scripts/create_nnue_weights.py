"""NNUE 初期重みを生成するスクリプト。

PST 評価 + ファイルボーナスを NN 重みとしてエンコードし、
models/nnue.npz に保存する。これにより NNUE は PST 相当の評価値を
初期状態で出力する。Phase 5 でこの重みを学習の起点として使用する。

実行:
    python scripts/create_nnue_weights.py
    python scripts/create_nnue_weights.py --output path/to/nnue.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import shogi

from core.types import PIECE_VALUES, PieceType, file_of, rank_of
from eval.nnue import INPUT_SIZE, L1_SIZE, L2_SIZE, OUTPUT_SCALE
from eval.pst import _RANK_BONUS  # type: ignore[attr-defined]  # package-internal use

_HAND_PT = (
    shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER,
    shogi.GOLD, shogi.BISHOP, shogi.ROOK,
)
_HAND_MAX = {
    shogi.PAWN: 18, shogi.LANCE: 4, shogi.KNIGHT: 4,
    shogi.SILVER: 4, shogi.GOLD: 4, shogi.BISHOP: 2, shogi.ROOK: 2,
}

# PST に加えてファイルボーナスを付与することで NNUE を PST より少し強くする。
# 正値 = ファイル5 (中央) に近いほど得。負値 = 端ファイルが得 (玉向き)。
_FILE_BONUS: dict[PieceType, int] = {
    PieceType.PAWN:       0,
    PieceType.LANCE:      0,
    PieceType.KNIGHT:     5,
    PieceType.SILVER:     8,
    PieceType.GOLD:       8,
    PieceType.BISHOP:     0,
    PieceType.ROOK:       5,
    PieceType.KING:      -15,  # 端に玉を置くことを優先
    PieceType.PRO_PAWN:   5,
    PieceType.PRO_LANCE:  5,
    PieceType.PRO_KNIGHT: 5,
    PieceType.PRO_SILVER: 5,
    PieceType.HORSE:      5,
    PieceType.DRAGON:     8,
}

_BIAS_SCALE = 10_000.0  # 入力を常に正にするためのバイアス規模


def _piece_sq_value(pt: PieceType, sq: int, is_black: bool) -> int:
    """PST 値 + ファイルボーナスを合算した駒-マス評価値 (先手視点)。"""
    rank_idx = (rank_of(sq) - 1) if is_black else (9 - rank_of(sq))
    rank_idx = max(0, min(8, rank_idx))
    file = file_of(sq)

    base = PIECE_VALUES.get(pt, 0)
    rank_b = _RANK_BONUS.get(pt, (0,) * 9)[rank_idx]
    # 中央 (file 5) に近いほど高い。dist=0(中央)〜4(端)
    dist = abs(file - 5)
    file_b = _FILE_BONUS.get(pt, 0) * (4 - dist) // 4

    return base + rank_b + file_b


def build_value_vector() -> np.ndarray:
    """PST + ファイルボーナスを表す INPUT_SIZE 次元ベクトルを構築する。

    v[idx] > 0: その特徴が先手に有利。
    v[idx] < 0: その特徴が後手に有利。
    """
    v = np.zeros(INPUT_SIZE, dtype=np.float32)

    # 盤面駒
    for color in (0, 1):
        sign = 1 if color == 0 else -1
        is_black = color == 0
        for pt_idx in range(14):
            pt_int = pt_idx + 1
            try:
                pt = PieceType(pt_int)
            except ValueError:
                continue
            for sq in range(81):
                val = _piece_sq_value(pt, sq, is_black)
                idx = color * 14 * 81 + pt_idx * 81 + sq
                v[idx] = float(sign * val)

    # 持ち駒 (正規化後の値が 1 のとき駒価値×最大枚数)
    for ci in range(2):
        sign = 1 if ci == 0 else -1
        for ti, pt_int in enumerate(_HAND_PT):
            try:
                pt = PieceType(pt_int)
            except ValueError:
                continue
            val = PIECE_VALUES.get(pt, 0) * _HAND_MAX.get(pt_int, 4)
            idx = 14 * 81 * 2 + ci * len(_HAND_PT) + ti
            v[idx] = float(sign * val)

    return v


def create_pst_weights(
    output_path: Path = Path("models/nnue.npz"),
    noise_scale: float = 0.02,
    rng_seed: int = 42,
) -> None:
    """PST 知識をエンコードした初期重みを生成して npz に保存する。

    数学的保証:
      net(x) * OUTPUT_SCALE ≈ v · x  (= PST + ファイルボーナス)
    ただし |v·x| < _BIAS_SCALE の範囲で正確。
    """
    rng = np.random.default_rng(rng_seed)
    v = build_value_vector()
    C = _BIAS_SCALE

    # W1: 全ニューロンが同じ PST 重みを持つ (rank-1 initialization)
    # b1 = 1.0 により h1 = ReLU(PST/C + 1) ≈ PST/C + 1 (通常局面では常に正)
    w1_row = (v / C).astype(np.float32)
    w1 = np.tile(w1_row, (L1_SIZE, 1)).T.copy()  # (INPUT_SIZE, L1_SIZE)
    noise = rng.normal(0.0, noise_scale * max(float(np.abs(w1_row).max()), 1e-6), w1.shape)
    w1 += noise.astype(np.float32)
    b1 = np.ones(L1_SIZE, dtype=np.float32)

    # W2: 平均化層 (h2 ≈ h1 の平均 ≈ PST/C + 1)
    w2 = (np.ones((L1_SIZE, L2_SIZE)) / L1_SIZE).astype(np.float32)
    w2 += rng.normal(0.0, noise_scale / L1_SIZE, w2.shape).astype(np.float32)
    b2 = np.zeros(L2_SIZE, dtype=np.float32)

    # W3 + b3: スケールバック
    # output = (PST/C + 1) * (C/OUTPUT_SCALE) - C/OUTPUT_SCALE = PST/OUTPUT_SCALE
    sf = C / OUTPUT_SCALE  # scale_factor = 10000/600 ≈ 16.67
    w3 = (np.ones((L2_SIZE, 1)) * sf / L2_SIZE).astype(np.float32)
    b3 = np.array([-sf], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

    print(f"Saved to {output_path}")
    print(f"  architecture: {INPUT_SIZE} → {L1_SIZE} → {L2_SIZE} → 1")
    print(f"  PST vector norm: {float(np.linalg.norm(v)):.1f}")
    print(f"  bias scale C={C}, output scale={OUTPUT_SCALE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NNUE 初期重み生成")
    parser.add_argument("--output", default="models/nnue.npz", help="出力パス")
    parser.add_argument("--noise", type=float, default=0.02, help="対称性破りノイズ")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_pst_weights(Path(args.output), args.noise, args.seed)


if __name__ == "__main__":
    main()
