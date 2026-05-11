"""KPテーブルの学習スクリプト (Bonanza方式SGD)。

使い方:
  # Floodgate CSAファイルを data/csa/ に置いてから実行
  python -m train.kpp_train --csa-dir data/csa --output models/kpp.npz

  # PST初期値から始める場合
  python -m train.kpp_train --csa-dir data/csa --output models/kpp.npz --init pst

  # 小規模テスト (--max-files で制限)
  python -m train.kpp_train --csa-dir data/csa --output models/kpp.npz --max-files 100

目的関数: MSE(sigmoid(score/600) - outcome)
  score は先手視点のcentipawnスコア
  outcome: 先手勝=1.0, 後手勝=0.0 (sigmoid出力との比較用に0/1に変換)

更新: テーブルは sparse update (使われた (king_sq, feat_idx) のみ更新)
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import shogi

from eval.kpp import KPPEvaluator
from train.dataset import PIECE_FEAT_SIZE, compute_kp_indices, load_csa_dir

_SCORE_SCALE = 600.0  # sigmoid の温度パラメータ
_DEFAULT_LR = 0.01
_DEFAULT_EPOCHS = 3
_DEFAULT_BATCH = 64
_GRAD_CLIP = 5.0


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _compute_score(table: np.ndarray, b: shogi.Board) -> float:
    """テーブルから先手視点スコアを計算する。"""
    k_b, idx_b = compute_kp_indices(b, shogi.BLACK)
    k_w, idx_w = compute_kp_indices(b, shogi.WHITE)
    score_b = float(table[k_b, idx_b].sum()) if idx_b else 0.0
    score_w = float(table[k_w, idx_w].sum()) if idx_w else 0.0
    return score_b - score_w


def _update_table(
    table: np.ndarray,
    b: shogi.Board,
    grad: float,
    lr: float,
) -> None:
    """勾配を先手・後手玉視点の両方の特徴に伝播する。"""
    k_b, idx_b = compute_kp_indices(b, shogi.BLACK)
    k_w, idx_w = compute_kp_indices(b, shogi.WHITE)

    if idx_b:
        idx_arr = np.array(idx_b, dtype=np.int32)
        np.add.at(table[k_b], idx_arr, -lr * grad)
    if idx_w:
        idx_arr = np.array(idx_w, dtype=np.int32)
        np.add.at(table[k_w], idx_arr, lr * grad)  # 後手視点は符号反転


def train(
    csa_dir: Path,
    output_path: Path,
    init: str = "zeros",
    lr: float = _DEFAULT_LR,
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = _DEFAULT_BATCH,
    max_files: int | None = None,
) -> KPPEvaluator:
    """
    CSAファイルからKPテーブルを学習する。

    Args:
        csa_dir: CSAファイルのディレクトリ
        output_path: 学習済みテーブルの保存先 (.npz)
        init: 初期化方法 ("zeros" or "pst")
        lr: 学習率
        epochs: エポック数
        batch_size: ミニバッチサイズ
        max_files: 読み込む最大CSAファイル数 (None=全件)
    """
    print(f"[kpp_train] CSAディレクトリ: {csa_dir}")

    # データ読み込み
    print("[kpp_train] データ読み込み中...")
    t0 = time.time()
    data = list(load_csa_dir(csa_dir, max_files=max_files))
    print(f"[kpp_train] {len(data):,}局面 ({time.time()-t0:.1f}s)")

    if not data:
        raise ValueError(f"CSAファイルが見つかりません: {csa_dir}")

    # テーブル初期化
    if init == "pst":
        evaluator = KPPEvaluator.from_pst_values()
        print("[kpp_train] PST値でテーブルを初期化")
    else:
        evaluator = KPPEvaluator.from_zeros()
        print("[kpp_train] ゼロでテーブルを初期化")

    table = evaluator._table

    # SGD学習
    sfens = [sfen for sfen, _ in data]
    # outcome を 0/1 に変換 (先手勝=1, 後手勝=0, 引き分け=0.5)
    outcomes = np.array(
        [(o + 1.0) / 2.0 for _, o in data], dtype=np.float32
    )
    n = len(data)
    rng = np.random.default_rng(42)

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        indices = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n, batch_size):
            batch_idx = indices[batch_start: batch_start + batch_size]
            batch_loss = 0.0

            for i in batch_idx:
                sfen = sfens[i]
                outcome = float(outcomes[i])

                try:
                    b = shogi.Board(sfen)
                except Exception:
                    continue

                score = _compute_score(table, b)
                pred = _sigmoid(score / _SCORE_SCALE)
                err = pred - outcome
                # d(MSE)/d(score) = 2*(pred-outcome) * pred*(1-pred) / SCALE
                grad = 2.0 * err * pred * (1.0 - pred) / _SCORE_SCALE
                # グラジェントクリッピング
                grad = max(-_GRAD_CLIP, min(_GRAD_CLIP, grad))
                batch_loss += err * err
                _update_table(table, b, grad, lr)

            total_loss += batch_loss
            n_batches += 1

        avg_loss = total_loss / n if n > 0 else 0.0
        elapsed = time.time() - t_ep
        print(f"[epoch {epoch}/{epochs}] loss={avg_loss:.6f}  ({elapsed:.1f}s)")

    # 保存
    evaluator.save(output_path)
    print(f"[kpp_train] 保存: {output_path}")
    return evaluator


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #

def _main() -> None:
    parser = argparse.ArgumentParser(description="KPテーブル学習 (Bonanza方式SGD)")
    parser.add_argument("--csa-dir", required=True, type=Path, help="CSAファイルのディレクトリ")
    parser.add_argument("--output", default=Path("models/kpp.npz"), type=Path, help="保存先")
    parser.add_argument("--init", choices=["zeros", "pst"], default="zeros", help="初期化方法")
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR, help="学習率")
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH, help="ミニバッチサイズ")
    parser.add_argument("--max-files", type=int, default=None, help="最大CSAファイル数")
    args = parser.parse_args()

    train(
        csa_dir=args.csa_dir,
        output_path=args.output,
        init=args.init,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    _main()
