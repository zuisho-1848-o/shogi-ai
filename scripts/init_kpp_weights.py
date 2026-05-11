"""PST初期値からKPテーブルを生成するスクリプト。

CSAファイルなしでもKPP評価関数を使い始められる。
学習前の初期値として使うか、そのまま対局に使うことができる。

使い方:
  python -m scripts.init_kpp_weights
  python -m scripts.init_kpp_weights --output models/kpp.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.kpp import KPPEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="PST初期値からKPテーブルを生成")
    parser.add_argument(
        "--output", type=Path, default=Path("models/kpp.npz"),
        help="保存先 (default: models/kpp.npz)",
    )
    args = parser.parse_args()

    print("[init_kpp_weights] PST値からKPテーブルを生成中...")
    evaluator = KPPEvaluator.from_pst_values()
    evaluator.save(args.output)
    print(f"[init_kpp_weights] 保存完了: {args.output}")
    print(f"  テーブル shape: {evaluator._table.shape}")
    print(f"  ファイルサイズ: {args.output.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
