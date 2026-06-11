"""Phase 1: 詰め将棋シードから確定ラベルDBを構築する。

使い方:
    python -m solve.run_phase1 [--depth N] [--save]

オプション:
    --depth N   探索深さ上限（デフォルト: 15）
    --save      結果を solve/data/confirmed.json に保存
    --verbose   手ごとの詳細を出力
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.board import PythonShogiBoard
from solve.db.confirmed import ConfirmedDB, Label
from solve.retrograde.solver import best_moves, solve_with_stats

# -----------------------------------------------------------------------
# 既知の詰め将棋シード局面（SFEN形式）
# 手番側（b=先手 or w=後手）が詰まされる直前の局面
# -----------------------------------------------------------------------
# 凡例: 「先手玉が詰まされる1手前」= 後手がある手を指すと先手が詰む局面
#       「後手の手番」なので sfen の2番目フィールドは w
# -----------------------------------------------------------------------

TSUME_SEEDS: list[dict] = [
    # --- 詰み局面（終端状態、合法手なし） ---
    {
        "name": "詰み_後手玉1一金2枚",
        "sfen": "8k/7GG/9/9/9/9/9/9/9 w - 1",
        "expected": Label.LOSS,
        "notes": "後手玉1一。先手金2二+1二でチェック+逃げ場なし。is_game_over=True",
    },
    {
        "name": "詰み_後手玉9一金2枚",
        "sfen": "k8/GG7/9/9/9/9/9/9/9 w - 1",
        "expected": Label.LOSS,
        "notes": "後手玉9一。先手金8二+9二でチェック+逃げ場なし。is_game_over=True",
    },
    # --- 1手詰（先手番、1手で詰ます） ---
    {
        "name": "1手詰_金打ち",
        "sfen": "8k/9/8G/9/9/9/9/9/9 b G 1",
        "expected": Label.WIN,
        "notes": "後手玉1一。先手金1三+持ち金1枚。G*1bで詰み。",
    },
    {
        "name": "1手詰_金移動",
        "sfen": "8k/9/7G1/9/9/9/9/9/9 b G 1",
        "expected": Label.WIN,
        "notes": "後手玉1一。先手金2三+持ち金1枚。複数の詰み手あり。",
    },
    # --- 3手詰（先手番） ---
    {
        "name": "3手詰_金移動から",
        "sfen": "8k/9/8G/9/9/9/9/9/9 b 2G 1",
        "expected": Label.WIN,
        "notes": "後手玉1一。先手金1三+持ち金2枚。多段階の詰み。",
    },
    # --- 後手番の詰み直前（深さ次第でunknownになりうる） ---
    {
        "name": "後手LOSS_深さ依存",
        "sfen": "8k/8G/9/9/9/9/9/9/9 w G 1",
        "expected": None,   # 深さが足りないとunknownになる。期待値なし（探索のみ）
        "notes": "後手玉1一。先手金1二でチェック中。後手持ち金1枚。深さ>=7で解決見込み。",
    },
]

VALID_SEEDS = TSUME_SEEDS


def run_phase1(depth: int, save: bool, verbose: bool) -> None:
    save_path = str(Path(__file__).parent / "data" / "confirmed.json") if save else None
    db = ConfirmedDB(path=save_path)

    print("=" * 60)
    print("Phase 1: 後退解析による確定ラベルDB構築")
    print(f"  探索深さ上限: {depth}")
    print(f"  シード局面数: {len(VALID_SEEDS)}")
    print("=" * 60)

    total_start = time.time()
    results: list[dict] = []

    for seed in VALID_SEEDS:
        name = seed["name"]
        sfen = seed["sfen"]
        expected = seed["expected"]

        try:
            board = PythonShogiBoard.from_sfen(sfen)
        except Exception as e:
            print(f"[SKIP] {name}: SFEN解析エラー - {e}")
            continue

        start = time.time()
        label, stats = solve_with_stats(board, db, depth_limit=depth)
        elapsed = time.time() - start

        # 最善手を取得（DBに登録済みの場合）
        bm = best_moves(board, db) if label == Label.WIN else []
        bm_str = ", ".join(m.to_usi() for m in bm) if bm else "-"

        ok = (expected is None) or (label == expected)
        mark = "✓" if ok else "✗"
        if expected is None:
            mark = "~"  # 期待値なし（探索のみ）

        result_str = label.value if label else "unknown"
        exp_str = expected.value if expected else "unknown"

        print(f"\n[{mark}] {name}")
        print(f"    SFEN    : {sfen}")
        print(f"    結果    : {result_str}  (期待値: {exp_str})")
        print(f"    最善手  : {bm_str}")
        print(f"    ノード  : {stats.nodes_visited:,} | キャッシュHit: {stats.cache_hits:,}")
        print(f"    確定数  : WIN={stats.confirmed_win} / LOSS={stats.confirmed_loss}")
        print(f"    時間    : {elapsed:.3f}s")

        if verbose and stats.confirmed_by_depth:
            depth_info = "  ".join(f"d{d}:{n}" for d, n in sorted(stats.confirmed_by_depth.items()))
            print(f"    深さ別  : {depth_info}")

        results.append({"name": name, "label": result_str, "ok": ok, "nodes": stats.nodes_visited})

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("DB 統計")
    stats_db = db.stats()
    print(f"  総局面数  : {len(db):,}")
    for k, v in stats_db.items():
        print(f"  {k:8s}: {v:,}")
    print(f"  総時間    : {total_elapsed:.2f}s")

    correct = sum(1 for r in results if r["ok"])
    print(f"\n正答率: {correct}/{len(results)}")

    if save:
        db.save()
        print(f"\nDB保存: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: 確定ラベルDB構築")
    parser.add_argument("--depth", type=int, default=15, help="探索深さ上限")
    parser.add_argument("--save", action="store_true", help="DBをファイルに保存")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ出力")
    args = parser.parse_args()

    run_phase1(depth=args.depth, save=args.save, verbose=args.verbose)
