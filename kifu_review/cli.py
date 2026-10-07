"""python -m kifu_review <kif> [options]"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .analyze import analyze_game, cache_key, eval_id_of
from .engine import UsiEngine
from .kif import KifError, load_kif
from .report import build_json, build_markdown
from .review import pick_key_moves, review_game

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENGINE = ROOT / "engines" / "yaneuraou-std" / "yaneuraou"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kifu_review", description="KIF棋譜をやねうら王で解析し、振り返りレポートを作る")
    ap.add_argument("kif", help="KIFファイル（棋桜・ShogiHome等で保存したもの）")
    ap.add_argument("--me", default=os.environ.get("KIFU_REVIEW_ME"), help="自分の対局者名（KIFの先手/後手名と部分一致）。環境変数 KIFU_REVIEW_ME でも可")
    ap.add_argument("--side", choices=["black", "white", "both"], help="--me を使わず手番で指定")
    ap.add_argument("--engine", default=str(DEFAULT_ENGINE), help="USIエンジンのパス")
    ap.add_argument("--eval-dir", help="評価関数ディレクトリ（既定: エンジンと同階層の eval/）")
    ap.add_argument("--movetime", type=int, default=1500, help="1局面あたりの思考時間 ms（既定1500）")
    ap.add_argument("--depth", type=int, help="時間でなく深さ固定で解析")
    ap.add_argument("--multipv", type=int, default=3)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--hash", type=int, default=1024, help="USI_Hash MB")
    ap.add_argument("--top", type=int, default=8, help="振り返る手の最大数（既定8）")
    ap.add_argument("--out", default=str(ROOT / "reports"), help="出力先ディレクトリ")
    ap.add_argument("--fv-scale", type=int, help="評価関数のFV_SCALE（水匠5は24）")
    args = ap.parse_args(argv)

    try:
        game = load_kif(args.kif)
    except (KifError, OSError) as e:
        print(f"KIF読み込みエラー: {e}", file=sys.stderr)
        return 2

    # 振り返り対象の側を決める
    focus_side: str | None = None
    if args.side in ("black", "white"):
        focus_side = "先手" if args.side == "black" else "後手"
    elif args.me:
        hit_b, hit_w = args.me in game.black_name, args.me in game.white_name
        if hit_b != hit_w:
            focus_side = "先手" if hit_b else "後手"
        else:
            print(f"警告: --me '{args.me}' が先手({game.black_name})/後手({game.white_name})のどちらにも一意に一致しません。両者を対象にします。", file=sys.stderr)
    focus_name = {"先手": game.black_name, "後手": game.white_name}.get(focus_side or "", "両者")

    engine_path = Path(args.engine).expanduser()
    eval_dir = Path(args.eval_dir).expanduser() if args.eval_dir else engine_path.parent / "eval"
    if not (eval_dir / "nn.bin").exists() and "material" not in engine_path.name.lower():
        print(f"評価関数が見つかりません: {eval_dir}/nn.bin\n  → README『セットアップ』の手順で水匠5などを配置してください。", file=sys.stderr)
        return 3

    stem = Path(args.kif).stem
    out_dir = Path(args.out) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with UsiEngine(engine_path, eval_dir, args.threads, args.hash, args.multipv, fv_scale=args.fv_scale) as eng:
        key = cache_key(game, eng.name, args.movetime, args.depth, args.multipv, eval_id_of(eval_dir / "nn.bin"))
        print(f"解析開始: {len(game.moves)}手 / {eng.name} / {'depth ' + str(args.depth) if args.depth else str(args.movetime) + 'ms'}", file=sys.stderr)

        def progress(i: int, n: int) -> None:
            print(f"\r  {i}/{n}", end="", file=sys.stderr, flush=True)

        pos = analyze_game(game, eng, args.movetime, args.depth, progress, out_dir / "analysis_cache.json", key)
        print(file=sys.stderr)
        engine_name = eng.name

    reviews = review_game(game, pos)
    keys = pick_key_moves(reviews, focus_side, args.top)
    meta = {
        "engine": engine_name, "limit": f"depth {args.depth}" if args.depth else f"{args.movetime}ms/局面",
        "multipv": args.multipv, "focus": f"{focus_side}（{focus_name}）" if focus_side else "両者", "source": str(args.kif),
    }  # fmt: skip
    (out_dir / "report.md").write_text(build_markdown(game, reviews, keys, pos, meta), encoding="utf-8")
    (out_dir / "report.json").write_text(build_json(game, reviews, keys, meta), encoding="utf-8")
    print(f"レポート: {out_dir / 'report.md'}")
    return 0
