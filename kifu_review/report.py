"""Markdown / JSON レポート生成。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

from .analyze import PosAnalysis, winrate
from .kif import KifGame
from .review import MoveReview, summary

BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(pos: list[PosAnalysis]) -> str:
    """先手視点の勝率を1手1文字で。手番が後手の局面は符号反転。"""
    chars = []
    for i, p in enumerate(pos):
        cp = p.cp if i % 2 == 0 else -p.cp  # 偶数ply=先手番（平手の場合）
        chars.append(BLOCKS[min(7, int(winrate(cp) * 8))])
    return "".join(chars)


def _fmt_cp(cp: int) -> str:
    return "詰み" if abs(cp) >= 29000 else f"{cp:+d}"


def build_markdown(game: KifGame, reviews: list[MoveReview], key_moves: list[MoveReview], pos: list[PosAnalysis], meta: dict) -> str:
    sm = summary(reviews)
    L: list[str] = []
    L.append(f"# 棋譜レビュー: {game.black_name}（先手） vs {game.white_name}（後手）")
    L.append("")
    info = [f"{k}：{v}" for k, v in game.headers.items() if k in ("対局日", "開始日時", "棋戦", "手合割", "場所")]
    L.append(" / ".join(info + [f"総手数：{len(game.moves)}", f"結果：{game.end_word or '不明'}"]))
    L.append(f"解析：{meta['engine']} / {meta['limit']} / MultiPV {meta['multipv']}")
    if meta.get("focus"):
        L.append(f"振り返り対象：{meta['focus']}")
    L.append("")
    L.append("## 形勢の推移（先手勝率・1手1文字）")
    L.append("")
    L.append("```")
    spark = _sparkline(pos)
    for i in range(0, len(spark), 40):
        L.append(f"{i:>3}手 {spark[i:i + 40]}")
    L.append("```")
    L.append("（▁=後手勝勢 … █=先手勝勢）")
    L.append("")
    L.append("## 全体サマリ")
    L.append("")
    L.append("| | 手数 | 最善手一致率 | 平均損失(cp) | 疑問手 | 悪手 | 大悪手 |")
    L.append("|---|---|---|---|---|---|---|")
    for side, name in (("先手", game.black_name), ("後手", game.white_name)):
        s = sm[side]
        L.append(f"| {side}（{name}） | {s['moves']} | {s['best_rate']}% | {s['avg_loss_cp']} | {s['疑問手']} | {s['悪手']} | {s['大悪手']} |")
    L.append("")
    L.append("※ 疑問手=勝率-5%以上 / 悪手=-10%以上 / 大悪手=-20%以上（評価値→勝率は 1/(1+e^(-cp/600))）")
    L.append("")
    L.append(f"## 振り返るべき手（{len(key_moves)}手・手数順）")
    L.append("")
    if not key_moves:
        L.append("大きな悪手は見つかりませんでした。")
    for r in key_moves:
        L.append(f"### {r.ply}手目 {r.kanji} {r.mark}（{r.label} / 勝率 -{r.loss_wr * 100:.0f}%）")
        L.append("")
        L.append(f"- 指した側：{r.side}　評価値：{_fmt_cp(r.cp_before)} → {_fmt_cp(r.cp_after)}（損失 {r.loss_cp}cp）")
        L.append(f"- **最善手：{r.best_kanji}**　読み筋：{r.best_pv}")
        if r.played_pv:
            L.append(f"- 実戦の手の読み筋：{r.played_pv}")
        L.append(f"- 実戦の手の後、相手の最善進行：{r.refutation_pv or '（なし）'}")
        if r.comment:
            L.append(f"- 棋譜コメント：{r.comment}")
        L.append("- 自動所見：")
        for t in r.tags:
            L.append(f"  - {t}")
        L.append("")
        L.append("指す前の局面（`*`=直前の相手の着手、`v`=後手の駒）：")
        L.append("")
        L.append("```")
        L.append(r.board_before)
        L.append("```")
        L.append("")
        L.append(f"SFEN: `{r.sfen_before}`")
        L.append("")
    L.append("## 全手の評価値一覧")
    L.append("")
    L.append("| 手数 | 指し手 | 評価値(指した側) | 最善手 | 判定 |")
    L.append("|---|---|---|---|---|")
    for r in reviews:
        best = r.best_kanji if r.best_usi and r.best_usi != r.usi else "◎"
        L.append(f"| {r.ply} | {r.kanji} | {_fmt_cp(r.cp_after)} | {best} | {r.mark} |")
    L.append("")
    return "\n".join(L)


def build_json(game: KifGame, reviews: list[MoveReview], key_moves: list[MoveReview], meta: dict) -> str:
    return json.dumps(
        {
            "meta": meta,
            "headers": game.headers,
            "end": game.end_word,
            "summary": summary(reviews),
            "key_moves": [asdict(r) for r in key_moves],
            "moves": [{k: v for k, v in asdict(r).items() if k not in ("board_before", "sfen_before")} for r in reviews],
            "generated": datetime.now().isoformat(timespec="seconds"),
        },
        ensure_ascii=False,
        indent=1,
    )
