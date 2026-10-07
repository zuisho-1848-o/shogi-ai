# shogi-ai — 棋譜レビューツール

棋桜・ShogiHome 等で保存した **KIF** を、やねうら王で全手解析し、
「振り返るべき手（疑問手/悪手/大悪手）」＋「なぜ悪いか・最善手の狙い」をレポートにする。

## セットアップ（初回のみ）

1. `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
2. 評価関数を `engines/yaneuraou-std/eval/nn.bin` に配置 → [engines/README.md](engines/README.md)（水匠5の公式配布URLあり）
3. （任意）ShogiHome は棋譜閲覧用。エンジン登録するなら `engines/yaneuraou-std/yaneuraou` を指定

## 使い方

```bash
.venv/bin/python -m kifu_review kifu/対局.kif --me 自分の名前 --fv-scale 24
# → reports/対局/report.md / report.json
```

主なオプション: `--movetime 1500`(ms/局面) `--depth N` `--top 8`(振り返る手数) `--side black|white`
`--me` は KIF の先手/後手名に部分一致（環境変数 `KIFU_REVIEW_ME` でも可）。結果は `analysis_cache.json` に保存され、再実行は高速。

## 理由まで解説してもらう

Claude Code で `/review-kifu kifu/対局.kif` — レポートを読んで、各悪手の
「何が起きたか／なぜ悪いか／最善手の狙い／次回の考え方」を日本語で解説する。

## 仕組み

KIF → USI手に変換（python-shogiで合法性検証）→ 全局面をMultiPV解析 → 1手ごとに
`指す前の最善評価 − 指した後の評価` を勝率換算 → 閾値でラベル付け。理由の材料として
駒損・詰みの見逃し/許容・取られる駒・王手・候補順位・消費時間・形勢区分を自動抽出し、読み筋を日本語表記で添える。

旧プロジェクト（独自AI・詰将棋ソルバ等）は `../shogi-ai-archive/` に退避。
