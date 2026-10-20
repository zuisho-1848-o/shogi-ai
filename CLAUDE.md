# shogi-ai（棋譜レビューツール）

自分の対局KIFをやねうら王で解析して、振り返るべき手と理由をレポートする。
旧・独自将棋AI/詰将棋ソルバ/盤面解説AIは `../shogi-ai-archive/` に退避済み（git履歴にも残っている）。

- 実行: `.venv/bin/python -m kifu_review <kif> --me <名前>` → `reports/<名前>/report.md`
- 解説は `/review-kifu` スキル（`.claude/skills/review-kifu/`）が report を読んで行う
- 評価関数 `engines/yaneuraou-std/eval/nn.bin` はgit管理外。入手は `engines/README.md`
- テスト: `.venv/bin/python -m pytest`

## 非NN将棋AIの研究（`research/`・`lab/`）
- 進め方の決まり・裏で計算を回すときの注意・環境まわりは `research/運用メモ.md`（必ず読む。新しく分かったことはここにも追記）
- 実験の記録は `research/実験計画_フェーズ0.md`、先行研究は `research/先行研究調査.md`
