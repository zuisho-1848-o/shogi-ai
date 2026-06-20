# DESIGN.md — 盤面解説AI 詳細設計

---

## データフロー（詳細）

```
1. クライアント
   POST /explain { sfen, candidates, history_sfens, level, backend }

2. api.py
   ├── リクエストバリデーション
   └── ExplainService.explain() を呼び出す

3. context_builder.py
   ├── SFEN → Board インスタンス化（core/board.py）
   ├── 盤面テキスト生成（9×9 グリッド表示）
   ├── 持ち駒テキスト生成
   ├── 手番・手数・局面フェーズ判定（序盤/中盤/終盤）
   ├── 候補手テキスト化（上位5件、評価値差分付き）
   └── 王手・詰み判定（core/rules.py）

4. knowledge.py
   ├── 戦型判定（joseki_map.json と駒配置パターンマッチ）
   ├── 有名局面照合（famous_positions.json と SFEN 完全一致）
   └── 格言選択（proverbs.json から局面特徴に応じて最大3件）

5. commentator.py
   ├── バックエンド選択（OllamaBackend / ClaudeBackend）
   ├── プロンプト構築（level に応じたシステムプロンプト）
   ├── LLM呼び出し
   └── レスポンス整形

6. クライアントへ返却
```

---

## context_builder.py — 設計

### 盤面テキスト表示形式

```
後手持ち駒: 飛 角 金
  ９ ８ ７ ６ ５ ４ ３ ２ １
一|香 桂 銀 金 玉 金 銀 桂 香|
二|・ 飛 ・ ・ ・ ・ ・ 角 ・|
三|歩 歩 歩 歩 歩 歩 歩 歩 歩|
四|・ ・ ・ ・ ・ ・ ・ ・ ・|
五|・ ・ ・ ・ ・ ・ ・ ・ ・|
六|・ ・ ・ ・ ・ ・ ・ ・ ・|
七|歩 歩 歩 歩 歩 歩 歩 歩 歩|
八|・ 角 ・ ・ ・ ・ ・ 飛 ・|
九|香 桂 銀 金 玉 金 銀 桂 香|
先手持ち駒: なし
（先手番 / 1手目 / 序盤）
```

### 候補手テキスト化

```
【候補手】
1位: ７六歩（+120cp）先手が角道を開ける。以下 ▲７六歩 △３四歩 ▲２六歩 ...
2位: ２六歩（+110cp）
3位: ３六歩（+80cp）
```

### フェーズ判定

| 手数 | フェーズ |
|------|---------|
| 1〜30 | 序盤 |
| 31〜80 | 中盤 |
| 81〜 | 終盤 |
| 詰み圏内（|評価値| > 2000cp） | 終盤・決戦 |

---

## knowledge.py — 設計

### 戦型判定ロジック

完全な定跡DBは重いため、**駒の配置特徴**でルールベース判定する。

```python
# 判定項目（例）
- 後手の角が7二にいる → 角換わりの可能性
- 先手の飛車が左に移動している → 振り飛車
- 先手の王が左辺（7〜9筋）にいる → 穴熊・美濃囲いの可能性
- 先手の王が中央付近 → 中住まい・早囲い
```

判定結果: `"矢倉"`, `"四間飛車"`, `"三間飛車"`, `"穴熊"`, `"雁木"`, `None`

### joseki_map.json の構造

```json
{
  "yagura": {
    "name": "矢倉",
    "description": "居飛車対居飛車の代表的な戦型。玉を金銀で固める。",
    "signature": {
      "black_king_file_range": [4, 5],
      "white_king_file_range": [4, 5]
    }
  },
  "shiken_bisha": {
    "name": "四間飛車",
    "description": "振り飛車の一種。飛車を4筋に移動させる。",
    "signature": {
      "black_rook_file": 6
    }
  }
}
```

### proverbs.json の構造

```json
[
  {
    "id": "ohwa_nige",
    "text": "玉の早逃げ八手の得",
    "meaning": "王将を早めに安全な場所へ逃がすと、後々大きなメリットがある",
    "trigger_conditions": ["endgame", "king_exposed"],
    "tags": ["王", "終盤", "基本"]
  },
  {
    "id": "hisha_mae_fu",
    "text": "飛車先の歩は交換せよ",
    "meaning": "飛車先の歩を早めに交換して飛車を活用する",
    "trigger_conditions": ["opening", "rook_pawn_not_exchanged"],
    "tags": ["飛車", "序盤"]
  }
]
```

### famous_positions.json の構造

```json
[
  {
    "name": "羽生の歩（1996年 羽生 vs 森内）",
    "event": "第54期名人戦",
    "sfen": "...",
    "description": "羽生善治が放った△3一銀が有名な一手。",
    "move": "3一銀"
  }
]
```

---

## commentator.py — 設計

### クラス構成

```python
class LLMBackend(ABC):
    @abstractmethod
    async def chat(self, system: str, user: str) -> str: ...

class OllamaBackend(LLMBackend):
    def __init__(self, host: str, model: str): ...

class ClaudeBackend(LLMBackend):
    def __init__(self, api_key: str, model: str): ...

class Commentator:
    def __init__(self, backend: LLMBackend): ...
    async def explain(self, context: BoardContext, knowledge: KnowledgeResult, level: str) -> str: ...
```

### プロンプトテンプレート（level 別）

#### beginner
```
あなたは将棋の家庭教師AIです。
駒の名前や基本的なルールも丁寧に説明しながら、初心者にも分かるように
現在の盤面を解説してください。難しい用語は使わず、平易な言葉で話しかけてください。
```

#### intermediate
```
あなたは将棋の解説AIです。
手筋・定跡名・形勢判断を使いながら、初段前後のプレイヤーに向けて
局面の特徴・候補手の意図・想定される展開を解説してください。
```

#### advanced
```
あなたはプロ棋士レベルの将棋解説AIです。
変化手順・深い読み・戦略的意図を詳細に解説してください。
評価値の根拠・定跡の分岐・プロの実戦例も積極的に引用してください。
```

### ユーザープロンプト構成

```
【現在の局面】
{board_text}

【手番・局面情報】
{turn}番 / {move_count}手目 / {phase}

【戦型・定跡】
{joseki or "判定中"}

【候補手（上位5件）】
{candidates_text}

【関連する格言】
{proverbs_text}

{famous_position_text}

以下の観点で解説してください：
1. 現在の局面の特徴（有利不利・駒の働き・陣形の評価）
2. 最善手の狙いと理由
3. この局面から想定される展開（2〜3手先）
4. 注意すべき相手の反撃
```

---

## api.py — エンドポイント設計

```python
POST /explain

Request:
{
  "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
  "candidates": [
    {"move": "7g7f", "score": 120, "pv": "7g7f 3c3d 2g2f"},
    {"move": "2g2f", "score": 110, "pv": "..."}
  ],
  "history_sfens": [],
  "level": "intermediate",
  "backend": "ollama"
}

Response:
{
  "commentary": "# 局面解説\n\nこれは初期局面です...",
  "joseki": "矢倉への移行が多い局面",
  "matched_position": null,
  "relevant_proverbs": ["飛車先の歩は交換せよ"],
  "backend_used": "ollama",
  "model_used": "llama3.2"
}
```

---

## フロントエンド連携

### 追加UI要素（web/index.html）

```
[盤面エリア] [解説パネル（新規）]
              ┌─────────────────────┐
              │ LLM: [ollama ▼]     │
              │ レベル: [中級 ▼]    │
              │ [解説を生成]        │
              ├─────────────────────┤
              │ 局面解説            │
              │ （Markdownレンダリング）│
              │                     │
              │ 定跡: 矢倉           │
              │ 格言: 飛車先の歩は... │
              └─────────────────────┘
```

### JS側のフロー

```javascript
async function fetchExplanation() {
  const res = await fetch('/explain', {
    method: 'POST',
    body: JSON.stringify({
      sfen: currentSfen,
      candidates: currentCandidates,
      level: selectedLevel,
      backend: selectedBackend
    })
  });
  const data = await res.json();
  renderMarkdown(data.commentary);
}
```

---

## 環境変数

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `LLM_BACKEND` | `ollama` | デフォルトバックエンド |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama サーバーURL |
| `OLLAMA_MODEL` | `llama3.2` | 使用モデル |
| `ANTHROPIC_API_KEY` | なし | Claude API利用時のキー |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude使用モデル |

---

## 懸念事項・制約

| 項目 | 内容 | 対策 |
|------|------|------|
| Ollama応答速度 | モデルにより5〜30秒かかる | ローディング表示 + Streaming対応（Phase 3） |
| 将棋用語の精度 | 汎用LLMは将棋専用語彙が弱い | システムプロンプトに用語集を埋め込む |
| 評価値の解釈 | LLMが評価値の大小を誤解する可能性 | 「+500cpは先手有利」のような変換を事前にテキスト化して渡す |
| 有名局面DB | 棋譜著作権の問題 | 公開済み・著作権フリーの局面のみ収録（1997年以前の棋譜等） |
| 詰み局面の解説 | LLMは詰み手順を正確に読めない | `solve/` モジュールと連携して詰み情報を事前計算してLLMに渡す |
