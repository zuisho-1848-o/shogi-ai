# どうぶつしょうぎ「後手必勝」の Lean4 による機械検証

定理 `Dobutsu.second_player_wins : Dobutsu.Loses Dobutsu.initKey`
（初期局面で手番の先手は、後手が最善を尽くせば必ず負ける）

## ファイル
| ファイル | 内容 |
|---|---|
| `Dobutsu/Rules.lean` | ルール（局面の表し方・合法手・即勝ち）。`lab/dobutsu/solve.c` と同じ規則 |
| `Dobutsu/Game.lean` | 勝ち負けの定義（`Wins` / `Loses`。有限の手数で決着がつく必勝・必敗） |
| `Dobutsu/Cert.lean` | 証明書の検査器 `check` と健全性の定理 `sound`（Lean のカーネルで検査。sorry なし） |
| `Dobutsu/Main.lean` | 証明書の読み込み、`check cert = true`（native_decide）、最終定理 |
| `Dobutsu/Data.lean` | 証明書（base64）。自動生成・約74MB・git 管理外 |

## 再現手順
```bash
# 1. 真値DB（lab/data/dobutsu/keys.bin, vals.bin）を作る: lab/dobutsu/solve（約15分）
# 2. 証明書を作る（276万局面、55.6MB、省電力コアで約15分）
cd lab/dobutsu && ../.venv/bin/python cert_build.py
# 3. Lean のソースに埋め込む
../.venv/bin/python cert_to_lean.py ../data/dobutsu/cert.bin
# 4. 検査（M5 の省電力コアで約27分、最大メモリ 約800MB）
cd ../lean && PATH=$HOME/.elan/bin:$PATH lake build Dobutsu.Main
```

## 信頼しているもの
- `sound`（検査器が通れば勝ち負けが正しい）と最終定理の組み立ては、Lean のカーネルが検査
- 276万局面の照合 `check cert = true` は `native_decide`（Lean のコンパイラ／インタプリタの計算を信頼する公理 `ofReduceBool` 系）
- `#print axioms Dobutsu.second_player_wins` → `propext, Classical.choice, Quot.sound` と native_decide の2件のみ

## ルールの前提（形式化した内容）
- 盤 3×4、駒はライオン・キリン・ぞう・ひよこ（成るとにわとり）、取った駒は持ち駒として空いているマスに打てる（ひよこを最奥に打ってもよい）
- 勝ち: 相手のライオンを取る / 自分のライオンを最奥の段に進めて相手の利きがない（トライ）/ 自分のライオンが最奥にいて相手が取らなかった
- 千日手の規則には依存しない（必勝は有限の手数で勝ち切る形で定義しているので、負ける側が千日手で逃げる余地がない）
