# engines/

ローカル配置のみ（git管理外）。

| パス | 内容 |
|---|---|
| `yaneuraou-std/yaneuraou` | やねうら王 V9.80 標準NNUE版（halfKP 256x2-32-32 / Apple Silicon）。ソースからビルド済み |
| `yaneuraou-std/eval/nn.bin` | **評価関数（要配置）**。水匠5など標準NNUE形式のもの |
| `yaneuraou/` | 旧：Material版・KPPT版（動作確認用。解析精度は低い） |

## 水匠5の入手（公式配布）

```bash
curl -L -o /tmp/Suisho5.7z https://github.com/yaneurao/YaneuraOu/releases/download/suisho5/Suisho5.7z
tar -xf /tmp/Suisho5.7z -C /tmp          # macOS標準tarで7zを展開可（無理なら brew install sevenzip）
cp /tmp/Suisho5/eval/nn.bin engines/yaneuraou-std/eval/nn.bin   # 展開先の構成に合わせて調整
```

FV_SCALE は水匠5で 24（`--fv-scale 24`）。

## エンジンを作り直す場合

```bash
git clone https://github.com/yaneurao/YaneuraOu && cd YaneuraOu/source
make -j8 TARGET_CPU=APPLEM1 YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE COMPILER=clang++
cp YaneuraOu-by-gcc ../../engines/yaneuraou-std/yaneuraou
```
