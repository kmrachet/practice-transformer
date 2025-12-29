# Transformerの理論的解説と実装

## 概要

本コースは、既存のライブラリ（Hugging Face Transformers等）や高レベルフレームワーク（PyTorch Lightning等）に頼らず、**PyTorchのプリミティブな機能（`torch.nn.Module`や`torch.matmul`など）のみ**を用いて、原論文 *"Attention Is All You Need"* のモデルをゼロから実装することを目的とします。

このプロセスを通じて、「なぜその次元操作が必要なのか」「なぜ学習が収束するのか」という数理的な裏付けを体感していただきます。

以下に、本コースのロードマップを提示します。


## Transformer 実装・学習ロードマップ

本コースは全6回のセクション（フェーズ）に分かれています。

### Phase 1: 入力表現と位置情報の埋め込み (Input Embedding & Positional Encoding)

Transformerは再帰構造を持たないため、単語の順序情報をどのようにベクトル空間に埋め込むかが最初の課題です。

* **トピック:** Word Embedding, 正弦波によるPositional Encoding
* **実装対象:** `PositionalEncoding` クラス
* **理論背景:** フーリエ級数的な位置情報の表現、次元間の関係性
* **主要参考文献:** Vaswani et al. (2017) Section 3.5

### Phase 2: 注意機構の核心 (The Heart: Scaled Dot-Product & Multi-Head Attention)

モデルの核となるAttentionメカニズムを実装します。ここが最も重要かつ複雑な部分です。

* **トピック:** Query, Key, Valueの概念、スケーリング、マスク処理、マルチヘッド化
* **実装対象:** `MultiHeadAttention` クラス
* **理論背景:** 行列積による類似度計算、部分空間（Subspace）への射影
* **主要参考文献:** Vaswani et al. (2017) Section 3.2

### Phase 3: 層内の構成要素 (Position-wise FFN & Layer Normalization)

Attention層の出力を処理し、学習を安定化させるための構造を作ります。

* **トピック:** Position-wise Feed-Forward Networks, Residual Connection (残差結合), Layer Normalization
* **実装対象:** `PositionwiseFeedForward`, `LayerNorm` (または `nn.LayerNorm` の挙動理解)
* **理論背景:** 共変量シフトの抑制、勾配消失の防止
* **主要参考文献:** Ba et al. (2016) *"Layer Normalization"*

### Phase 4: アーキテクチャの構築 (Encoder, Decoder, and Transformer)

Phase 1～3で作った部品を組み立て、EncoderとDecoder、そして全体モデルを構築します。

* **トピック:** Encoder Layerのスタック、DecoderにおけるMasked Attention、Source-Target Attention
* **実装対象:** `EncoderLayer`, `DecoderLayer`, `Transformer` (メインクラス)
* **理論背景:** 自己回帰モデル（Auto-regressive）としてのDecoderの性質

### Phase 5: 学習のメカニズム (Training Mechanics: Masking & Scheduler)

Transformerの学習を成功させるための特殊な処理（マスキングと学習率スケジューリング）を実装します。

* **トピック:** Padding Mask, Look-ahead Mask (未来の情報を隠す), Label Smoothing, Noam Scheduler (Warmup)
* **実装対象:** バッチ処理関数, カスタムOptimizerスケジューラ, 損失関数
* **理論背景:** 可変長シーケンスの並列処理における制約

### Phase 6: 学習ループと検証 (Training Loop & Validation)

PyTorch Lightningを使わず、純粋なPyTorchで学習ループを書き、トイタスク（小規模な翻訳や文字列操作）で動作検証を行います。

* **トピック:** 勾配降下、バックプロパゲーション、推論（Greedy Decoding / Beam Search）
* **実装対象:** `train()` 関数, `evaluate()` 関数
* **検証:** 数千ステップでLossが下がり、意味のある出力が出ることを確認

---

### 開発環境の推奨設定

* **Python:** 3.8 以上
* **PyTorch:** 2.0 以上 (最新の安定版推奨)
* **NumPy, Matplotlib:** データの可視化用