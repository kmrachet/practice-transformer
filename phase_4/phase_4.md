```python
Phase 4: アーキテクチャの構築 (Encoder, Decoder, and Transformer)
```

これまでPhase 1〜3で、Transformerを構成する主要な部品（Positional Encoding, Multi-Head Attention, FFN）を実装してきました。今回の**Phase 4**では、これらを統合して**Encoder**と**Decoder**を定義し、最終的な**Transformerモデル全体**のアーキテクチャを完成させます。

Phase 1〜3で作成した部品は、いわば「エンジン」や「タイヤ」です。このフェーズではそれらを組み上げて「車体（Encoder/Decoder）」を作り、最終的に走れる「車（Transformer）」を完成させます。

---- 

### 1. 理論背景
Transformerの全体構造は **Encoder-Decoder** モデルです。

1. **Encoder（符号化器）:**
	- 入力文（Source）を受け取り、その意味内容を圧縮した「文脈ベクトル（Memory）」に変換します。
	- $N$ 個の `EncoderLayer` を積み重ねて構成されます。
2. **Decoder（復号化器）:**
	- Encoderが作った「文脈ベクトル」と、これまでに出力した単語（Target）を受け取り、次の単語を予測します。
	- $N$個の `DecoderLayer` を積み重ねて構成されます。
	- **重要:** Decoderには2種類のAttentionがあります。
		- **Masked Self-Attention:** 自身の過去の出力だけを見る（未来の単語をカンニングしないためのマスクが必要）。
		- **Cross Attention (Source-Target Attention):** QueryをDecoderから、Key/ValueをEncoder（Memory）から取ってくることで、翻訳元の情報を参照する。

---- 

### 2. PyTorchによる実装
これまでのフェーズで実装した以下のクラスが利用可能であると仮定して進めます。
- `PositionalEncoding` (Phase 1)
- `MultiHeadAttention` (Phase 2)
-  `PositionwiseFeedForward` (Phase 3)

#### 2.1 Encoder Layer
まずはEncoderの1層分である `EncoderLayer` を実装します。構造: `Self-Attention` → `Add & Norm` → `FFN` → `Add & Norm`

```python
import torch
import torch.nn as nn
import copy

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 1. Self-Attention layer
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # 2. Feed-Forward Network layer
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 3. Layer Normalization & Dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Padding Maskなど. Defaults to None.

        Returns:
            torch.Tensor: [batch_size, seq_len, d_model]
        """

        # 1. Sublayer 1: Self-Attention
        # Residual Connection: x + Sublayer(x)
        # Post-LN: Norm(x + Sublayer(x))
        # Attentionの入出力は同じshape
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Sublayer 2: Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x
```

#### 2.2 Decoder Layer
次にDecoderの1層分です。Encoderとの違いは、真ん中に `Cross Attention` が挟まる点です。

```python
class DecorderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 1. Masked Self-Attention layer
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # 2. Cross-Attention layer (Source-Target Attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # 3. Feed-Forward Network layer
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 4. Normalization & Dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.tensor): Decorderへの入力テンソル、shapeは[batch_size, tgt_len, d_model]
            memory (torch.Tensor): Encoderの出力テンソル、shapeは[batch_size, src_len, d_model]
            src_mask (torch.Tensor): Memoryに対するマスク(Padding Mask)
            tgt_mask (torch.Tensor): Self-Attention用のマスク(Look-Ahead Mask + Padding Mask)

        Returns:
            torch.Tensor: Decorderの出力テンソル、shapeは[batch_size, tgt_len, d_model]
        """

        # 1. Sublayer 1: Masked Self-Attention
        # 未来の単語を見ないように tgt_mask を適用
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Sublayer 2: Cross-Attention
        # Query = x(Decorderの出力), Key = Value = memory(Endocderの出力)
        # Encoder側のパディングを見ないように src_mask を適用
        attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 3. Sublayer 3: Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x
```

#### 2.3 Encoder & Decoder (Stacks)
レイヤーを  個積み重ねるためのコンテナを作ります。PyTorchの `nn.ModuleList` を使うと便利です。

ここでは、`Transformer` クラスの中にまとめるのではなく、役割を明確にするために `Encoder` クラスと `Decoder` クラスを定義し、そこで Embedding と Positional Encoding も管理させます。

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_len: int, dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        # EncoderLayerをnum_layers個積み重ねる
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1. Embedding & Positional Encoding
        # 論文通り sqrt(d_model)でスケーリング
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 2. Apply all layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_len: int, dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self. pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        # DecoderLayerをnum_layers個積み重ねる
        self.layers = nn.ModuleList([
            DecorderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # 1. Embedding & Positional Encoding
        # 論文通り sqrt(d_model)でスケーリング
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 2. Apply all layers
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        return x
```

#### 2.4 Transformer (Main Model)
最後にこれらを統合するメインクラスです。出力層として、`d_model` から `target_vocab_size` に変換する線形層（Projection）を追加します。

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, num_layers: int = 6, num_heads: int = 8, d_ff: int = 2048, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)

        # 最終出力層の線形変換(Linear Projection)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            src (torch.Tensor): [batch, src_len] Encoderへの入力単語ID列
            tgt (torch.Tensor): [batch, tgt_len] Decoderへの入力単語ID列
            src_mask (torch.Tensor, optional): Encoder用のマスク. Defaults to None.
            tgt_mask (torch.Tensor, optional): Decoder用のマスク. Defaults to None.

        Returns:
            torch.Tensor: [batch, tgt_len, tgt_vocab_size] 出力単語の確率分布
        """

        # 1. Encode
        # memory: [batch, src_len, d_model]
        memory = self.encoder(src, src_mask)

        # 2. Decode
        # decoder_output: [batch, tgt_len, d_model]
        decoder_output = self.decoder(tgt, memory, src_mask, tgt_mask)

        # 3. Final linear layer
        # logits: [batch, tgt_len, tgt_vocab_size]
        logits = self.fc_out(decoder_output)

        return logits
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """推論時にEncoderのみを動かすためのヘルパー"""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """推論時にDecoderのみを動かすためのヘルパー"""
        return self.decoder(tgt, memory, src_mask, tgt_mask)
```

---- 

### 3. 実装のポイント解説
1. **Post-LN vs Pre-LN:**
	今回は原論文("Attention Is All You Need")に従い、`LayerNorm(x + Sublayer(x))` という順序（Post-LN）で実装しました。

	近年のLLM（GPT-2/3, LLaMAなど）では、`x + Sublayer(LayerNorm(x))` という順序（Pre-LN）が採用されることが多いです。Pre-LNの方が深層モデルでの学習が安定しやすいと言われていますが、Post-LNの方が正しく学習できた場合の性能は高い場合があるとも議論されています。
2. **Cross Attentionの引数:**
	`DecoderLayer` 内の2つ目のAttentionでは、`query` はDecoderからの入力 `x` ですが、`key` と `value` はEncoderの出力 `memory` を使用します。これが「翻訳元の情報を検索する」という動作になります。
3. **重み共有 (Weight Tying):**
	原論文では、Source Embedding, Target Embedding, Final Linear Layer の3つの重み行列を共有（Weight Tying）しています。今回の実装ではわかりやすさを優先して独立させていますが、パラメータ数を削減したい場合は `self.decoder.embedding.weight = self.encoder.embedding.weight` のように共有させることも可能です。

---- 

### 4. 動作確認
組み上げたTransformerが正しく動くか、ダミーデータを入れて確認します。

```python
def verify_transformer():
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model=d_model)
    
    # ダミー入力 (Batch=2, SeqLen=10)
    src = torch.randint(0, src_vocab_size, (2, 10))
    tgt = torch.randint(0, tgt_vocab_size, (2, 10))
    
    # マスクは一旦Noneで動作確認
    output = model(src, tgt)
    
    print("Source shape:", src.shape)
    print("Target shape:", tgt.shape)
    print("Output shape:", output.shape)
    
    # 出力は [Batch, Tgt_SeqLen, Tgt_VocabSize] になるはず
    expected_shape = (2, 10, tgt_vocab_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"
    print("Transformer architecture verification passed!")

verify_transformer()

```

**期待される結果:**
`Output shape: torch.Size([2, 10, 1000])` と表示されれば成功です。

---- 

次回 **Phase 5** では、今回 `None` で済ませてしまった重要な要素、**「マスキング（Padding Mask, Look-ahead Mask）」** の作成と、学習を安定させるための **「学習率スケジューリング」** について実装します。