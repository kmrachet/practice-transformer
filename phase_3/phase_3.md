```python
Phase 3: 層内の構成要素 (Position-wise FFN & Layer Normalization)
```
Phase 2では、Transformerの核心である「Attention機構」を実装しました。しかし、Attentionだけでは非線形な変換能力が不足しており、また層を深く積み重ねると学習が不安定になるという問題があります。
Phase 3では、これらを解決するために必要な **Position-wise Feed-Forward Networks (FFN)** と、学習を安定させるための **Layer Normalization** および **残差結合 (Residual Connection)** について解説・実装します。

---- 

### 1. 理論背景
#### 1.1 Position-wise Feed-Forward Networks (FFN)
Multi-Head Attention層の後には、各位置（単語）ごとに独立して適用される全結合ニューラルネットワークが接続されます。これを **Position-wise Feed-Forward Networks** と呼びます。数式は以下の通りです（Vaswani et al., 2017）。
$$FFN(x) = \text{max} (0, xW_1+b_1)W_2+b_2$$

ここで重要な特徴が2つあります。
1. **2層の線形変換とReLU:** 入力を一度高い次元（$d_{ff}$）に拡大し、ReLU（活性化関数）を通してから元の次元（$d_{model}$）に戻します。一般的に隠れ層の次元 $d_{ff}$ は $d_{model}$ の4倍程度（例: $d_{model} = 512, d_{ff} = 2048$）に設定されます。
2. **Position-wise（位置ごとに独立）:** この変換は、シーケンス内の「すべての単語（位置）」に対して **「同一の重み ($W_1,W_2,b_1,b_2$)」** で適用されます。RNNのように前の時刻の隠れ状態を引き継ぐことはしません。カーネルサイズ1のCNN（Convolution）と見なすこともできます。

**役割:** Attentionが「単語間の関係性（Mix）」を計算するのに対し、FFNは「その単語自身の特徴抽出・加工」を担います。

### 1.2 Residual Connection (残差結合)
深層学習において層が深くなると「勾配消失」や「情報の劣化」が起きやすくなります。これを防ぐため、ResNet (He et al., 2016) で提案されたSkip Connection（入力 $x$ を出力にそのまま足す）を採用します。
$$\text{Output} = x + \text{Sublayer}(x)$$

これにより、勾配が近道（ショートカット）を通って下層まで伝わりやすくなり、学習が劇的に安定します。

### 1.3 Layer Normalization (層正規化)
バッチサイズ方向に正規化を行う Batch Normalization (BN) は、NLPのようにバッチサイズが小さかったり可変長であったりするタスクでは不安定になりがちです。そこで、**「1つのサンプル内の全特徴量」** を使って正規化を行う **Layer Normalization (LN)** (Ba et al., 2016) を使用します。
$$LN(x) = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\cdot\gamma+\beta$$

- $\mu, \sigma$ : そのベクトル内の平均と分散
- $\epsilon$ : ゼロ除算を防ぐ微小値
- $\gamma, \beta$ : 学習可能なパラメータ（スケーリングとシフト）

Transformerでは、各サブレイヤー（Attention, FFN）の出力に対して、**「残差結合 → Layer Norm」** の順で適用するのが一般的です（論文では "Add & Norm" と呼ばれます）。

---- 

### 2. PyTorchによる実装
それでは、`PositionwiseFeedForward` クラスを実装しましょう。`LayerNorm` については、PyTorch標準の `nn.LayerNorm` の挙動を理解して利用します。

```python
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): モデルの次元数
            d_ff (int): FFNの中間層の次元数
            dropout (float, optional): ドロップアウト率. Defaults to 0.1.
        """
        super().__init__()

        # 一層目 d_model -> d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # 二層目 d_ff -> d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        # 活性化関数 ReLU
        # 元論文ではReLUが使われているが、近年のLLMではGELUがよく使われている
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: [batch_size, seq_len, d_model]
        """
        # Linear -> ReLU -> Dropout -> Linear
        # x: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff]
        hidden = self.activation(self.w_1(x))
        hidden = self.dropout(hidden)

        # [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
        output = self.w_2(hidden)
        return output
```

#### 実装のポイント解説
1. **`d_ff` のサイズ:** 通常、`d_model` よりも大きく取ります。これにより、モデルは入力情報を一度高次元空間に写像し、より豊かな特徴表現を獲得してから元の次元に戻すことができます。
2. **`nn.Linear` の適用:** PyTorchの `nn.Linear` は、入力が `[Batch, Seq_Len, Dim]` のような3次元テンソルであっても、末尾の次元 `Dim` に対して線形変換を行う仕様になっています。したがって、ループを使わずとも自動的に **Position-wise** に計算が行われます。

---- 

### 3. Add & Norm の構成（参考実装）
Phase 4 で `EncoderLayer` を作る際に組み合わせますが、ここで「Attention」と「FFN」をどう接続するか、**Add & Norm** の構造を予習しておきましょう。各サブレイヤー（Attention や FFN）の出力は以下のように処理されます。

```python
# 概念コード（Phase 4で正式にクラス化します）

# 1. Multi-Head Attention
# 残差結合のために元の x を保持
residual = x 
x = attention_layer(x)
x = dropout(x)
# Add (残差結合) & Norm (正規化)
x = layer_norm_1(x + residual)

# 2. Feed-Forward Network
residual = x
x = ffn_layer(x)
x = dropout(x)
# Add & Norm
x = layer_norm_2(x + residual)

```

この「入力をそのまま足す」処理があるおかげで、層を6個、12個と重ねても、入力された単語のEmbedding情報が消えることなく深層まで伝播します。

---- 

### 4. 動作確認
実装した `PositionwiseFeedForward` の入出力シェイプを確認します。

```python
def verify_ffn():
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    ffn = PositionwiseFeedForward(d_model, d_ff)
    
    # ダミー入力
    x = torch.rand(batch_size, seq_len, d_model)
    output = ffn(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    
    assert output.shape == x.shape, "Output shape must match input shape"
    print("PositionwiseFeedForward verification passed!")

verify_ffn()

```

**期待される結果:**
入力と同じ `[2, 10, 512]` が出力されれば成功です。内部で次元が `2048` に膨らんで戻っていることは、外からは見えませんが、モデルの表現力向上に寄与しています。

---- 

### 5. 理論的な補足：なぜ Layer Normalization なのか？
画像処理（CNN）では Batch Normalization (BN) が主流ですが、Transformer ではなぜ Layer Normalization (LN) なのでしょうか？
1. **バッチサイズへの依存性:** BNはバッチ内の統計量（平均・分散）を使うため、バッチサイズが小さいと統計量が不安定になります。Transformerはメモリ消費が激しく、大きなモデルではバッチサイズを小さくせざるを得ないことが多いため、BNは不利です。
2. **可変長シーケンス:** NLPでは文の長さがバラバラで、パディング（穴埋め）が多く含まれます。BNだとパディング部分も統計計算に含まれてしまい、正確な正規化が難しくなります。LNは「その単語（位置）のベクトル」だけで正規化が完結するため、他の単語やパディングの影響を受けません。

これで、Transformerを構成する主要な「部品」がすべて揃いました。
次回 **Phase 4** では、これら（Embedding, Positional Encoding, Multi-Head Attention, FFN, Layer Norm）をすべて組み上げて、**Encoder** と **Decoder**、そして **Transformer全体** のアーキテクチャを構築します。