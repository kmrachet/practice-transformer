```python
Phase 2: 注意機構の核心 (The Heart: Scaled Dot-Product & Multi-Head Attention)
```

### 1. 理論背景：なぜ "Attention" なのか？
RNNやLSTMが「過去の隠れ状態」を順次受け渡すことで文脈を保持していたのに対し、Transformerは**「全ての単語同士の関連度（Attention Score）」を一挙に計算**します。

#### 1.1 Query, Key, Value の概念
TransformerのAttentionは「検索システム」のアナロジーで説明されます。
ある単語（**Query**）の情報を取り込むために、他の全ての単語が持つ見出し（**Key**）と照らし合わせ、その一致度に基づいて中身（**Value**）を合成する、というプロセスです。
- **Query ($Q$):** 「検索クエリ」。情報を探している側のベクトル。
- **Key ($K$):** 「検索対象のインデックス」。検索される側のベクトル。
- **Value ($V$):** 「コンテンツ」。実際に抽出される情報のベクトル。

#### 1.2 Scaled Dot-Product Attention
Attentionの計算式は以下の通りです（Vaswani et al., 2017）。
$$\text{Attention}(Q,V,K) = 
\text{softmax} \left (
\frac{QK^T}{\sqrt{d_k}}
\right )V$$

ここには2つの重要な数理的操作が含まれています。
1. 内積による類似度 ($QK^T$): ベクトル同士の内積は「類似度」を表します。$Q$と$K$の向きが近いほど値が大きくなり、Attention（重み）が大きくなります。
2. スケーリング ($\frac{1}{\sqrt{d_k}}$): ベクトルの次元数が大きくなると、内積の和も増大します。するとSoftmax関数の勾配が極端に小さくなり（勾配消失）、学習が進まなくなります。これを防ぐために$\sqrt{d_k}$で割って分散を正規化します。

#### 1.3 Multi-Head Attention
「誰が」「いつ」「どこで」など、文には複数の着眼点があります。1つのAttentionですべてを捉えるのは困難です。
そこで、ベクトルを複数の「ヘッド」に分割し、それぞれ異なる部分空間（Subspace）でAttentionを計算させます。これにより、「あるヘッドは主語と動詞の関係」「別のヘッドは形容詞と名詞の関係」といった異なる文脈を並列に学習できるようになります。

---- 

### 2. PyTorchによる実装
それでは実装に移ります。`MultiHeadAttention` クラスを作成します。
ここでは `nn.MultiheadAttention` を使わず、行列演算 `torch.matmul` を使って仕組みをゼロから構築します。

以下のコードを作成してください。

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): モデルの隠れ層の次元数
            num_heads (int): ヘッドの数
            dropout (float, optional): ドロップアウト率. Defaults to 0.1.
        """

        super().__init__()

        # d_modelがnum_headsで割り切れることを確認
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 各ヘッドの次元数

        # Q, K, Vの線形変換
        # 実際には全ヘッド分を一度に計算するため、出力次元はd_modelのまま
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            query (torch.Tensor): [batch_size, seq_len, d_model]
            key (torch.Tensor):   [batch_size, seq_len, d_model]
            value (torch.Tensor): [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): [batch_size, 1, 1, seq_len] または [batch_size, 1, seq_len, seq_len]
                                           (0: マスクなし、1: マスクありなどの定義によるが、ここでは加算マスクを想定)
                                           Defaults to None.

        Returns:
            torch.Tensor: [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)

        # 1. 線形変換
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. ヘッドの分割
        # [batch_size, seq_len, num_heads] -> [batch_size, seq_len, num_heads, d_k]
        # その後、計算しやすいようにヘッドの次元を先頭に移動させる(転置) -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # 3.1. スコアの計算 Q * K^T / sqrt(d_k)
        # Q: [..., seq_len_q, d_k], K^T: [..., d_k, seq_len_k] -> scores: [..., seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3.2. マスクの適用(optional)
        if mask is not None:
            # ここではマスクが0の場所を非常に小さい値(-1e9)でマスクすると仮定
            # 実装により1と0の定義が異なる場合があるため注意する
            # 非常に小さい値(-1e9)で埋めることで、softmax後にほぼ0になるようにする
            # scores = scores.masked_fill(mask == 0, -1e9)
            scores = scores + mask # 加算マスクの場合 (maskが0の場所に-1e9が入っている想定)
        
        # 3.3. softmax & dropout
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 3.4. Valueとの積
        # attention_weights: [..., seq_len_q, seq_len_k] * V: [..., seq_len_k, d_k] -> [...,seq_lenn_q, d_k]
        output = torch.matmul(attention_weights, V)

        # 4. ヘッドの結合
        # [batch_size, num_beads, seq_len, d_k] -> [batch_size, seq_len, num_heads, d_k]
        output = output.transpose(1, 2).contiguous()

        # [batch_size, seq_len, d_model]に戻す
        output = output.view(batch_size, -1, self.d_model)

        # 5. 線形変換
        output = self.fc_out(output)
        return output
```

#### 実装のポイント解説
1. テンソルの変形 (`view` と `transpose`):
	ヘッド分割の処理 `Q.view(...).transpose(1, 2)` が核心です。これにより、`num_heads` の次元がバッチ次元の隣に来ます。PyTorchの `matmul` は先頭の次元をバッチとして扱い、末尾2次元で行列積を行うため、これにより**全てのヘッドのAttention計算を一度の行列演算で並列処理**できます。
2. `contiguous()`:
	`transpose` を行うとメモリ上の配置が不連続になるため、`view` で形状を戻す前に `contiguous()` を呼んでメモリを再配置する必要があります。これを忘れるとランタイムエラーになります。
3. スケーリング:
	`math.sqrt(self.d_k)` で割る操作を忘れないでください。これがないと、`d_model` が大きい場合に学習初期で勾配がほぼゼロになり、収束しません。

---- 

### 3. 動作確認
実装したクラスが正しく動作し、入出力のシェイプ（形状）が整合しているか確認します。

```python
def verify_multi_head_attention():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)

    # ダミー入力の作成
    # self-attentionなのでkey, valueもqueryと同じにする
    x = torch.rand(batch_size, seq_len, d_model)
    output = mha(x, x, x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # 出力の形状が入力と同じであることを確認
    assert output.shape == x.shape, "Output shape must match input shape"
    print("MultiHeadAttention verification passed!")

verify_multi_head_attention()

```

**期待される結果:**

```text
Input shape:  torch.Size([2, 10, 512])
Output shape: torch.Size([2, 10, 512])
Verification Successful: Output shape matches input shape.

```

出力の形状が入力と同じ `[Batch, Seq_Len, D_Model]` に戻っていれば成功です。

---- 

### 4. 発展：マスク処理について（予告）
コード内に `mask` 引数がありましたが、これはPhase 4以降で重要になります。
- Padding Mask: 文の長さが足りない部分（`<pad>`トークン）にAttentionを向けないようにする。
- Look-ahead Mask (Causal Mask): Decoderにおいて、「未来の単語（カンニング）」を見ないようにする。