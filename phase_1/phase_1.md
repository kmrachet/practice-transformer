## Phase 1: 入力表現と位置情報の埋め込み (Input Embedding & Positional Encoding)

Transformerにおいて、このフェーズは「テキスト（離散的なID）」を「ニューラルネットワークが扱える連続的な数値（ベクトル）」に変換し、さらに「順序情報」を付与する極めて重要なステップです。

---- 

### 1. 理論背景
#### 1.1 Input Embedding (単語の埋め込み)
まず、入力された単語ID列を、 次元のベクトルに変換します。これは通常の `nn.Embedding` ですが、Transformerの論文では一つ重要な工夫があります。

- **スケーリング:** Embeddingの出力ベクトルに$\sqrt{d_{model}}$を掛けます。
- **理由:** 後述する Positional Encoding の値（-1〜1の範囲）を加算する際、Embeddingされたベクトルの値が小さすぎると、位置情報に埋もれてしまう（あるいは分散のバランスが悪くなる）ため、スケールを調整して学習を安定させます。

#### 1.2 Positional Encoding (位置エンコーディング)
RNNやLSTMと異なり、TransformerのAttention機構は入力を並列に処理するため、そのままでは「単語の順序」を認識できません（"I hit him" と "Him hit I" が区別できない）。
そこで、単語ベクトルに「位置を表すベクトル」を**加算**します。論文では以下の正弦波・余弦波関数を用いた固定のベクトルを採用しています。
位置 $pos$、次元 $i$ における Positional Encoding  は以下の通りです：
$$\begin{align}
PE(pos, 2i) &= \sin \left( \frac{pos}{10000^\frac{2i}{d_{model}}} \right)
\\
PE(pos, 2i+1) &= \cos \left( \frac{pos}{10000^\frac{2i}{d_{model}}} \right)
\end{align}$$

- **偶数次元 ($2i$)**: 正弦波 ($\sin$) を使用
- **奇数次元 ($2i+1$)**: 余弦波 ($\cos$) を使用
- **波長:** 次元が上がるにつれて波長が長くなります ($2\pi$から$10000 \cdot 2\pi$まで)。

これにより、モデルは相対的な位置関係（ある位置の単語が、別の位置の単語とどのくらい離れているか）を容易に学習できるとされています。

---- 

### 2. PyTorchによる実装

では、実装に移ります。ここでは、`InputEmbedding` クラスを作るのではなく、PyTorch標準の `nn.Embedding` を使いつつ、`PositionalEncoding` クラスを定義して組み合わせる形をとります。

以下のコードを作成してください。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Arguments:
            d_model: モデルの隠れ層の次元数
            dropout: ドロップアウト率
            max_len: 想定される入力シーケンス最大長
        """

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding行列[max_len, d_model]の初期化
        pe = torch.zeros(max_len, d_model)

        # 位置情報のベクトル
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(2i/d_model)の計算
        # 対数空間で計算してからexpで戻すことで数値安定性を確保
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数次元にsin、奇数次元にcosを適用
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # バッチ次元を追加してshapeを[1, max_len, d_model]に変形
        pe = pe.unsqueeze(0)

        # モデルのパラメータとして登録（学習されない）
        # state_dictに保存されるが、勾配計算optimizerの対象にはならない
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Enbeddingされた入力テンソル、shapeは[batch_size, seq_len, d_model]
        
        Returns:
            Positional Encodingが加算されたテンソル、shapeは[batch_size, seq_len, d_model]
        """
        # 入力テンソルの長さに合わせてPositional Encodingをスライスして加算
        x = x + self.pe[:, :x.size(1), :]

        # ドロップアウトを適用して出力
        return self.dropout(x)
```

#### 実装のポイント解説

1. **`div_term` の計算:** 定義式の分母$10000^{\frac{2_i}{d_{model}}}$は、指数法則により$e^{\ln(10000)\cdot-\frac{-2i}{d_{model}}}$と変形して計算しています。これは数値計算上の慣例です。
2. **`register_buffer`:** Positional Encodingは学習する重み（Parameter）ではなく、固定値です。しかし、GPUへの転送やモデルの保存時には一緒に扱いたいため、`self.pe = ...` ではなく `register_buffer` を使用します。
3. **Broadcasting:** `pe` の形状は `[1, max_len, d_model]` です。入力 `x` `[batch_size, seq_len, d_model]` と加算する際、バッチ次元は自動的にブロードキャスト（複製）されます。

---- 

### 3. 動作確認と可視化

実装が正しいか、視覚的に確認しましょう。Positional Encodingは独特な幾何学的パターンを持ちます。

```python
def visualize_pe(d_model: int = 128, max_len: int = 100):
    pe_module = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

    # ダミーの入力テンソルを作成
    dummy_input = torch.zeros(1, max_len, d_model)
    output = pe_module(dummy_input)

    # 可視化のために1つ目のバッチを取得
    pe_image = output[0].detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(pe_image, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.title('Positional Encoding Visualization')
    plt.show()

visualize_pe(128, 100)
```

**期待される結果:**

横軸が次元、縦軸が単語の位置（シーケンス）を表すヒートマップが表示されます。左側（低次元）ほど波の周波数が高く（縞模様が細かい）、右側（高次元）ほど周波数が低い（色がゆっくり変化する）様子が確認できれば成功です。

---- 

### 4. Positional Encodingの数理的解説
Transformerが登場する以前のCNNベースのモデル（例：FacebookのConvS2S）などでは、学習可能なEmbedding層を使って単純に位置情報を学習させる手法もありました。しかし、Vaswaniらの原論文で「固定の三角関数」が採用されたのには、**「整数のインデックス」や「単純な正規化（0〜1）」では解決できない、機械学習特有の3つの課題**があるからです。

#### 4.1. 整数インデックスの問題点：スケールの爆発と一般化の欠如
単純に `[0, 1, 2, ..., T]` という整数を埋め込みベクトルに加算する場合を考えてみましょう。
- **スケールの不均衡 (Scale Issue):**
	ニューラルネットワークの重みやアクティベーションは、通常 -1〜1 や 0〜1 付近の小さな値で分布するように設計されます（Batch Normalization等の存在理由もこれです）。
	もしシーケンス長が100や1000になると、位置情報の値が `100` や `1000` になり、単語のEmbedding（意味情報）の値が位置情報という巨大なノイズにかき消されてしまいます。
- **学習データの範囲外への対応 (Extrapolation):**
	学習時に最大長さが50の文しか見ていない場合、テスト時に長さ100の文が来ると、モデルは `51` 以降の値をどう処理していいか全くわかりません。

#### 4.2. 単純な正規化（0〜1）の問題点：可変長への不適応
では、「最大長で割って 0〜1 に正規化すればよいのでは？」 (`pos / max_len`) と思うかもしれません。
- **ステップ幅の不変性欠如 (Variable Step Size):**
	長さ10の文では隣り合う単語との差は `0.1` ですが、長さ100の文では `0.01` になります。
	モデルにとって「隣の単語」という意味関係を学習したいのに、入力される文の長さによって「隣」を表す数値の差が変わってしまうと、一貫した学習が極めて困難になります。

#### 4.3. 三角関数を採用した最大の理由：相対位置の線形表現
これが最も重要な理論的根拠です。
TransformerのAttention機構にとって重要なのは、「絶対的な位置（私は10番目にいる）」よりも、**「相対的な位置（AはBの3つ後ろにいる）」** です。正弦波・余弦波を用いると、**位置  のベクトルを、位置  のベクトルの線形変換（回転行列の積）として表現できる** という数学的特性があります。

#### 4.4. 数理的証明
位置$pos$におけるエンコーディング$PE_{pos}$は正弦波と余弦波のペアで構成されています。
ある周波数$\omega_i$に着目すると、その値は$\sin(\omega_i \cdot pos)$と$\cos(\omega_i \cdot pos)$です。ここで、位置が$k$だけずれた$pos+k$の値を加法定理で展開してみましょう。
$$\sin(\omega_i (pos+k)) = \sin(\omega_i pos)\cos(\omega_i k)+\cos(\omega_i pos)\sin(\omega_i k) \\
\cos(\omega_i (pos+k)) = \cos(\omega_i pos)\cos(\omega_i k)-\sin(\omega_i pos)\sin(\omega_i k)$$

これを行列形式で書くと以下のようになります。
$$\begin{pmatrix}
\sin(\omega_i (pos + k)) \\
\cos(\omega_i (pos + k))
\end{pmatrix}
=
\begin{pmatrix}
\cos(\omega_i k) & \sin(\omega_i k) \\
-\sin(\omega_i k) & \cos(\omega_i k)
\end{pmatrix}
\begin{pmatrix}
\sin(\omega_i pos) \\
\cos(\omega_i pos)
\end{pmatrix}$$

この右辺の行列（回転行列）は、$pos$に依存せず、**相対的な距離$k$のみに依存** します。つまり、モデル（重み行列$W_Q, W_V, W_K$）は、入力された$PE_{pos}$に対して線形変換を行うだけで、**「 個離れた位置の情報」に容易にアクセスできる（Attentionを向けられる）** 構造になっているのです。

#### 4.5. まとめ
1. **有界性:** 値が常に -1 〜 1 に収まり、Embeddingの意味情報を破壊しない（勾配も安定する）。
2. **一貫性:** 文の長さに関わらず、隣り合う位置との距離関係が一定である。
3. **相対位置の表現力:** 線形変換によって相対的な位置関係を容易に計算できる（これがAttention機構と相性が良い）。

---- 

### 5. より詳しい数理的解説
数理的展開は、Transformerが「なぜ相対位置を学習しやすいのか」を理解する上で最も美しい部分の一つです。

#### 5.1. 目標：何を示したいのか？
位置$pos$の情報を持つベクトル$PE_{pos}$があるとします。そこから$k$個ずれた位置のベクトル$PE_{pos+k}$を計算したいとき、元の$PE_{pos}$に対して **「$pos$に依存しない、ある一定の行列」を掛けるだけで求められる** ということを証明します。
これが示せれば、「モデルは$k$という距離関係を、単純な行列演算（線形変換）として捉えることができる」と言えます。

#### 5.2. ステップ1: 設定と定義
Positional Encodingの$d_{model}$次元のうち、ある特定の周波数を持つ **2つの次元のペア ($2i$と$2i+1$)** だけを取り出して考えます。周波数（角速度）を $w_i$ と置きます。
$$w_i = \frac{1}{10000^{\frac{2i}{d_{model}}}}$$

このとき、位置 $pos$ におけるベクトル  $\text p_{pos}$（2次元ベクトル）は以下のようになります。
$$\text p_{pos} = 
\begin{pmatrix}
PE_{(pos, 2i)} \\
PE_{(pos, 2i+1)}
\end{pmatrix}
=
\begin{pmatrix}
\sin(\omega_i (pos)) \\
\cos(\omega_i (pos))
\end{pmatrix}$$

同様に、位置 $pos+k$ におけるベクトル $\text p_{pos+k}$ は以下のようになります。
$$\text p_{pos+k} =
\begin{pmatrix}
\sin(\omega_i (pos + k)) \\
\cos(\omega_i (pos + k))
\end{pmatrix}$$

#### 5.3. ステップ2: 三角関数の加法定理による展開
ここで、 $\text p_{pos + k}$の各成分を展開します。高校数学で習う「加法定理」を使います。
**上段：$\sin$ の展開**
$$\begin{align}
\sin(\omega_i (pos + k)) &= \sin(\omega_i \cdot pos + \omega_i \cdot k) \\
&=\sin(\omega_i \cdot pos)\cos(\omega_i \cdot k)+\cos(\omega_i \cdot pos)\sin(\omega_i \cdot k)
\end{align}$$
**下段：$\cos$ の展開**
$$\begin{align}
\cos(\omega_i (pos + k)) &= \cos(\omega_i \cdot pos + \omega_i \cdot k) \\
&=\cos(\omega_i \cdot pos)\cos(\omega_i \cdot k)-\sin(\omega_i \cdot pos)\sin(\omega_i \cdot k)
\end{align}$$

#### 5.4. ステップ3: 行列形式への変換
展開した式をよく見ると、$\sin(\omega_i \cdot pos)$ と $\cos(\omega_i \cdot pos)$ という共通項（つまり元の $\text p_{pos}$ の成分）が含まれています。これをわかりやすく、$u=\sin(\omega_i \cdot pos)$、 $v=\cos(\omega_i \cdot pos)$と置いて書き直してみましょう。また、$A=\cos(\omega_i \cdot k)$、 $B=\sin(\omega_i \cdot k)$と定数扱いします（$k$は固定されているため）。
すると、先程の連立方程式はこう見えます。
$$\begin{align}
上段 &= u \cdot A + v \cdot B \\
下段 &= v \cdot A - u \cdot B = - u \cdot B + v \cdot A
\end{align}$$

これを線形代数の形式（$\text y = M \text x$）に書き換えます。
$$\begin{pmatrix}
上段 \\
下段
\end{pmatrix}
=
\begin{pmatrix}
u \cdot A + v \cdot B  \\
- u \cdot B + v \cdot A
\end{pmatrix}
=
\begin{pmatrix}
A & B  \\
-B & A
\end{pmatrix}
\begin{pmatrix}
u  \\
v
\end{pmatrix}$$

元の記号に戻すと、以下の行列式が完成します。
$$\begin{pmatrix}
\sin(\omega_i (pos + k)) \\
\cos(\omega_i (pos + k))
\end{pmatrix}
=
\underbrace {
\begin{pmatrix}
\cos(\omega_i k) & \sin(\omega_i k) \\
-\sin(\omega_i k) & \cos(\omega_i k)
\end{pmatrix}
}_{回転行列M_k}
\underbrace {
\begin{pmatrix}
\sin(\omega_i pos) \\
\cos(\omega_i pos)
\end{pmatrix}
}_{元の位置ベクトル\text p_{pos}}$$

#### 5.5. ステップ4: この数式の意味するところ
導出された行列 $M_k$ を見てください。
$$M_k = 
\begin{pmatrix}
\cos(\Phi) & \sin(\Phi) \\
-\sin(\Phi) & \cos(\Phi)
\end{pmatrix}
\qquad
(ただし\Phi = \omega_i k)$$

これは、2次元平面における回転行列（Rotation Matrix）そのものです。この結果から以下の重要な事実がわかります。
1. $pos$ が消えた: 行列 $M_k$ の中身には $pos$（現在の絶対位置）が含まれていません。含まれているのは $k$ （相対距離）と $\omega_i$ （周波数）だけです。
2. 線形変換: 位置を $k$ ずらすという操作は、ベクトル空間上で「ある角度だけ回転させる」という線形変換と等価になります。

#### 5.6. 結論
ニューラルネットワークの層（例えばLinear層）は、入力ベクトルに行列を掛けて変換する機構を持っています。Positional Encodingに三角関数を採用したことで、Transformerは「単語間の相対的な距離（ $k$ ）」を「ベクトルの回転」として捉えることができ、それを自身の重み行列（ $W_Q, W_K$ など）の学習によって容易に獲得・識別できるというわけです。
これが、整数インデックスにはない、三角関数特有の強力な数理的特性です。

---- 

### 6. なぜ偶数・奇数で交互に$\sin$と$\cos$を配置するのか
偶数・奇数で交互に配置した主な理由は、**「同じ周波数を持つ $\sin$ と $\cos$ を隣同士のペアにして、2次元の部分空間（回転面）を形成させるため」** です。先ほどの行列による証明を思い出していただくと、その意図がより明確になります。

#### 6.1. 2次元の部分空間（ペア）の形成
先ほどの証明で、相対位置の計算には **「同じ周波数 $\omega_i$ を持つ $\sin$ と $\cos$ のセット」** が必要でした。この2つの値は、数式上「切っても切り離せない関係」にあります。
これらをベクトルのインデックス $2i$ と $2i+1$ に隣接して配置することで、$d_{model}$ 次元のベクトル空間の中に、$\frac{d_{model}}{2}$ 個の独立した「2次元の回転平面（部分空間）」が存在する、という構造を作っています。

もし、前半をすべて $\sin$、後半をすべて $\cos$ にしてしまった場合（例：0〜255番目が $\sin$、256〜511番目が $\cos$ ）、同じ周波数のペアがベクトル内で遠く離れてしまいます。これでも全結合層（Linear層）を通せば理論的には計算可能ですが、**「局所的な構造」としてペアにしておく方が、情報の表現として自然であり、実装もしやすい**という利点があります。

#### 6.2. 複素数表現（オイラーの公式）との対応
数学的には、これは**複素数**として捉えると非常に綺麗です。オイラーの公式 $e^{i\theta} = \cos \theta + i \sin \theta$ を考えると、位置 $pos$ における $k$ 番目の周波数成分は、以下のような複素数表現とみなせます。
$$z_{pos, k} = e^{j \cdot \omega_k \cdot pos} = \cos (\omega_k pos) + j \sin (\omega_k pos)$$

(ここでは虚数単位を $j$ とします)
TransformerのPositional Encodingは、実数空間でこれを表現するために、実部と虚部を交互に並べていると解釈できます。つまり、ベクトル全体は **「異なる速度で回転する複数の複素数のリスト」** なのです。

#### 6.3. ニューラルネットワークにとってのメリット
ニューラルネットワークの重み行列  がこの入力  を受け取って計算する際、隣接する要素  を組み合わせて新しい特徴量を作ることは（畳み込み層的な視点で見ても）非常に容易です。もしこれが離れていると、ネットワークは「遠く離れた  と  を結びつける」ような重みを学習しなければなりませんが、隣接させておけば、局所的なパターンの組み合わせとして回転情報を抽出・操作しやすくなります。

#### 6.4. まとめ
偶数・奇数で分けているのは、**「 $\sin$ と $\cos$ をセットにして、一つの回転情報（複素数）として扱いたいから」** というのが最大の理由です。「前半に全部$\sin$、後半に全部$\cos$」でも数学的には同等の情報量ですが、「ペアである」という構造を明示的にベクトルの中に埋め込んでいる、と理解していただくと良いかと思います。

---- 

### 7. 偶数次元が$\sin$、奇数次元が$\cos$であることの理由はない
偶数が $\cos$、奇数が $\sin$ でも全く問題ありません。実のところ、$\sin$ と $\cos$ のどちらを先にするかは、数学的な必然性というよりは **「慣習（Convention）」や「定義の問題」** に過ぎません。なぜ逆でも良いのか、その理由を3つの視点で解説します。

#### 7.1. 数学的な視点：単なる「位相のズレ」である
三角関数の性質として、$\cos\theta$ は $\sin\theta$ を単に90度（$\pi / 2$）ずらしたものに過ぎません。つまり、偶数番目を $\cos$、奇数番目を $\sin$ に入れ替えるということは、**「スタート地点の角度を90度ずらした」** というだけの話です。

円周上の点を表現するのに、 座標を $(\cos\theta,\sin\theta)$ と書こうが $(\sin\theta,\cos\theta)$ と書こうが、**「円を描く」という構造自体は変わらない** のと同じです。

#### 7.2. 相対位置の計算への影響：回転の向きが変わるだけ
先ほどの回転行列の話を思い出してください。もし $\sin$ と $\cos$ を逆に定義した場合、導出される回転行列  は以下のようになります（符号の位置が変わります）。

- **元々の定義（偶数$\sin$/ 奇数$\cos$）:**
$$M_k = 
\begin{pmatrix}
\cos(\omega k) & \sin(\omega k) \\
-\sin(\omega k) & \cos(\omega k)
\end{pmatrix}$$

これは「時計回り（あるいは座標系によっては反時計回り）」の回転を表します。

- **逆の定義（偶数$\cos$ / 奇数$\sin$）:**
$$M_k = 
\begin{pmatrix}
\cos(\omega k) & -\sin(\omega k) \\
\sin(\omega k) & \cos(\omega k)
\end{pmatrix}$$


これは「逆回転」を表します。

回転する方向が逆になったとしても、「線形変換によって相対位置を特定できる」という**本質的な機能は完全に維持されます**。

#### 7.3. モデルの学習への影響：重みが吸収する
TransformerのAttention層には、入力に対して $W_Q, W_K, W_v$ という重み行列（Linear層）が掛かります。機械学習モデルは非常に柔軟なので、もし人間が Positional Encoding の $\sin$ と $\cos$ を逆にして入力したとしても、**学習の過程で重み行列 $W$ がその変化に合わせて数値を自動調整**してくれます。
つまり、モデルにとっては「右回りの情報」だろうが「左回りの情報」だろうが、一貫してさえいれば、どちらでも学習可能です。

#### 7.4. 結論
論文の著者が「偶数=$\sin$, 奇数=$\cos$」を選んだのは、おそらく複素平面での一般的な表記  や、あるいは単なる実装者の好みによる偶然の選択と考えられます。

**実装上の注意点:**
唯一重要なのは **「一貫性」** です。学習時には「偶数sin」でやったのに、推論（テスト）時には「偶数cos」にしてしまうと、モデルは混乱して性能が出ません。どちらかに決めたら、それを貫き通す必要があります。
