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

---- 

### 5. ヘッドを複数用いても違う結果が得られる理由
「ヘッド内の計算が自動で行われる」という感覚は、Deep Learningの「魔法」のように見える部分ですが、その裏側には明確な数理的・アルゴリズム的な理由があります。

結論から言うと、**「人間が『ヘッド1は文法を見ろ』『ヘッド2は代名詞を見ろ』と指示するのではなく、初期値のランダム性と学習プロセスによって、結果的に異なる役割（部分空間）が自動的に形成される」** というのが正解です。

なぜそうなるのか、数式と直感の両面から解説します。

#### 1. なぜ「異なる」計算結果になるのか？ (数理的理由)
コードの中で定義した線形層を見てみましょう。

```python
self.w_q = nn.Linear(d_model, d_model)
self.w_k = nn.Linear(d_model, d_model)
self.w_v = nn.Linear(d_model, d_model)
```

そして、これを `view` と `transpose` で `num_heads` 個に分割しました。これは数学的には、各ヘッド $i$ に対して個別の重み行列 $W_i^Q, W_i^K, W_i^K$ を用意していることと同義です。
ここで重要なのが以下の2点です。

##### ① 初期化のランダム性 (Random Initialization)
学習開始前、これらの重み行列 $W$ の中身は乱数（正規分布など）で初期化されます。
確率的に、$W_1^Q$ と $W_2^Q$ が全く同じ値になることはありえません。
- ヘッド1 ($W_1^Q$): 入力ベクトルを「ある方向」に回転・伸縮させる。
- ヘッド2 ($W_2^Q$): 入力ベクトルを「別の方向」に回転・伸縮させる。

スタート地点が異なるため、同じ入力  を入れても、射影された空間（Subspace）での座標は最初から異なります。

##### ② 勾配降下法による「棲み分け」 (Specialization via Optimization)
学習が進むにつれて、モデルは損失（Loss）を減らそうと重みを更新します。
もし、ヘッド1とヘッド2が全く同じ情報（例：隣の単語を見るだけ）に注目してしまったらどうなるでしょうか？
それは「無駄」です。モデルにとっては、限られたリソース（ヘッド数）を使って、できるだけ多様な情報を拾い集める方がLossを下げやすくなります。

結果として、オプティマイザ（Adam等）は、それぞれのヘッドが**互いに異なる有用な特徴**を捉えるように、重みを異なる方向へ誘導していきます。これが「自動的に役割分担される」メカニズムです。

#### 2. 「部分空間 (Subspace)」とはどういうことか？
「異なる部分空間」という言葉を、具体的な言語の例でイメージしてみましょう。
単語「**Bank**」というベクトルが入力されたとします。
- **ヘッド1（意味の空間）:** 
	- このヘッドの重み行列 $W_1$ は、文脈から「川岸(River bank)」か「銀行(Financial bank)」かを区別する成分を強調するように学習されます。
	-  この空間では、「Bank」は「Water」や「Money」に近い位置に配置されます。
- **ヘッド2（文法の空間）:**
	- このヘッドの重み行列 $W_2$ は、意味を無視して「品詞」や「構文」を強調するように学習されます。
	-  この空間では、「Bank」は（名詞として使われているなら）他の名詞「Apple」や「Desk」に近い位置に配置されます。

このように、**「同じ単語ベクトルに対しても、どの行列  を掛けるかによって、取り出される特徴（投影される先の座標）が変わる」**。これが Multi-Head Attention における「異なる部分空間」の意味です。

#### 3. 可視化による実例
実際に学習済みのTransformerを解析すると、以下のような役割分担が自然発生していることが多くの研究で報告されています（例：*Clark et al., 2019 "What Does BERT Look At?"*）。
- **Head 1-1:** 直前の単語（Previous token）に常にAttentionを向ける。
-  **Head 3-5:** 文末のピリオドにAttentionを向ける（文の区切りを認識）。
-  **Head 8-10:** 直接目的語から動詞へAttentionを向ける（構文解析的な役割）。
-  **Head 9-2:** 代名詞（it, he）から、その指している名詞へAttentionを向ける（照応解析）。

これらは人間が設計したルールではなく、**「翻訳や言語モデリングというタスクを解くために、モデルが試行錯誤した結果、最も効率的だった戦略」** が重み行列として焼き付いたものです。

#### まとめ
- **自動で行われるか？:** はい。重み行列  の値が学習によって更新されることで、自動的に決まります。
-  **なぜ結果が異なるのか？:**
	- **初期値の違い:** 最初にランダムな値が入っているため、スタート地点が違う。
	- **線形変換の違い:** 各ヘッドは独自の「色眼鏡（重み行列）」を持っており、入力データの異なる側面（意味、文法、位置関係など）を抽出・強調するため。

この「勝手に賢い特徴を見つけ出す」能力こそが、Deep Learning（表現学習）の最大の強みです。

---- 

### モデルの次元数とMulti-Headの関係
結論から申し上げますと、**「シングルヘッドで次元数を増やしても、マルチヘッドと同じ結果（性能・振る舞い）は得られません」**。
その理由を、$d_{model}$とヘッド数の関係、そして「なぜ分割するのか」という数理的な違いから解説します。

#### 1.  と Multi-Head の関係
まず、定義上の関係を確認します。Transformerにおいて、各ヘッドの次元数 $d_k$ は通常、モデル全体の次元数 $d_{model}$ をヘッド数 $h$ で割った値に設定されます。
$$d_k = d_{model} / h$$

例えば、「$d_{model} = 512, h = 8$」の場合、各ヘッドは「64次元」のベクトルを扱います。
パラメータ数（重み行列の合計サイズ）で見ると、以下の2つはほぼ同じです。
1. **シングルヘッド:** $512 \cdot 512$ の行列で計算する。
2. **マルチヘッド:** $512 \cdot 64$ の行列を  個用意して計算する（合計は $512 \cdot 512$ 相当）。

しかし、**計算される中身（Attentionの挙動）は全く異なります**。

#### 2. なぜシングルヘッドでは同じ結果にならないのか？
最大の理由は、**「Softmax関数の非線形性と、注意（Attention）の独立性」** にあります。

##### ① Softmaxによる情報の「混合」 vs 「独立」
Attentionの核心は、以下の式にある通り、内積の結果をSoftmaxで確率（重み）に変換することです。
$$\text{Attention}(Q,K,V) = \text{softmax}
\left (
\frac{QK^T}{\sqrt{d_k}}
\right )$$

- **シングルヘッドの場合:**
	512次元すべての情報を使って「たった1つのAttention分布（どこを見るかの確率）」を作ります。この場合、**「文法的に重要な単語」と「意味的に重要な単語」が別々の場所にあったとしても、それらを混ぜ合わせて1つのランキング（確率分布）にしなければなりません**。結果として、平均的でぼやけたAttentionになってしまうリスクがあります。
- **マルチヘッドの場合:**
	64次元ごとに分割し、「8個の独立したAttention分布」を作ります。
	- ヘッド1: 「文法」に基づいて、主語が動詞に強く注目する（確率100%）。
	- ヘッド2: 「意味」に基づいて、代名詞が名詞に強く注目する（確率100%）。
	このように、**互いに干渉することなく、異なる場所を同時に「凝視」する**ことができます。

##### ② アンサンブル効果
Multi-Head Attentionは、**「8つの異なる小さなモデルのアンサンブル（多数決や協調）」** のような効果を持ちます。
シングルヘッドで次元数だけを増やしても、それは「1つの巨大な視点」に過ぎません。複数のヘッドを持つことで、モデルは入力文章の「異なる側面（部分空間）」を並列に処理し、最後にそれらを結合することで豊かな表現を獲得できます。

#### 3. 直感的な例え
- **シングルヘッド（巨大な1つのヘッド）:**
	「1人の超天才」が文章を読みます。彼は非常に賢いですが、一度に1つの解釈しか出力できません。「文法的にはこうだけど、意味的には…」と迷った場合、その中間の曖昧な解釈を出してしまうかもしれません。
- **マルチヘッド（複数の専門家）:**
	「8人の専門家（文法学者、歴史学者、心理学者…）」が同時に文章を読みます。
	- 文法学者は「構文」だけを見ます。
	- 心理学者は「感情」だけを見ます。
	最後にリーダー（出力層）が全員の意見をまとめ上げます。これにより、多角的な理解が可能になります。

#### まとめ
- **関係:**  をヘッド数で分割して計算します。パラメータ数は同程度に保たれます。
- **結果の違い:** シングルヘッドで計算してもマルチヘッドと同じ結果にはなりません。**Softmaxによって確率分布を作るプロセスが「1回」か「複数回（並列）」か**という決定的な違いがあり、マルチヘッドの方が多様な特徴を捉える能力（表現力）が高くなります。

この「独立した視点を持つ」という特性をコード上で実現しているのが、前回実装した `view` で次元を分割し、`transpose` で独立させて行列演算を行う部分になります。

### モデルの次元数と`torch.nn.Embedding`について
「1つの値を $d_{model}$ 個コピーして次元を増やす」という認識は、根本的に異なります。もし全ての次元に同じ値が入っていると、512次元あっても実質的には「1次元（スカラー）」の情報しか持っていないことになり、Transformerのような巨大なモデルを使う意味がなくなってしまいます。正確なイメージを持っていただくために、「モデルの次元数 ($d_{model}$)」とは何なのか、**「RPGのキャラクターステータス」**に例えて解説します。

#### 1. 正しいイメージ：単語の「意味の成分表」
「単語（Token）」をモデルに入力する際、最初は単なる「ID（背番号）」です（例: "りんご" = ID 45）。Embedding（埋め込み）層を通すと、この「ID 45」は、**512個のそれぞれ異なる数値が並んだリスト（ベクトル）** に変換されます。「同一の値」ではなく、**「512個全てが異なる意味を持つ数値」** です。

##### わかりやすい例：RPGのステータス画面
「戦士」というキャラクター（単語）を表現することを想像してください。
- **ご認識されていたイメージ（同一値のコピー）:**
	- 戦士 = `[10, 10, 10, 10, 10]`
	- これでは「攻撃力」も「魔力」も「素早さ」も全部同じ値です。これでは個性を表現できません。

- **実際の  次元（Embedding）:**
	- 戦士 = `[攻撃:99, 魔力:5, 素早さ:50, 運:20, ...]`
	- 魔法使い = `[攻撃:10, 魔力:99, 素早さ:40, 運:80, ...]`

Transformerにおける  とは、1つの単語に対して **「512種類の異なる評価項目（パラメータ）」** を持たせている状態です。
- 次元1: 「人間かどうか」を表す数値
-  次元2: 「ポジティブかネガティブか」を表す数値
-  次元3: 「過去形かどうか」を表す数値
-  ...
-  次元512: 「王族に関係するか」を表す数値

（※実際には、ディープラーニングが学習過程で勝手に意味を決めるため、人間には「次元Xが何を表しているか」は解読不能なことが多いですが、概念としてはこのように**分散して意味を保持**しています）。

#### 2. なぜ次元を増やす（512個も用意する）のか？
「りんご」という単語一つをとっても、文脈によって意味が変わります。
1. 「**赤い**、**果物**、**甘い**」 （食べるりんご）
2. 「**IT企業**、**スマホ**、**Mac**」 （Apple社）
3. 「**椎名林檎**、**歌手**」 （人物）

もし次元数が少なかったら（例：2次元）、これらの微妙なニュアンスの違いを空間上で「区別して配置」することができません。「果物のりんご」と「Apple社」が重なってしまい、モデルが混乱します。512次元という広大な空間があれば、
- 「果物」の軸では遠く離す
-  「名詞」という軸では近くに置く
といった具合に、**単語同士の複雑な関係性（意味の距離）を、衝突させることなく立体的に配置できる**ようになります。

#### 3. Phase 1 の資料における記述
Phase 1 の資料 にて、「Input Embedding」について以下のように記述されています。
> まず、入力された単語ID列を、 次元のベクトルに変換します。

これは、プログラム上では `nn.Embedding` という層を使います。これは巨大な「変換テーブル（辞書）」です。
- **入力:** `[1, 5, 9]` (単語IDの列。サイズは $1 \cdot 3$)
-  **処理:** 辞書の 1番目、5番目、9番目の行を取り出す。
-  **出力:**
```
[
  [0.1, -0.5, 0.9, ...],  # ID 1 のベクトル (512個の固有の値)
  [0.8,  0.1, 0.0, ...],  # ID 5 のベクトル (512個の固有の値)
  [-0.2, 0.4, 1.2, ...]   # ID 9 のベクトル (512個の固有の値)
]
```

(サイズは $3 \cdot 512$ になります)

#### まとめ
- **間違い:** 1つの値を512個コピーするわけではありません。
- **正解:** 1つの単語IDを、**512個の異なる数値のセット（ベクトル）** に置き換えます。
- **役立つ理由:** 単語の持つ複雑な意味（攻撃力、魔力、属性...のような多面的な特徴）を表現し、文脈による微妙な違いを計算可能にするためです。

この「512個の数値のセット」同士を、Phase 2で実装したAttention（内積）を使って「似ているかどうか」比較することで、Transformerは文脈を理解しています。