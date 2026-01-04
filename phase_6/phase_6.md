## Phase 6: 学習ループと検証 (Training Loop & Validation)
これまでのフェーズで、Transformerのモデルアーキテクチャと、学習に必要な周辺コンポーネント（マスキング、スケジューラ）の実装が完了しました。
Phase 6では、これらを統合し、実際にモデルを学習させるための **「学習ループ」** と、学習したモデルを使って新しい文を生成する **「推論（Decoding）」** のプロセスを実装します。

本フェーズのゴールは、巨大なデータセットを使わずとも、**「モデルが正しく学習できているか」** を検証できる最小限の構成（トイタスク）を作成し、Lossが下がり、意味のある出力が得られることを確認することです。

---- 

### 1. 理論背景：Teacher Forcing と推論のギャップ
Transformer（および多くのSeq2Seqモデル）において、学習時と推論時ではデータの与え方が異なります。

#### 1.1 学習時：Teacher Forcing
学習時は、正解データ（Target）が手元にすべてあります。モデルが学習を効率よく進めるために、**「一つ前の時刻でモデルが何を予測したかに関わらず、次の入力には必ず正解（正解ラベル）を与える」** という手法をとります。これを **Teacher Forcing** と呼びます。

- **Decoderへの入力:** `<sos>`, $y_1, y_2, y_3$
- **モデルの予測対象:** $y_1, y_2, y_3$, `<eos>`

    `<sos>`は「Start of Sentence」、`<eos>`は「End of Sentence」の略。

このように、入力と正解（ターゲット）を1トークンずらして与えるのがポイントです。

#### 1.2 推論時：Auto-regressive Generation
推論（テスト）時には、正解ラベルは存在しません。したがって、**「自分の出力した単語を、次の時刻の入力として再利用する」** 必要があります。
1. Encoderで入力を処理し、Memoryを作る。
2. Decoderに初期トークン `<sos>` とMemoryを入れる → 最初の単語 $\hat{y}_1$ を予測。
3. Decoderに `[<sos>,` $\hat{y}_1$ `]` とMemoryを入れる → 次の単語 $\hat{y}_2$ を予測。
4. `<eos>` が出るまで繰り返す。

この処理を **Greedy Decoding**（常に確率最大の単語を選ぶ）と呼びます。

---- 

### 2. 実装：学習と推論の関数
これまでのPhaseで作成したクラス（`Transformer`, `create_masks`, `NoamScheduler` など）が利用可能である前提で進めます。

#### 2.1 学習および評価ステップの実装
PyTorchの標準的な学習ループを関数化します。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import random

def train_epoch(model: nn.Module, data_loader: list, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """1エポック分の学習を行う関数

    Args:
        model (nn.Module): transformerモデル
        data_loader (list): バッチデータのリスト
        optimizer (torch.optim.Optimizer): オプティマイザ (NoamSchedulerでラップされている想定)
        criterion (nn.Module): 損失関数 (LabelSmoothingCrossEntropy)
        device (torch.device): 計算デバイス(CPU or GPU)

    Returns:
        float: 1エポック分の平均損失
    """
    model.train() # 学習モード(Dropout有効)
    total_loss = 0.0
    
    for i, batch in enumerate(data_loader):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        # targetの処理
        # device_input: <sos>, w1, w2, ... (最後を含まない)
        # target_label: w1, w2, ..., <eos> (最初を含まない)
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        # マスクの作成
        # tgt_inputを使ってマスクを作る点に注意
        src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx=0)

        # Forward pass
        # output: [batch_size, tgt_len, vocab_size]
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Loss計算
        # CrossEntropyLoss は [N, C] の入力を期待するため、出力とラベルを変形
        # output: [batch_size * tgt_len, vocab_size]
        # label: [batch_size * tgt_len]
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_label.contiguous().view(-1))

        # Backward & Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # NoamScheduler.step()内でoptimizer.step()が呼ばれる

        total_loss += loss.item()
    
    return total_loss / len(data_loader)
```

#### 2.2 推論（Greedy Decoding）の実装
学習したモデルを使って文を生成する関数です。これまでの `model(src, tgt)` を呼ぶのではなく、Encoderを一度だけ回し、Decoderをループさせる点に注目してください。

```python
def greedy_decode(model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int, end_symbol: int, device: torch.device) -> torch.Tensor:
    """Greedy Decodingによる推論を行う関数

    Args:
        model (nn.Module): 学習済みTransformerモデル
        src (torch.Tensor): [1, src_len] エンコードする入力シーケンス
        src_mask (torch.Tensor): [1, 1, 1, src_len] エンコード入力に対するマスク
        max_len (int): 最大生成長
        start_symbol (int): 開始トークン<sos>のID
        end_symbol (int): 終了トークン<eos>のID
        device (torch.device): 計算デバイス(CPU or GPU)

    Returns:
        torch.Tensor: [1, gen_len] 生成されたシーケンスのトークンID列
    """
    model.eval() # 推論モード(Dropout無効)

    # 1. Encode(一度だけ実行)
    # memory: [1, seq_len, d_model]
    memory = model.encode(src, src_mask)

    # 2. Decode (ループ処理)
    # 生成シーケンスの初期化 (開始トークン <sos> のみ)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len):
        # 現在のysに対するマスクを作成
        tgt_mask = create_look_ahead_mask(ys.size(1)).to(device)

        # Decoderを通す
        # out: [1, current_seq_len, d_model]
        out = model.decode(ys, memory, src_mask, tgt_mask)

        # 最後の時刻の出力を使って次の単語を予測
        # prob: [1, vocab_size]
        prob = model.fc_out(out[:, -1])

        # 確率最大の単語IDを取得 (Greedy選択)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # 生成された単語をリストに追加
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        # 終了トークン<eos>が生成されたら終了
        if next_word == end_symbol:
            break
    
    return ys
```

---- 

### 3. 統合と実験：トイタスクによる検証
実際に動かすために、**「ランダムな数字の列をコピーする」** という単純なタスク（Copy Task）を用意します。これはAttention機構が正しく動作しているかを確認するための標準的なベンチマークです。

#### 3.1 データ生成とハイパーパラメータ
```python
class CopyTaskDataset(Dataset):
    def __init__(self, num_samples: int, max_len: int, vocab_size: int):
        """ランダムな文字列のペア(src, tgt)を生成・保持するデータセットクラス
        Args:
            num_samples (int): 生成するサンプル数
            max_len (int): 各サンプルの最大長
            vocab_size (int): 単語IDの語彙数
        """
        self.num_samples = num_samples
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.data = self._generate_data()

    def _generate_data(self) -> list:
        data = []

        # <sos>と<eos>のIDを定義
        start_symbol = self.vocab_size
        end_symbol = self.vocab_size + 1
        for _ in range(self.num_samples):
            # 1〜vocab_size-1 のランダムな数列(0はpad, vocab_size+2はstart/end token用に空けておく
            seq_len = random.randint(1, self.max_len)

            # ランダムな長さの数字列
            seq = torch.randint(1, self.vocab_size, (seq_len,))

            # Padding処理
            # src: [seq_len] -> [max_len] に0埋め
            src = torch.zeros(self.max_len + 2, dtype=torch.long)
            src[:seq_len] = seq

            # tgt: [<sos>, ..., <eos>] -> [max_len + 2] に0埋め
            tgt = torch.zeros(self.max_len + 2, dtype=torch.long)
            tgt[0] = start_symbol
            tgt[1:seq_len + 1] = seq
            tgt[seq_len + 1] = end_symbol
            # 残りは0でパディングされたまま

            data.append((src, tgt))
        return data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        # インデックスに対応する(src, tgt)ペアを返す
        return self.data[idx]

def collate_fn(batch) -> tuple:
    """DataLoaderがミニバッチを作成する際に呼ばれるコールバック関数

    Args:
        batch (list): (src, tgt)ペアのリスト [(src1, tgt1), (src2, tgt2), ...]

    Returns:
        tuple: パディングされたsrcとtgtのテンソル
    """
    # バッチ内のsrcとtgtのペアを分離
    src_list, tgt_list = zip(*batch)

    # リストをTensorに変換してスタック
    # [batch_size, seq_len]になる
    src_batch = torch.stack(src_list)
    tgt_batch = torch.stack(tgt_list)

    return src_batch, tgt_batch
```

#### 3.2 メイン実行ブロック
これまでの全てのPhaseを統合して実行します。

```python
# モデル設定
SRC_VOCAB = 100
TGT_VOCAB = 100 + 2  # +2 for <sos> and <eos>
D_MODEL = 128
D_FF = 512
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1
MAX_LEN = 20

# 学習設定
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# 1. モデルの構築
model = Transformer(
    src_vocab_size=SRC_VOCAB, 
    tgt_vocab_size=TGT_VOCAB, 
    d_model=D_MODEL, 
    num_layers=NUM_LAYERS, 
    num_heads=NUM_HEADS, 
    d_ff=D_FF, 
    max_len=MAX_LEN + 2, # +2 for <sos> and <eos>
    dropout=DROPOUT,
).to(DEVICE)

# 重みの初期化 Xavier Initialization
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# 2. オプティマイザとスケジューラ
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(optimizer=optimizer, d_model=D_MODEL, warmup_steps=1000)

# 3. 損失関数 Label Smoothing Cross Entropy
# pad_idx=0を無視するように設定
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)

# 4. ダミーデータの生成とデータローダの作成
dataset = CopyTaskDataset(num_samples=1000, max_len=MAX_LEN, vocab_size=SRC_VOCAB)
data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn, # カスタムコールバック関数(バッチ化の処理)を指定
    drop_last=True # 最後の端数バッチが小さい場合に破棄(サイズを固定したい場合)
)

# 5. 学習ループ実行
print("Start training...")
for epoch in range(EPOCHS):
    loss = train_epoch(model, data_loader, scheduler, criterion, DEVICE)
    print(f"Epoch: {epoch+1: 02} | Loss: {loss:.4f} | LR: {scheduler.get_lr():.6f}")

# 6. 推論の確認
print("Start inference...")
model.eval()
test_src = torch.tensor([[1, 2, 3, 4, 5]]).to(DEVICE)  # [1, src_len]
src_mask = create_padding_mask(test_src, pad_idx=0).to(DEVICE)

# 推論実行
# start_symbol=100, end_symbol=101を指定
generated = greedy_decode(model, test_src, src_mask, max_len=10, start_symbol=100, end_symbol=101, device=DEVICE)
print(f"Input Sequence: {test_src.cpu().numpy()}")
print(f"Generated Sequence: {generated.cpu().numpy()}")

# コピータスクなので、Inputと同じ数列（最後に101）が出れば成功
```

#### 3.3. 期待される結果
学習初期はLossが高く、ランダムな出力になりますが、Epochが進むにつれてLossが急激に下がり、最終的に `Generated` の結果が `Input` と（`<sos>`、`<eos>`を除いて）一致するようになります。

```text
Input Sequence: [[1 2 3 4 5]]
Generated Sequence: [[100   1   2   3   4   5 101]]
```

(100は`<sos>`、101は`<eos>`として定義した場合)

---- 

### 4. ハイパーパラメータの調整
Phase 6でのトラブルシューティングを通じて行った調整は、Transformerの学習における非常に本質的なテクニックを含んでいます。今回の「Lossは下がるが正しく止まらない（`<eos>`が出ない）」という現象を解決するために行った4つの調整について、その**性質**と**調整のポイント**をまとめます。

#### 4.1. D\_MODEL（モデルの次元数）
**性質:**
- 単語を表現するベクトルのサイズであり、モデルの「表現力（頭の良さ）」を決定します。
- *重要:** Noam Schedulerの学習率計算式  に組み込まれており、**この値が小さいほど、計算される学習率は高くなる**という反比例の関係があります。**

**今回の調整 (32 -\> 128):**
- 当初の `32` はTransformerとしては小さすぎました。これにより、計算される学習率が跳ね上がってしまい、学習が不安定になっていました。
- サイズを大きくすることで、表現力を確保すると同時に、ベースとなる学習率を適正範囲まで引き下げました。

**調整のポイント:**
- トイタスクであっても、最低でも `64` や `128` 程度確保するのが無難です。
- 逆に大きくしすぎると（512以上）、学習に時間がかかり、データが少ない場合は過学習しやすくなります。

#### 4.2. Warmup Steps（ウォームアップステップ数）
**性質:**
- 学習開始から学習率を「徐々に上げていく（線形増加させる）」期間の長さです。
- この期間が終わった瞬間が**「最大学習率（ピーク）」**となり、そこから徐々に下がっていきます。つまり、**Warmupを長くすればするほど、ピーク時の学習率は低く抑えられます。**

**今回の調整 (200 -\> 1000):**
- Lossは下がっているのに細かい挙動（止まる・止まらない）が安定しない場合、学習率が高すぎて最適解を飛び越えている可能性があります。
- Warmupを長く取ることでピーク学習率を下げ、モデルがより慎重にパラメータを調整できるようにしました。

**調整のポイント:**
- **学習が不安定（Lossが振動する）な場合:** Warmupを増やしてください。
- **学習が遅すぎる場合:** Warmupを減らしてください。
- 総ステップ数の 5%〜10% 程度に設定するのが一般的ですが、今回のように繊細な収束が求められる場合は長め（20%〜）に取ることも有効です。

#### 4.3. Label Smoothing（ラベル平滑化）
**性質:**
- 正解ラベルに対して「100%の自信を持つな（正解率を0.9などにして、残りを分散させる）」という制約を課す正則化手法です。
- モデルの過信を防ぎ、未知のデータに対する汎化性能を高める効果があります。

**今回の調整 (0.1 -\> 0.0):**
- **自然言語（翻訳など）:** 「答えが一つとは限らない（曖昧性がある）」ため、Smoothingが有効です。
- **コピー・計算・論理:** 「答えが完全に一つに定まる（決定論的）」ため、Smoothingは邪魔になります。モデルが「ここで絶対に終わるんだ！」と確信を持つのを阻害してしまいます。

**調整のポイント:**
- 翻訳や対話生成なら `0.1` が定石。
- 今回のようなコピー課題、算数、プログラミング言語生成など、厳密さが求められるタスクでは `0.0`（無効）にします。

#### 4.4. 可変長データ（Variable Length Data）
**性質:**
- 学習データの長さ（シーケンス長）を固定にするか、バラバラにするかというデータ戦略です。

**今回の調整 (固定長  ランダム長):**
- 固定長で学習すると、モデルは「データの終わり（`<eos>`）」ではなく「決まった文字数（位置情報）」を学習してしまいます（**長さへの過学習**）。
- 長さをランダムにすることで、モデルに「文字数に頼るな、パディングの境界線を見ろ」と強制し、本質的な停止条件を学習させました。

**調整のポイント:**
- 系列データ（時系列、テキスト）を扱う際は、**必ず長さにバリエーションを持たせる**ことが、推論時のバグを防ぐ鉄則です。

#### 4.5. まとめ

| パラメータ               | 役割            | 症状と対策                                |
| ------------------- | ------------- | ------------------------------------ |
| **D\_MODEL**        | モデルの容量＆学習率の基準 | 小さすぎると学習率が高騰し不安定になる。適度な大きさが必要。       |
| **Warmup Steps**    | 学習率のブレーキ      | Lossは下がるが精度が出ない時は、ここを増やしてピーク学習率を下げる。 |
| **Label Smoothing** | 曖昧さの許容        | 「正解が一つ」の厳密なタスクでは `0.0` にする。          |
| **データ長**            | 停止条件の学習       | 常に固定長だと「長さ」を暗記してしまう。ランダム長で汎化させる。     |

これらのパラメータ調整は、Transformerに限らず、多くの深層学習モデルにおいて共通する重要なチューニング勘所です。

---- 

## 講義のまとめ
全6回にわたり、Transformerをゼロから実装してきました。
1. **Embedding & Positional Encoding:** 文字列をベクトル化し、順序情報を加えました。
2. **Multi-Head Attention:** 文脈を並列かつ多角的に捉える仕組みを作りました。
3. **FFN & LayerNorm:** 情報を加工し、深層学習を安定させる層を作りました。
4. **Encoder-Decoder:** これらをスタックし、モデル全体を構築しました。
5. **Masking & Scheduler:** バッチ学習と収束のための重要なテクニックを導入しました。
6. **Training Loop:** 実際にデータを流し、モデルがパターンを学習することを確認しました。

この実装は、BERTやGPT、そして現在のLLM（Large Language Models）のすべての基礎となっています。ここで学んだ「Q, K, Vの相互作用」や「マスクの役割」といった根本原理は、最新のモデルを理解する上でも変わらぬ羅針盤となるはずです。

## 参考文献
- Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS.
- Harvard NLP. "The Annotated Transformer".
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization".