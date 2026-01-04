## Phase 5: 学習のメカニズム (Training Mechanics: Masking & Scheduler)

Phase 4まででTransformerのモデル構造自体は完成しました。しかし、実際にこの車を走らせる（学習させる）ためには、燃料の供給方法（データバッチ処理）や、アクセルワーク（学習率制御）を適切に設計する必要があります。Phase 5では、Transformerの学習において不可欠な**「マスキング」**と、学習を安定させるための**「Noamスケジューラ」**および**「Label Smoothing」**について解説・実装します。

可変長の自然言語データをバッチ処理で学習させ、かつ高い性能を出すためには、以下の3つの重要なメカニズムを実装する必要があります。

1. **マスキング (Masking):** パディング部分や未来の単語を計算から除外する。
2. **学習率スケジューラ (Noam Scheduler):** 非常に大きなモデルを安定して収束させるための特殊な学習率制御。
3. **ラベルスムージング (Label Smoothing):** モデルの過信を防ぎ、汎化性能を高める正則化。

本フェーズでは、これらをPyTorchのプリミティブな機能を用いて実装します。

---- 

### 1. マスキングの理論と実装
Transformerは、RNNのようにシーケンシャルに処理するのではなく、行列演算で行列全体を一気に処理します。そのため、**「計算してはいけない場所」**を明示的に隠す（マスクする）処理が必須となります。Phase 4の `MultiHeadAttention` クラスでは `scores = scores + mask` という処理を入れましたが、ここで加算するための適切なマスクを作成します。

#### 1.1 Padding Mask (パディングマスク)
バッチ学習を行う際、長さの異なる文を同じ長さに揃えるために `<pad>` トークン（通常はID 0）で埋めます。Attention機構がこの `<pad>` に注意を向けないように、スコアを無限小（$-\infty$、実装上は 1e-9）にしてSoftmaxの結果を0にします。
- **適用場所:** EncoderのSelf-Attention, DecoderのSelf-Attention, DecoderのSource-Target Attentionすべて。
- **形状:** `[batch_size, 1, 1, seq_len]` (ブロードキャスト用)

#### 1.2 Look-ahead Mask (未来情報のマスク)
DecoderのSelf-Attentionでは、位置 $i$ の単語を予測する際に、位置 $i$ 以降（未来）の単語を見てはいけません（カンニング防止）。そのため、行列の上三角部分をマスクします。
- **適用場所:** DecoderのSelf-Attentionのみ。
- **形状:** `[1, 1, seq_len, seq_len]` または `[batch_size, 1, seq_len, seq_len]`

#### 1.3 実装
以下のコードを記述します。マスク生成関数はモデルの外で呼び出し、`forward` に渡します。

```python
import torch
import torch.nn as nn
import numpy as np

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """パディングマスクを作成する関数。<pad>の部分を1e-9、それ以外を0にする。

    Args:
        seq (torch.Tensor): [batch_size, seq_len]の形状を持つテンソル。入力単語のID列
        pad_idx (int, optional): パディングを表すID。 Defaults to 0.

    Returns:
        torch.Tensor: [batch_size, 1, 1, seq_len] Mult-head Attentionのスコアに加算するためのマスク
    """

    # seq == pad_idx の部分はTrue、それ以外はFalse
    mask = (seq == pad_idx)

    # shapeを [batch_size, 1, 1, seq_len] に変形
    # Trueを1e-9、Falseを0に変換
    # float型に変換しないと加算時にエラーになる
    return mask.unsqueeze(1).unsqueeze(2).float() * -1e9


def create_look_ahead_mask(seq_len: int) -> torch.Tensor:
    """未来の単語を見えなくするための上三角マスクを作成する関数

    Args:
        seq_len (int): シーケンス長

    Returns:
        torch.Tensor: [1, 1, seq_len, seq_len] 対角成分より上が1e-9、それ以外が0の行列
    """
    # torch.triuで上三角行列を取り出す (diagonal=1で対角線のひとつ上から)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

    # マスク部分を1e-9、それ以外を0に変換
    return mask.unsqueeze(0).unsqueeze(0).float() * -1e9


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0):
    """EncoderとDecoderに必要なすべてのマスクを一括生成するヘルパー関数

    Args:
        src (torch.Tensor): [batch_size, src_len]
        tgt (torch.Tensor): [batch_size, tgt_len]
        pad_idx (int, optional): パディングID。 Defaults to 0.
    """
    # 1. Encoder用のマスク(Padding Maskのみ)
    src_mask = create_padding_mask(src, pad_idx)

    # 2. Decoder用のマスク(Padding Mask + Look-ahead Mask)
    # 2.1. TargetのPadding Mask [batch_size, 1, 1, tgt_len]
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)

    # 2.2. Look-ahead Mask [1, 1, tgt_len, tgt_len]
    tgt_len = tgt.size(1)
    look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device)

    # 2.3. Padding MaskとLook-ahead Maskを結合 (どちらかが-1e9なら-1e9になるように加算またはmaxを取る)
    # ここで単純に和を取ると-2e9になる箇所ができるが、Softmaxにおいては十分小さいので計算には影響しない
    # 論理和(OR)的にマスクしたいので、最小値を取る実装もよくある(torch.min)
    tgt_mask = torch.min(tgt_pad_mask, look_ahead_mask)

    return src_mask, tgt_mask

```

##### 実装のポイント
- **加算マスク (Additive Mask):** Attentionスコア（内積値）に直接足し算するため、マスクする場所には `-1e9` (マイナス無限大に近い値) を、マスクしない場所には `0` を設定しています。
- **形状:** `MultiHeadAttention` 内部でのスコア形状 `[batch, num_heads, seq_len, seq_len]` にブロードキャストできるように、次元 `1` (head用) を挿入しています。

---- 

### 2. Noam Scheduler (学習率のウォームアップ)
Transformerはパラメータ数が多く、初期学習率が大きいと勾配が爆発したり、局所解に陥りやすかったりします。そこで、Vaswani et al. (2017) では、**「学習率を線形に上げてから（Warmup）、徐々に下げる（Decay）」**という特殊なスケジューリングを採用しました。これを一般に **Noam Scheduler** と呼びます。

#### 2.1 理論式
学習ステップ数  における学習率  は以下の式で計算されます。
$$lrate=d_{model}^{-0.5}\cdot\text{min}(step\_num^{-0.5}, step\_min\cdot warmap\_steps^{-1.5})$$

* **Warmup期:** $step\_num$ が小さい間は、右側の項 $step\_min\cdot warmap\_steps^{-1.5}$ が選ばれ、学習率が線形に上昇します。
* **Decay期:** $step\_num$ が $warmap\_steps$ を超えると、左側の項 $step\_num^{-0.5}$ が選ばれ、学習率がルート逆数で減衰していきます。

#### 2.2 実装
PyTorchの標準的な `LambdaLR` を使っても実装できますが、ここでは理論の理解を深めるため、Optimizerをラップするクラスとして実装します。

```python
class NoamScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        """
        Transformer用学習率スケジューラ (Noam Scheduler)

        Args:
            optimizer: PyTorchのOptimizer
            d_model (int): モデルの次元数
            warmup_steps (int): ウォームアップのステップ数. Defaults to 4000.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self):
        """
        1ステップ進め、学習率を更新し、optimizer.step()を実行する
        """
        self.current_step += 1
        lr = self.get_lr()
        
        # Optimizerの学習率を更新
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # 重み更新
        self.optimizer.step()

    def get_lr(self) -> float:
        """現在のステップ数に基づき学習率を計算"""
        step = self.current_step
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
    
    def zero_grad(self):
        """Optimizerの勾配初期化"""
        self.optimizer.zero_grad()

```

---- 

### 3. Label Smoothing (ラベルスムージング)
Transformerの学習では、正解ラベルに対する確信度を下げ、汎化性能を向上させるために **Label Smoothing** (Szegedy et al., 2016) が用いられます。通常、分類問題では正解クラスに `1.0`、それ以外に `0.0` を割り当てる One-hot ベクトルを教師データとします。しかし、これをそのまま学習させると、モデルはロジット（Softmax前の値）を無限大にしようとして過学習を起こしやすくなります。

Label Smoothingでは、正解クラスの確率を少し下げ（例: `0.9`）、残りの確率（`0.1`）を他の全単語に均等配分します。

#### 実装
PyTorchの `nn.CrossEntropyLoss` はバージョン1.10以降、標準で `label_smoothing` 引数をサポートしています。これを利用するのが最も効率的でバグが少ない方法です。

```python
# 損失関数の定義例
# padding_idxを指定することで、<pad>トークンに対する損失を自動的に0にしてくれます
criterion = nn.CrossEntropyLoss(
    ignore_index=0,        # pad_idx
    label_smoothing=0.1    # ラベルスムージング係数
)

```

**数理的な背景（参考）:**
数式で書くと、ターゲット分布 $q(k)$ を以下のように変更しています。
$$q'(k) = (1 - \epsilon) \cdot q(k) + \frac{\epsilon}{V}$$

ここで $\epsilon$ はスムージング係数、$V$ は語彙数です。

---- 

### 4. 動作確認
作成したマスク生成機能とスケジューラが正しく動作するか確認します。

#### 4.1 マスクの可視化
特にLook-ahead Maskが正しく上三角になっているかを確認することは重要です。

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_masks():
    src_len = 5
    tgt_len = 5
    
    # ダミーデータ: 0はpad_idxとする
    # src: [1, 2, 3, 0, 0] (長さ3, pad2)
    src = torch.tensor([[1, 2, 3, 0, 0]])
    # tgt: [1, 2, 3, 4, 0] (長さ4, pad1)
    tgt = torch.tensor([[1, 2, 3, 4, 0]])
    
    src_mask, tgt_mask = create_masks(src, tgt, pad_idx=0)
    
    print("Source Mask Shape:", src_mask.shape)
    print("Target Mask Shape:", tgt_mask.shape)

    # グラフ描画
    plt.figure(figsize=(10, 4))
    
    # Source Mask (Paddingのみ)
    plt.subplot(1, 2, 1)
    # [1, 1, 5] -> [1, 5] 表示のため次元削除
    sns.heatmap(src_mask[0, 0, 0, :].unsqueeze(0).numpy(), cbar=False, square=True)
    plt.title("Source Padding Mask")
    
    # Target Mask (Padding + Look-ahead)
    plt.subplot(1, 2, 2)
    # [1, 5, 5]
    sns.heatmap(tgt_mask[0, 0, :, :].numpy(), cbar=False, square=True)
    plt.title("Target Mask (Look-ahead + Pad)")
    
    plt.show()

# 実行
visualize_masks()

```

**期待される結果:**
* **Left:** `[0, 0, 0, -1e9, -1e9]` のようなヒートマップ（後半がマスク）。
* **Right:** 対角線より上がマスクされ（上三角）、かつ最後の列（パディング部分）が縦にすべてマスクされている状態。

#### 4.2 スケジューラの確認
学習率の変化をグラフにします。

```python
def visualize_scheduler():
    # ダミーモデルとオプティマイザ
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0) # 初期lrは0でOK

    d_model = 512
    warmup_steps = 4000
    scheduler = NoamScheduler(optimizer, d_model, warmup_steps)

    lrs = []
    for _ in range(20000):
        scheduler.current_step += 1
        lrs.append(scheduler.get_lr())
    
    plt.plot(lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Noam Scheduler")
    plt.grid(True)
    plt.show()

visualize_scheduler()
```

**期待される結果:**

* 0ステップから4000ステップまで直線的に上昇し、その後緩やかに下降するカーブが描かれます。

---

これで、Transformerを正しく学習させるための周辺機構がすべて整いました。
次回 **Phase 6** では、いよいよこれらを統合し、翻訳タスク（あるいは文字列反転などのトイタスク）を用いた**学習ループ（Training Loop）**と**推論（Inference）**を実装して、モデルが実際に学習する様子を観察します。