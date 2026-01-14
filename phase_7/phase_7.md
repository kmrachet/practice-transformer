## Phase 7: PyTorch Lightningによるリファクタリングと実践 (Refactoring with PyTorch Lightning)

これまでのフェーズで、Transformerの理論と実装は完結しました。最後の仕上げとして、この「素のPyTorch」で書かれたコードを、モダンな深層学習フレームワークである **PyTorch Lightning** を用いて書き直します。

なぜこれが必要なのでしょうか？ Phase 6 の学習ループを見直してみると、`model.train()`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, そしてデバイス管理 (`.to(device)`) など、本質的なモデルのロジックとは関係のない**定型処理（Boilerplate）**が多く含まれていることに気づくはずです。

PyTorch Lightningは、これらのエンジニアリング部分を抽象化し、研究者が「モデルの設計」と「データ」に集中できるように設計されています。

### 1. PyTorch Lightningの導入

まず、PyTorch Lightning における主要なコンポーネントである `LightningModule` を実装します。これは `nn.Module` のスーパーセットであり、モデル本体だけでなく、**「どう学習するか（train\_step）」** や **「どう最適化するか（configure\_optimizers）」** というロジックまでを1つのクラスにカプセル化します。

#### 1.1 LightningModule の実装

Phase 6 までの `Transformer` クラスや `NoamScheduler` のロジックを再利用しつつ、Lightning の作法に従って実装します。ただし、Phase 6 で定義した `Transformer` クラス、`create_masks` 関数、`CopyTaskDataset` クラス等はすでに定義されているものとしてインポートして使用します。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import math
import random

# これまでのフェーズで作成したクラスや関数を想定
# from phase_6_code import Transformer, create_masks, CopyTaskDataset, collate_fn

class LitTransformer(pl.LightningModule):
    """
    TransformerモデルをPyTorch Lightning用にラップしたクラス。
    学習ステップ、オプティマイザ設定、推論ロジックをカプセル化します。
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        warmup_steps: int = 4000,
        pad_idx: int = 0,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        # ハイパーパラメータを保存 (self.hparamsでアクセス可能になり、ログにも残る)
        self.save_hyperparameters()

        # 1. モデル本体の初期化 (Phase 4-6で作ったもの)
        self.transformer = Transformer(
            src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout
        )
        
        # 重み初期化 (Xavier Initialization)
        self._init_weights()

        # 損失関数
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.hparams.pad_idx,
            label_smoothing=self.hparams.label_smoothing
        )

    def _init_weights(self):
        """重みの初期化"""
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理。
        LightningModuleのforwardは推論時や予測時に使われることを想定するのが一般的です。
        
        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
        """
        # マスク作成 (Phase 5の実装を利用)
        # tgtは入力用(最後を含まない)として渡される想定
        src_mask, tgt_mask = create_masks(src, tgt, pad_idx=self.hparams.pad_idx)
        return self.transformer(src, tgt, src_mask, tgt_mask)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        1バッチ分の学習ステップ。
        Phase 6の train_epoch 関数の中身に相当します。
        """
        src, tgt = batch
        
        # Targetの分割: 入力(<sos>...w_n) と 正解ラベル(w_1...<eos>)
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        # Forward
        # 内部でcreate_masksを呼ぶようにforwardを定義したのでシンプルに呼び出せます
        logits = self(src, tgt_input)

        # Loss計算
        # [batch * seq_len, vocab_size] に変形
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)), 
            tgt_label.reshape(-1)
        )

        # ログ記録 (プログレスバーとロガーに表示)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        オプティマイザとスケジューラの設定。
        Phase 5の NoamScheduler のロジックを、PyTorch標準のLambdaLRを使って再現します。
        Lightningではstep毎のスケジューリングを自動化できます。
        """
        optimizer = optim.Adam(
            self.parameters(), 
            lr=1.0, # LambdaLRで係数をかけるため、ベースは1.0にするか、あるいはbeta1/beta2のみ設定
            betas=(0.9, 0.98), 
            eps=1e-9
        )

        # Noam Schedulerの計算式: 
        # lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        def noam_lambda(step: int) -> float:
            # stepが0だとゼロ除算になるため1から始める等の対処、あるいはstep+1
            if step == 0: step = 1
            d_model = self.hparams.d_model
            warmup = self.hparams.warmup_steps
            
            return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda),
            'interval': 'step', # epochごとではなくstepごとに更新
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def encode(self, src, src_mask):
        return self.transformer.encode(src, src_mask)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.transformer.decode(tgt, memory, src_mask, tgt_mask)
```

#### 1.2. 推論関数(Greedy Decode)
```python
def greedy_decode_lightning(model: LitTransformer, src: torch.Tensor, max_len: int, start_symbol: int, end_symbol: int, device: torch.device):
    """
    LightningModuleを用いた推論関数
    """
    model.eval() # 推論モード
    model.to(device)
    src = src.to(device)

    # マスク作成
    # pad_idxはハイパーパラメータから取得可能
    pad_idx = model.hparams.pad_idx
    src_mask = create_padding_mask(src, pad_idx).to(device)

    # 1. Encode
    with torch.no_grad():
        memory = model.encode(src, src_mask)

    # 2. Decode Loop
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len):
        # Look-ahead Mask
        tgt_mask = create_look_ahead_mask(ys.size(1)).to(device)

        with torch.no_grad():
            out = model.decode(ys, memory, src_mask, tgt_mask)
            # 最後の単語の確率分布を取得
            # model.transformer.fc_out にアクセス
            prob = model.transformer.fc_out(out[:, -1])
            
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

        # 生成された単語を追加
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        if next_word == end_symbol:
            break
            
    return ys
```

### 2. 学習の実行 (Trainer)

LightningModule を定義したら、学習ループを自分で書く必要はありません。`pl.Trainer` が全て（GPUへの転送、勾配の計算、ログの保存、チェックポイントの作成など）を代行してくれます。

```python
# --- 1. 設定 ---
pl.seed_everything(42) # 再現性のためのシード固定

SRC_VOCAB = 100
TGT_VOCAB = 100 + 2 # <sos>, <eos>
D_MODEL = 128
BATCH_SIZE = 32
MAX_LEN = 20
EPOCHS = 30 # Lightningなら早く収束する場合が多いが、適宜調整

# --- 2. データ準備 ---
dataset = CopyTaskDataset(num_samples=2000, max_len=MAX_LEN, vocab_size=SRC_VOCAB)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=0 # 環境によっては0推奨
)

# --- 3. モデル初期化 ---
model = LitTransformer(
    src_vocab_size=SRC_VOCAB,
    tgt_vocab_size=TGT_VOCAB,
    d_model=D_MODEL,
    warmup_steps=1000,
    pad_idx=0,
    label_smoothing=0.0
)

# --- 4. 学習実行 (Trainer) ---
# GPUが使えるなら自動で使用
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto", 
    devices=1,
    enable_progress_bar=True,
    enable_checkpointing=False, # 実験用なのでチェックポイント保存なし
    logger=False # ログファイル出力なし
)

print(">>> Start Training with PyTorch Lightning...")
trainer.fit(model, dataloader)
print(">>> Training Finished!")

# --- 5. 推論による検証 (Inference) ---
print(">>> Start Inference Check...")

# テストデータ: [1, 2, 3, 4, 5] という数列
test_src = torch.tensor([[1, 2, 3, 4, 5]])

# 推論実行 (<sos>=100, <eos>=101 と仮定)
device = model.device # Trainerが割り当てたデバイスを取得
generated = greedy_decode_lightning(
    model, 
    test_src, 
    max_len=10, 
    start_symbol=100, 
    end_symbol=101, 
    device=device
)

print(f"Input:     {test_src.cpu().numpy()}")
print(f"Generated: {generated.cpu().numpy()}")

# 成功判定
# <sos>1, 2, 3, 4, 5<eos> の形になっていれば成功
expected_part = [1, 2, 3, 4, 5]
gen_list = generated.cpu().numpy()[0].tolist()

# <sos>を除去して比較
if gen_list[1:len(expected_part)+1] == expected_part:
    print("✅ SUCCESS: Copy Task Completed!")
else:
    print("❌ FAILED: Incorrect output.")
```

### 3. 素のPyTorch vs PyTorch Lightning の比較

Phase 6 のコードと今回のコードを比較してみましょう。

| 項目                 | 素の PyTorch (Phase 6)                      | PyTorch Lightning (Phase 7)                |
| ------------------ | ----------------------------------------- | ------------------------------------------ |
| **学習ループ**          | `for epoch in range...` から全て手書き           | `trainer.fit()` 1行のみ                       |
| **デバイス管理**         | `.to(device)` を各テンソル・モデルに記述               | 自動 (CPU/GPU/TPU/MPS 自動判別)                  |
| **Optimizer/Loss** | ループ内で `zero_grad`, `backward`, `step` を記述 | `training_step` で Loss を返すだけ               |
| **スケジューラ**         | 手動で `scheduler.step()` を呼ぶ場所を管理           | `configure_optimizers` で設定するだけ             |
| **可読性・保守性**        | 学習ロジックが手続き的に分散しやすい                        | モデル定義の中に学習ロジックが凝集される                       |
| **機能拡張**           | 混合精度学習(16bit)や分散学習の実装は困難                  | `precision=16` や `strategy='ddp'` フラグのみで実現 |

### 4. 講義の総括

本コースでは、Phase 1〜6 を通じて、Transformer の中身（Attention機構、Positional Encoding、Encoder-Decoder構造）を数式とPyTorchのプリミティブな実装から深く理解しました。これは「車輪の再発明」ですが、研究者としてモデルの挙動を完全に把握するためには不可欠なプロセスです。

そして Phase 7 では、その知識を持った上で、現代的なツール（PyTorch Lightning）を用いて効率的に実装する方法を学びました。

* **基礎力:** 素のPyTorchでどんなモデルでもゼロから書ける力。
* **応用力:** ツールを使って、大規模な実験や複雑な学習を効率よく回す力。

この両方を兼ね備えることが、優れた機械学習エンジニア・研究者への第一歩です。
### 参考文献 (Phase 7)

* Falcon, W., & The PyTorch Lightning Team. (2019). *PyTorch Lightning*. GitHub.
* Vaswani, A., et al. (2017). "Attention Is All You Need". *NeurIPS*. (Noam Schedulerの理論)