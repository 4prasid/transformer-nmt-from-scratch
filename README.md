# Transformer for German to English Neural Machine Translation

A from-scratch PyTorch implementation of the Transformer from *"Attention Is All You Need"* (Vaswani et al., 2017), trained for German→English translation on Multi30k. Multi-head attention, positional encoding, the Noam learning-rate schedule, label smoothing and greedy decoding are all written by hand, with no `nn.MultiheadAttention`. Five controlled experiments probe why each design choice matters.

**📊 [Interactive W&B report](https://api.wandb.ai/links/prasid-indian-institute-of-technology-madras/pae80rij)**

## Repository structure

```
.
├── model.py                         # Full Transformer: attention, masks, positional encoding, encoder/decoder, greedy inference
├── train.py                         # Label smoothing loss, training loop, BLEU evaluation, checkpointing
├── dataset.py                       # Multi30k dataset, spaCy tokenization, vocabulary building, collate function
├── lr_scheduler.py                  # Noam learning-rate scheduler
├── Experiments/
│   ├── Noam_scheduler_1.py          # Noam vs fixed learning rate
│   ├── Ablation_2.py                # 1/√d_k scaling ablation with Q/K gradient-norm logging
│   ├── Head_specialization_3.py     # Per-head attention heatmaps, entropy and similarity analysis
│   ├── Encodings_4.py               # Sinusoidal vs learned positional embeddings
│   └── Decoder_sensitivity_5.py     # Label smoothing and prediction-confidence tracking
├── Findings.md                      # Written analysis of all five experiments
├── requirements.txt
└── LICENSE
```

## Key results

| | |
|---|---|
| **Test BLEU** (Multi30k test set, 1,000 pairs, greedy decoding, corpus-level sacreBLEU) | **32.76** |
| Final model | d_model 512 · 6 enc/dec layers · 8 heads · d_ff 2048 · dropout 0.3 · 50 epochs |
| Training data | 29,000 sentence pairs (validation: 1,014 · test: 1,000) |
| Optimization | Adam (β₁=0.9, β₂=0.98, ε=1e-9) · Noam schedule (4,000 warmup steps) · label smoothing 0.1 · gradient clipping at 1.0 |


## Implementation details

### `model.py`

- **`scaled_dot_product_attention(Q, K, V, mask)`** computes softmax(QKᵀ/√d_k)·V and returns both the output and the attention weights. Masked positions are filled with `-inf` before the softmax, so their weight is exactly zero.
- **`make_src_mask` / `make_tgt_mask`** build the padding mask (`[batch, 1, 1, src_len]`) and the combined padding + causal mask (`[batch, 1, tgt_len, tgt_len]`) via `torch.triu`.
- **`MultiHeadAttention`** projects Q, K and V into `num_heads` subspaces of size `d_k = d_model // num_heads`, attends in parallel, concatenates, and applies the output projection. `torch.nn.MultiheadAttention` is not used.
- **`PositionalEncoding`** is the sinusoidal encoding, precomputed up to `max_len=5000` and registered as a non-trainable buffer. Token embeddings are scaled by √d_model before it is added (§3.4 of the paper).
- **`PositionwiseFeedForward`** is `max(0, xW₁ + b₁)W₂ + b₂` with dropout between the projections.
- **`EncoderLayer` / `DecoderLayer`** use Post-LayerNorm residual blocks (`nn.LayerNorm`). The decoder layer has masked self-attention, cross-attention over the encoder memory, and the feed-forward network.
- **`Encoder` / `Decoder`** stack `N` deep-copied layers with a final `nn.LayerNorm`.
- **`Transformer`** is the full model, exposing `encode`, `decode`, `forward` and `infer`. `infer(src_sentence)` tokenizes with spaCy, decodes greedily and re-attaches punctuation.

### `lr_scheduler.py`

`NoamScheduler` subclasses `torch.optim.lr_scheduler.LRScheduler` and implements

```
lrate = d_model^-0.5 · min(step^-0.5, step · warmup_steps^-1.5)
```

The optimizer's base lr is set to 1.0 so the scheduler has full control. `get_lr_history` simulates the lr trajectory without training.

### `dataset.py`

`Multi30kDataset` wraps [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k), tokenizes with spaCy (`de_core_news_sm` / `en_core_web_sm`), builds word↔index vocabularies **from the training split only**, and wraps sentences in `<sos>` / `<eos>`. Validation and test sets reuse the training vocabularies. `collate_fn` pads each batch to its own maximum length.

### `train.py`

- **`LabelSmoothingLoss`** assigns `1 − ε` to the gold token and `ε / (vocab_size − 2)` to every other non-pad token. Pad targets are excluded from the mean.
- **`run_epoch`** handles both training and evaluation (gradient clipping at 1.0).
- **`greedy_decode`** encodes the source once and generates token-by-token until `<eos>` or the length limit.
- **`evaluate_bleu`** reports corpus-level BLEU with `sacrebleu`, falling back to NLTK if it is unavailable.
- **`save_checkpoint` / `load_checkpoint`** store model, optimizer and scheduler state plus the model config and vocabularies.
- **`run_training_experiment`** is the main entry point: W&B logging, validation BLEU every 2 epochs, best-checkpoint saving, and a final test-set BLEU.

## Setup

```bash
git clone https://github.com/4prasid/transformer-nmt-from-scratch.git
cd transformer-nmt-from-scratch
pip install -r requirements.txt

# spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Training the final model

```bash
python train.py
```

Log in to W&B first (`wandb login`). Hyperparameters live in the `config` dict inside `run_training_experiment`:

| Hyperparameter | Value |
|---|---|
| `d_model` | 512 |
| `N` (layers) | 6 |
| `num_heads` | 8 |
| `d_ff` | 2048 |
| `dropout` | 0.3 |
| `batch_size` | 128 |
| `num_epochs` | 50 |
| `warmup_steps` | 4000 |
| label smoothing | 0.1 |

The best checkpoint (by validation BLEU) is saved to `checkpoint_best.pth`, and a per-epoch checkpoint is written for recovery.

## Inference

```python
from model import Transformer

model = Transformer()   # builds vocabularies; tries to download the trained checkpoint (see note)
print(model.infer("Ein Mann sitzt auf einer Bank."))
# → "A man is sitting on a bench."
```

> **Note:** `Transformer()` tries to download the trained 512/6 checkpoint from Google Drive via `gdown` on construction. If the download fails, the model initializes with random weights and prints a warning, so train your own with `python train.py` in that case.

## Design 

- **Post-LayerNorm.** The residual connection is applied first, then LayerNorm, matching the original paper. Pre-LayerNorm is often easier to train, but Post-LN combined with the Noam warmup converges well on Multi30k and keeps the implementation faithful to the paper.
- **No data leakage.** Vocabularies are built from the training split only and reused for validation and test.
- **Embedding scaling.** Token embeddings are multiplied by √d_model before positional encodings are added.

## Experiments & Findings

Ablations use a smaller model (d_model 256, 3 layers, d_ff 512, dropout 0.1, 30 epochs, seed 42) so that paired runs finish quickly. The full analysis, including caveats, is in [Findings.md](Findings.md).

| Topic | Key Finding |
|---|---|
| Learning-rate warmup | Noam reaches **28.8 BLEU** vs **25.5** and a lower training loss (2.05 vs 2.92); validation loss ends equal |
| Attention scaling | Unscaled Q/K gradients are larger and much noisier; no vanishing gradients at d_k = 32 |
| Head specialization | Several heads show distinct syntactic roles; pairwise similarity is low (0.16–0.44), so little redundancy |
| Positional encodings | Indistinguishable in-distribution (28.80 vs 28.77 BLEU); they differ in extrapolation |
| Label smoothing | Smoothing lowers mean gold-token probability (0.53 vs 0.58) with slightly higher BLEU (28.6 vs 27.8) |

## Reproducing the experiments

The experiment scripts import from the repository root, so run them from there with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python Experiments/Noam_scheduler_1.py        # Noam vs fixed lr   (saves checkpoint_best_noam_scheduler.pth)
PYTHONPATH=. python Experiments/Ablation_2.py              # 1/√d_k ablation
PYTHONPATH=. python Experiments/Head_specialization_3.py   # needs checkpoint_best_noam_scheduler.pth (--checkpoint to override)
PYTHONPATH=. python Experiments/Encodings_4.py             # sinusoidal vs learned PE
PYTHONPATH=. python Experiments/Decoder_sensitivity_5.py   # label smoothing ε = 0.1 vs 0.0
```

Run `Noam_scheduler_1.py` first, because the head-specialization script analyzes its checkpoint. All ablation runs share: d_model 256, 3 layers, 8 heads, d_ff 512, dropout 0.1, batch size 128, 4,000 warmup steps, seed 42, and 30 epochs (1,000 steps for the scaling ablation).


## Background

Built using concepts taught in the course *DA6401: Introduction to Deep Learning* (IIT Madras).

Part of a deep learning project series:
[MLP from Scratch](https://github.com/4prasid/MLP-from-scratch) · [Multi-task Vision](https://github.com/4prasid/multitask-vision-vgg11) · [Transformer NMT](https://github.com/4prasid/transformer-nmt-from-scratch)

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). *NeurIPS 2017*.
- Dataset: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)

## License

MIT, see [LICENSE](LICENSE).
