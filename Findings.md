# Findings

Written analysis of five controlled experiments on the Transformer in this repository. The interactive version, with all plots, is in the [W&B report](https://api.wandb.ai/links/prasid-indian-institute-of-technology-madras/pae80rij). This file is a durable copy of the analysis.

**Conventions**

- Every experiment uses a smaller model than the final one: d_model 256, 3 encoder/decoder layers, 8 heads, d_ff 512, dropout 0.1, batch size 128, 4,000 warmup steps, seed 42, trained on Multi30k German→English (29,000 training pairs).
- BLEU is corpus-level, computed with greedy decoding on the 1,014-sentence validation set every 2 epochs. 
- Every comparison is a **single run with one seed**. Differences of under about 1 BLEU should be read as suggestive, not conclusive.

---

## 1. The necessity of the Noam Scheduler

**Setup.** Two identical models trained for 30 epochs. One uses the Noam schedule (linear warmup for 4,000 steps, then inverse-square-root decay). The other uses a constant learning rate of 10⁻⁴ with no warmup. Everything else is identical.

**Results (final epoch).**

| Run | Train loss | Val loss | Val BLEU |
|---|---|---|---|
| Noam (4,000 warmup) | 2.049 | 3.2535 | **28.80** |
| Fixed lr = 1e-4 | 2.924 | 3.2537 | 25.49 |

**What the curves show**

- **Early epochs.** The fixed-lr run is ahead only for the first 2–3 epochs, which is the stretch where the Noam lr is still tiny. Noam's lr at step 1 is d_model⁻⁰·⁵ · 4000⁻¹·⁵ ≈ 2 × 10⁻⁷.
- **After the crossover.** Noam's training loss falls below the fixed run's by epoch ≈3–4 and stays lower to the end. Its validation BLEU leads from about epoch 4 and plateaus around 28–29 from epoch ≈16, while the fixed-lr BLEU is still rising slowly at epoch 30.
- **Validation loss.** The two runs end at the same value (3.2535 vs 3.2537), and they arrive differently. Noam's validation loss bottoms out around epoch ≈13–18 and drifts slightly upward while its training loss keeps falling. The fixed-lr run is still improving at epoch 30. So Noam learns faster, but the model has started to overfit slightly by the end, and the fixed run is still catching up rather than plateaued.

**Why 1e-4 isn't "aggressive" here.** With 29,000 pairs and batch size 128, one epoch is about 227 steps, so warmup lasts roughly 17–18 epochs and the peak lr is d_model⁻⁰·⁵ · 4000⁻⁰·⁵ ≈ 1 × 10⁻³ at step 4,000. The Noam lr passes 10⁻⁴ at around step 400, about epoch 2. For essentially the whole run, Noam's lr is **higher** than the fixed 10⁻⁴ (about 7.6 × 10⁻⁴ at the last epoch). The fixed-lr run was stable and simply too conservative. It did not diverge. Noam's advantage in this experiment is that it safely reaches a much larger learning rate.

**Why warmup matters for Transformers.**

- At initialization the query/key/value projections are random, so QKᵀ produces arbitrary attention patterns and the gradients flowing through the softmax are noisy and poorly directed.
- Adam's second-moment estimates are also unreliable in the first few hundred steps, so full-size updates early on can push the weights into a bad region.
- Warmup keeps the early steps small while these statistics settle and attention begins to carry signal, and then the schedule ramps up to a large lr.
- Post-LayerNorm architectures, as used here, are known to be especially sensitive to this.

---

## 2. Ablation: The Scaling Factor 

**Setup.** Two models (d_k = 32 per head) trained for 1,000 steps. Run A uses standard scaled attention softmax(QKᵀ/√d_k). Run B uses raw dot products. The gradient norms of the Query and Key weight matrices in the **first encoder layer's self-attention** were logged at every step. The Noam schedule is in its warmup phase for the whole run (lr ≈ 2.5 × 10⁻⁴ at step 1,000).

**What the curves show**

| | Scaled (√d_k) | Unscaled |
|---|---|---|
| Gradient norm at step 0 | ≈ 0.11 | ≈ 0.38–0.40 |
| Early dip | ≈ 0.05 near step 130 | ≈ 0.2 over steps 100–200 |
| Steps 600–1,000 | ≈ 0.3–0.35, smooth | ≈ 0.4 on average, spikes to ≈ 0.55 |
| Train loss at step 999 | 4.25 | 4.34 |
| Val loss after 1,000 steps | 4.08 | 4.17 |

Both W_Q and W_K show the same pattern. The unscaled run has larger gradient norms throughout, with the biggest gap at the start (about 3.5×) narrowing to about 1.3× by step 1,000, and its gradients are much more volatile from step to step.

**Relation to the "vanishing gradient" argument in §3.2.1 of the paper.** The paper argues that for large d_k the dot products grow in magnitude (their variance scales with d_k), pushing the softmax into saturated regions where its gradient is tiny. **We do not observe vanishing gradients** at d_k = 32. The unscaled run's Q/K gradients are *larger* and noisier. Two reasons this is consistent with the paper's argument rather than contradicting it:

1. The paper's argument is about *large* d_k, and the effect should grow with d_k. At d_k = 32 the dot-product standard deviation is about √32 ≈ 5.7, which is large enough to sharpen the softmax but not catastrophic.
2. Without the 1/√d_k factor, the gradient path to Q and K is also not divided by √d_k. That alone could inflate their gradients by up to about 5.7× relative to the scaled run, which may partly offset any reduction from saturation. Softmax saturation making different batches produce very different gradient magnitudes is a plausible explanation for the volatility. It is **not directly measured** here, and logging attention entropy or the maximum softmax probability per step would test it.

The scaled run shows a smooth, steady rise in gradient norm that follows the lr warmup. The practical benefit after 1,000 steps is a **modest** loss improvement (≈0.09 in validation loss) from a single seed.

**Caveat.** The norms are read *after* `clip_grad_norm_` (max norm 1.0) is applied to the whole model. If the total gradient norm exceeded 1.0 in some steps, the logged Q/K norms are rescaled, which would compress the true difference between the two runs.

---

## 3. Attention Rollout & Head Specialization

**Setup.** Attention weights from the **last encoder layer's self-attention** were captured with a forward hook on `MultiHeadAttention`, using the model from the Noam run (d_model 256, 3 layers, 8 heads). The source sentence is German: *"Ein Mann sitzt auf einer Bank im Park."* ("A man is sitting on a bench in the park."). One heatmap was logged per head, together with per-head attention entropy and a head-to-head cosine-similarity matrix.

This is single-layer attention from one sentence. It is not attention rollout across layers, and it is an illustration rather than a statistical study.

**Head behaviors** (attention weights in parentheses; tokens are `<sos> Ein Mann sitzt auf einer Bank im Park . <eos>`) 
| Head | Observed pattern | Interpretation |
|---|---|---|
| 0 | `.` → `.` (0.97), `<eos>` → `.` (1.00) | Sentence-boundary anchoring |
| 1 | `auf` → `Park` (0.86), `Bank` → `im` (0.77) | Long-range dependency (skipping 4 tokens) |
| 2 | `Mann` → `Ein` (0.97), `auf` → `einer` (0.81), `im` → `Mann` (0.76) | determiner head & mostly adjacent-token attention in both directions, often onto an article.|
| 3 | `<sos>` → `Park` (0.80), `sitzt` → `Ein` (0.73), `.` → `.` (0.71) | Mixed pattern with no clear single function |
| 4 | `<eos>` → `Mann` (0.87), `im` → `Park` (0.66); highest entropy (1.70) | Diffuse, global summary |
| 5 | `auf` → `sitzt` (0.99) | Preposition ↔ governing verb |
| 6 | `auf` → `Bank` (1.00), `<sos>` → `sitzt` (0.95) | Preposition ↔ its object |
| 7 | `im` → `Park` (0.91), `Ein` → `Park` (0.77) | Location-phrase cohesion |

**Head redundancy.** Pairwise cosine similarities between the flattened attention maps range from 0.16 to 0.44. The most similar pair is H3–H5 (0.44), followed by H3–H0 (0.42), and the least similar is H2–H7 (0.16). No pair approaches a conventional redundancy threshold of about 0.85, so **no head redundancy is observed** in this layer. Heads 5 and 6 are a clear illustration: both attend from `auf`, but one to the verb `sitzt` (0.99) and the other to the noun `Bank` (1.00), so they cover complementary syntactic relations for the same query token.

Entropy is fairly uniform across heads (≈1.45–1.70). This fits a small model on a small dataset: even so, Heads 0, 2, 5 and 6 show interpretable, sharply peaked patterns.

**Caveats.** The roles above are read from one sentence and describe tendencies, not guaranteed functions. Redundancy should be re-checked over many sentences, and across layers, before drawing general conclusions.

---

## 4. Positional Encoding vs. Learned Embeddings

**Setup.** Two models trained for 30 epochs. One uses the standard sinusoidal encoding. The other replaces it with `torch.nn.Embedding` (learned positions, table size 256 × d_model, separate for encoder and decoder).

**Results.**

| Run | Last logged val BLEU | Final val loss | Parameters |
|---|---|---|---|
| Sinusoidal | 28.80 | 3.2535 | 14,432,853 |
| Learned | 28.77 | 3.2539 | 14,563,925 |

The BLEU difference is 0.03, which is noise. The learned variant has 131,072 extra parameters (two tables of 256 × 256). The sinusoidal run reproduces the Noam run from the warmup experiment to every logged decimal (same seed, same configuration), which is a useful sanity check on the pipeline's determinism.

This agrees with Table 3 (row E) of the original paper, where learned and sinusoidal encodings gave nearly identical results.

**Why they match on Multi30k.** Multi30k sentences are short image captions, on the order of a dozen tokens. In that narrow range, a learned table has enough training signal for every position it will ever see, and the sinusoidal encoding supplies equivalent position information. Both reach the same ceiling.

**Extrapolation to longer sequences (theory).** They differ in what happens at inference time on sequences longer than any seen in training.

- **Sinusoidal.** PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(·) are closed-form functions defined for *any* integer position. The wavelengths form a geometric progression from about 6 positions up to about 60,000. Because of the angle-addition identities, PE(pos + k) is a fixed linear transformation (a rotation) of PE(pos) for every offset k. Attention can therefore learn to depend on **relative** distance, a relationship that does not rely on having seen that absolute position.
- **Learned.** A lookup table has no entry beyond its size. With MAX_LEN = 256, position 300 raises an index error. Even in an over-allocated table, rows for unseen positions never received gradient and stay at their random initialization, so the model has no basis for interpreting them.

**Caveat.** Sinusoidal encoding makes extrapolation *possible*, but the trained network is not guaranteed to extrapolate well, and quality often degrades on sequences much longer than the training ones. Extrapolation was **not tested** here. Training on short sentences and evaluating on longer ones would measure it.

---

## 5. Decoder Sensitivity: Label Smoothing

**Setup.** Two models trained for 30 epochs, with label smoothing ε = 0.1 (as in the paper) and ε = 0.0 (standard cross-entropy). "Prediction confidence" is the mean softmax probability assigned to the gold token over all non-pad positions of the validation set (teacher forced).

**Results.**

| ε | Prediction confidence (final) | Last logged val BLEU |
|---|---|---|
| 0.1 | 0.528 | **28.57** |
| 0.0 | **0.578** | 27.76 |

**What the curves show**

- **Confidence.** From epoch 2 onward the ε = 0.0 model assigns higher probability to the gold token, and the gap of about 0.05 persists until the end.
- **BLEU.** The two runs track each other closely until epoch ≈8–10. After that, ε = 0.1 is usually ahead, by up to about 2 BLEU around epochs 20–28. Both runs show dips (ε = 0.0 at epochs ≈16 and ≈22, ε = 0.1 at epoch ≈16), so the epoch-to-epoch noise is as large as the final 0.8 BLEU gap.
- **Losses are not comparable.** The reported training and validation losses of the two runs use different objectives (smoothed vs plain cross-entropy), so a lower loss for ε = 0.0 (2.57 vs 3.26 validation) says nothing about which model is better. This is also why smoothing raises "perplexity" computed from the training objective.

**How label smoothing regularizes.**

- With one-hot targets, cross-entropy is minimized only as p(gold) → 1, which requires the gold logit to exceed all others by an ever-growing margin. The model is rewarded for becoming arbitrarily certain, including on noisy or ambiguous examples. Natural language has many valid translations, so this encourages overfitting to the training references.
- With smoothing, the target puts 1 − ε = 0.9 on the gold token and spreads ε over the others. The loss is minimized at p(gold) ≈ 0.9, which corresponds to a **finite** logit margin. The model is structurally discouraged from extreme confidence and keeps some probability on plausible alternatives. This bounds the logits much as weight decay bounds weights, and it matches the trade-off in §5.4 of the paper: perplexity gets worse while accuracy and BLEU improve.

**Caveats on the "overconfidence" reading.** Mean gold-token probability is not a calibration measure. A model can assign higher p(gold) because it is more correct, not only because it is overconfident. Showing overconfidence needs confidence compared against accuracy (for example reliability diagrams or expected calibration error). Also, both models' validation confidence (≈0.53–0.58) is well below the 0.9 ceiling that smoothing imposes, so on held-out data the regularizing effect shows up only moderately. The results are consistent with smoothing reducing overconfidence and with a small BLEU gain, but one seed and a 0.8 BLEU gap do not establish either claim.

---
