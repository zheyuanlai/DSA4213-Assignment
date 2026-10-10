# PLAN.md — DSA4213 Assignment 1 (Stage 0 output)

Status: **approved by the user on 2026-10-01** with the changes marked "Changed by user" in NOTES.md (D28–D35), all folded in below. Numbers in Section 1 and
Section 7 are *measurements* taken on 2026-10-01 (scratch scripts, not committed); numbers in
Section 9 are *estimates* and are labelled as such. They will be replaced by measured values
in `results/` as the stages run.

---

## 1. Environment (measured 2026-10-01)

| Item | Value |
|---|---|
| Host OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-90-generic |
| GPUs | 4 × NVIDIA H200 NVL (143,771 MiB each), driver 580.105.08, CUDA 13.0 driver API |
| GPU sharing | **Shared node.** At check time GPU 0 (99 % util, 115 GB used), GPU 1 (95 %, 29 GB) and GPU 3 (83 %) were running other users' python jobs; **GPU 2 was idle (0 %, 539 MiB)**. |
| GPU policy | **Exactly one GPU** for every run (`CUDA_VISIBLE_DEVICES=<one id>`, GPU 2 at present). Timing runs record an `nvidia-smi` snapshot + load average so contention is visible. |
| CPU | 2 × AMD EPYC 9554 64-core (128 physical cores, 256 threads, 2 NUMA nodes) — shared |
| RAM | 1,511 GiB total, 1,313 GiB available |
| Disk | 471 GB free on `/` (NVMe) |
| Scheduler | None (no SLURM/PBS binaries; `SLURM_JOB_ID` unset). Jobs run directly; `scripts/*.sh` use `nohup`-style foreground commands, no `sbatch`. |
| Internet | huggingface.co HTTP 200; nlp.stanford.edu → 301 → `downloads.cs.stanford.edu/nlp/data/glove.6B.zip` HTTP 200 (862,182,613 bytes); pypi.org 200. Downloads can be done on this node. |
| Python env | fresh conda env `dsa4213`, **Python 3.11.15** (base is 3.13; 3.11 chosen for gensim/datasets wheel compatibility) |
| Libraries | torch 2.14.1+cu126 (CUDA 12.6 runtime, cuDNN 9.10.02), numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, gensim 4.4.0, datasets 5.0.1, matplotlib 3.11.2, PyYAML 6.0.3, tqdm 4.70.1, pytest 9.1.1 — pinned in `requirements.txt` |
| CUDA check | `torch.cuda.is_available()=True`, device H200 NVL (sm_90), 138.8 GiB free on GPU 2, `F.scaled_dot_product_attention` available |
| Precision | fp32 parameters/activations; `torch.backends.cuda.matmul.allow_tf32=True`, `cudnn.allow_tf32=True` for all neural models; no AMP, no `torch.compile` |

Activation: `source ~/miniconda3/bin/activate dsa4213` (or `conda activate dsa4213`).

---

## 2. Repository layout and file responsibilities

```
README.md  TASK.md  PLAN.md  NOTES.md  requirements.txt  .gitignore
configs/
  data.yaml                                  # paths, cleaning/tokeniser options, vocab size 25000, L=256, seeds
  part1/ngram.yaml                           # orders {3,4}, discount grid, selection on LM-val
  part1/{rnn,lstm,transformer}_seed{0,1,2}.yaml
  probes/{rnn,lstm,transformer}_lr{3e-4,1e-3,2e-3}.yaml   # 1-epoch LR probes, seed 0
  smoke/{rnn,lstm,transformer}.yaml          # 1 % of LM-train, 2 epochs
  smoke/lstm_{own,glove}frozen.yaml          # same 1 %/2-epoch setting with the frozen tables (exercises freeze/checksum path)
  smoke/ngram.yaml  smoke/word2vec.yaml      # n-gram selection and word2vec (epochs=1) on the 1 % stream
  smoke/part3.yaml                           # 500 train / 100 val / 200 test reviews through features + classifier + exports
  part2/word2vec.yaml   part2/glove.yaml
  part2/lstm_ownfrozen_seed{0,1,2}.yaml   part2/lstm_glovefrozen_seed{0,1,2}.yaml
  part3/classify.yaml                        # classifier split, truncation, C grid, bootstrap
src/
  common.py        # seeding, YAML config loading, paths, JSONL logger, env capture; generic `limit_reviews` subset
                   # option (first N reviews of each split) honoured by train_lm.py, ngram.py, word2vec.py, features.py, classify.py
  data.py          # HF download, cleaning (Sec. 2), de-dup, LM splits (seed 0), IMDb classifier split
  tokenize.py      # deterministic regex PTB-style tokeniser, clitic splitting, 5 raw-vs-tokenised examples
  vocab.py         # vocab from LM-train (25k + 3 specials), <unk> rates, stream encoding → data/streams/*.npy
  ngram.py         # interpolated / modified Kneser–Ney on numpy sorted-key tables; block scoring; dense P(.|h); sampling
  models.py        # RNNLM, LSTMLM, TransformerLM; parameter breakdown; embedding init/load/freeze
  train_lm.py      # config-driven training loop (AdamW groups, warmup+cosine, clipping, early stopping, JSONL, ckpt)
  evaluate.py      # THE common protocol: blocks, masked CE/PPL, unk fraction, loss-by-position, frequency buckets
  generate.py      # greedy / T=0.8 top-k 50 for neural + n-gram; generations.md; distinct-1/2, rep-4
  timing.py        # training-time summary, workload A (scoring) and B (generation) timing; results/env.json
  word2vec.py      # gensim skip-gram on the vocab-mapped LM-train stream → embedding table (.npy) + timing
  glove.py         # load glove.6B.300d, coverage report, seeded init of missing words → table (.npy)
  features.py      # frozen seed-0 LMs → last / mean features for IMDb train/test (first-256 setting)
  classify.py      # StandardScaler+LogReg(lbfgs), C selection on classifier-val, baselines, bootstrap CIs, error exports
  analysis/
    plots.py         # shared matplotlib style; save PDF+PNG at half-page width
    part1_tables.py  # main_table.csv, loss curves per model, combined val curves, loss-by-position figure
    part2_tables.py  # Part II table, stability stats, curves, frequency-bucket figure, nearest neighbours
    part3_tables.py  # Part III tables, LM-vs-downstream table, length-bucket accuracy
scripts/
  prepare_data.sh  run_part1.sh  run_part2.sh  run_part3.sh   # exact reproduction commands (no sbatch: no SLURM)
tests/
  test_causality.py  test_ngram_normalisation.py  test_padding.py  test_overfit.py
results/                       # committed, small files only
  env.json  SUMMARY.md
  runs/<run_name>/log.jsonl    # per-run JSONL logs (config+env record, 50-step records, epoch records) ≈ tens of KB each
  runs/<run_name>/steps.npz    # per-step train loss and pre-clip grad norm (≈ 9k floats) for spike statistics
  part1/data_stats.json        # de-dup counts, split sizes (reviews/tokens), |V|, <unk> rates on the five splits
  part1/ngram_selection.csv    # val CE of all 14 candidates (+ count-of-counts and discounts per level)
  part1/ngram_model.json       # chosen order/discounts, n-grams per order, nbytes, build time, LM-test record
  part3/input_stats.json       # % truncated and <unk> rate of the Part III inputs
  part3/best_pair.json         # val-based ranking and the two representations used for error exports
  part1/  part2/  part3/  figures/
data/                          # git-ignored: raw HF cache, cleaned reviews, streams, GloVe, derived arrays
checkpoints/                   # git-ignored: best-val checkpoints (*.pt)
```

`src/common.py` is the only file not named in TASK.md §7; it holds shared helpers so the named
modules stay single-purpose.

---

## 3. Data pipeline (Stage 1)

1. **Download** `stanfordnlp/imdb` with `datasets.load_dataset` into `data/hf_cache/` (train 25k,
   test 25k, unsupervised 50k). README gives the command; data is never committed.
2. **Cleaning** (exact order of TASK §2): `<br />|<br/>|<br>` → space; `html.unescape`; character
   normalisation (NOTES D41): C1 controls re-interpreted as Windows-1252 (\x91 \x92 → `'`,
   \x93 \x94 → `"`, \x96 → `-`, \x97 → `--`, \x85 → `...`), curly/angle quotes, prime, back-tick
   and the standalone acute accent → `'`, curly/low double quotes and the standalone diaeresis →
   `"`, Unicode hyphens/en dash → `-`, em dash → `--`, ellipsis → `...`, remaining control and
   zero-width characters removed; lowercase; collapse whitespace. The cleaned string is the
   "normalised text" used for de-duplication. Other HTML tags are not stripped (0.023 % of
   reviews, NOTES D44).
3. **Tokeniser** (`tokenize.py`, deterministic regex, PTB/GloVe-style; 14 edge cases in
   `tests/test_tokenize.py`):
   - clitics first: `(\w)n't\b → \1 n't`; `(\w)'(s|re|ve|ll|d|m)\b → \1 '\2` (so `can't → ca n't`,
     `it's → it 's`, as in PTB/GloVe);
   - token regex, in priority order: clitic tokens | dotted acronyms `(?:letter\.){2,}` (h.p.,
     u.s., e.g.) | titles `mr. mrs. ms. dr. jr. sr. st. vs.` | numbers with internal separators
     `\d+(?:[.,:/]\d+)+` | Unicode-aware words `[^\W_]+(?:[-'][^\W_]+)*` (internal hyphens and
     apostrophes kept: `well-made`, `o'clock`, `café`) | `...` | `--` | any other single non-space
     character;
   - print 5 random reviews (seed 0) raw vs tokenised to `results/part1/tokenisation_examples.md`.
4. **De-duplication** by cleaned text: remove from `unsupervised` any review whose cleaned text
   occurs in labelled train ∪ test; drop exact duplicates within `unsupervised` (keep first);
   report all counts plus the number of cleaned-text duplicates shared between labelled train and
   test (labelled splits are not modified).
5. **LM splits**: shuffle de-duplicated unsupervised reviews with `numpy.random.default_rng(0)`;
   first 2,000 → LM-val, next 2,000 → LM-test, rest → LM-train. Split by review.
6. **Vocabulary** from LM-train only: 25,000 most frequent tokens (ties broken alphabetically for
   determinism); ids: `<pad>`=0, `<unk>`=1, `<eos>`=2, words 3…25,002 → **|V| = 25,003**.
   Report token-level `<unk>` rate on LM-train, LM-val, LM-test, IMDb-train, IMDb-test.
7. **Streams**: `<eos>`, then each review followed by `<eos>`; saved as `int32 .npy`
   (`data/streams/{lm_train,lm_val,lm_test}.npy`), plus per-review token-id lists for IMDb
   train/test (`data/streams/imdb_{train,test}.pkl`) so nothing is re-tokenised later.

All Stage-1 statistics are written by `data.py`/`vocab.py` to `results/part1/data_stats.json`
(raw HF split sizes; unsupervised reviews removed because they appear in labelled train/test;
exact duplicates removed within unsupervised; cleaned-text duplicates shared between labelled
train and test; LM-train/val/test sizes in reviews and tokens incl. `<eos>`; |V|; token-level
`<unk>` rate on LM-train, LM-val, LM-test, IMDb-train, IMDb-test). The Stage-1 stop-point report
quotes this file plus the 5 tokenisation examples.

---

## 4. Common evaluation protocol (`src/evaluate.py`, Stage 2)

- `make_blocks(stream, L=256, pad=0)` → `inputs[N,L]`, `targets[N,L]` exactly as TASK §3
  (`inputs = s[kL:kL+L]`, `targets = s[kL+1:kL+L+1]`, last block right-padded with `<pad>`).
  Number of blocks N = ceil((len(s) − 1)/L).
- `score(logprob_fn, inputs, targets, batch=64)` → per-token CE array `[N,L]` (NaN at pad
  targets). `logprob_fn(inputs) → log P(targets | inputs[0..j])` is supplied by each model:
  neural = `log_softmax(logits).gather(targets)` with hidden state reset per block (stateless
  across blocks); n-gram = `ngram.logprob_blocks`.
- Reported: mean CE (nats) over non-pad targets, PPL = exp(CE), fraction of `<unk>` targets,
  per-position mean CE (positions 1…256), CE by LM-train frequency bucket of the target.
- Loss masking everywhere uses `ignore_index=0` / explicit masks; pad never enters an average.
- Per-token CE arrays for LM-test are cached under `data/derived/` (git-ignored) so loss-by-position
  and frequency-bucket analyses re-use the single evaluation pass.

---

## 5. n-gram model (`src/ngram.py`, Stage 2)

- **Interpolated Kneser–Ney**, implemented here. Counts are stored as **sorted `int64` keys**
  (n-gram packed base-|V|; a 4-gram needs 4 × 15 bits) with `int32` counts; lookups via
  `np.searchsorted`; per-history aggregates (`c(h)`, `N1+(h•)`, and `N1(h•)`, `N2(h•)`, `N3+(h•)`
  for modified KN) stored in parallel sorted arrays keyed by history. Lower orders use
  **continuation counts** `N1+(•h w)`; the unigram level is `N1+(•w)/N1+(••)` discounted and
  interpolated with the uniform distribution `1/|V|`, so every level is exactly normalised and no
  probability is zero.
- **Short contexts** (block positions `j < n−1`): use the last `min(n−1, j+1)` input tokens and
  evaluate the KN recursion starting at order `j+2` — i.e. the model's own lower-order
  (continuation-count) estimate, as in SRILM/KenLM. Nothing outside the block is used.
- **Candidates** (all scored on LM-val, every val CE recorded in `results/part1/ngram_selection.csv`):
  orders {3, 4} × discounts { single D ∈ {0.5, 0.6, 0.7, 0.75, 0.8, 0.9} applied at every level,
  plus "modified KN" with count-of-count-estimated (D1, D2, D3+) per level } = 14 candidates.
  Modified-KN discounts are the Chen–Goodman (1998) estimates computed separately at each level
  from that level's own stored table (raw counts at the top order, continuation counts below):
  Y = n1/(n1+2n2), D1 = 1−2Y·n2/n1, D2 = 2−3Y·n3/n2, D3+ = 3−4Y·n4/n3, with back-off weight
  γ(h) = (D1·N1(h•) + D2·N2(h•) + D3+·N3+(h•))/c(h). If at some level any n_k (k ≤ 4) is 0 or an
  estimate is negative (expected at the continuation-unigram level), that level falls back to
  D1, D2, D3+ = 0.5, 1.0, 1.5 (KenLM's fallback defaults); the per-level n1…n4, the discounts used
  and any fallback are recorded in `results/part1/ngram_selection.csv`.
  The best val CE picks order and discount; LM-test is scored once with that choice.
- **Outputs**: `ngram.py` times the count build (wall-clock, CPU, single process; validation
  scorings excluded) and writes `results/part1/ngram_model.json`: selected order/discounts,
  distinct n-grams stored per order, total nbytes, build seconds, LM-train tokens, CPU model, and
  the LM-test record (CE, PPL, number of targets, `<unk>`-target fraction).
- **Model size**: number of stored n-grams per order (distinct n-grams of each order) and memory
  footprint = sum of `nbytes` of all arrays.
- **Dense distribution** `P(·|h)` over |V| for sampling and the normalisation test: start from the
  dense unigram vector and, for each higher order, add the sparse discounted term (contiguous
  key range for history h) and scale by the back-off weight — O(|V|) numpy work per step.
- **Sampling/greedy** uses that dense vector with `<pad>`/`<unk>` masked, temperature 0.8,
  top-k 50 (identical to the neural decoding; §4.5).
- **Workload A on CPU**: single process, `OMP_NUM_THREADS=1`; vectorised scoring of all LM-test
  blocks (keys for every position computed at once, order chosen per position).

Test (b) sums the dense vector for 20 random histories of each length 0…n−1 (tolerance 1e-6).
Test (c) runs for the n-gram as well as the neural models.

---

## 6. Neural models (`src/models.py`, Stage 3)

Shared: `nn.Embedding(|V|, 300)` (trainable, random init) → core → **300-d pre-head vector** →
`nn.Linear(300, |V|)` with bias (identical head for all three models); **no weight tying**. For
RNN/LSTM the pre-head vector is `Linear(H, 300)` (bias, no nonlinearity) applied to the dropped-out
top-layer output; for the Transformer it is the final-LayerNorm output. Loss
`F.cross_entropy(..., ignore_index=0)`. (Changed by user, D29.)

| Model | Core | Dropout | Init |
|---|---|---|---|
| RNN | `nn.RNN(300, 1072, num_layers=2, nonlinearity='tanh', dropout=0.3, batch_first=True)` → `Linear(1072, 300)` | 0.3 on embedding output, between layers, and on the top-layer output before the projection | embedding U(−0.1, 0.1); RNN/projection/head PyTorch defaults |
| LSTM | `nn.LSTM(300, 512, num_layers=2, dropout=0.3, batch_first=True)` → `Linear(512, 300)` | same as RNN | same as RNN |
| Transformer | GPT-style decoder written here: 4 pre-LN blocks, d_model 300, 6 heads (head dim 50), FFN 1200 + GELU, learned absolute positions (256), final LayerNorm, `F.scaled_dot_product_attention(is_causal=True)` | **0.1** on embeddings, attention weights, residual branches | N(0, 0.02) for Linear/Embedding weights, zero biases, residual projections scaled by 1/√(2·4) (GPT-2) |

Hidden state is zero at the start of every block (no state carried across blocks) in training
and evaluation; in generation the RNN/LSTM carry state across the generated tokens (equivalent to
re-reading the prefix), while the Transformer re-encodes the full prefix each step.

### 6.1 Parameter counts (measured with `torch` module instantiation, |V| = 25,003, d = 300) — final configs

| Model | Embedding | Core | Projection | Head | Total (= trainable in Part I) |
|---|---:|---:|---:|---:|---:|
| RNN tanh, 2 L, H = 1072 | 7,500,900 | 3,773,440 | 321,900 | 7,525,903 | 19,122,143 |
| LSTM, 2 L, H = 512 | 7,500,900 | 3,768,320 | 153,900 | 7,525,903 | 18,949,023 |
| Transformer, 4 L, d 300 (core incl. 76,800 positional + 600 final-LN) | 7,500,900 | 4,413,000 | 0 | 7,525,903 | 19,439,803 |

Measured step times with these configs (batch 64 × 256, fp32+TF32, GPU 2): RNN 29.0 ms, LSTM
27.6 ms, Transformer 25.9 ms per training step; eval forward 10.1 / 8.1 / 7.4 ms per batch;
peak memory 5.3 / 5.5 / 7.3 GiB.

Earlier candidates without the projection (kept for the record):

| Model | Embedding | Core | Head | Total (= trainable in Part I) |
|---|---:|---:|---:|---:|
| LSTM, 2 L, H = 512 | 7,500,900 | 3,768,320 | 12,826,539 | 24,095,759 |
| Transformer, 4 L, d 300, 6 h, FFN 1200 (core incl. 76,800 positional + 600 final-LN) | 7,500,900 | 4,413,000 | 7,525,903 | 19,439,803 |
| RNN tanh, 2 L, H = 768 | 7,500,900 | 2,002,944 | 19,227,307 | 28,731,151 |
| **RNN tanh, 2 L, H = 1024** | 7,500,900 | **3,457,024** | 25,628,075 | 36,585,999 |
| **RNN tanh, 2 L, H = 1072** | 7,500,900 | **3,773,440** | 26,828,219 | 38,102,559 |
| RNN tanh, 2 L, H = 1152 | 7,500,900 | 4,331,520 | 28,828,459 | 40,660,879 |

**Decisions (user, 2026-10-01):** `H_rnn = 1072` (RNN core matches the LSTM core within
0.14 %; both ≈ 15 % below the spec-fixed Transformer core). A `Linear(H, 300)` projection (bias,
no nonlinearity) sits between the dropped-out top recurrent layer and the head for RNN and LSTM,
so all three models share an identical 300 × |V| head and totals are within 2.6 % of each other.
The projection is reported as its own column in every parameter breakdown.

---

## 7. Training procedure (`src/train_lm.py`), identical for all neural runs

- **Data**: each epoch draws offset `o ~ U{0..255}` (seeded), cuts LM-train into non-overlapping
  blocks `s[o+kL : o+kL+L]` with targets shifted by one, drops the final incomplete block (no pad
  in training), shuffles block order, batches 64 blocks.
- **Optimiser**: AdamW(β = (0.9, 0.999), ε = 1e-8), peak LR per family from the probe, weight
  decay 0.01 on parameters with `ndim ≥ 2` (embedding table, RNN/LSTM weight matrices, Linear
  weights, positional table), 0 on biases and LayerNorm parameters.
- **Schedule**: linear warm-up over 300 steps to peak, then cosine decay to 10 % of peak at the
  planned horizon = 12 epochs × steps/epoch (early stopping may end the run earlier).
- **Clipping**: `clip_grad_norm_(…, 1.0)`; the returned pre-clip total norm is logged every step
  (saved to `steps.npz`) and every 50 steps (JSONL).
- **LR probe**: for each family, seeds 0, one epoch with LR ∈ {3e-4, 1e-3, 2e-3} following the
  *first epoch of the real 12-epoch schedule* (warm-up + start of cosine), scored on LM-val;
  results to `results/part1/lr_probes.csv`; the best LR is written into the configs.
- **LR fallback rule (fixed in advance, user D31)**: if any full run hits a NaN/inf loss, training
  stops and the failure is reported to the user; the fix is to rerun that model family with the
  next-lower LR from the grid for all three seeds, keeping the failed logs. LRs are not changed for
  any other reason.
- **Epochs / stopping**: max 12; LM-val CE after every epoch; early stopping with patience 3 on
  val CE; the best-val checkpoint is kept (`checkpoints/<run>/best.pt`) and used for all test
  evaluation, timing, generation and Part III features.
- **Seeds** {0, 1, 2}: `torch`, `numpy`, `random` seeded; the seed also drives the per-epoch
  offset and block shuffle. Seed 0 runs first. cuDNN kernels are not bit-deterministic (README).
- **Logging** (JSONL `results/runs/<run>/log.jsonl`): record 0 = resolved config + environment
  info; every 50 steps: step, epoch, mean train loss over the last 50 steps, LR, pre-clip grad
  norm (this step and mean of the last 50), elapsed *training* time (validation excluded), tokens
  seen; every epoch: val CE/PPL, epoch train time, cumulative train time, best-so-far flag;
  one final `test` record written when `evaluate.py` scores `best.pt` on LM-test (CE, PPL,
  number of targets, `<unk>`-target fraction, best epoch), so every reported test number has a
  committed primary record.
- **Timing of training** (`timing.py` reads the logs): total train time, per-epoch time, time up
  to the best epoch, tokens/s, epochs run. GPU timers use `torch.cuda.synchronize()`.
- **Smoke tests** before any full run (TASK rule 6 applies to *every* pipeline): the generic
  `limit_reviews` option restricts each split to its first N reviews. `configs/smoke/*.yaml` cover
  the three Part I models (1 % of LM-train, 2 epochs), the two frozen-embedding LSTM variants
  (same setting, exercising the freeze/checksum/optimiser-group code), n-gram selection and
  word2vec (epochs=1) on the 1 % stream, and the whole Part III pipeline (500/100/200 reviews,
  one LM, both poolings, full C grid, bootstrap, error exports). GloVe loading has no heavy run.

Run names: `part1/{rnn,lstm,transformer}_s{0,1,2}`, `probes/{model}_lr{…}`,
`part2/lstm_{own,glove}frozen_s{0,1,2}`, `smoke/{model}`.

---

## 8. Part I measurements and outputs (Stage 4)

- **Workload A (scoring)**: entire LM-test, batch 64 blocks, 2 warm-ups then 5 timed repeats,
  `torch.cuda.synchronize()` before each timer read; tokens/s = non-pad target tokens / time;
  median reported. N-gram: same blocks, CPU, single process (`OMP_NUM_THREADS=1`), CPU model
  recorded.
- **Workload B (generation)**: batch 1, the 5 prompts, 60 new tokens, greedy decoding for timing;
  per-step times (synchronised) pooled over the 300 steps → median ms/token. Transformer
  recomputes the full prefix (no KV cache); RNN/LSTM carry state; n-gram on CPU.
- **Generation** (`generate.py`): prompts prefixed with `<eos>`, all prompt tokens checked to be
  in-vocabulary; greedy and sampling (T = 0.8, top-k 50); exactly 60 new tokens (no early stop at
  `<eos>`; `<eos>` is rendered as `⏎`); `<pad>` and `<unk>` logits set to −∞ (n-gram: probability
  zeroed and renormalised) for every model (README states this).
  `results/part1/generations.md`: per prompt, every model's greedy output and samples with
  sampling seeds 0 and 1, side by side.
  Quantitative: 5 prompts × 20 samples (sampling seeds 0…19) per model → corpus-level distinct-1
  and distinct-2 over all generated tokens, and the mean within-sample repeated-4-gram rate
  (fraction of a sample's 4-grams that already occurred earlier in the same sample); the same for
  the 5 greedy outputs. → `results/part1/generation_metrics.csv`.
- **Which checkpoint feeds what**: everything that needs a single checkpoint per model —
  workload A, workload B, `generations.md`, `generation_metrics.csv` and the Part III features —
  uses the **seed-0** best-val checkpoint (the n-gram is a single deterministic model). Test
  CE/PPL (mean ± std), the train/val curve figures, the loss-by-position figure and the Part II
  frequency-bucket figure use **all three seeds** (curves = mean over the three seeds' per-token
  LM-test CE, which is already computed for the test score). The seed is never chosen by LM-test
  CE. The checkpoint/seed used is named in a header line of `generations.md` and in a column of
  every CSV.
- **Loss by position**: mean LM-test CE at positions 1…256 for all four models (mean over seeds
  for neural models), one figure; plotted values dumped to `results/part1/loss_by_position.csv`.
- **Outputs**: `results/part1/main_table.csv`, one row per model with columns: `params_embedding,
  params_core, params_head, params_total, params_trainable, memory_mb, test_ce_mean, test_ce_std,
  test_ce_s0, test_ce_s1, test_ce_s2, test_ppl_mean, test_ppl_std, unk_target_fraction,
  best_epoch_s0/s1/s2, epochs_run_s0/s1/s2, train_time_s_s0/s1/s2 (to best epoch and total),
  train_tokens_per_s, workloadA_tokens_per_s (seed 0), workloadB_ms_per_token (seed 0), device`.
  n-gram row: embedding/core/head = NA, `params_total` = stored n-grams summed over orders, extra
  column `ngrams_per_order` ("1:…;2:…;3:…;4:…"), `memory_mb` from `ngram_model.json` (neural rows:
  4 bytes × params), std = NA (single deterministic run), best epoch = NA, training time = count
  build seconds, workload A/B measured on CPU. Figures: per-model train/val curves (all seeds),
  combined val-curve figure, loss-by-position figure (`results/figures/*.pdf|png`).

---

## 9. Part II plan (Stage 5)

- **A** = Part I LSTM runs; `part2_tables.py` asserts the Part I LSTM configs equal the Part II
  base config except the embedding fields.
- **B (own-frozen)**: `word2vec.py` trains gensim `Word2Vec(vector_size=300, window=5,
  negative=10, sg=1, epochs=5, min_count=1, seed=0, workers=16)` on the **vocab-mapped LM-train
  stream**, one review (ending in `<eos>`) per sentence, tokens as id-strings; `<unk>` and `<eos>`
  therefore get vectors; `<pad>` (never observed) is set to zeros. Wall-clock training time and
  worker count recorded separately in `results/part2/word2vec_training.json`; README notes
  multi-worker gensim is not bit-reproducible.
- **C (GloVe-frozen)**: `glove.py` reads `glove.6B.300d.txt` (download command in README; time not
  counted); coverage as % of vocabulary types and % of LM-train tokens; 30 most frequent uncovered
  words; missing words ~ N(0, σ²) with σ = per-dimension std of covered vectors
  (`default_rng(0)`); `<pad>` = 0; `<unk>` and `<eos>` are *not* GloVe words → initialised like
  other missing words (documented); also reports per-dimension std of each table.
- **Freezing**: `embedding.weight.requires_grad=False`, table excluded from the optimiser; SHA-256
  of the table bytes before training and after loading the best checkpoint are logged and must
  match; trainable counts reported.
- **Outputs**: `results/part2/main_table.csv` with the same schema as Part I (settings A, B, C;
  `params_trainable` differs; test CE/PPL mean ± std, per-seed CE, best epoch, training time, read
  from the per-run `test` records; plus the per-dimension std of each embedding table: A at init
  and after training (per seed), B, C — user D34); stability (across-seed
  std of test CE, max and mean pre-clip grad norm over all steps, number of spikes — **spike =
  per-step train loss > 1.5 × trailing mean of the previous 50 steps, counted from step 51**);
  val/train curves for all settings and seeds; generations (§4.5 protocol); frequency-bucket CE
  figure (buckets by LM-train frequency rank of the target: 1–100, 101–1k, 1k–5k, 5k–25k, and
  `<unk>`; `<eos>` ranks by its own count; mean over the three seed checkpoints per setting, values
  dumped to `results/part2/freq_buckets.csv`) comparing A, B, C; nearest-neighbour sanity check for
  "good", "bad", "boring", "actor", "plot" in both tables.

---

## 10. Part III plan (Stage 6)

- **Data**: `train_test_split(stratify=label, test_size=2500, random_state=0)` → 22,500 /
  2,500; IMDb test used once. Same cleaning/tokeniser/vocab.
- **Input**: `<eos>` + the first **255** review tokens = 256 positions (the Transformer's maximum
  context); truncated flag if the review has > 255 tokens (Q3). `features.py` writes
  `results/part3/input_stats.json` (number and % of reviews truncated, token-level `<unk>` rate of
  the inputs, for IMDb train and test); `part3_tables.py` includes them in the Part III tables.
- **Features** (`features.py`): seed-0 best checkpoints, `eval()`, `no_grad`; the **300-d pre-head
  vector** of every model (RNN/LSTM: projection output; Transformer: final-LayerNorm output; user
  D30), so all feature sets have the same dimension; `last` = state at the last
  non-pad position; `mean` = masked mean over non-pad positions → 5 LMs × 2 poolings = 10 sets,
  cached as `.npy` under `data/derived/`.
- **Classifier**: `Pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=2000,
  C=c))`, C ∈ {0.01, 0.1, 1, 10} chosen by classifier-val accuracy; the fit on 22,500 with the
  chosen C is evaluated once on test.
- **Baselines**: (i) `Pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=100000),
  LogisticRegression(solver='lbfgs', max_iter=2000, C=c))` on the same `<eos>`+first-255 vocab-mapped
  token sequences — **no StandardScaler** for TF-IDF because its rows are already L2-normalised
  (user D32; documented in README); for each C the whole pipeline (vocabulary, IDF, top-100k cut,
  classifier) is fit on the 22,500 classifier-train reviews only, and classifier-val and IMDb-test
  are only transformed. lbfgs accepts sparse input; any non-convergence within 2,000 iterations is
  reported, not hidden.
  (ii) masked mean of the frozen GloVe table (the Part II C table) over the same inputs.
- **Metrics**: test accuracy and macro-F1 with 95 % bootstrap CIs (1,000 resamples, `default_rng(0)`).
  Robustness check (user D46): the same metrics on the IMDb-test subset excluding reviews whose
  cleaned text appears in IMDb-train; official splits unchanged.
- **Tables**: all 12 feature sets; LM-vs-downstream table with, per LM, the **seed-0** LM-test PPL
  (the checkpoint the features come from) and the 3-seed mean ± std PPL as an extra column, next
  to downstream accuracy for both poolings (user D33).
- **Error exports**: the "two best LM representations" are the two of the 10 LM feature sets
  (baselines excluded) with the highest **classifier-val accuracy** at their selected C (already
  computed during C selection, so IMDb-test is never consulted); ties by classifier-val macro-F1,
  then table order. A = the higher-ranked, B = the other; the ranking and the pair go to
  `results/part3/best_pair.json` and the first line of the markdown sample file. CSV for that
  pair (label, both probabilities, review length in tokens, truncated flag); markdown samples of
  15 reviews (first 120 words) for
  both-wrong / A-right-B-wrong / B-right-A-wrong; accuracy by length bucket (≤128, 129–256, >256
  tokens) for all 10 LM feature sets.

---

## 11. Tests (`pytest tests/`)

| Test | Design |
|---|---|
| (a) causality | each neural model, eval mode, fp32 CPU (TF32 off): random batch, random corruption of positions > j for several j; `allclose(logits[:, :j+1], atol=1e-5)` |
| (b) normalisation | KN model trained on a synthetic Zipf stream (small |V|) and, if data exists, on LM-train; 20 random histories of each length 0…n−1; `|ΣP − 1| < 1e-6` and `min P ≥ 0` (so a negative discount cannot pass) |
| (c) padding | a sequence of length T < 256 evaluated unpadded vs right-padded to 256 (neural and n-gram): masked CE identical within 1e-5 |
| (d) overfit | each neural model built with every dropout = 0 (architecture otherwise as §6); one fixed seeded batch of 4 blocks × 64 tokens; Adam(lr 1e-3, betas 0.9/0.999), no weight decay, clipping or schedule; full-batch steps on the GPU (CPU fallback: 2 blocks × 32 tokens), up to 2,000 steps, stop as soon as loss < 0.5 nats; assert it happened and that the step-1 loss ≈ ln 25,003 ≈ 10.1 |

---

## 12. Runtime estimates (ESTIMATES — to be replaced by measurements)

Measured micro-benchmark with the final configs (random tokens, batch 64 × 256, fp32+TF32, GPU 2,
one fwd+bwd+AdamW step): RNN-1072+proj 29 ms, LSTM-512+proj 28 ms, Transformer 26 ms; eval
forward per batch 10 / 8 / 7 ms; peak memory 5.3–7.3 GiB.

Data-size assumption (to be measured in Stage 1): ≈ 48–49k unsupervised reviews after de-dup,
≈ 270 tokens/review after punctuation splitting → LM-train ≈ 12–13 M tokens ≈ 760–810
steps/epoch at 64 × 256 tokens.

| Item | Estimate |
|---|---|
| Data download + cleaning + tokenising 100k reviews | 5–10 min (pure Python regex) |
| KN counting (orders ≤ 4) + 14 val scorings | < 5 min, < 1 GB RAM |
| One epoch of training (train only) | RNN ≈ 23 s, LSTM ≈ 22 s, Transformer ≈ 21 s; val pass ≈ 1 s |
| One full run (≤ 12 epochs) | 4–8 min |
| 9 LR probes (1 epoch each) | ≈ 5 min |
| Part I: 9 full runs | ≈ 1 h |
| Part II: word2vec (12.5 M tokens × 5 epochs, 16 workers) | ≈ 3–10 min (CPU) |
| Part II: 6 frozen-embedding LSTM runs | ≈ 40 min |
| Timing workloads, generation (5 × 23 × 60 tokens per model incl. n-gram) | ≈ 10 min |
| Part III feature extraction (50k reviews × 5 LMs) + 12 classifier selections + bootstrap | ≈ 15–30 min (TF-IDF lbfgs is the slowest) |
| **Total compute** | **≈ 3 h on one GPU + CPU**, well within budget |

---

## 13. Decisions taken in this plan (all logged in NOTES.md; tell me if you disagree)

D1 One GPU only (`CUDA_VISIBLE_DEVICES`), currently GPU 2. D2 Python 3.11 env, torch cu126 wheel.
D3 Weight decay on all `ndim ≥ 2` parameters incl. embedding and head weights.
D4 Transformer dropout 0.1. D5 RNN/LSTM dropout 0.3 on embedding output, between layers and
before the head. D6 Final incomplete training block dropped each epoch (eval blocks are padded).
D7 Short-context n-gram estimates = KN lower-order (continuation) recursion. D8 KN unigram level
interpolated with uniform. D9 Discount grid as in §5. D10 LR probe = first epoch of the real
schedule. D11 Tokens/s in workload A counts non-pad target tokens. D12 Generation does not stop
at `<eos>`; timing workload B uses greedy decoding. D13 Distinct-n is corpus-level over all
samples of a model; rep-4 is within-sample, averaged. D14 Spike definition (§9).
D15 gensim `workers=16`. D16 Per-step loss/grad-norm arrays saved per run for exact statistics.
D17 Run logs committed under `results/runs/`; heavy arrays under `data/derived/` (ignored).
D21 Seed-0 checkpoints feed workloads A/B, generations and Part III; loss-by-position and
frequency-bucket curves average the three seeds; seeds are never chosen on LM-test.
D22 Part III "two best" representations chosen by classifier-val accuracy (ties: val macro-F1,
table order). D23 TF-IDF vectoriser inside the Pipeline, fit on the 22,500 only.
D24 Modified-KN discounts per Chen–Goodman per level with the KenLM fallback (0.5/1.0/1.5).
D25 Overfit test fixture as in §11 (dropout 0, Adam 1e-3, ≤ 2,000 steps). D26 Every pipeline has
a `limit_reviews` smoke config; `data_stats.json`, `input_stats.json`, `ngram_model.json` and
per-run `test` records make every reported number traceable to a committed file.

User decisions (2026-10-01, NOTES D28–D35): H_rnn = 1072; Linear(H,300) projection before a shared
300 × |V| head for RNN/LSTM; Part III input = `<eos>` + first 255 tokens; Part III features = the
300-d pre-head vector; fixed LR fallback rule; TF-IDF baseline without StandardScaler; LM-vs-
downstream table uses seed-0 PPL plus 3-seed mean ± std; Part II table reports per-dimension std
of each embedding table; no multi-agent review panels unless asked — a single self-check against
TASK.md at each stop point.

## 14. Questions (resolved 2026-10-01)

- **Q1** H_rnn → 1072. **Q2** → add the `Linear(H, 300)` projection (see §6). **Q3** → `<eos>` +
  first 255 tokens confirmed. **Q4** → D1–D27 kept, with additions D31–D35.

## 15. Stage schedule and commits

| Stage | Commit message | Stop-point deliverables |
|---|---|---|
| 0 | `Stage 0: environment check, plan, pinned env` | this file, env summary, Q1–Q4 |
| 1 | `Stage 1: data pipeline, vocab, streams` | `data_stats.json` (de-dup counts, split sizes, unk rates), 5 examples |
| 2 | `Stage 2: evaluation protocol, KN n-gram, tests b/c` | smoke on the 1 % stream first; then n-gram val CE table, test results |
| 3 | `Stage 3: neural models, tests a/d, LR probes, smoke` | parameter table, probe CSV — wait for approval |
| 4 | `Stage 4: Part I training, timing, generation` | main_table.csv, figures, generations.md |
| 5 | `Stage 5: Part II embeddings ablation` | smoke (word2vec epochs=1 on 1 %, two frozen smoke configs) first; then coverage report, Part II table/figures |
| 6 | `Stage 6: Part III classification` | smoke (`smoke/part3.yaml`) first; then Part III tables, error-analysis files |
| 7 | `Stage 7: README, summary, reproduction check` | checklist with evidence |

## 16. Risks and mitigations

- Shared GPU/CPU → noisy timings: 5 repeats + median, `nvidia-smi`/load-average snapshot saved
  with every timing result; timing reruns if the GPU became busy.
- cuDNN non-determinism: 3 seeds, documented. gensim multi-worker non-determinism: documented.
- tanh-RNN instability at high LR: caught by the probe; clipping at 1.0; spikes counted.
- GloVe/tokeniser mismatch: our clitic and punctuation conventions follow the Stanford/PTB style
  used for GloVe 6B; coverage is measured and reported, not assumed.
- Disk: 471 GB free; the project needs < 10 GB (GloVe zip 0.86 GB + txt ~1 GB, HF cache < 0.5 GB,
  checkpoints 24 × ≤ 160 MB).
