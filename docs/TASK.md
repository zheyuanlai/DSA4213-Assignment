# TASK.md — DSA4213 Assignment 1: Implementation Spec for the Coding Agent

You are helping me implement an individual NLP course assignment (DSA4213, "Language Models,
Embeddings, and Representations").  Read all of it before doing anything.

## 0. Ground rules

2. **No fabrication.** Every number in any output must come from an actual run in this repo.
   If a run fails or is incomplete, say so plainly. Never fill a table by hand or estimate.
3. **No test-set tuning.** LM test data and IMDb test data are used only for final evaluation.
   All choices (n-gram order, discounts, learning rates, early stopping, classifier C) use
   validation data.
4. **Work in stages and stop at every checkpoint** (Section 10). At each stop, report what you
   did, sanity-check results, key numbers and any problem, then wait for me to say "continue".
5. **Plan before executing.** Stage 0 produces `PLAN.md`; do not write project code before I
   approve it.
6. **Small verified steps.** Run every pipeline on a tiny subset (smoke test) before a full run.
7. **Git discipline.** Initialise a git repo, commit after each stage with a clear message, and
   never commit data, GloVe files, checkpoints or credentials (use `.gitignore`).
8. **Decision log.** Keep `NOTES.md`: every design choice, the reason, and anything I changed.
9. **Stay in scope.** Do not add features that are not requested (see Section 9).

## 1. Environment

- Remote Linux server with one NVIDIA H200 GPU. In Stage 0, check and record: `nvidia-smi`
  (is the GPU shared or busy?), CPU model and core count, RAM, Python, PyTorch and CUDA
  versions, and whether a job scheduler (e.g. SLURM) is used.
- Check internet access to Hugging Face and to `nlp.stanford.edu`. If the GPU node has no
  internet, tell me. We will download data on a login node into `data/`.
- Use a fresh virtual environment or conda env. Pin versions in `requirements.txt`.
- Allowed libraries: `torch`, `numpy`, `pandas`, `scikit-learn`, `gensim`, `datasets`
  (only for downloading IMDb), `matplotlib`, `pyyaml`, `tqdm`. Do not use pretrained language
  models from `transformers` or anywhere else.
- Numerical precision: **fp32 for all neural models** (TF32 matmuls allowed), the same for every
  model. No `torch.compile`. The models are small; consistency matters more than speed.

## 2. Data (shared by all parts)

**Source.** IMDb Large Movie Review Dataset (Maas et al., 2011), via Hugging Face
`stanfordnlp/imdb`. Splits: `train` (25,000 labelled), `test` (25,000 labelled),
`unsupervised` (50,000 unlabelled). Do not redistribute the data; the README gives a download
command.

**Usage.**
- The language models (Parts I and II) are trained **only on the `unsupervised` split**.
- Part III uses the labelled `train` and `test` splits.

**Cleaning.**
1. Replace `<br />`, `<br/>` and `<br>` with a space.
2. Apply `html.unescape`.
3. Normalise curly quotes and apostrophes to ASCII.
4. Lowercase.
5. Collapse whitespace.

**Tokenisation (GloVe-compatible, word level).**
- Use a deterministic regex or PTB-style tokenizer.
- Split punctuation into separate tokens.
- Split clitics: `don't → do n't`, `it's → it 's`, and likewise for `'re 've 'll 'd 'm`.
- Keep internal hyphens (`well-made` stays one token). Keep numbers.
- Print 5 random reviews raw vs. tokenised for my review.

**De-duplication.**
- Compare reviews by normalised text.
- Remove from `unsupervised` any review that also appears in labelled `train` or `test`, and
  drop exact duplicates within `unsupervised`.
- Report the counts. Do not modify the labelled splits; just report the number of duplicates
  shared between labelled `train` and `test`.

**LM splits.**
- After de-duplication, shuffle the unsupervised reviews with seed 0.
- Split by review, never by sentence: 2,000 for LM-val, 2,000 for LM-test, the rest for LM-train.

**Vocabulary.**
- Built from LM-train only: the 25,000 most frequent tokens.
- Add the specials `<pad>` (id 0), `<unk>` and `<eos>`. Every other token maps to `<unk>`.
- Report the `<unk>` rate (token level) on LM-train, LM-val, LM-test, IMDb-train and IMDb-test.

**Streams.**
- Each split becomes one token stream: `<eos>` at the start, then each review followed by
  `<eos>`.
- Save the streams as numpy/torch tensors so training never re-tokenises.

## 3. Common evaluation protocol (all four Part I models and all Part II models)

This protocol is the core of a fair comparison. Implement it once in `src/evaluate.py` and use
it for every model.

**Blocks.**
- Block length L = 256.
- For block k: inputs = `s[kL : kL+L]`, targets = `s[kL+1 : kL+L+1]`.
- The final incomplete block is right-padded with `<pad>`. Pad targets are excluded from the loss
  (`ignore_index`) and from all averages.

**Context rule.**
- For the target at position j of a block, every model may condition only on `inputs[0..j]` of
  that same block. Nothing from earlier blocks, nothing from the future.
- Neural models are stateless across blocks (RNN/LSTM hidden state reset to zero for each block).
- The n-gram model uses the last min(n−1, j+1) tokens of that window. When fewer than n−1 are
  available, it uses its lower-order estimate.

**Metric.**
- Mean cross-entropy in nats over all non-pad target tokens, and PPL = exp(CE).
- `<unk>` targets are included because all models share the vocabulary. Also report the
  fraction of targets that are `<unk>`.

**Required tests (`tests/`, run with pytest).**
- (a) **Causality**, for each neural model: changing tokens after position j leaves the logits
  at positions ≤ j unchanged (tolerance 1e-5).
- (b) **n-gram normalisation**: the probabilities sum to 1 over the vocabulary for 20 random
  histories of each length.
- (c) **Padding**: adding extra pad positions does not change the loss.
- (d) **Overfit check**: each neural model can overfit a single batch.

## 4. Part I — four language models on the same data

### 4.1 n-gram model
- Interpolated Kneser–Ney, implemented in this repo. A library is acceptable only if it follows
  the exact protocol in Section 3.
- Orders 3 and 4. Choose the order and the discount(s) on LM-val; record the val CE for every
  candidate.
- Report model size as the number of stored n-grams per order and the approximate memory
  footprint.
- It must be able to sample for generation.

### 4.2 Neural models (randomly initialised, trainable input embeddings)

**Shared structure.**
- Embedding (|V| × 300) → core → linear head to |V|.
- Embedding dim d = 300 for all models, chosen so that GloVe 300d fits Part II.
- **No weight tying** (Part II freezes the input embedding, which must not freeze the output
  layer).

**Starting configurations.** Confirm in `PLAN.md` after counting parameters.

| Model | Configuration |
|---|---|
| RNN | `nn.RNN` (tanh), 2 layers, hidden H_rnn, dropout 0.3 |
| LSTM | `nn.LSTM`, 2 layers, hidden 512, dropout 0.3 |
| Transformer | GPT-style decoder-only (written by you): 4 layers, d_model 300, 6 heads, FFN 1200 with GELU, learned absolute positional embeddings (max 256), pre-LayerNorm with a final LayerNorm, dropout 0.1–0.2, causal masking (e.g. `F.scaled_dot_product_attention(is_causal=True)`) |

**Parameter matching.**
- Print a parameter breakdown for every model: embedding / core / head / total / trainable.
- Propose H_rnn (and any other change) so that **core** parameter counts are roughly comparable,
  then wait for my decision.

### 4.3 Training procedure (identical unless stated)
- Training data: non-overlapping L = 256 blocks from the LM-train stream. Use a random start
  offset in [0, L) each epoch and shuffle the block order.
- Batch size: 64 blocks.
- Optimiser: AdamW with weight decay 0.01 on weight matrices only. Linear warmup (300 steps),
  then cosine decay to 10% of peak.
- Gradient-norm clipping at 1.0. Log the pre-clip gradient norm.
- Peak LR per model family: pick from {3e-4, 1e-3, 2e-3} using a 1-epoch probe scored on LM-val.
  Log the probe results.
- Max 12 epochs. Evaluate LM-val at the end of every epoch. Early stopping with patience 3.
  Keep the best-val checkpoint.
- Seeds {0, 1, 2} for every neural model. Run seed 0 first, then the others. Note in README
  that cuDNN kernels are not bit-deterministic.
- Logging (JSONL per run):
  - train loss averaged over the last 50 steps, LR, grad norm and elapsed time, every 50 steps;
  - val CE every epoch;
  - the run config and the environment info.

### 4.4 Cost measurements (`src/timing.py`)
- **Hardware:** record GPU, CPU and library versions in `results/env.json`.
- **Training:** wall-clock time excluding validation (total, per epoch, and up to the best
  epoch), tokens/s, number of epochs run.
- **Inference workload A (scoring):**
  - Score the entire LM-test set with batch size 64.
  - 2 warm-up runs, then 5 timed repeats with `torch.cuda.synchronize()` before each timer read.
  - Report the median tokens/s.
  - The n-gram model runs the same workload on CPU (single process); report the CPU model.
- **Inference workload B (generation):**
  - Batch size 1, the 5 prompts below, 60 new tokens each.
  - Report the median ms/token.
  - The Transformer recomputes the full prefix each step (no KV cache); state this.

### 4.5 Generation
- **Prompts:**
  - "this movie was"
  - "i have never seen"
  - "the acting in this film"
  - "if you are looking for"
  - "the plot of this"

  Check that all prompt tokens are in the vocabulary. Prepend `<eos>`.
- **Decoding (identical for all models):**
  - (i) greedy;
  - (ii) sampling with temperature 0.8 and top-k 50.

  Generate exactly 60 new tokens. Suppress `<pad>` and `<unk>` equally for every model, and say
  so in the README.
- **Outputs:**
  - `results/part1/generations.md`: per prompt, every model's greedy output plus 2 samples
    (seeds 0 and 1), side by side.
  - Quantitative support: for each model, 5 prompts × 20 samples. Report distinct-1, distinct-2,
    and the within-sample repeated 4-gram rate. Also report the same metrics for the greedy
    outputs.

### 4.6 Extra analysis
- **Loss by position:** mean CE at each block position 1..256 on LM-test for all four models.
  Make one figure.

### 4.7 Part I outputs
- `results/part1/main_table.csv`, with one row per model:
  - parameter breakdown;
  - test CE and PPL (mean ± std over seeds);
  - best epoch;
  - training time;
  - workload A tokens/s;
  - workload B ms/token.
- Figures:
  - train and val loss curves (all seeds) per neural model;
  - one combined val-curve figure;
  - the loss-by-position figure.

## 5. Part II — embedding ablation on the LSTM

**Settings.** Everything except the input embedding is identical to the Part I LSTM: same
architecture, d = 300, data, training procedure, LR and seeds {0, 1, 2}.

| Setting | Input embedding |
|---|---|
| A. random-trainable | The Part I LSTM runs, reused directly; verify the configs match exactly |
| B. own-frozen | Your own word2vec embeddings, frozen |
| C. GloVe-frozen | Public GloVe embeddings, frozen |

**Setting B: own word2vec embeddings.**
- gensim Skip-gram: `vector_size=300`, `window=5`, `negative=10`, `sg=1`, `epochs=5`,
  `min_count=1`, `seed=0`.
- Train on the **LM-train token stream after mapping to the vocabulary**, so `<unk>` and
  `<eos>` get vectors and coverage is 100% by construction. Use no val or test text.
- Record the training time **separately** from LM training, plus the worker count. Note that
  multi-worker gensim is not bit-reproducible.

**Setting C: GloVe 6B 300d.**
- Use `glove.6B.300d` (Wikipedia 2014 + Gigaword 5, uncased). Its download time is not part of
  the training cost.
- Report coverage as the % of vocabulary types and the % of LM-train tokens found in GloVe.
- List the 30 most frequent uncovered words.
- Initialise missing words from N(0, σ²), where σ is the per-dimension std of the covered GloVe
  vectors (seeded). Set `<pad>` to zeros. Document the initialisation of every special token.

**Freezing.**
- In B and C the **entire** embedding table is frozen (`requires_grad=False`).
- Verify by comparing an embedding checksum before and after training.
- Report trainable parameter counts.

**Part II outputs.**
- A table with test CE/PPL (mean ± std), best epoch and training time.
- Stability: across-seed std, max and mean grad norm, and the number of loss spikes. A spike is
  a train loss more than 1.5× its running mean; report the definition.
- Val and train curves for all settings and seeds.
- Generations using the Section 4.5 protocol.
- **Frequency-bucket analysis:** test CE by the LM-train frequency rank of the target, in buckets
  1–100, 101–1k, 1k–5k, 5k–25k and `<unk>`. Make one figure comparing A, B and C.
- Optional sanity check: the 10 nearest neighbours (cosine) of "good", "bad", "boring",
  "actor" and "plot" in the word2vec and GloVe tables.

## 6. Part III — downstream sentiment classification with frozen LMs

**Data.**
- Split IMDb `train` into a stratified 22,500 for classifier training and 2,500 for classifier
  validation (seed 0). IMDb `test` (25,000) is used once, at the end.
- Use the same cleaning, tokenisation and vocabulary as the LMs.

**Input format.**
- `<eos>` + review tokens, truncated to the **first 256 tokens** (this is the main setting).
- Report the % of reviews truncated and the `<unk>` rate.

**Representations.** All LMs are frozen, in eval mode, using the seed-0 checkpoints. Use the
top-layer hidden states; for the Transformer, take them after the final LayerNorm.

| Representation | Definition |
|---|---|
| last | the state at the last non-pad position |
| mean | the mean over non-pad positions (mask out pads) |

Extract features for 5 LMs × 2 poolings = 10 feature sets. The 5 LMs are: RNN, LSTM (= Part II
A), Transformer, LSTM-B and LSTM-C. The n-gram model has no representation and is excluded.

**Classifier (identical for every feature set).**
- `StandardScaler` + `LogisticRegression` (lbfgs, max_iter 2000).
- Choose C from {0.01, 0.1, 1, 10} by accuracy on classifier-val. Fit on the 22,500 with the
  chosen C, then evaluate on test once.

**Baselines (same classifier and selection procedure).**
- (i) TF-IDF of unigrams and bigrams (top 100k features). Its classifier may need a sparse-
  friendly solver; document it.
- (ii) Mean of frozen GloVe vectors.

**Metrics.** Test accuracy and macro-F1, each with a 95% bootstrap CI (1,000 resamples).

**LM vs. downstream table.** For each LM: LM test PPL next to downstream accuracy for both
poolings.

**Error-analysis exports** (I will read and categorise the examples myself).
- For the two best LM representations, write a CSV of all test reviews with:
  - the true label;
  - both models' predicted probabilities;
  - review length;
  - a flag for whether the review was truncated.
- Write a markdown sample of 15 reviews from each group (first 120 words of each):
  - both wrong;
  - A right, B wrong;
  - B right, A wrong.
- Accuracy by review-length bucket (≤128, 129–256, >256 tokens) for all 10 feature sets.

**Optional ablation, only if I ask:** truncate to the last 256 tokens instead of the first,
for the best LM.

## 7. Repository layout and reproducibility

```
README.md  TASK.md  PLAN.md  NOTES.md  requirements.txt  .gitignore
configs/        # one YAML per run (model, data, optimiser, seed); every run is config-driven
src/            # data.py, tokenize.py, vocab.py, ngram.py, models.py, train_lm.py,
                # evaluate.py, generate.py, timing.py, word2vec.py, glove.py,
                # features.py, classify.py, analysis/
scripts/        # prepare_data.sh, run_part1.sh, run_part2.sh, run_part3.sh (+ sbatch if SLURM)
tests/
results/        # part1/ part2/ part3/ figures/ env.json SUMMARY.md (committed; small files only)
data/ checkpoints/  # git-ignored
```

**README.md must include:**
- data access and download commands (IMDb, GloVe);
- every preprocessing step;
- dependencies and installation;
- the exact commands to reproduce each part;
- the configs and seeds used;
- hardware and approximate runtimes;
- a section **"Use of AI tools"** with a placeholder that I will fill in;
- attribution for any external code adapted.

**No secrets.** No API keys, tokens or credentials anywhere in the repo.

**Figures.** Matplotlib, saved as both PDF and PNG, readable at half a page width. Every axis
labelled with units.

**`results/SUMMARY.md`.** Links to every table and figure, with a one-line factual description
of each. No interpretation.

## 8. Final checklist (verify before declaring done)

- [ ] All four Part I models were evaluated by the same `evaluate.py` with the same blocks and
      vocabulary.
- [ ] All tests pass (causality, normalisation, padding, overfit).
- [ ] No choice was made using LM-test or IMDb-test.
- [ ] Part II settings differ only in the embedding table; the frozen checksum is unchanged.
- [ ] The word2vec training cost is reported separately.
- [ ] Part III uses the same classifier family and procedure for every feature set.
- [ ] Every number in the results is traceable to a log file or CSV.
- [ ] The README reproduces everything from a clean clone.

## 9. Non-requirements (do not do these)

- No subword/BPE tokenisation, no pretrained language models, no weight tying.
- No fine-tuning of LMs in Part III, and no hyperparameter searches beyond those specified.
- No extra architectures and no testing of every embedding setting on every architecture.
- No report text or analysis prose.

## 10. Stages and stop points

| Stage | Work | Stop and show me |
|---|---|---|
| 0 | Read this file; check the environment (Section 1); write `PLAN.md` with the file list, configs, parameter estimates and runtime estimates | `PLAN.md`, the environment summary, any questions |
| 1 | Data pipeline (Section 2), vocabulary, streams | the de-dup counts, split sizes, `<unk>` rates, 5 tokenisation examples |
| 2 | `evaluate.py`, the n-gram model, tests (b) and (c) | the n-gram val CE per order and discount, test results |
| 3 | The three neural models plus tests (a) and (d); parameter breakdown; LR probes; smoke test on 1% of the data | the parameter table and proposed H_rnn, probe results — **wait for approval before full training** |
| 4 | Full Part I training (seed 0, then seeds 1–2), timing, generation, loss by position | `main_table.csv`, the figures, `generations.md` |
| 5 | Part II (word2vec, GloVe, coverage, frozen runs, analyses) | the coverage report, the Part II table and figures |
| 6 | Part III (features, classifiers, baselines, error exports) | the Part III tables, the error-analysis files |
| 7 | README, `SUMMARY.md`, final checklist, clean-clone reproduction test | the checklist with evidence |
