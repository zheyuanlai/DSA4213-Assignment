# NOTES.md — decision log

Format: `D<n> (stage, date) — decision. Why. Changed by user: …`

## Stage 0 (2026-10-01)

- D1 — **Use exactly one GPU** (`CUDA_VISIBLE_DEVICES=2` at present; GPU 2 was the only idle
  device on the shared 4×H200 node). Why: user instruction (shared cluster); also keeps timing
  comparable. The device id is a config/env setting, not hard-coded.
- D2 — Fresh conda env `dsa4213` with **Python 3.11.15** and `torch==2.14.1+cu126`. Why: base
  env is Python 3.13; 3.11 maximises wheel availability for gensim/datasets; the driver (CUDA
  13.0) supports the 12.6 runtime wheel. Versions pinned in `requirements.txt`.
- D3 — Weight decay 0.01 on every parameter with `ndim ≥ 2` (embedding, recurrent and linear
  weight matrices, positional table); none on biases/LayerNorm. Why: TASK says "weight matrices
  only"; this is the standard GPT-style split and treats all models identically.
- D4 — Transformer dropout 0.1 (TASK allows 0.1–0.2). Why: smallest allowed value; early
  stopping on val handles over-fitting; keeps the Transformer closest to its spec defaults.
- D5 — RNN/LSTM dropout 0.3 applied to the embedding output, between the two layers (`nn.RNN`/
  `nn.LSTM` `dropout=`), and to the top-layer output before the head. Why: standard Zaremba-style
  placement; one value (0.3) as specified.
- D6 — Training drops the final incomplete 256-block of each epoch; evaluation pads it. Why:
  avoids pad tokens in training batches; loses < 256 tokens per epoch.
- D7 — n-gram at block positions with fewer than n−1 context tokens uses the KN recursion at the
  available order (continuation-count lower-order estimates), exactly as SRILM/KenLM do for
  short contexts. Why: this is "its lower-order estimate" in TASK §3; no information outside the
  block is used.
- D8 — The KN unigram level is the continuation unigram, discounted and interpolated with the
  uniform distribution 1/|V|. Why: exact normalisation at every level and no zero probabilities.
- D9 — n-gram candidates: orders {3,4} × ({single discount D ∈ {0.5,0.6,0.7,0.75,0.8,0.9}} ∪
  {modified KN with count-of-count-estimated D1/D2/D3+}) = 14 candidates, all scored on LM-val.
- D10 — LR probe = the first epoch of the real 12-epoch schedule (300 warm-up steps then cosine),
  scored on LM-val. Why: the probe epoch is then identical to epoch 1 of the chosen full run.
- D11 — Workload A tokens/s counts non-pad target tokens of LM-test.
- D12 — Generation never stops early at `<eos>` (exactly 60 new tokens); `<eos>` is shown as `⏎`.
  Workload B timing uses greedy decoding.
- D13 — distinct-1/2 are corpus-level (over all generated tokens of a model's 100 samples, and
  separately over its 5 greedy outputs); repeated-4-gram rate is within-sample, averaged.
- D14 — Loss spike = per-step train loss > 1.5 × trailing mean of the previous 50 steps, counted
  from step 51 onwards.
- D15 — gensim `workers=16` (of 256 threads on a shared node); documented as not bit-reproducible.
- D16 — Per-step train loss and pre-clip grad norm are saved per run (`steps.npz`, ≈ 9k values)
  in addition to the 50-step JSONL records. Why: exact spike counts and max/mean grad norm.
- D17 — Run logs are committed under `results/runs/`; large per-token arrays and features live in
  `data/derived/` (git-ignored). Why: traceability of every reported number with a small repo.
- D18 — Vocabulary ties at the 25,000 cut-off are broken alphabetically. Why: determinism.
- D19 — Part III input = `<eos>` + first 255 review tokens (256 positions, the Transformer's
  maximum). Pending user confirmation (PLAN Q3).
- D20 — H_rnn proposal 1072 (matches LSTM core parameters within 0.14 %); alternative 1024.
  Pending user decision (PLAN Q1).

### Added after the Stage-0 plan review (2026-10-01)

A four-lens adversarial review of PLAN.md against TASK.md (spec coverage, fairness/hygiene,
engineering soundness, reproducibility) confirmed 12 findings; the following decisions close them.

- D21 — Seed-0 best-val checkpoints feed everything that needs one checkpoint per model
  (workload A, workload B, generations, generation metrics, Part III features). Test CE/PPL,
  the curve figures, loss-by-position and the frequency-bucket figure use all three seeds
  (curves = mean over seeds). A seed is never chosen on LM-test. Why: an unspecified choice
  here would either triple the timing work or invite a test-set choice.
- D22 — Part III "two best LM representations" = the two LM feature sets with the highest
  classifier-val accuracy at their selected C (ties: val macro-F1, then table order); recorded
  in `results/part3/best_pair.json`. Why: TASK rule 3 — IMDb-test must not drive any choice.
- D23 — The TF-IDF vectoriser is a Pipeline step fit on the 22,500 classifier-train reviews only
  (vocabulary, IDF, top-100k cut, scaler, classifier). Why: identical fitting procedure to the LM
  feature sets; no contamination of classifier-val or IMDb-test.
- D24 — Modified-KN discounts: Chen–Goodman count-of-count estimates per level from that level's
  own table (raw counts at the top order, continuation counts below); fallback D1/D2/D3+ =
  0.5/1.0/1.5 if any n_k is 0 or an estimate is negative. Why: negative discounts would give
  negative probabilities; test (b) also asserts min P ≥ 0.
- D25 — Overfit test fixture: dropout 0, one seeded batch of 4 × 64 tokens, Adam lr 1e-3, no
  decay/clipping/schedule, ≤ 2,000 steps, pass when loss < 0.5 nats. Why: a 300-step budget with
  dropout active can fail for reasons unrelated to correctness.
- D26 — Every pipeline gets a smoke config via a generic `limit_reviews` option (Part I models,
  frozen LSTM variants, n-gram, word2vec, Part III). Why: TASK rule 6 applies to every pipeline.
- D27 — Traceability files: `results/part1/data_stats.json`, `results/part1/ngram_model.json`,
  `results/part3/input_stats.json`, a `test` record in every run's JSONL, per-seed test CE columns
  and `unk_target_fraction` in `main_table.csv`, and CSV dumps of plotted figure values. Why:
  TASK §8 — every number must trace to a committed log or CSV, not to the chat transcript.

### User decisions at the Stage-0 stop point (2026-10-01)

- D28 — **Changed by user:** H_rnn = 1072 (Q1).
- D29 — **Changed by user:** a `Linear(H, 300)` projection (with bias, no nonlinearity) between
  the top recurrent layer and the head for both RNN and LSTM; dropout stays on the top-layer
  output, before the projection. All three models therefore share an identical 300 × |V| head and
  totals are roughly equal (RNN 19,122,143 / LSTM 18,949,023 / Transformer 19,439,803). The
  projection is its own column in the parameter breakdown. Re-measured step times: RNN 29.0 ms,
  LSTM 27.6 ms, Transformer 25.9 ms (batch 64 × 256, GPU 2).
- D19 (confirmed by user) — Part III input = `<eos>` + first 255 review tokens (Q3).
- D30 — **Changed by user:** Part III representation for every model is the 300-d pre-head vector
  (projection output for RNN/LSTM, final-LayerNorm output for the Transformer), with `last` and
  `mean` pooling.
- D31 — **Changed by user (added):** LR fallback rule. If any full run hits a NaN/inf loss, stop
  and report; the fix is to rerun that model family with the next-lower LR from the grid for all
  three seeds, keeping the failed logs. LRs are not changed for any other reason.
- D32 — **Changed by user:** TF-IDF baseline drops the StandardScaler (TF-IDF rows are already
  L2-normalised); the rest of its pipeline is unchanged. Documented in README.
- D33 — **Changed by user (added):** LM-vs-downstream table uses each model's seed-0 LM-test PPL
  (the checkpoint the features come from) plus the 3-seed mean ± std as an extra column.
- D34 — **Changed by user (added):** Part II table includes the per-dimension std of each
  embedding table (A at init and after training, B, C).
- D35 — **Changed by user (process):** no more multi-agent review panels unless asked; a single
  self-check against TASK.md at each stop point.

## Stage 1 (2026-10-01) — data pipeline

- D36 — Tokeniser regex (see `src/tokenize.py` docstring): clitics split first (`n't`, `'s 're 've
  'll 'd 'm`, so `can't → ca n't`, `won't → wo n't` as in PTB/GloVe); numbers with internal
  separators (`3.5`, `10,000`, `8/10`) are one token; words keep internal hyphens and apostrophes
  (`well-made`, `o'clock`); `...` and `--` are single tokens; any other non-space character is its
  own token (so `!!` → `! !`, non-ASCII letters stay inside words via Unicode `\w`). Back-tick is
  normalised to `'` together with the curly quotes. Other HTML tags than `<br>` are not stripped
  (TASK lists only `<br>` variants); they are rare and tokenise as `< i >` etc.
- D37 — "Normalised text" for de-duplication = the cleaned string (after step 5 of cleaning),
  before tokenisation. Within-unsupervised duplicates keep the first occurrence. Labelled
  duplicates are reported both as distinct shared texts and as review pairs, plus within-split
  duplicate counts; labelled splits are untouched.
- D38 — Smoke mode (`--limit N`) keeps the first N reviews of every HF split and redirects outputs
  to `data/smoke/` and `results/smoke/` so the real data is never clobbered; held-out sizes are
  scaled to 10 % in smoke mode. `results/smoke/` is git-ignored.
- D39 — `results/part1/vocab.tsv` (id, token, LM-train count; 25,003 rows, ~0.4 MB) is committed
  for traceability of the vocabulary and of the frequency-bucket analysis. The `<eos>` count is
  set to (#LM-train reviews + 1), i.e. its actual frequency in the LM-train stream.
- D40 — IMDb train/test are stored per review as ragged int32 arrays (`imdb_{train,test}.npz`:
  ids, offsets, labels) with the full review length, so Part III can truncate to 255 tokens and
  compute length buckets on the untruncated length.
- Observations: 641 unsupervised reviews duplicated labelled train/test and 474 were within-split
  duplicates → 48,885 kept; 123 texts are shared between labelled train and test (reported only).
  |V| = 25,003 with minimum in-vocabulary count 12 of 126,896 distinct LM-train types.

### Stage 1 revision after user review (2026-10-01)

- D41 — **Changed by user:** character normalisation extended (cleaning step 3). C1 controls
  U+0080–U+009F are re-interpreted as Windows-1252 (\x91 \x92 → ', \x93 \x94 → ", \x96 → -,
  \x97 → --, \x85 → ..., \x84 → ", \x82 → ', \x8b \x9b → ', the printable rest such as € ™ • œ
  kept as those characters; undefined bytes removed); Unicode hyphens/en dash (U+2010–U+2013,
  U+2212) → -, em dash/horizontal bar (U+2014, U+2015) → --, ellipsis U+2026 → ...; remaining
  control (Cc) and format/zero-width (Cf) characters removed. Review 42527 contained U+0096 ×2 and
  U+0085 (code points printed in the Stage-1 revision report).
  Added on evidence from the census (not in the user's list, flagged for veto): the standalone
  acute accent U+00B4 (804 occurrences, used as an apostrophe: "don´t", "80´s") → ', and the
  standalone diaeresis U+00A8 (168, used as a double quote: "¨cabaret¨") → ". Effect visible in
  the vocabulary: the fragments `doesn`, `didn`, `ve` and the stray `´ ¨ – \x96 \x97 \x91 \x84 \x95 …`
  tokens dropped out.
  Census (all 100,000 reviews): raw text 146 distinct non-ASCII characters, 22,717 occurrences,
  9,526 reviews; after the previous cleaning 120 / 19,039 / 8,786; after the new cleaning 109 /
  11,187 / 6,656, with 0 control/format characters left. The remaining non-ASCII characters are
  accented letters and symbols (é 6,758; á 457; è 389; ö 281; £ 241; » « ½ ¡ · •). No mojibake
  sequences (ã©, â€™, …) exist; the "ã/â" hits are Portuguese/French words (são paulo, château).
  Full census with top-30 lists before/after: `results/part1/data_stats.json` →
  `character_census_all_splits`.
- D42 — **Changed by user:** word tokens are Unicode-aware. The implementation already used
  `[^\W_]` (Unicode letters/digits), so `café`, `amélie`, `naïve` were whole tokens; PLAN.md §3
  wrongly described the pattern as `[a-z0-9]` and is corrected. Cases added to the test suite.
- D43 — **Changed by user:** dotted acronyms/initials `(?:letter\.){2,}` (h.p., u.s., e.g., i.e.,
  a.k.a.) and the titles mr. mrs. ms. dr. jr. sr. st. vs. are single tokens (GloVe-compatible). A
  sentence-final period after such a token is absorbed into it, as in GloVe. 53 vocabulary entries
  changed (53 in, 53 out; 24,950 unchanged): in = mr. (1,940) dr. ms. mrs. u.s. jr. i.e. vs. st.
  e.g. l.a. t.v. p.s. a.k.a. …; out = the C1/accent fragments above plus 44 words with LM-train
  count 12 at the cut-off (e.g. siegel, splashing, stalingrad).
- D44 — HTML left unchanged (user rule: < 0.1 %): after cleaning, 23 of 100,000 reviews (0.023 %)
  still contain a tag-like pattern `<[a-z/][^>]*>`; top tags `<i>` 8, `</i>` 8, `<p>` 3, `<grin>` 2,
  `<sighs>` 2, `<shakes his head>` 2, `<hr>` 1, a URL in angle brackets 1, `<spoiler>` 1,
  `</spoiler>` 1. These tokenise as `< i >` etc.
- D45 — Prompt-token clarification: the five prompts contain 21 tokens, 18 unique, all in the
  vocabulary. The earlier "19" in the Stage-1 report was a miscount in the prose, not in code.
- D46 — **User request for Stage 6 (recorded now, no action yet):** additionally report accuracy
  and macro-F1 on the IMDb-test subset that excludes reviews whose cleaned text appears in
  IMDb-train (123 shared texts), as a robustness check; the official splits stay unchanged.
- D47 — `tests/test_tokenize.py` (14 cases) is the edge-case check: clitics, hyphens, numbers,
  accented words, acronyms/titles, titles not firing inside words (1st. cast. films.), C1
  controls, Unicode dashes/ellipsis/accent-as-apostrophe/diaeresis-as-quote, zero-width/control
  removal, `<br>` variants and entities, curly quotes.
- Re-run results (full pipeline, smoke first): de-dup counts unchanged (641 / 474 / 48,885 kept;
  123 shared labelled texts); LM-train 12,132,567 review tokens (was 12,147,522), LM-val 536,486,
  LM-test 545,241; |V| = 25,003 with cut-off count 12 of 127,311 distinct types; `<unk>` rates
  LM-train 2.011 %, LM-val 2.199 %, LM-test 2.186 %, IMDb-train 2.425 %, IMDb-test 2.419 %.
  44 vocabulary tokens contain non-ASCII characters (all accented words or symbols such as
  cliché, café, £, ½); 0 contain non-printable characters.

## Stage 2 (2026-10-01) — evaluation protocol and Kneser–Ney n-gram

- D48 — `src/evaluate.py` is the single scorer for every model: `make_blocks` (inputs
  `s[kL:kL+L]`, targets `s[kL+1:kL+L+1]`, last block right-padded), `score_blocks` (per-token CE
  with NaN at pad targets, batches of 64 blocks), `summarize` (mean CE, PPL, `<unk>`-target
  fraction, `<eos>`-target fraction, CE by position), `ce_by_frequency_bucket` (ranks by LM-train
  count; `<eos>` ranked by its own count; `<pad>`/`<unk>` unranked). Each model only supplies a
  `logprob_fn(inputs, targets)` obeying the context rule.
- D49 — n-gram tables are trie-keyed: a k-gram is stored as `row_{k-1}(history) * |V| + w` in a
  sorted int64 array, so order 5 fits in int64 (plain base-|V| packing would overflow at order 5)
  and all continuations of a history are one contiguous key range. Lookups are
  `np.searchsorted`; scoring of all 547k LM-test positions is fully vectorised (one lookup pass
  per level), so a candidate is scored on LM-val in ≈ 1.5 s.
- D50 — Continuation counts N1+(•hw) are computed from the distinct (k+1)-grams (first-occurrence
  positions → suffix row), per-history aggregates c(h), N1(h•), N2(h•), N3+(h•) by `bincount`.
  Top level raw counts, lower levels continuation counts; c(h) is the sum of continuation counts
  with that history so every level normalises exactly. The unigram level is interpolated with
  the uniform 1/|V|; `<pad>` therefore has a tiny non-zero probability like every unseen event,
  which never matters because `<pad>` is masked in evaluation and suppressed in generation.
- D51 — Modified-KN discounts per level (Chen–Goodman) with the KenLM fallback 0.5/1.0/1.5
  when any n_k = 0 or an estimate ≤ 0. The fallback fired only for the order-1 reference row
  (raw unigram counts have no entries with count 1–4 because the vocabulary cut-off is 12).
  All discounts used are recorded per level in `results/part1/ngram_selection.csv`.
- D52 — **User addition:** reference rows (orders 1, 2, 5 with modified KN) in
  `ngram_selection.csv`, never selected. Observation: order 5 (val CE 4.8262) beats the selected
  order-4 modified KN (4.8325) by 0.006 nats at 1.7× the memory (1,107 MB vs 623 MB); reported to
  the user for a decision, selection unchanged.
- D53 — **User addition:** uniform-model harness check `tests/test_evaluate_uniform.py` (CE must
  equal ln 25,003 within 1e-12, PPL must equal 25,003 within 1e-12 relative).
- D54 — Test (b) uses 20 histories per length 0…order−1, alternating observed (from the stream)
  and random (mostly unseen) histories, asserting |ΣP − 1| < 1e-6 and min P ≥ 0 on six synthetic
  KN variants and on the real selected model; a second test asserts the dense distribution and
  the vectorised block scorer agree to 1e-9. Test (c) scores a 97-token sequence unpadded (L=96)
  and right-padded to 256 and requires identical CE (< 1e-12); neural models join it in Stage 3.
- D55 — The n-gram "training time" is the wall-clock of building the selected order's tables
  (count tables + discounts + aggregates), CPU, single process, `OMP_NUM_THREADS=1`: 5.57 s for
  order 4 on 12.18 M tokens (shared node, 1-min load average 11 at the time). Validation
  scorings are excluded.
- Results: selected order 4 modified KN; LM-val CE 4.8325 (PPL 125.53); LM-test CE 4.8555
  (PPL 128.45) over 547,241 targets, `<unk>`-target fraction 2.18 %; stored n-grams
  25,003 / 1,598,095 / 5,736,546 / 9,379,021 (orders 1–4, total 16,738,665); memory 622.9 MB.
  Model saved to `checkpoints/ngram/kn_selected.npz` (ignored); per-token LM-test CE cached in
  `data/derived/part1/ngram_test_ce.npy`.

### User decisions at the Stage-2 stop point (2026-10-01)

- D56 — **User decision:** keep order 4 with modified KN as the selected n-gram. Order 5 stays a
  reference row only: it is 0.006 nats better on LM-val (4.8262 vs 4.8325) at 1.8× the memory
  (1,107 MB vs 623 MB), which does not justify the extra size.
- D57 — **Changed by user (added figure for Stage 4):** LM-val PPL vs n-gram order (1–5, modified
  KN) with memory (MB) on a secondary axis, from `results/part1/ngram_selection.csv`.
- D58 — **Changed by user (LR tie rule, fixed before the probes):** if the two best peak LRs of a
  model family are within 0.01 nats on LM-val after the probe epoch, choose the smaller LR.
- D59 — **User instruction:** full training does not start until the Stage-3 report (parameter
  breakdown, tests a/d with step-1 loss and steps-to-0.5-nats, 9-probe table with max grad
  norm / spike count / NaN flags, smoke results, nvidia-smi snapshot) is approved.

## Stage 3 (2026-10-01) — neural models, tests a/d, LR probes, smoke

- D60 — `src/models.py`: RNNLM (nn.RNN tanh / nn.LSTM, 2 layers, dropout 0.3 on embedding output,
  between layers and on the top-layer output, then Linear(H, 300) projection, then the shared
  Linear(300, |V|) head) and TransformerLM (4 pre-LN blocks, d 300, 6 heads, FFN 1200 GELU, learned
  positions ≤ 256, final LayerNorm, `F.scaled_dot_product_attention(is_causal=True)`, dropout 0.1
  on embeddings / attention / residual branches; N(0, 0.02) init with residual projections scaled
  by 1/√8). `features()` returns the 300-d pre-head vector (D30). Parameter breakdown
  (`results/part1/param_breakdown.csv`): RNN 7,500,900 / 3,773,440 / 321,900 / 7,525,903 =
  19,122,143; LSTM 7,500,900 / 3,768,320 / 153,900 / 7,525,903 = 18,949,023; Transformer
  7,500,900 / 4,413,000 / 0 / 7,525,903 = 19,439,803 (embedding / core / projection / head / total;
  all trainable in Part I).
- D61 — `src/train_lm.py` implements TASK §4.3 exactly as planned (D3, D6, D10, D14, D16, D31):
  steps per epoch are fixed at floor(((N − L) // L) / 64) = 743 for the full stream (worst-case
  offset) so every epoch has the same length and the cosine horizon is 12 × 743 = 8,916 steps;
  the final incomplete batch of an epoch is dropped (≤ 63 blocks). The JSONL has `config`,
  `step` (every 50 steps), `epoch`, `failure`, `test` and `summary` records; `steps.npz` holds the
  per-step loss and pre-clip grad norm. A non-finite loss or grad norm aborts the run with a
  `failure` record and exit code 1 (user rule D31). Weight decay goes to ndim ≥ 2 parameters
  (RNN: 19,092,552 decayed / 29,591 not; LSTM 18,915,528 / 33,495; Transformer 19,398,600 / 41,203).
- D62 — Test (a) causality runs on CPU fp32 in eval mode with the real architectures, 5 cut
  positions (0, 1, 7, 100, 254): all three models pass at 1e-5 (and later logits do change).
  Test (d) overfit (fixture D25, GPU): step-1 loss RNN 10.117, LSTM 10.133, Transformer 10.180
  (ln 25,003 = 10.127); steps to < 0.5 nats: RNN 213, LSTM 215, Transformer 29
  (`results/part1/test_overfit.json`).
- D63 — Smoke configs (1 % of LM-train = 121,774 tokens, 7 steps/epoch, 2 epochs, warm-up 5 steps,
  no LM-test evaluation) ran for all three models: val CE after 2 epochs RNN 6.79, LSTM 7.10,
  Transformer 7.44; no spikes, no non-finite values; checkpoints and logs written
  (`results/runs/smoke/*`).
- D64 — LR probes (`results/part1/lr_probes.csv`): 1 epoch of the real 12-epoch schedule, seed 0,
  LM-val CE after the epoch — RNN 5.559 / **5.242** / 5.720 for 3e-4 / 1e-3 / 2e-3; LSTM 5.912 /
  5.422 / **5.152**; Transformer 5.116 / 4.800 / **4.749**. No NaN/inf, no spikes; max pre-clip
  grad norm ≤ 6.0 (RNN at 2e-3 is the largest, 5.81, and also the worst CE — the tanh RNN is the
  only family where the largest LR hurts). Selection by lowest val CE; the tie rule D58 (two best
  within 0.01 nats → smaller LR) did not trigger (closest gap: Transformer 0.051 nats). Selected
  peak LRs: RNN 1e-3, LSTM 2e-3, Transformer 2e-3 (`results/part1/lr_selected.json`), filled
  into `configs/part1/*` and `configs/part2/*` by `scripts/make_configs.py`.
- D65 — Probe configs are also "one YAML per run" and committed, as are the smoke configs. The
  Part II frozen-run configs reuse the LSTM LR (TASK §5: same LR).

### User decisions at the Stage-3 stop point (2026-10-01)

- D66 — **User decision:** keep the selected LRs (RNN 1e-3, LSTM 2e-3, Transformer 2e-3); the
  grid is not extended. **Limitation:** the LSTM and Transformer optima lie at the upper edge of
  the pre-specified grid {3e-4, 1e-3, 2e-3}, so a larger LR might have been better for them; this
  is not explored (TASK §9: no hyper-parameter searches beyond those specified).
- D67 — **Changed by user (Stage-4 additions):** (1) consistency check — the seed-0 full run's
  epoch-1 LM-val CE is compared with its probe result (expected ≈ 0 difference up to cuDNN
  non-determinism); (2) every run reports whether early stopping triggered and its best epoch,
  flagging best epoch = 12 (still improving); (3) train–val gap at the best epoch per model (mean
  over seeds), where train = mean training loss of that epoch (dropout active) and val = LM-val
  CE (eval mode); (4) an nvidia-smi snapshot is taken immediately before every timing run — if
  GPU 2 is not idle the run waits and retries; the n-gram CPU timing records the load average;
  (5) rule D31 applies to any NaN/inf.
- D68 — Full runs are executed strictly sequentially on GPU 2, seed 0 for RNN, LSTM, Transformer,
  then seeds 1 and 2, via the loop appended to `scripts/run_part1.sh`.

## Stage 4 (2026-10-01) — Part I training, timing, generation, analysis

- D69 — `src/generate.py`: one torch sampler for every model (`<pad>`/`<unk>` log-probs set to −∞,
  temperature 0.8, top-k 50, `torch.multinomial` with a `torch.Generator` seeded by the sample
  index 0…19, the same seed set for every prompt); the n-gram's dense distribution is fed to the
  same sampler, so the decoding procedure is identical. RNN/LSTM decode incrementally (identical
  to re-reading the prefix); the Transformer re-encodes the full prefix at every step (no KV cache).
  Metrics per D13; `generations_all.jsonl` keeps every generated sample for traceability.
- D70 — `src/timing.py`: workload A times forward + log-softmax + gather + masked sum over all 2,138
  LM-test blocks in batches of 64 with inputs/targets resident on the device (2 warm-ups, 5 timed
  repeats, `torch.cuda.synchronize()` before each timer read, median tokens/s over non-pad targets;
  the CE accumulated inside the timed loop is written as `ce_check` and must match the evaluation
  CE). Workload B times each greedy generation step (synchronised) for 5 prompts × 60 tokens after
  one untimed warm-up generation; median over the 300 steps. Before every GPU timing run an
  nvidia-smi snapshot is stored and the run waits while GPU 2 shows > 5 % utilisation or
  > 1,500 MiB used (another user's idle 528 MiB context is always present). The n-gram runs on CPU
  with `OMP_NUM_THREADS=1` and records the load average (D67).
- D71 — `main_table.csv` std over seeds uses ddof = 1 (sample std of the 3 seeds); PPL mean ± std is
  computed over the per-seed PPLs. Neural `memory_mb` = 4 bytes × parameters (fp32 weights).
  `run_summary.csv` holds the per-run early-stopping status, best epoch with the
  best-is-epoch-12 flag, epoch-1 val CE vs the probe (D67 consistency check), and the train–val gap
  at the best epoch.
- D72 — Loss-by-position and the combined val-curve figure use a log-scaled x axis for position
  (the interesting region is the first few positions) and plot the neural curves as the mean over
  the three seeds' per-token test CE (D21); the n-gram's LM-val CE is drawn as a dashed reference
  line in the val-curve figure.
- D73 — Observations from the nine full runs (`results/part1/run_summary.csv`): no NaN/inf, no loss
  spikes, rule D31 not needed. **Every run has best epoch = 12 and none early-stopped**, i.e. all
  models were still improving when the 12-epoch budget (user-fixed) ended; the per-epoch gains at
  epochs 10→12 were 0.005–0.015 nats. Consistency check: the seed-0 epoch-1 LM-val CE equals the
  probe CE exactly (difference 0.0000 for all three families) — identical seed, data order and
  schedule; cuDNN non-determinism did not show up at this precision. Train–val gap at the best
  epoch (val CE − mean train loss of the epoch, dropout active in training): RNN −0.036, LSTM
  −0.012, Transformer +0.325 (mean over seeds) — the recurrent models with dropout 0.3 train
  above their val loss, the Transformer with dropout 0.1 fits the training data markedly better
  than val. RNN seed 1 had one pre-clip gradient norm of 71.1 (clipped to 1.0; no loss spike).
- D74 — Timing conditions (`results/part1/timing.json`): every GPU timing run passed the idle check
  on the first attempt (GPU 2 at 0 % utilisation, 528 MiB held by another user's idle context);
  the 5 timed repeats agree within 0.5 %. The CPU n-gram timing ran at load average 16.2 (1-min,
  shared 256-thread node). The first timing attempt was aborted and rerun after fixing the idle
  check, which had counted this process's own GPU memory as "busy" (no numbers from that attempt
  are used).
- D75 — Workload-A `ce_check` values (RNN 4.5952, LSTM 4.3249, Transformer 4.1670, n-gram 4.8555)
  equal the seed-0 LM-test CEs in the run logs, confirming the timed code path scores exactly
  what `evaluate.py` scores.

### User decisions at the Stage-4 stop point (2026-10-01, evening)

- D76 — **User decision:** keep the 12-epoch budget; no Part I run is extended or rerun. Best
  epoch = 12 in every run is expected because the cosine schedule keeps decaying until the last
  epoch, and the final-epoch gains (≤ 0.015 nats) are an order of magnitude smaller than the gaps
  between models (0.15–0.28 nats). **Flagged as a training-budget limitation.**
- D77 — **User instruction (overnight autonomous mode):** work through Stage 5, Stage 6 and the
  safe parts of Stage 7 without waiting at stop points; stop-point reports go to
  `results/stage_reports/stage<N>.md`; progress to `results/OVERNIGHT_LOG.md`; long jobs via
  `nohup` with log files; smoke test before every full run; one idle GPU (prefer GPU 2, poll every
  10 min, another idle H200 allowed after 30 min; never a busy GPU); never modify/rerun Part I;
  no history rewriting; no test-set decisions; uncovered decisions → most conservative option,
  marked **PENDING USER REVIEW**. Failure rules: NaN/inf in a frozen run → D31 for that setting;
  GloVe download → 3 retries over 30 min; unfixable test → stop that branch, record, continue.
  Extra work requested: D46 robustness subset in Part III, the TASK §6 optional ablation (last 255
  tokens, best LM by classifier-val accuracy, both poolings, own table), README/SUMMARY/checklist,
  clean-clone smoke reproduction, and `results/MORNING_REPORT.md` at the end.

## Stage 5 (2026-10-01, overnight) — Part II

- D78 — word2vec (setting B): gensim 4.4.0 skip-gram exactly as specified, on the vocabulary-mapped
  LM-train stream (44,885 sentences = reviews each ending in `<eos>`, 12,177,452 tokens, ids as
  strings); 16 workers; **training time 105.9 s**, recorded separately in
  `results/part2/word2vec_training.json`; coverage 100 % by construction (`<pad>` = zeros). Smoke
  run first (1 % of the stream, 1 epoch).
- D79 — GloVe (setting C): `glove.6B.300d` read from the zip; coverage 24,690 / 25,000 word types
  (98.76 %) and 99.95 % of LM-train word tokens (99.84 % if `<unk>` tokens count as uncovered);
  310 missing words + `<unk>` + `<eos>` initialised N(0, σ_d²) per dimension with σ_d from the
  covered vectors (mean σ 0.358, seed 0); `<pad>` zeros. The GloVe file contains no literal
  `<unk>`/`<eos>`/`<pad>` strings. The 30 most frequent uncovered words are hyphenated compounds
  (camera-work, god-awful, film-noir, over-acting …), rare names (snowblood, nukie, nanon …),
  ratings written as numbers (0/10, 8.5/10) and slang (imho, cheezy) — see
  `results/part2/glove_coverage.json`.
- D80 — **PENDING USER REVIEW:** classifier C ties on classifier-val accuracy are broken towards the
  smaller C (most regularised, conservative). Implemented in `src/classify.py`; the tie flag is
  recorded per feature set in `results_table.csv`.
- D81 — Smoke subsets for Part III are stratified (first N/2 negatives + first N/2 positives of each
  split) because the HF IMDb splits are sorted by label; the smoke run uses the real streams and
  vocabulary and only slices the data (no stream redirect). Pipeline check: 8 of the 12 feature
  sets on 500/500 reviews ran end to end (Part II checkpoints did not exist yet).
- D82 — Overnight tooling: `scripts/wait_gpu.py` (prefer GPU 2, poll 10 min, switch after 30 min, idle
  = ≤ 5 % util and ≤ 1,500 MiB used; two readings 5 s apart), `scripts/run_part2.sh` /
  `scripts/run_part3.sh` run under `nohup`. A first launch of `run_part2.sh` hung in a `pgrep`
  wait loop (the pattern matched the launching shell's own command line); it was fixed to a file
  check and relaunched before any training started.

## Stage 6 (2026-10-01, overnight) — Part III

- D83 — Features: `src/features.py` builds `<eos>` + first 255 tokens (256 positions, right-padded in
  batches of 256 reviews), extracts the 300-d pre-head vector of each seed-0 LM in eval mode and
  pools `last` / masked `mean` over the non-pad positions (including the `<eos>` position). Inputs
  are shared with the baselines (`data/derived/part3/inputs_*.npz`). 36.15 % of IMDb-train and
  34.68 % of IMDb-test reviews are truncated; `<unk>` rate of the inputs 2.33 %.
- D84 — Classifiers exactly as specified (`src/classify.py`); the TF-IDF baseline consumes the same
  `<eos>`+first-255 vocabulary-mapped token strings (whitespace tokenizer, no lowercasing) so it
  sees the same information as the LM features; the GloVe-mean baseline averages the frozen
  setting-C table over the same positions. Bootstrap resample indices are generated once
  (`default_rng(0)`) and shared by all sets. All 48 + 8 fits converged.
- D85 — D46 robustness subset: 123 test reviews whose cleaned text occurs in IMDb-train are excluded
  (24,877 kept); metrics change by ≤ 0.0006. Length buckets use the untruncated review length.
- D86 — Best pair by classifier-val accuracy (D22): A = lstm_own_mean (0.8736), B = lstm_glove_mean
  (0.8688). Error exports written for this pair. Observation (not analysis): the LM-perplexity
  ranking (Transformer < LSTM-A < LSTM-B < LSTM-C < RNN) does not match the downstream ranking of
  the `mean` features (LSTM-B > Transformer > LSTM-C > LSTM-A > RNN); `mean` pooling beats `last`
  for every LM, and the gap is largest for the RNN.
- D87 — **User-requested ablation (TASK §6 optional):** last 255 tokens instead of the first 255,
  for the best LM (lstm_own = LSTM-B), both poolings, same classifier procedure; reported in its own
  table `results/part3/ablation_last255.csv`. The `last`-255 inputs were extracted with
  `--truncate last` into separate files (`*_last255.npy`), so the main results are untouched.
- D88 — Code hygiene: a dead statement in `classify.error_exports` that sorted by
  `["val_acc", "test_macro_f1"]` and was immediately overwritten by the val-only sort was removed
  after the Part III run. It never affected results (the overriding sort used `val_acc` only, as
  `best_pair.json` shows: ranking by val accuracy 0.8736 > 0.8688 > 0.8660 …); no rerun needed.

## Stage 7 (2026-10-01, overnight) — safe parts

- D89 — README §8 "Use of AI tools" left as a placeholder for the author (TASK §7). README runtimes are
  approximate and point to the logs for exact values. `results/SUMMARY.md` contains descriptions only,
  no interpretation (TASK §7).
- D90 — Clean-clone reproduction executed in smoke mode only (`scripts/clean_clone_smoke.sh`, 165 s,
  GPU 2): full data preparation is run (it is cheap and required by every smoke config), the GloVe
  zip is copied from the source checkout instead of re-downloaded, and the smoke word2vec table
  stands in for the full one so the frozen smoke configs can run. The full experiments were not
  rerun (user D77). The temporary clone is left at `/tmp/dsa4213_clean_clone` (1.8 GB) for inspection.

## Follow-ups after the overnight run (2026-10-02)

- D80 — **Approved by user:** ties on classifier-val accuracy go to the smaller C.
- D91 — **Changed by user:** classifier C grid extended to {0.01, 0.1, 1, 10, 100, 1000} for all 12
  feature sets and the ablation (several sets had chosen the upper edge C = 10; TF-IDF val accuracy
  was still rising 0.869 → 0.883 from C = 1 to 10). Selection on classifier-val only. The extended
  grid is the main table; the original 4-value results are kept as `*_Cgrid4.*` files. The D22 best
  pair is recomputed and the error exports regenerated if it changes. Non-converged fits are reported.
- D92 — **Changed by user (added):** paired bootstrap of the accuracy difference on IMDb-test (same
  resampled indices for both models, 10,000 resamples, seed 0, 95 % percentile CI) for 12 pairs →
  `results/part3/paired_bootstrap.csv`.
- D93 — **Changed by user (added):** one extra error export chosen by design (the Part II question):
  lstm_mean (A) vs lstm_own_mean (B), same format as the best-pair exports.
- D94 — **Changed by user (added):** Part II frequency-bucket analysis also by the LM-train frequency
  rank of the immediately preceding input token (`inputs[j]` for the target at position j), same
  buckets plus `<unk>`, A/B/C mean over seeds, from the cached per-token CE arrays; table + figure.
- D95 — **Changed by user (added, extra ablation, outside the main Part II table):** setting C-scaled =
  the GloVe table (random-init rows included, `<pad>` stays 0) multiplied by one scalar so that its
  mean per-dimension std equals the word2vec table's (0.1347, same definition as the logged
  `per_dim_std`: std over all rows, mean over dimensions). Frozen, same procedure and LR, 3 seeds;
  test CE/PPL mean ± std, stability and both frequency-bucket tables reported next to B and C; no
  Part III features.
- D96 — **User instruction:** `/tmp/dsa4213_clean_clone` deleted.
- D97 — **Changed by user (added check):** IMDb-test split by whether the model input (`<eos>` + first 255
  tokens) contains ≥ 1 of the 310 vocabulary words missing from GloVe; per subset n, accuracy of
  lstm_mean / lstm_own_mean / lstm_glove_mean and a paired-bootstrap 95 % CI (10,000 resamples, seed 0)
  of (lstm_own_mean − lstm_glove_mean) → `results/part3/glove_uncovered_subsets.csv`.
- D98 — **Changed by user (added check):** truncated share per error group (both wrong, A right/B wrong,
  B right/A wrong, both right) next to the overall test share, and for the two disagreement groups the
  share where the wrong model's P(positive) ∈ [0.4, 0.6], for every error-analysis CSV of the main
  (6-value grid) run → `results/part3/error_group_truncation.csv`.
- D99 — **Changed by user (added check):** bracket/quote closure in the 100 sampled generations per model
  (Part I: ngram, rnn, lstm, transformer; Part II: A, B, C): number of "(" opened within the first 40
  generated tokens, the fraction later closed by a matching ")" in the same sample (stack matching over
  the whole 60-token sample), and the fraction of samples with an even number of `"` tokens →
  `results/part1/generation_brackets.csv`. Implemented in `src/analysis/followup_checks.py`.
- D100 — Results of the extended grid (`results/part3/results_table.csv`): all 72 fits converged (max
  lbfgs iterations 448 of 2,000). Chosen C changed for lstm_last (10 → 100), lstm_mean (10 → 1),
  transformer_mean (1 → 1000, still the upper edge: val 0.8664 vs 0.8660 at C = 1, test accuracy
  identical 0.8791), lstm_own_last (0.01 → 0.1) and tfidf (10 → 100; val 0.8852, test 0.8808 vs 0.8816
  before). Test accuracies move by ≤ 0.0008. The best pair is unchanged (lstm_own_mean 0.8736,
  lstm_glove_mean 0.8688), so the error exports were regenerated with the new predictions (group
  counts 2,175 / 1,015 / 822). Ablation (last 255 tokens, lstm_own) under the 6-value grid:
  mean 0.8937, last 0.8884 (both C = 10). Numerical note: refitting the *same* C gives val/test
  accuracies that differ in the 4th decimal between the two runs (e.g. rnn_last test 0.6509 →
  0.6510, lstm_mean val at C = 1: 0.8344 → 0.8356), because lbfgs on 32-thread BLAS is not
  bit-reproducible; the 4-grid files are kept for transparency, the 6-grid files are the main results.
- D101 — Paired bootstrap (`results/part3/paired_bootstrap.csv`, 10,000 resamples): CI excludes 0 for
  own−A (+0.032), glove−A (+0.025), own−glove (+0.008), transformer−A (+0.031), A−rnn (+0.048) and
  every mean−last pair; it includes 0 for own vs transformer (+0.001) and tfidf vs own (+0.001).
- D102 — Context-frequency buckets (`results/part2/context_freq_buckets.csv`): A/B/C differ little
  when the preceding token is frequent (1–100: 4.703 / 4.688 / 4.706) but the frozen settings are
  worse when it is rare (5k–25k: 3.839 / 4.077 / 4.076; 1k–5k: 3.757 / 3.837 / 3.867), whereas the
  target-bucket differences are small everywhere.
- D103 — GloVe-uncovered subsets (`results/part3/glove_uncovered_subsets.csv`): 1,049 of 25,000 test
  inputs contain ≥ 1 of the 310 uncovered words (1.11 such tokens on average). own − glove is
  +0.0114 [−0.0029, +0.0257] on that subset and +0.0076 [+0.0042, +0.0110] on the other 23,951.
- D104 — Error-group truncation (`results/part3/error_group_truncation.csv`): wrong-prediction groups
  are enriched in truncated reviews (both wrong 45–46 %, disagreement groups 39–46 %) vs 34.7 %
  overall and 32.7–32.8 % among both-right; in the disagreement groups the wrong model is borderline
  (P ∈ [0.4, 0.6]) in 41–46 % of cases.
- D105 — Bracket/quote closure (`results/part1/generation_brackets.csv`): the n-gram closes 1 of 13 early
  "(" (7.7 %) and has even `"` counts in 85 % of samples; every neural model closes 100 % of its 4–10
  early "(" and has even `"` counts in 94–100 % of samples.
- D106 — C-scaled control results (`results/part2/ablation_glove_scaled.csv`, scalar 0.3766 → per-dim
  std 0.1347): test CE 4.3734 ± 0.0074 (PPL 79.31 ± 0.58) vs C-glove 4.3580 ± 0.0058 and B-own
  4.3406 ± 0.0031; all seeds best epoch 12, checksums unchanged, 0 spikes, max grad norm ≤ 6.97.
  Rescaling GloVe to the word2vec std does not close the gap to B; it is slightly worse than the
  unscaled GloVe in every target bucket and every context bucket except `<unk>` contexts.
  Observation only; kept outside the main Part II table.

## Final optional batch (2026-10-02) — then freeze

- D107 — **Changed by user (added baselines):** two embedding-only Part III baselines with exactly the
  glove_mean procedure (masked mean over the same `<eos>`+first-255 inputs, StandardScaler + LR,
  6-value C grid, classifier-val selection): `w2v_mean` (setting-B table) and `lstmA_emb_mean`
  (trained input-embedding table of the Part I LSTM seed 0 = setting A after training). Added to
  `results_table.csv` / `length_buckets.csv` as `kind = baseline` via `classify --mode extra_baselines`,
  which fits only the new sets and recomputes `paired_bootstrap.csv` from the saved predictions, so
  the existing rows are untouched. Extra paired rows: w2v_mean vs glove_mean, w2v_mean vs lstmA_emb_mean.
- D108 — **Changed by user:** `nearest_neighbours.md` now has a third column: the trained LSTM seed-0
  input-embedding table.
- D109 — **Changed by user:** README documents the follow-up analyses and the C-scaled control and has a
  "Known limitations" list (budget, LR grid edge, seed-0 checkpoints, first-255 truncation, lbfgs
  non-reproducibility, single dataset, A = Part I LSTM seed 0 in the Part II generation tables).
- D110 — **User instruction:** final checks (test suite; no data/checkpoint/GloVe files tracked; total
  tracked size) and git tag `submission-candidate`; no further experiments afterwards.
- D111 — Results of the extra baselines: w2v_mean test accuracy 0.8597 (C = 0.01), lstmA_emb_mean
  0.8189 (C = 1000, upper edge), vs glove_mean 0.8210; paired bootstrap w2v_mean − glove_mean
  +0.0387 [+0.0344, +0.0430], w2v_mean − lstmA_emb_mean +0.0408 [+0.0363, +0.0453]. All fits
  converged. Nearest neighbours of the trained A table are sentiment-polar (good ↔ bad 0.72,
  bad → terrible/awful/horrible) and topical for plot (story, storyline, premise, script).

## Figure revision (2026-10-02)

- D112 — **Changed by user:** no figure text mentions seeds; curves over several runs are drawn as the mean
  with a ± std band (ddof = 1), bucket bars keep mean ± std error bars. One fixed colour-blind-safe colour
  (Okabe–Ito), line style, marker and bar hatch per model across all figures (`src/analysis/plots.py`);
  A-random shares the Part I LSTM's colour because it is the same model. Titles say "mean ± std" only — no repetition count or seed appears in any figure text or caption (user instruction).
- D113 — **Changed by user (new figures):** `part1_learning_curves` (1×3 panels, shared y, mean over runs
  with a min–max band, n-gram LM-val CE as dashed reference) and `part3_ppl_vs_accuracy` (seed-0 LM-test
  PPL vs IMDb-test accuracy per LM and pooling, two marker shapes, 95 % CI error bars, point labels,
  dashed tfidf_uni_bi and w2v_mean lines). Every figure's plotted values are dumped to CSV (listed in
  `results/figures/CAPTIONS.md`, which holds one factual caption per figure; the learning-curve captions
  state that training loss is computed with dropout active). Tables are unchanged (regenerated
  deterministically from the same logs).

## Public release (2026-10-10)

- D114 — **User request:** public GitHub repository https://github.com/zheyuanlai/DSA4213-Assignment created
  from a clean snapshot of tag `submission-candidate-v2` (one commit `2fdd070`, 260 files, 11.0 MB), so the
  private history is not published. Left out as process artefacts (conservative default, can be added on
  request): `TASK.md`, `PLAN.md`, `NOTES.md`, `results/OVERNIGHT_LOG.md`, `results/MORNING_REPORT.md`,
  `results/CHECKLIST.md`, `results/stage_reports/`, the pipeline console logs `results/runs/*.log` and the
  smoke-run logs `results/runs/smoke/`. In the public copy only `README.md` and `results/SUMMARY.md` were
  edited (references to the omitted files removed; one sentence explains the "TASK §…"/"D…" code comments);
  code, configs and results are byte-identical to the tag. No data, checkpoints, embeddings or GloVe files
  are included. Checks: anonymous HTTPS clone works; on a copy of the snapshot without data the test suite
  gives 40 passed, 1 skipped (the real-model normalisation test needs a checkpoint).

## Final polish (2026-10-10) — plots and docs only, no experiments

- D115 — **Changed by user (figures):** no in-figure titles (captions live in `results/figures/CAPTIONS.md`,
  15 entries, no seed or run counts; the Part III x-axis is described as "LM-test perplexity of the
  checkpoint the features were extracted from"). Part II bucket figures now plot the difference to A
  (B − A, C − A, C-scaled − A lighter) with error bars ±√(std_A² + std_X²) — the settings are separate
  training runs, so their spreads are combined as independent — and a zero line; the absolute versions
  are kept as `*_abs`. `part1_loss_by_position` = token-weighted mean CE per log-spaced position bin
  (1, 2–3, 4–7, …, 128–256), neural error bars = std of the per-run bin means, n-gram without error bar;
  the per-position version is kept as `part1_loss_by_position_raw`. `part1_learning_curves` uses a
  constrained layout with in-panel model labels and the legend below the panels. `part3_ppl_vs_accuracy`
  has a broken y-axis (0.83–0.89 above, 0.636–0.815 below), fixed label positions with leader lines that
  touch no other label, marker, error bar or dashed line, and legend names "TF-IDF uni+bigram" and
  "word2vec mean". New value dumps: `results/part1/loss_by_position_binned.csv`,
  `results/part2/freq_buckets_diff_vs_A.csv`, `results/part2/context_freq_buckets_diff_vs_A.csv`;
  `results/part3/ppl_vs_accuracy.csv` gained label/panel columns. Every result table regenerated
  byte-identically.
- D116 — **Changed by user (README):** §5 now states the 6-value C grid (an earlier README patch had not
  applied — it used an unchecked string replace) and that the 4-value grid survives only as `*_Cgrid4.*`;
  §5 lists all four baselines and the paired bootstrap; §4 run descriptions include the C-scaled control
  and the follow-up outputs; §6 runtimes include the C-scaled runs; §7 lists `docs/` and every analysis
  module; §9 notes that `lstmA_emb_mean` also chose C = 1000; §10 holds the author's declaration of AI
  use ([name] filled with "Claude Code by Anthropic, running the Claude Fable 5.1 and Claude Opus 5.5
  models"; the self-reference "README §10" dropped because the text sits in README §10). A scripted
  cross-check of 24 README statements against configs and result files passes.
- D117 — `scripts/run_part3.sh` was incomplete relative to the results it claims to reproduce; it now also
  runs `classify --mode extra_baselines` (w2v_mean, lstmA_emb_mean and their paired-bootstrap rows) and,
  in full mode only, `followup_checks all`, `part3_tables` and `captions`. No result changed.
- D118 — **Changed by user:** the working documents moved to `docs/` (`docs/TASK.md`, `docs/PLAN.md`,
  `docs/NOTES.md`) in the working repo and are published in the public repo. README and SUMMARY are now
  identical in both; the public repo omits only the process files (overnight log, morning report,
  checklist, stage reports, pipeline console logs, smoke-run logs).
- D119 — **Changed by user:** the tie-rule comment in `configs/part3/classify.yaml` reads "approved, D80".
- D120 — Provenance of `results/part1/test_overfit.json`: every test-suite run rewrites it. The committed
  version comes from CPU runs (steps to < 0.5 nats: RNN 221, LSTM 214, Transformer 29; reproduced exactly
  by the final run, 41 passed); the Stage-3 GPU run recorded in D62 gave 213 / 215 / 29. Both pass.
