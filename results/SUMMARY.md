# results/SUMMARY.md — index of every table and figure (factual descriptions only)

Paths are relative to the repository root. Per-run logs: `results/runs/<run>/log.jsonl` (records:
config+env, every-50-step training stats, per-epoch LM-val, LM-test, summary) and `steps.npz`
(per-step loss and pre-clip gradient norm).

## Environment and data
- `results/env.json` — GPU, CPU, library versions and an nvidia-smi snapshot at timing time.
- `results/part1/data_stats.json` — raw split sizes, de-duplication counts, LM split sizes (reviews/tokens), vocabulary size and cut-off, `<unk>` rates on five splits, character census before/after cleaning, residual HTML tags.
- `results/part1/tokenisation_examples.md` — five random unsupervised reviews, raw vs tokenised.
- `results/part1/vocab.tsv` — the 25,003-entry vocabulary with LM-train counts.
- `results/part1/test_overfit.json` — step-1 loss and steps to < 0.5 nats for test (d), per model.

## Part I — four language models
- `results/part1/ngram_selection.csv` — LM-val CE/PPL, stored n-grams, memory and per-level discounts for all 14 KN candidates and the 3 reference rows (orders 1, 2, 5).
- `results/part1/ngram_model.json` — selected n-gram (order 4, modified KN): discounts per level, LM-val and LM-test CE/PPL, `<unk>`-target fraction, stored n-grams per order, memory, count-build time, CPU.
- `results/part1/param_breakdown.csv` — embedding / core / projection / head / total / trainable parameters of RNN, LSTM, Transformer.
- `results/part1/lr_probes.csv`, `results/part1/lr_selected.json` — 1-epoch LR probes (val CE, grad norms, spikes) and the selected peak LR per family.
- `results/part1/run_summary.csv` — one row per full run: status, early stopping, best epoch (flag if 12), epoch-1 val CE vs probe, train–val gap, test CE/PPL, times, grad-norm stats, spikes.
- `results/part1/main_table.csv` — one row per model: parameter breakdown, memory, test CE/PPL mean ± std over seeds (per-seed values), `<unk>`-target fraction, best epochs, training time, training tokens/s, workload A tokens/s, workload B ms/token, device.
- `results/part1/timing.json` — raw timing records of workloads A and B (repeats, GPU snapshots, load averages, CE checks).
- `results/part1/generations.md` — per prompt, greedy output and two samples (seeds 0, 1) of all four models side by side.
- `results/part1/generations_all.jsonl` — every generated sample (5 prompts × 20 seeds + greedy per model).
- `results/part1/generation_metrics.csv` — distinct-1, distinct-2 and within-sample repeated-4-gram rate for samples and greedy outputs, per model.
- `results/part1/loss_by_position.csv` — mean LM-test CE at block positions 1–256 per model (neural: mean over 3 seeds).
- `results/figures/part1_curves_{rnn,lstm,transformer}.{pdf,png}` — training loss (50-step means, dropout active) and LM-val CE per epoch, mean ± std (values: `results/part1/curves_{model}.csv`).
- `results/figures/part1_learning_curves.{pdf,png}` — 1×3 panels (RNN, LSTM, Transformer; shared y-axis): training loss and LM-val CE per epoch as the mean with a min–max band, plus the n-gram LM-val CE as a dashed reference (values: `results/part1/learning_curves.csv`).
- `results/figures/part1_val_curves.{pdf,png}` — LM-val CE per epoch for the three neural models, mean ± std, with the n-gram's val CE as a reference line (values: `results/part1/val_curves.csv`).
- `results/figures/part1_loss_by_position.{pdf,png}` — mean LM-test CE by block position for all four models (neural: mean ± std; values incl. std columns in `loss_by_position.csv`).
- `results/figures/part1_ngram_order.{pdf,png}` — LM-val PPL vs n-gram order (1–5, modified KN) with memory (MB) on a secondary axis (values: `results/part1/ngram_order_figure.csv`).

## Part II — embedding variants on the LSTM
- `results/part2/word2vec_training.json` — gensim parameters, worker count, training time, coverage, per-dimension std of the own embeddings.
- `results/part2/glove_coverage.json` — GloVe type and token coverage, 30 most frequent uncovered words, missing-word initialisation details, per-dimension std.
- `results/part2/config_check.json` — Part I LSTM configs vs Part II configs (identical except embedding fields).
- `results/part2/main_table.csv` — settings A/B/C: trainable parameters, frozen-checksum flags, test CE/PPL mean ± std and per seed, best epochs, training time, embedding per-dimension std at init and after training.
- `results/part2/stability.csv` — across-seed std, max/mean pre-clip gradient norm, loss-spike counts (definition included), non-finite steps.
- `results/part2/embedding_std.csv` — per-dimension std of every embedding table (A per seed at init/after; B; C covered and random-init parts).
- `results/part2/freq_buckets.csv` — LM-test CE by LM-train frequency bucket of the target (1–100, 101–1k, 1k–5k, 5k–25k, `<unk>`), mean ± std over seeds, per setting.
- `results/part2/nearest_neighbours.md` — 10 cosine nearest neighbours of good, bad, boring, actor, plot in the word2vec, GloVe and trained LSTM seed-0 embedding tables.
- `results/part2/generations.md`, `generations_all.jsonl`, `generation_metrics.csv` — Section 4.5 generation protocol for A, B, C.
- `results/figures/part2_val_curves.{pdf,png}`, `results/figures/part2_train_curves.{pdf,png}` — LM-val CE per epoch and training loss (dropout active) for settings A, B, C, mean ± std (values: `results/part2/val_curves.csv`, `train_curves.csv`).
- `results/figures/part2_freq_buckets.{pdf,png}` — LM-test CE by target frequency bucket for A, B, C (mean ± std).
- `results/part2/context_freq_buckets.csv`, `results/figures/part2_context_freq_buckets.{pdf,png}` — LM-test CE bucketed by the LM-train frequency rank of the immediately preceding input token (same buckets plus `<unk>`), A/B/C mean ± std over seeds.
- `results/part2/glove_scaled.json` — the scalar applied to the GloVe table for the C-scaled control (target per-dimension std = word2vec table's).
- `results/part2/ablation_glove_scaled.csv`, `stability_glove_scaled.csv`, `freq_buckets_with_scaled.csv`, `context_freq_buckets_with_scaled.csv` — extra control C-scaled (rescaled frozen GloVe) reported next to A, B, C; outside the main Part II table.

## Part III — downstream classification
- `results/part3/input_stats.json` (and `input_stats_last255.json`) — reviews truncated, lengths, `<unk>` rate of the inputs for IMDb train/test.
- `results/part3/features.json`, `features_last255.json` — feature-extraction provenance (checkpoints, epochs, seconds).
- `results/part3/results_table.csv` — all 12 feature sets plus the two extra embedding-mean baselines (`w2v_mean`, `lstmA_emb_mean`) with the 6-value C grid {0.01, 0.1, 1, 10, 100, 1000}: dimension, val accuracy per C, chosen C (tie flag), lbfgs iterations/convergence, test accuracy and macro-F1 with 95 % bootstrap CIs, D46 robustness-subset metrics, accuracy and counts per length bucket.
- `results/part3/*_Cgrid4.*`, `results/part3/predictions_Cgrid4/` — the earlier 4-value C-grid results (results table, LM-vs-downstream, length buckets, ablation, best pair, error exports, predictions), kept for transparency.
- `results/part3/paired_bootstrap.csv` — paired bootstrap (10,000 resamples, seed 0, identical indices for both models) of the IMDb-test accuracy difference with 95 % CI for 12 model pairs.
- `results/part3/error_analysis_lstm_mean_vs_lstm_own_mean.csv`, `error_samples_lstm_mean_vs_lstm_own_mean.md`, `pair_lstm_mean_vs_lstm_own_mean.json` — extra error export for the design-chosen pair A (trainable embeddings) vs B (own frozen embeddings).
- `results/part3/lm_vs_downstream.csv` — per LM: seed-0 LM-test PPL/CE, 3-seed PPL mean ± std, downstream accuracy and macro-F1 for `last` and `mean` pooling.
- `results/part3/length_buckets.csv` — accuracy by full-review length bucket (≤128, 129–256, >256) for all feature sets.
- `results/part3/best_pair.json` — val-based ranking of the LM feature sets, the selected pair and error-group counts.
- `results/part3/error_analysis_lstm_own_mean_vs_lstm_glove_mean.csv` — all test reviews with label, both probabilities and predictions, length, truncated flag.
- `results/part3/error_samples.md` — 15 sampled reviews (first 120 words) per group: both wrong, A right/B wrong, B right/A wrong.
- `results/part3/predictions/*.npy` — P(positive) on the test set for every feature set (and the ablation).
- `results/part3/ablation_last255.csv` — optional ablation: last 255 vs first 255 tokens for the best LM (lstm_own), both poolings.
- `results/part3/classify_meta.json` — split sizes, bootstrap settings, C grid, tie rule, selected pair.
- `results/part3/glove_uncovered_subsets.csv` — IMDb-test split by whether the input contains a GloVe-uncovered vocabulary word: n, accuracy of lstm_mean / lstm_own_mean / lstm_glove_mean, paired-bootstrap CI of (own − glove).
- `results/part3/error_group_truncation.csv` — per error group (both wrong, A right/B wrong, B right/A wrong, both right): n, truncated share vs the overall test share, and the borderline share (wrong model's P(positive) in [0.4, 0.6]) for the disagreement groups, for each error-analysis pair.
- `results/part1/generation_brackets.csv` — per model (Part I and Part II): "(" opened within the first 40 generated tokens, fraction closed by a matching ")" in the same sample, fraction of samples with an even number of `"` tokens.

## Part III figure and figure index
- `results/figures/part3_ppl_vs_accuracy.{pdf,png}` — seed-0 LM-test perplexity vs IMDb-test accuracy per LM and pooling (two marker shapes), 95 % CI error bars, dashed TF-IDF and word2vec-mean baseline lines (values: `results/part3/ppl_vs_accuracy.csv`).
- `results/figures/CAPTIONS.md` — one factual caption per figure with the CSV holding its plotted values; all figures use one fixed colour-blind-safe colour, line style, marker and hatch per model.

## Smoke and reproduction
- `results/smoke/` (git-ignored) — outputs of the smoke-mode runs (`scripts/clean_clone_smoke.sh` runs every pipeline in smoke mode from a fresh clone).
