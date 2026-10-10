"""Write results/figures/CAPTIONS.md: one factual line per figure (no interpretation), plus the CSV holding its values."""
from src.common import ROOT

CAPTIONS = [
    ("part1_curves_rnn", "results/part1/curves_rnn.csv",
     "RNN: training loss (50-step means, computed with dropout active) and LM-val cross-entropy per epoch; mean ± std."),
    ("part1_curves_lstm", "results/part1/curves_lstm.csv",
     "LSTM: training loss (50-step means, computed with dropout active) and LM-val cross-entropy per epoch; mean ± std."),
    ("part1_curves_transformer", "results/part1/curves_transformer.csv",
     "Transformer: training loss (50-step means, computed with dropout active) and LM-val cross-entropy per epoch; mean ± std."),
    ("part1_val_curves", "results/part1/val_curves.csv",
     "LM-val cross-entropy per epoch for RNN, LSTM and Transformer (mean ± std) with the KN 4-gram's LM-val cross-entropy as a dashed reference line."),
    ("part1_learning_curves", "results/part1/learning_curves.csv",
     "Three panels (RNN, LSTM, Transformer; shared y-axis): training loss (50-step means, computed with dropout active) and LM-val cross-entropy per epoch as the mean with a min–max band; dashed line = KN 4-gram LM-val cross-entropy."),
    ("part1_loss_by_position", "results/part1/loss_by_position.csv",
     "Mean LM-test cross-entropy at each position 1–256 of the evaluation blocks for the KN 4-gram and the three neural models (neural: mean ± std); log-scaled position axis."),
    ("part1_ngram_order", "results/part1/ngram_order_figure.csv",
     "LM-val perplexity of modified Kneser–Ney n-gram models of order 1–5 (log scale) with the memory footprint in MB of each model's tables on the secondary axis."),
    ("part2_val_curves", "results/part2/val_curves.csv",
     "LM-val cross-entropy per epoch for the LSTM with random-trainable (A), own frozen word2vec (B) and frozen GloVe (C) input embeddings; mean ± std."),
    ("part2_train_curves", "results/part2/train_curves.csv",
     "Training loss (50-step means, computed with dropout active) per epoch for settings A, B and C; mean ± std."),
    ("part2_freq_buckets", "results/part2/freq_buckets.csv",
     "Mean LM-test cross-entropy for settings A, B and C by LM-train frequency rank of the target token (1–100, 101–1k, 1k–5k, 5k–25k, <unk>); bars = mean, error bars = std."),
    ("part2_context_freq_buckets", "results/part2/context_freq_buckets.csv",
     "Mean LM-test cross-entropy for settings A, B and C by LM-train frequency rank of the immediately preceding input token (same buckets); bars = mean, error bars = std."),
    ("part3_ppl_vs_accuracy", "results/part3/ppl_vs_accuracy.csv",
     "IMDb-test accuracy of the logistic-regression classifier on each frozen LM representation (circles = mean pooling, squares = last pooling; error bars = 95 % bootstrap CI) against the LM's seed-0 LM-test perplexity; dashed lines = TF-IDF and word2vec-mean baselines."),
]


def main() -> None:
    lines = ["# Figure captions (factual; one line per figure)\n",
             "Every figure is saved as PDF and PNG at half-page width in `results/figures/`; the plotted values are in the CSV named in the third column. "
             "Colours, line styles, markers and hatches are fixed per model across all figures (colour-blind-safe Okabe–Ito palette); "
             "setting A is the same model as the Part I LSTM and shares its colour.\n",
             "| Figure | Values | Caption |", "|---|---|---|"]
    for stem, csv, cap in CAPTIONS:
        lines.append(f"| `{stem}.pdf/.png` | `{csv}` | {cap} |")
    (ROOT / "results/figures/CAPTIONS.md").write_text("\n".join(lines) + "\n")
    print("wrote results/figures/CAPTIONS.md with", len(CAPTIONS), "captions")


if __name__ == "__main__":
    main()
