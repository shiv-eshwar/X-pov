"""Record-level boundary and failure summaries from saved metrics."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts" / "raw"

def main():
    paths = [
        RAW / "provenance_metrics.csv",
        RAW / "provenance_metrics_scifact-dev-v2.csv",
        RAW / "provenance_metrics_pubhealth-dev.csv",
        RAW / "provenance_metrics_pubhealth-evidence-dev.csv",
    ]
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    d = pd.concat(frames, ignore_index=True)
    d = d.sort_values(["dataset", "record_id", "strategy", "seed"])
    d = d.drop_duplicates(["dataset", "record_id", "strategy"])
    label = d.groupby(["dataset", "label", "strategy"], dropna=False).agg(
        n_records=("record_id", "nunique"),
        pair_f1_mean=("pair_f1", "mean"),
        pair_f1_sd=("pair_f1", "std"),
        rephrase_mean=("rephrase_match", "mean"),
        composable_mean=("composable_source_rate", "mean"),
    ).reset_index()
    source_count = d.groupby(["dataset", "n_gold_sources", "strategy"]).agg(
        n_records=("record_id", "nunique"),
        pair_f1_mean=("pair_f1", "mean"),
        rephrase_mean=("rephrase_match", "mean"),
        composable_mean=("composable_source_rate", "mean"),
    ).reset_index()
    low = d[d.strategy == "xprov_mas"].groupby("dataset", as_index=False).agg(
        n_records=("record_id", "nunique"),
        q10_pair_f1=("pair_f1", lambda x: x.quantile(.10)),
        median_pair_f1=("pair_f1", "median"),
        zero_pair_f1_rate=("pair_f1", lambda x: (x == 0).mean()),
        q10_rephrase=("rephrase_match", lambda x: x.quantile(.10)),
    )
    label.to_csv(RAW / "failure_by_label.csv", index=False)
    source_count.to_csv(RAW / "failure_by_gold_source_count.csv", index=False)
    low.to_csv(RAW / "failure_summary.csv", index=False)
    print("By label")
    print(label[label.strategy == "xprov_mas"].to_string(index=False))
    print("\nLow-tail summary")
    print(low.to_string(index=False))

if __name__ == "__main__":
    main()
