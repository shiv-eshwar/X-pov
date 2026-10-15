"""Record-level bootstrap uncertainty and paired strategy differences.

The retrieval/graph implementation is deterministic for a fixed encoder.  The
main benchmark therefore reports record-level bootstrap intervals rather than
pretending duplicated deterministic seeds are independent stochastic runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts" / "raw"


def bootstrap_frame(frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = [
        "pair_precision",
        "pair_recall",
        "pair_f1",
        "rephrase_match",
        "source_precision",
        "source_recall",
        "source_f1",
        "composable_source_rate",
    ]
    rows = []
    frame = frame.reset_index(drop=True)
    if len(frame) == 0:
        return pd.DataFrame()
    for (dataset, strategy), group in frame.groupby(["dataset", "strategy"], sort=True):
        group = group.reset_index(drop=True)
        draws = rng.integers(0, len(group), size=(n_boot, len(group)))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            means = values[draws].mean(axis=1)
            rows.append({
                "dataset": str(dataset),
                "strategy": strategy,
                "metric": metric,
                "estimate": float(values.mean()),
                "ci_low": float(np.quantile(means, 0.025)),
                "ci_high": float(np.quantile(means, 0.975)),
                "n_records": len(group),
                "n_boot": n_boot,
            })
    return pd.DataFrame(rows)


def paired_differences(frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_names = ["pair_f1", "rephrase_match", "composable_source_rate"]
    rows = []
    for dataset, dframe in frame.groupby("dataset", sort=True):
        wide = dframe.drop_duplicates(["record_id", "strategy"]).pivot(
            index="record_id", columns="strategy", values=metric_names
        )
        for metric in metric_names:
            if metric not in wide.columns.get_level_values(0):
                continue
            for challenger in ["dense", "xprov_mas"]:
                if (metric, challenger) not in wide or (metric, "lexical") not in wide:
                    continue
                values = (wide[(metric, challenger)] - wide[(metric, "lexical")]).dropna().to_numpy(float)
                if len(values) == 0:
                    continue
                draws = rng.integers(0, len(values), size=(n_boot, len(values)))
                means = values[draws].mean(axis=1)
                rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "comparison": f"{challenger}-lexical",
                    "estimate": float(values.mean()),
                    "ci_low": float(np.quantile(means, 0.025)),
                    "ci_high": float(np.quantile(means, 0.975)),
                    "n_records": len(values),
                    "n_boot": n_boot,
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", default="artifacts/raw/bootstrap_cis.csv")
    ap.add_argument("--paired-output", default="artifacts/raw/paired_differences.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    frames = []
    for item in args.inputs:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    # A deterministic benchmark may be run with several manifest seeds.  Keep
    # one record per strategy so the bootstrap unit is always the query record.
    frame = frame.sort_values(["dataset", "record_id", "strategy", "seed"])
    frame = frame.drop_duplicates(["dataset", "record_id", "strategy"])

    ci = bootstrap_frame(frame, args.n_boot, args.seed)
    paired = paired_differences(frame, args.n_boot, args.seed + 1)
    for path, data in [(args.output, ci), (args.paired_output, paired)]:
        out = Path(path)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(out, index=False)
    print(ci.to_string(index=False))
    print("\nPaired differences")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
