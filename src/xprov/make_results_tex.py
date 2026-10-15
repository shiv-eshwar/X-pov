"""Render manuscript result macros from immutable CSV outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts" / "raw"
OUT = ROOT / "manuscript" / "results.tex"


def fmt(value, digits=3):
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def main():
    summaries = {
        "FEVER": RAW / "provenance_summary.csv",
        "SciFact": RAW / "provenance_summary_scifact-dev-v2.csv",
        "PUBHEALTH": RAW / "provenance_summary_pubhealth-dev.csv",
        "PUBHEALTH-EVIDENCE": RAW / "provenance_summary_pubhealth-evidence-dev.csv",
    }
    metric_paths = {
        "FEVER": RAW / "provenance_metrics.csv",
        "SciFact": RAW / "provenance_metrics_scifact-dev-v2.csv",
        "PUBHEALTH": RAW / "provenance_metrics_pubhealth-dev.csv",
        "PUBHEALTH-EVIDENCE": RAW / "provenance_metrics_pubhealth-evidence-dev.csv",
    }
    rows = []
    for dataset, path in summaries.items():
        d = pd.read_csv(path)
        metric_means = pd.read_csv(metric_paths[dataset]).groupby("strategy", as_index=False).agg(
            pair_precision=("pair_precision", "mean"),
            pair_recall=("pair_recall", "mean"),
        )
        d = d.merge(metric_means, on="strategy", how="left")
        for _, row in d.iterrows():
            row = row.to_dict()
            row["dataset"] = dataset
            rows.append(row)
    summary = pd.DataFrame(rows)
    ci = pd.read_csv(RAW / "bootstrap_cis.csv")
    paired = pd.read_csv(RAW / "paired_differences.csv")
    paws = pd.read_csv(RAW / "paws_rephrase_metrics.csv")
    sens = pd.read_csv(RAW / "threshold_sensitivity_scifact.csv")
    train_sens = pd.read_csv(RAW / "threshold_sensitivity_scifact_train.csv")

    lines = ["% Generated from artifacts/raw CSV outputs; do not hand edit."]
    macro_names = {"FEVER": "FEVER", "SciFact": "SciFact", "PUBHEALTH": "PubHealth", "PUBHEALTH-EVIDENCE": "PubHealthEvidence"}
    for dataset, prefix in macro_names.items():
        d = summary[summary.dataset == dataset]
        for strategy, label in [("dense", "Dense"), ("lexical", "Lexical"), ("xprov_mas", "XProv")]:
            r = d[d.strategy == strategy].iloc[0]
            lines += [
                f"\\newcommand{{\\{prefix}{label}Precision}}{{{fmt(r.pair_precision)}}}",
                f"\\newcommand{{\\{prefix}{label}Recall}}{{{fmt(r.pair_recall)}}}",
                f"\\newcommand{{\\{prefix}{label}FOne}}{{{fmt(r.pair_f1_mean)}}}",
                f"\\newcommand{{\\{prefix}{label}Rephrase}}{{{fmt(r.rephrase_match_mean)}}}",
                f"\\newcommand{{\\{prefix}{label}Composable}}{{{fmt(r.composable_source_rate_mean)}}}",
                f"\\newcommand{{\\{prefix}{label}SourceFOne}}{{{fmt(r.source_f1_mean)}}}",
            ]
        lines.append(f"\\newcommand{{\\{prefix}N}}{{{int(d.n_records.max())}}}")

    def get_ci(dataset, strategy, metric):
        r = ci[(ci.dataset == dataset) & (ci.strategy == strategy) & (ci.metric == metric)].iloc[0]
        return f"{fmt(r.estimate)} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]"

    for dataset, prefix in macro_names.items():
        for strategy, label in [("dense", "Dense"), ("lexical", "Lexical"), ("xprov_mas", "XProv")]:
            lines.append(f"\\newcommand{{\\{prefix}{label}FOneCI}}{{{get_ci(dataset, strategy, 'pair_f1')}}}")

    for dataset, prefix in macro_names.items():
        for metric, label in [("pair_f1", "PairFOne"), ("rephrase_match", "Rephrase"), ("composable_source_rate", "Composable")]:
            for comparison, cprefix in [("dense-lexical", "Dense"), ("xprov_mas-lexical", "XProv")]:
                r = paired[(paired.dataset == dataset) & (paired.metric == metric) & (paired.comparison == comparison)].iloc[0]
                lines.append(f"\\newcommand{{\\{prefix}{cprefix}{label}Delta}}{{{fmt(r.estimate)} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]}}")

    for split in ["train", "validation", "test"]:
        d = paws[paws.split == split]
        for scorer, label in [("lexical", "Lex"), ("dense", "Dense"), ("xprov_hybrid", "Hybrid")]:
            r = d[d.scorer == scorer].iloc[0]
            lines.append(f"\\newcommand{{\\PAWS{split.title()}{label}AUC}}{{{fmt(r.auc)}}}")
    best = sens.loc[sens.pair_f1_mean.idxmax()]
    train_best = train_sens.loc[train_sens.pair_f1_mean.idxmax()]
    lines += [
        f"\\newcommand{{\\SciFactSensitivityBest}}{{{fmt(best.pair_f1_mean)}}}",
        f"\\newcommand{{\\SciFactSensitivityPrimary}}{{{fmt(sens[(sens.dense_threshold == .60) & (sens.hybrid_threshold == .55)].pair_f1_mean.iloc[0])}}}",
        f"\\newcommand{{\\SciFactTrainBest}}{{{fmt(train_best.pair_f1_mean)}}}",
        f"\\newcommand{{\\SciFactTrainBestDense}}{{{fmt(train_best.dense_threshold, 2)}}}",
        f"\\newcommand{{\\SciFactTrainBestHybrid}}{{{fmt(train_best.hybrid_threshold, 2)}}}",
        f"\\newcommand{{\\FEVERAssertions}}{{299970}}",
        f"\\newcommand{{\\SciFactAssertions}}{{4500}}",
        f"\\newcommand{{\\PubHealthAssertions}}{{18315}}",
        f"\\newcommand{{\\EvidenceAssertions}}{{495}}",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(lines)} macros")


if __name__ == "__main__":
    main()
