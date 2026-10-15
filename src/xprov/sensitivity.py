"""Threshold sensitivity for the XProv-MAS graph linker.

This script reuses saved agent assertions and evaluates a grid of graph-edge
thresholds without changing retrieval.  It is intentionally separate from
the benchmark runner so threshold selection cannot overwrite the primary raw
outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from xprov.experiment import (
    ROOT,
    cluster_assertions,
    load_fever,
    load_pubhealth,
    load_scifact,
    pair_features,
    pairwise_metrics,
    source_metrics,
)


def load_records(names: list[str], scifact_split: str):
    records = []
    if "scifact" in names:
        records.extend(load_scifact(scifact_split)[0])
    if "fever" in names:
        records.extend(load_fever())
    if "pubhealth" in names:
        records.extend(load_pubhealth())
    return records


def threshold_clusters(assertions, dense_threshold, hybrid_threshold, emb_cache, encoder):
    parent = list(range(len(assertions)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(assertions)):
        for j in range(i + 1, len(assertions)):
            _, dense, hybrid = pair_features(assertions[i], assertions[j], emb_cache, encoder)
            same = (hybrid >= hybrid_threshold and dense >= dense_threshold) or dense >= 0.80
            if same:
                union(i, j)
    return [find(i) for i in range(len(assertions))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assertions", required=True)
    ap.add_argument("--datasets", default="scifact,pubhealth")
    ap.add_argument("--scifact-split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dense", default="0.55,0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--hybrid", default="0.45,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--output", default="artifacts/raw/threshold_sensitivity.csv")
    args = ap.parse_args()

    assertion_path = Path(args.assertions)
    if not assertion_path.is_absolute():
        assertion_path = ROOT / assertion_path
    assertions = [json.loads(line) for line in assertion_path.open(encoding="utf-8") if line.strip()]
    records = load_records(args.datasets.split(","), args.scifact_split)
    wanted = {(r.dataset, r.record_id): r for r in records}
    assertions = [a for a in assertions if (a["dataset"], a["record_id"]) in wanted]
    by_key = {}
    for a in assertions:
        by_key.setdefault((a["dataset"], a["record_id"]), []).append(a)
    surfaces = sorted({a["surface"] for a in assertions})
    encoder = SentenceTransformer(args.model)
    embs = encoder.encode(surfaces, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    cache = dict(zip(surfaces, embs))
    rows = []
    dense_values = [float(x) for x in args.dense.split(",")]
    hybrid_values = [float(x) for x in args.hybrid.split(",")]
    for dense_threshold in dense_values:
        for hybrid_threshold in hybrid_values:
            per_record = []
            for key, group in by_key.items():
                clusters = threshold_clusters(group, dense_threshold, hybrid_threshold, cache, encoder)
                metrics = pairwise_metrics(group, clusters)
                metrics.update(source_metrics(group, wanted[key].gold_sources, clusters))
                per_record.append(metrics)
            if not per_record:
                continue
            frame = pd.DataFrame(per_record)
            rows.append({
                "dense_threshold": dense_threshold,
                "hybrid_threshold": hybrid_threshold,
                "n_records": len(frame),
                "pair_precision_mean": frame.pair_precision.mean(),
                "pair_recall_mean": frame.pair_recall.mean(),
                "pair_f1_mean": frame.pair_f1.mean(),
                "rephrase_match_mean": frame.rephrase_match.mean(),
                "source_f1_mean": frame.source_f1.mean(),
                "composable_source_rate_mean": frame.composable_source_rate.mean(),
            })
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).sort_values("pair_f1_mean", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
