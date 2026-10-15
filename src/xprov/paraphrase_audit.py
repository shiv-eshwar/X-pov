"""Evaluate the semantic edge scorer on the real PAWS paraphrase benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "raw"


def toks(x: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", x.lower()))


def overlap(a: str, b: str) -> float:
    aa, bb = toks(a), toks(b)
    return len(aa & bb) / len(aa | bb) if aa and bb else 0.0


def score_split(frame: pd.DataFrame, encoder: SentenceTransformer, split: str) -> pd.DataFrame:
    a = frame.sentence1.tolist()
    b = frame.sentence2.tolist()
    ea = encoder.encode(a, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    eb = encoder.encode(b, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    dense = np.sum(ea * eb, axis=1)
    lex = np.array([overlap(x, y) for x, y in zip(a, b)])
    hybrid = 0.65 * dense + 0.35 * lex
    y = frame.label.to_numpy(dtype=int)
    rows = []
    for name, values in [("lexical", lex), ("dense", dense), ("xprov_hybrid", hybrid)]:
        best = None
        for threshold in np.linspace(0.05, 0.95, 181):
            pred = (values >= threshold).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            item = (f, threshold, p, r)
            if best is None or item[0] > best[0]:
                best = item
        rows.append({
            "split": split,
            "scorer": name,
            "auc": roc_auc_score(y, values),
            "best_f1": best[0],
            "best_threshold": best[1],
            "best_precision": best[2],
            "best_recall": best[3],
            "n_pairs": len(frame),
            "positive_rate": float(y.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    ds = load_dataset("google-research-datasets/paws", "labeled_final")
    enc = SentenceTransformer(args.model)
    frames = []
    for split in ["train", "validation", "test"]:
        frames.append(score_split(ds[split].to_pandas(), enc, split))
    out = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "paws_rephrase_metrics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
