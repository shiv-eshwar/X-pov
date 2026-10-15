"""Generate manuscript figures from saved raw outputs only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts" / "raw"
FIG = ROOT / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)


def save(fig, name: str):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def framework():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, 0.67, 0.20, 0.19, "Agent 1\nlexical retrieval", "#dbeafe"),
        (0.03, 0.405, 0.20, 0.19, "Agent 2\ndense retrieval", "#dcfce7"),
        (0.03, 0.14, 0.20, 0.19, "Agent 3\nhybrid retrieval", "#fef3c7"),
        (0.36, 0.35, 0.25, 0.30, "Provenance\nassertions\n\nclaim, evidence,\nrank, score", "#ede9fe"),
        (0.77, 0.35, 0.19, 0.30, "XProv-MAS\nsemantic graph\n\nclusters, links,\nconfidence", "#fee2e2"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", linewidth=1.0, edgecolor="#334155", facecolor=color))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=8.4, color="#0f172a")
    for y in [0.765, 0.50, 0.235]:
        ax.annotate("", xy=(0.36, 0.50), xytext=(0.235, y), arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.1))
    ax.annotate("", xy=(0.77, 0.50), xytext=(0.61, 0.50), arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))
    ax.text(0.665, 0.54, "surface-invariant\nlinking", ha="center", va="bottom", fontsize=7.2, color="#475569")
    ax.text(0.865, 0.27, "reusable evidence\nlineage", ha="center", va="top", fontsize=7.2, color="#475569")
    ax.text(0.50, 0.93, "Independent channels are composed after retrieval; canonical source IDs remain hidden from the linker", ha="center", va="center", fontsize=8.2, weight="bold")
    save(fig, "framework")


def performance():
    files = [
        RAW / "provenance_summary.csv",
        RAW / "provenance_summary_scifact-dev-v2.csv",
        RAW / "provenance_summary_pubhealth-dev.csv",
        RAW / "provenance_summary_pubhealth-evidence-dev.csv",
    ]
    frames = [pd.read_csv(p) for p in files if p.exists()]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df = df[df.strategy.isin(["lexical", "dense", "xprov_mas"])].copy()
    # Retain one file per dataset, preferring the calibrated reruns.
    df = df.drop_duplicates(["dataset", "strategy"], keep="last")
    order = [x for x in ["FEVER", "SciFact", "PUBHEALTH", "PUBHEALTH-EVIDENCE"] if x in set(df.dataset)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    sns.barplot(data=df, x="dataset", y="pair_f1_mean", hue="strategy", order=order, ax=axes[0], palette="Set2")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Pairwise provenance F1")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(title="Linker", fontsize=7, title_fontsize=7, loc="lower left")
    sns.barplot(data=df, x="dataset", y="rephrase_match_mean", hue="strategy", order=order, ax=axes[1], palette="Set2", legend=False)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Cross-agent rephrase match")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout(w_pad=1.2)
    save(fig, "performance")


def robustness():
    p = RAW / "paws_rephrase_metrics.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df.scorer.isin(["lexical", "dense", "xprov_hybrid"])].copy()
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    sns.barplot(data=df, x="split", y="auc", hue="scorer", ax=ax, palette="Set2")
    ax.set_ylim(0.45, 0.70)
    ax.set_ylabel("ROC AUC on PAWS")
    ax.set_xlabel("")
    ax.legend(title="Surface scorer", fontsize=6.7, title_fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "paws_robustness")


def sensitivity():
    p = RAW / "threshold_sensitivity_scifact.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    pivot = df.pivot(index="dense_threshold", columns="hybrid_threshold", values="pair_f1_mean")
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.2, vmax=0.8, cbar_kws={"label": "Pairwise F1"}, ax=ax)
    ax.set_xlabel("Hybrid edge threshold")
    ax.set_ylabel("Dense edge threshold")
    fig.tight_layout()
    save(fig, "threshold_sensitivity")


if __name__ == "__main__":
    framework()
    performance()
    robustness()
    sensitivity()
