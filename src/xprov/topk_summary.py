from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts" / "raw"

def main():
    rows = []
    for k in [1, 3, 5, 10]:
        tag = "scifact-dev-v2" if k == 5 else f"scifact-topk{k}"
        d = pd.read_csv(RAW / f"provenance_summary_{tag}.csv")
        d = d[d.strategy.isin(["lexical", "dense", "xprov_mas"])].copy()
        d["top_k"] = k
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(RAW / "topk_sensitivity.csv", index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
