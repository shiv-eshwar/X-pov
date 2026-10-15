"""Verify DOI metadata through Crossref and write a citable journal registry."""

from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def get_doi(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(url, headers={"User-Agent": "XProv-MAS research verification mailto:research@example.org"})
    with urllib.request.urlopen(request, timeout=8) as response:
        import json
        return json.load(response)["message"]


def year(item: dict) -> str:
    for key in ["published-print", "published-online", "issued", "created"]:
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def bib_key(item: dict) -> str:
    authors = item.get("author", [])
    surname = authors[0].get("family", "Anon") if authors else "Anon"
    title = re.sub(r"[^a-z0-9]+", "", item.get("title", ["work"])[0].lower())[:16]
    doi_suffix = re.sub(r"[^A-Za-z0-9]", "", item.get("doi", ""))[-10:]
    return f"{surname}{year(item)}{title}{doi_suffix}"


def bib_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi-file", default="artifacts/literature_dois.txt")
    ap.add_argument("--output", default="artifacts/literature_registry.csv")
    ap.add_argument("--bib", default="manuscript/references.bib")
    args = ap.parse_args()
    doi_path = Path(args.doi_file)
    if not doi_path.is_absolute():
        doi_path = ROOT / doi_path
    dois = [x.strip() for x in doi_path.read_text().splitlines() if x.strip() and not x.startswith("#")]
    rows = []
    for doi in dict.fromkeys(dois):
        try:
            item = get_doi(doi)
            container = "; ".join(item.get("container-title", []))
            is_journal = item.get("type") == "journal-article" and bool(container)
            authors = "; ".join(
                " ".join(x for x in [a.get("given", ""), a.get("family", "")] if x).strip()
                for a in item.get("author", [])
            )
            rows.append({
                "doi": doi,
                "verified": int(is_journal),
                "title": item.get("title", [""])[0],
                "authors": authors,
                "year": year(item),
                "journal": container,
                "volume": item.get("volume", ""),
                "issue": item.get("issue", ""),
                "pages": item.get("page", ""),
                "article_number": item.get("article-number", ""),
                "publisher": item.get("publisher", ""),
                "url": item.get("URL", ""),
            })
        except Exception as exc:
            rows.append({"doi": doi, "verified": 0, "error": repr(exc)})
        time.sleep(0.05)
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    bib = Path(args.bib)
    if not bib.is_absolute():
        bib = ROOT / bib
    bib.parent.mkdir(parents=True, exist_ok=True)
    with bib.open("w", encoding="utf-8") as f:
        for row in rows:
            if not row.get("verified"):
                continue
            key = re.sub(r"[^A-Za-z0-9:_-]", "", bib_key(row))
            author = " and ".join(x.strip() for x in row["authors"].split(";"))
            f.write(f"@article{{{key},\n")
            f.write(f"  author = {{{author}}},\n")
            f.write(f"  title = {{{bib_escape(row['title'])}}},\n")
            f.write(f"  journal = {{{bib_escape(row['journal'])}}},\n")
            if row.get("year"): f.write(f"  year = {{{row['year']}}},\n")
            if row.get("volume"): f.write(f"  volume = {{{row['volume']}}},\n")
            if row.get("issue"): f.write(f"  number = {{{row['issue']}}},\n")
            if row.get("pages"): f.write(f"  pages = {{{row['pages']}}},\n")
            if row.get("article_number"): f.write(f"  eid = {{{row['article_number']}}},\n")
            f.write(f"  doi = {{{row['doi']}}}\n}}\n\n")
    verified = [row for row in rows if row.get("verified")]
    print(f"DOIs checked: {len(rows)}; Crossref journal articles: {len(verified)}")
    for row in verified:
        print(f"{row['doi']} | {row['year']} | {row['journal']} | {row['title']}")


if __name__ == "__main__":
    main()
