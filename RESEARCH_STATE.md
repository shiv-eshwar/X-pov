# XProv-MAS research state

## Scientific scope

The working study is an experiment-first evaluation of a provenance graph
layer for multi-agent evidence systems. Agents are represented by independent
lexical, dense, and hybrid retrieval channels; the graph hides canonical source
IDs from the linker and evaluates whether semantically equivalent reports are
composed without merging conflicting evidence branches.

## Verified data acquired

- FEVER official development set: 19,998 labeled claims; the official June
  2017 Wikipedia snapshot was downloaded from `fever.ai` and extracted locally.
- SciFact official release: 1,409 claims and 5,183 scientific abstracts;
  train/dev/test files and corpus are under `data/raw/scifact/data`.
- PUBHEALTH official repository: 12,279 claims across train/dev/test, with
  evidence-bearing text and expert-written explanations; data and evidence
  documents are under `data/raw/pubhealth/repo/data`.
- PAWS labeled paraphrase benchmark: 49,401 train, 8,000 validation, and
  8,000 test pairs acquired through the official Hugging Face mirror.

## Executed evidence

- FEVER development: 19,998 claims and 299,970 assertions.
- SciFact development: 300 claims and 4,500 assertions.
- PUBHEALTH development: 1,220 evaluable claims and 18,315 assertions.
- PUBHEALTH official evidence subset: 33 claims and 495 assertions.
- PAWS train/validation/test semantic-edge audit: dense matching is more
  informative than lexical overlap on the hard paraphrase benchmark, but the
  initial hybrid scorer is not superior.
- SciFact threshold and top-$k$ sensitivity, record-level bootstrap intervals,
  source-retrieval ceilings, and label/source-count failure analyses.
- Literature registry: 55 Crossref-verified journal articles, plus official
  dataset, model, and provenance specifications.

## Manuscript status

The IEEE Access manuscript is complete and compiles to a 20-page PDF. It uses
the official IEEE Access class/template structure, includes raw-output-derived
tables and figures, and keeps the primary thresholds fixed while reporting
sensitivity and boundary results separately. The author name, affiliation, and
email in the template are placeholders and must be replaced before submission.
