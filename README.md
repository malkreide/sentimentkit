# sentimentkit

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)

> Sentiment toolkit with swappable backends, evidence atoms, an eval harness and policy-as-code — built for German-language public-sector text.

🇩🇪 [Deutsche Version](README.de.md)

## Overview

sentimentkit judges the tone of *texts about topics* — media coverage, open survey answers, incoming requests, council debates, an organisation's own communication — and never the emotional state of a person. Every judgement is an **evidence atom** with full provenance (backend, model version, prompt hash, confidence, evidence tokens), every run is checked against a **policy file** before and after the model runs, and every backend is measured on the same **gold set** before it may become the default.

Stage 0 ships a transparent lexicon-and-rules baseline for German (negation incl. *Satzklammer*, compounds, intensifiers, contrast, Swiss-German flagging) plus the eval harness. Encoder, local LLM scoring (fixed-answer / typed-question pattern) and cloud LLM backends are scheduled for stage 2 — see [docs/ROADMAP.md](docs/ROADMAP.md).

## Features

- **Swappable backends** behind one typed-question interface: `lexicon` (available), `encoder`, `llm-score`, `llm-cloud` (stage 2); a typo in `--backend` fails loudly
- **Evidence atoms** (`schema/judgement.schema.json`): polarity, signed intensity, confidence, per-option distribution, evidence cues, provenance, human-verification fields
- **Aspect-based** judgements from a YAML of typed questions, with clause-level scoping (`"gut, aber der Mittagstisch ein Chaos"`)
- **Policy as code** (`policy.yaml`): declared purpose, forbidden dimensions (political leaning, person emotion, biometrics), no per-person profiles, k-anonymity floor for aggregation, anonymisation requirement for cloud backends — violations exit with code 2
- **Eval harness**: accuracy, macro-F1, Cohen's kappa, ECE, tag slices, and the Alternative Annotator Test (Calderon et al. 2025) in pure Python
- **Human gate**: confidence threshold and top-2 margin route judgements to `review_queue.csv`
- **Outputs**: `judgements.jsonl`, `run.json`, `review_queue.csv`, `report.md`
- **Claude skill** `sentiment-scan` in `skills/`

## Prerequisites

- Python 3.10+
- No model downloads in stage 0; the lexicon backend runs on a Raspberry Pi 5 or any CI container

## Installation

```bash
git clone https://github.com/malkreide/sentimentkit.git
cd sentimentkit
pip install -e ".[dev]"
```

## Usage / Quickstart

```bash
sentimentkit backends                       # what exists in this stage
sentimentkit policy-check                   # print the policy and its hash

# judge a synthetic survey export (24 rows, no person identifiers)
sentimentkit analyze fixtures/survey_synthetic.csv \
  --purpose survey_open_answers \
  --aspects fixtures/aspects_schule.yaml \
  -o out/
cat out/report.md

# evaluate the baseline on the synthetic gold set
sentimentkit eval fixtures/de_synthetic_gold.jsonl --backend lexicon
```

Baseline on the stage-0 fixture set (30 synthetic German sentences, two synthetic annotators):

| Metric | lexicon (seed:120) |
|---|---|
| accuracy | 0.933 |
| macro-F1 | 0.777 |
| Cohen's κ vs. gold | 0.892 |
| ECE | 0.203 (uncalibrated by design) |
| irony slice | 0.00 — documented weakness, the stage-2 benchmark |
| alt-test ω | 0.50 (pass) |

These numbers describe a fixture, not a domain. Stage 1 replaces the fixture with a real, public gold set.

## Available Tools

| Command | Description |
|---|---|
| `sentimentkit analyze INPUT --purpose P [--backend B] [--aspects A.yaml] [-o DIR]` | Judge every text; write atoms, run meta, review queue and report |
| `sentimentkit eval GOLD.jsonl [--backend B] [--json]` | Metrics, tag slices, alt-test, failure list |
| `sentimentkit backends` | List backends and their roadmap stage |
| `sentimentkit policy-check [--policy FILE]` | Load the policy and print its hash |
| `sentimentkit validate judgements.jsonl` | Validate atoms against the JSON schema |

Exit codes: `0` ok · `1` error · `2` policy violation.

## Configuration

| File | Purpose |
|---|---|
| `policy.yaml` | Governance rules; hashed into every run |
| `lexicon/de_seed.tsv`, `lexicon/de_modifiers.yaml` | Baseline lexicon (own work, MIT) and German rules |
| `fixtures/aspects_schule.yaml` | Example typed questions for school/parent feedback |
| `schema/*.schema.json` | Judgement, run and gold-set schemas (draft 2020-12) |

Input formats: CSV/TSV (column `text`), JSONL/JSON (`{"text": …}`), Markdown/TXT (one item per paragraph).

## Project Structure

```
sentimentkit/
├── src/sentimentkit/
│   ├── models.py          # TextItem, Judgement (evidence atom), RunMeta
│   ├── policy.py          # Policy, PolicyViolation, checks
│   ├── preprocess.py      # sentence split, tokens, language guess, decompound
│   ├── backends/          # base.py (interface), lexicon.py, registry
│   ├── eval/              # metrics.py, alt_test.py, gold-set loader, gates
│   ├── pipeline.py        # policy → backend → atoms → gate
│   ├── gate.py            # human review routing
│   ├── report.py          # aggregation with k-anonymity floor, Markdown
│   ├── io.py              # readers/writers
│   └── cli.py
├── schema/                # JSON schemas
├── lexicon/               # seed lexicon and modifiers (German)
├── fixtures/              # synthetic gold set, survey CSV, aspects example
├── skills/sentiment-scan/ # Claude skill
├── docs/                  # ROADMAP.md, ARCHITECTURE.md
├── tests/
└── policy.yaml
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md)

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Please report vulnerabilities as described in [SECURITY.md](SECURITY.md). The toolkit processes text only; it must never be pointed at person-level data — the policy refuses, and so should you.

## License

MIT License — see [LICENSE](LICENSE)

## Author

Hayal Özkan · [malkreide](https://github.com/malkreide)
