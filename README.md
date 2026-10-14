# sentimentkit

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)

> Sentiment toolkit with swappable backends, evidence atoms, an eval harness and policy-as-code — built for German-language public-sector text.

🇩🇪 [Deutsche Version](README.de.md)

## Overview

sentimentkit judges the tone of *texts about topics* — media coverage, open survey answers, incoming requests, council debates, an organisation's own communication — and never the emotional state of a person. Every judgement is an **evidence atom** with full provenance (backend, model version, prompt hash, confidence, evidence tokens), every run is checked against a **policy file** before and after the model runs, and every backend is measured on the same **gold set** before it may become the default.

Stage 0 ships a transparent lexicon-and-rules baseline for German (negation incl. *Satzklammer*, compounds, intensifiers, contrast, Swiss-German flagging) plus the eval harness. Stage 2 adds a local **encoder** backend (fine-tuned transformer classifiers, CPU by default); local LLM scoring (fixed-answer / typed-question pattern) and a cloud LLM backend follow — see [docs/ROADMAP.md](docs/ROADMAP.md).

## Features

- **Swappable backends** behind one typed-question interface: `lexicon` and `encoder` (available), `llm-score`, `llm-cloud` (stage 2); a typo in `--backend` fails loudly
- **Backend comparison** on one gold set: `sentimentkit eval GOLD --compare lexicon,encoder` prints metrics, tag slices and the stage-2 gate `eval.beats_baseline()`
- **Evidence atoms** (`schema/judgement.schema.json`): polarity, signed intensity, confidence, per-option distribution, evidence cues, provenance, human-verification fields
- **Aspect-based** judgements from a YAML of typed questions, with clause-level scoping (`"gut, aber der Mittagstisch ein Chaos"`)
- **Policy as code** (`policy.yaml`): declared purpose, forbidden dimensions (political leaning, person emotion, biometrics), no per-person profiles, k-anonymity floor for aggregation, anonymisation requirement for cloud backends — violations exit with code 2
- **Eval harness**: accuracy, macro-F1, Cohen's kappa, ECE, tag slices, and the Alternative Annotator Test (Calderon et al. 2025) in pure Python
- **Human gate**: confidence threshold and top-2 margin route judgements to `review_queue.csv`
- **Outputs**: `judgements.jsonl`, `run.json`, `review_queue.csv`, `report.md`
- **Claude skill** `sentiment-scan` in `skills/`

## Prerequisites

- Python 3.10+
- The lexicon backend needs no model download and runs on a Raspberry Pi 5 or any CI container
- The encoder backend needs the optional extra `encoder` (PyTorch, transformers) and downloads its model once: 436 MB for the German default, 1.1 GB for the multilingual alternative

## Installation

```bash
git clone https://github.com/malkreide/sentimentkit.git
cd sentimentkit
pip install -e ".[dev]"
pip install -e ".[encoder]"   # optional: encoder backend (PyTorch, transformers)
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

# compare backends on the same gold set; the first one is the baseline
sentimentkit eval fixtures/de_synthetic_gold.jsonl --compare lexicon,encoder
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

## Stage 1: Building the Gold Set

Stage 1 replaces the fixture with 300–500 real sentences from public sources, each labelled by two people. Both scripts are rules only — no model, no network.

1. **Collect sources** as JSONL, one document per line with `id`, `source`, `date`, `title`, `text`: media articles via news-monitor-mcp, council speeches via zurich-opendata-mcp. Person fields (`author`, `name`, …) must be removed first; otherwise the policy aborts with exit code 2. A synthetic example: `fixtures/sources_synthetic.jsonl`.
2. **Build the annotation template.** Sentences are split (with repair of German false splits such as "3. März" or "12 Mio. Franken"), kept only with a school reference (`fixtures/keywords_schule.yaml`) and 5–60 tokens, de-duplicated (exact and near-duplicate), then sampled round-robin across sources and shuffled.
   ```bash
   python scripts/build_gold_candidates.py sources.jsonl --target 400
   ```
   Output: `fixtures/gold_v1_candidates.csv` with `text_id, text, source, a1, a2, tags, note`. Rule-based hints such as `negation?` stay unconfirmed until an annotator removes the `?`.
3. **Annotate** following [docs/ANNOTATION.md](docs/ANNOTATION.md) — each person in an own copy, only their own column.
4. **Merge into the gold set.** Copies are merged by `text_id`; `gold` is the majority label, `unclear` on dissent (an optional third column `a3` adjudicates). Every line is validated against `schema/gold.schema.json`; the script prints Cohen's κ (target ≥ 0.7) and the tag slices the stage-1 gate counts.
   ```bash
   python scripts/candidates_to_gold.py gold_v1_a1.csv gold_v1_a2.csv -o fixtures/gold_v1.jsonl
   ```
5. **Measure the baseline:** `sentimentkit eval fixtures/gold_v1.jsonl --backend lexicon`

## Encoder Backend (Stage 2)

`--backend encoder` runs a fine-tuned transformer classifier locally. The softmax over its three classes is the judgement's `distribution`, the top probability its `confidence`, P(positive) − P(negative) its `intensity`. Texts never leave the machine; only the weights are downloaded once from the Hugging Face Hub (or read from a local directory). `decide_many` batches for real, and `analyze` and `eval` group their questions per aspect so that the encoder sees whole batches. Aspects are scoped to the clauses that mention them, exactly as in the lexicon backend; without a hit the answer is `not_mentioned`, without a forward pass.

| Model | Languages | Licence | Pinned revision |
|---|---|---|---|
| `oliverguhr/german-sentiment-bert` (default) | German | MIT | `b1177ff59e30` |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` (French, Italian) | 8, incl. de, fr, it | not declared on the model card | `f2f1202b1bde` |

Configuration via environment variables (or constructor arguments of `EncoderBackend`):

| Variable | Default | Meaning |
|---|---|---|
| `SENTIMENTKIT_ENCODER_MODEL` | `oliverguhr/german-sentiment-bert` | Hub id or local directory |
| `SENTIMENTKIT_ENCODER_REVISION` | pinned commit of the two presets | commit, tag or branch |
| `SENTIMENTKIT_ENCODER_DEVICE` | `cpu` | `cpu`, `cuda`, `mps` |
| `SENTIMENTKIT_ENCODER_BATCH_SIZE` | `16` | texts per forward pass |

`model_version` is `<model>@<commit>`; a branch or tag is resolved to the commit the weights were actually loaded from. Both presets are pinned, so a re-upload on the Hub cannot change results unnoticed. Each preset gets the input cleaning its training data had (`germansentiment`: letters only, digits spelled out, lower case; XLM-R: user and link placeholders). Evidence lists the class probabilities plus flags: `DIALECT?`, `LANG?:fr` (language outside the model's languages), `TRUNCATED?` (input longer than 512 tokens). Offline after the first download: `HF_HUB_OFFLINE=1`. The backend switches off transformers' background safetensors conversion, which would otherwise contact the Hub and download a converted copy from an open pull request on the model repo.

Measured on the stage-0 fixture (30 synthetic German sentences; transformers 5.19, torch 2.14 CPU):

| Metric | lexicon | encoder, german-sentiment-bert | encoder, twitter-xlm-roberta |
|---|---|---|---|
| accuracy | 0.933 | 0.633 | 0.900 |
| macro-F1 | 0.777 | 0.700 | 0.881 |
| Cohen's κ vs. gold | 0.892 | 0.488 | 0.840 |
| ECE | 0.203 | 0.351 | 0.151 |
| irony slice (n=2) | 0.00 | 0.00 | 1.00 |
| negation slice (n=7) | 1.00 | 0.86 | 0.57 |
| alt-test ω | 0.50 (pass) | 0.00 (fail) | 1.00 (pass) |
| `eval.beats_baseline()` | — | no (−0.077) | yes (+0.104) |

Read these figures with care:

- **Not the stage-2 gate.** The fixture has 30 sentences and was written alongside the lexicon rules, which flatters the lexicon. The gate counts on the stage-1 gold set (`fixtures/gold_v1.jsonl`), which does not exist yet.
- **The German default is confidently wrong.** 10 of its 11 errors have confidence ≥ 0.9; mostly it calls evaluative but formally worded sentences `neutral`. On the synthetic survey it sends 0 of 144 judgements to review (lexicon: 10). Uncalibrated, it bypasses the human gate — do not use it for reports before calibration (stage-2 gate: ECE < 0.10).
- **The multilingual model** is ahead on this fixture, also for German, and its errors are low-confidence. Its model card declares no licence; clarify that before productive use.

## Available Tools

| Command | Description |
|---|---|
| `sentimentkit analyze INPUT --purpose P [--backend B] [--aspects A.yaml] [-o DIR]` | Judge every text; write atoms, run meta, review queue and report |
| `sentimentkit eval GOLD.jsonl [--backend B] [--json]` | Metrics, tag slices, alt-test, failure list |
| `sentimentkit eval GOLD.jsonl --compare B1,B2[,…] [--json]` | Same gold set for every backend, side by side, plus `eval.beats_baseline()` against B1 |
| `sentimentkit backends` | List backends: available, missing extra, or roadmap stage |
| `sentimentkit policy-check [--policy FILE]` | Load the policy and print its hash |
| `sentimentkit validate judgements.jsonl` | Validate atoms against the JSON schema |

Exit codes: `0` ok · `1` error · `2` policy violation.

## Configuration

| File | Purpose |
|---|---|
| `policy.yaml` | Governance rules; hashed into every run |
| `lexicon/de_seed.tsv`, `lexicon/de_modifiers.yaml` | Baseline lexicon (own work, MIT) and German rules |
| `fixtures/aspects_schule.yaml` | Example typed questions for school/parent feedback |
| `fixtures/keywords_schule.yaml` | School-reference filter for stage-1 candidates; doubles as `aspect_terms` of the gold set |
| `schema/*.schema.json` | Judgement, run and gold-set schemas (draft 2020-12) |
| `SENTIMENTKIT_ENCODER_*` | Model, revision, device and batch size of the encoder backend (see above) |

Input formats: CSV/TSV (column `text`), JSONL/JSON (`{"text": …}`), Markdown/TXT (one item per paragraph).

## Project Structure

```
sentimentkit/
├── src/sentimentkit/
│   ├── models.py          # TextItem, Judgement (evidence atom), RunMeta
│   ├── policy.py          # Policy, PolicyViolation, checks
│   ├── preprocess.py      # sentence split, tokens, language guess, decompound
│   ├── backends/          # base.py (interface), lexicon.py, encoder.py, registry
│   ├── eval/              # metrics.py, alt_test.py, gold-set loader, gates
│   ├── pipeline.py        # policy → backend → atoms → gate
│   ├── gate.py            # human review routing
│   ├── report.py          # aggregation with k-anonymity floor, Markdown
│   ├── io.py              # readers/writers
│   └── cli.py
├── schema/                # JSON schemas
├── lexicon/               # seed lexicon and modifiers (German)
├── fixtures/              # synthetic gold set, survey CSV, source documents, aspects and keywords
├── scripts/               # build_gold_candidates.py, candidates_to_gold.py, validate_repo.py
├── skills/sentiment-scan/ # Claude skill
├── docs/                  # ROADMAP.md, ARCHITECTURE.md, ANNOTATION.md
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
