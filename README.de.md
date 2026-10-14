# sentimentkit

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)

> Sentiment-Toolkit mit austauschbaren Backends, Evidenz-Atomen, Eval-Harness und Governance-als-Code — gebaut für deutschsprachige Texte aus dem öffentlichen Sektor.

🇬🇧 [English Version](README.md)

## Übersicht

sentimentkit bestimmt die Tonalität von *Texten über Themen* — Medienberichte, offene Befragungsantworten, Anfragen, Ratsdebatten, die eigene Kommunikation — und nie den emotionalen Zustand einer Person. Jedes Urteil ist ein **Evidenz-Atom** mit vollständiger Provenienz (Backend, Modellversion, Prompt-Hash, Konfidenz, auslösende Tokens), jeder Lauf wird vor und nach dem Modell gegen eine **Policy-Datei** geprüft, und jedes Backend wird auf demselben **Gold-Set** gemessen, bevor es Default werden darf.

Stufe 0 liefert eine transparente Lexikon-und-Regel-Baseline für Deutsch (Negation inkl. Satzklammer, Komposita, Verstärker, Kontrast, Schweizerdeutsch-Erkennung) und den Eval-Harness. Encoder, lokales LLM-Scoring (Fixed-Answer / Typed-Question-Muster) und Cloud-LLM folgen in Stufe 2 — siehe [docs/ROADMAP.md](docs/ROADMAP.md).

## Funktionen

- **Austauschbare Backends** hinter einer Typed-Question-Schnittstelle: `lexicon` (verfügbar), `encoder`, `llm-score`, `llm-cloud` (Stufe 2); ein Tippfehler in `--backend` schlägt laut fehl
- **Evidenz-Atome** (`schema/judgement.schema.json`): Polarität, Intensität mit Vorzeichen, Konfidenz, Verteilung je Option, Evidenz-Hinweise, Provenienz, Felder für die menschliche Prüfung
- **Aspektbasierte** Urteile aus einem YAML mit Typed Questions, mit Scoping auf Teilsatzebene («gut, aber der Mittagstisch ein Chaos»)
- **Governance als Code** (`policy.yaml`): deklarierter Zweck, verbotene Dimensionen (politische Ausrichtung, Emotion einer Person, Biometrie), keine Personenprofile, k-Anonymitäts-Untergrenze bei der Aggregation, Anonymisierungspflicht für Cloud-Backends — Verstösse enden mit Exit 2
- **Eval-Harness**: Accuracy, Macro-F1, Cohen's Kappa, ECE, Tag-Slices und der Alternative Annotator Test (Calderon et al. 2025) in reinem Python
- **Human Gate**: Konfidenzschwelle und Top-2-Margin leiten Urteile in `review_queue.csv`
- **Ausgaben**: `judgements.jsonl`, `run.json`, `review_queue.csv`, `report.md`
- **Claude-Skill** `sentiment-scan` in `skills/`

## Voraussetzungen

- Python 3.10+
- Keine Modell-Downloads in Stufe 0; das Lexikon-Backend läuft auf einem Raspberry Pi 5 oder jedem CI-Container

## Installation

```bash
git clone https://github.com/malkreide/sentimentkit.git
cd sentimentkit
pip install -e ".[dev]"
```

## Verwendung / Schnellstart

```bash
sentimentkit backends                       # was in dieser Stufe existiert
sentimentkit policy-check                   # Policy und ihren Hash anzeigen

# synthetischen Befragungs-Export beurteilen (24 Zeilen, keine Personenkennungen)
sentimentkit analyze fixtures/survey_synthetic.csv \
  --purpose survey_open_answers \
  --aspects fixtures/aspects_schule.yaml \
  -o out/
cat out/report.md

# Baseline auf dem synthetischen Gold-Set messen
sentimentkit eval fixtures/de_synthetic_gold.jsonl --backend lexicon
```

Baseline auf dem Fixture-Set der Stufe 0 (30 synthetische deutsche Sätze, zwei synthetische Annotierende):

| Kennzahl | lexicon (seed:120) |
|---|---|
| Accuracy | 0.933 |
| Macro-F1 | 0.777 |
| Cohen's κ vs. Gold | 0.892 |
| ECE | 0.203 (absichtlich unkalibriert) |
| Slice Ironie | 0.00 — dokumentierte Schwäche, Massstab für Stufe 2 |
| alt-test ω | 0.50 (bestanden) |

Diese Zahlen beschreiben ein Fixture, keine Domäne. Stufe 1 ersetzt das Fixture durch ein reales, öffentliches Gold-Set.

## Verfügbare Tools

| Befehl | Beschreibung |
|---|---|
| `sentimentkit analyze INPUT --purpose P [--backend B] [--aspects A.yaml] [-o DIR]` | Jeden Text beurteilen; Atome, Run-Meta, Review-Queue und Report schreiben |
| `sentimentkit eval GOLD.jsonl [--backend B] [--json]` | Kennzahlen, Tag-Slices, alt-test, Fehlerliste |
| `sentimentkit backends` | Backends und ihre Roadmap-Stufe auflisten |
| `sentimentkit policy-check [--policy DATEI]` | Policy laden und Hash ausgeben |
| `sentimentkit validate judgements.jsonl` | Atome gegen das JSON-Schema prüfen |

Exit-Codes: `0` ok · `1` Fehler · `2` Policy-Verstoss.

## Konfiguration

| Datei | Zweck |
|---|---|
| `policy.yaml` | Governance-Regeln; wird in jeden Lauf gehasht |
| `lexicon/de_seed.tsv`, `lexicon/de_modifiers.yaml` | Baseline-Lexikon (eigene Arbeit, MIT) und deutsche Regeln |
| `fixtures/aspects_schule.yaml` | Beispiel-Typed-Questions für Schul-/Elternfeedback |
| `schema/*.schema.json` | Schemas für Urteil, Lauf und Gold-Set (Draft 2020-12) |

Eingabeformate: CSV/TSV (Spalte `text`), JSONL/JSON (`{"text": …}`), Markdown/TXT (ein Eintrag je Absatz).

## Projektstruktur

```
sentimentkit/
├── src/sentimentkit/
│   ├── models.py          # TextItem, Judgement (Evidenz-Atom), RunMeta
│   ├── policy.py          # Policy, PolicyViolation, Prüfungen
│   ├── preprocess.py      # Satzsplit, Tokens, Sprachschätzung, Decompound
│   ├── backends/          # base.py (Interface), lexicon.py, Registry
│   ├── eval/              # metrics.py, alt_test.py, Gold-Set-Loader, Gates
│   ├── pipeline.py        # Policy → Backend → Atome → Gate
│   ├── gate.py            # Routing in die menschliche Prüfung
│   ├── report.py          # Aggregation mit k-Anonymitäts-Untergrenze, Markdown
│   ├── io.py              # Leser/Schreiber
│   └── cli.py
├── schema/                # JSON-Schemas
├── lexicon/               # Seed-Lexikon und Modifikatoren (Deutsch)
├── fixtures/              # synthetisches Gold-Set, Befragungs-CSV, Aspekte-Beispiel
├── skills/sentiment-scan/ # Claude-Skill
├── docs/                  # ROADMAP.md, ARCHITECTURE.md
├── tests/
└── policy.yaml
```

## Changelog

Siehe [CHANGELOG.md](CHANGELOG.md)

## Mitwirken

Beiträge sind willkommen — siehe [CONTRIBUTING.md](CONTRIBUTING.md).

## Sicherheit

Sicherheitslücken bitte wie in [SECURITY.md](SECURITY.md) beschrieben melden. Das Toolkit verarbeitet nur Text; es darf nie auf personenbezogene Daten gerichtet werden — die Policy verweigert das, und das sollte auch die Anwenderin oder der Anwender tun.

## Lizenz

MIT-Lizenz — siehe [LICENSE](LICENSE)

## Autor

Hayal Özkan · [malkreide](https://github.com/malkreide)
