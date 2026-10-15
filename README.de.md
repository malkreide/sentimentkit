# sentimentkit

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)

> Sentiment-Toolkit mit austauschbaren Backends, Evidenz-Atomen, Eval-Harness und Governance-als-Code — gebaut für deutschsprachige Texte aus dem öffentlichen Sektor.

🇬🇧 [English Version](README.md)

## Übersicht

sentimentkit bestimmt die Tonalität von *Texten über Themen* — Medienberichte, offene Befragungsantworten, Anfragen, Ratsdebatten, die eigene Kommunikation — und nie den emotionalen Zustand einer Person. Jedes Urteil ist ein **Evidenz-Atom** mit vollständiger Provenienz (Backend, Modellversion, Prompt-Hash, Konfidenz, auslösende Tokens), jeder Lauf wird vor und nach dem Modell gegen eine **Policy-Datei** geprüft, und jedes Backend wird auf demselben **Gold-Set** gemessen, bevor es Default werden darf.

Stufe 0 liefert eine transparente Lexikon-und-Regel-Baseline für Deutsch (Negation inkl. Satzklammer, Komposita, Verstärker, Kontrast, Schweizerdeutsch-Erkennung) und den Eval-Harness. Stufe 2 ergänzt ein lokales **Encoder**-Backend (feinjustierte Transformer-Klassifikatoren, standardmässig auf der CPU) und **LLM-Scoring** (Typed Questions mit festen Antworten, bewertet über SGLang oder Ollama, kein Freitext); ein Cloud-LLM folgt — siehe [docs/ROADMAP.md](docs/ROADMAP.md).

## Funktionen

- **Austauschbare Backends** hinter einer Typed-Question-Schnittstelle: `lexicon`, `encoder` und `llm-score` (verfügbar), `llm-cloud` (Stufe 2); ein Tippfehler in `--backend` schlägt laut fehl
- **Backend-Vergleich** auf einem Gold-Set: `sentimentkit eval GOLD --compare lexicon,encoder` gibt Kennzahlen, Tag-Slices und das Stufe-2-Gate `eval.beats_baseline()` aus
- **Evidenz-Atome** (`schema/judgement.schema.json`): Polarität, Intensität mit Vorzeichen, Konfidenz, Verteilung je Option, Evidenz-Hinweise, Provenienz, Felder für die menschliche Prüfung
- **Aspektbasierte** Urteile aus einem YAML mit Typed Questions, mit Scoping auf Teilsatzebene («gut, aber der Mittagstisch ein Chaos»)
- **Governance als Code** (`policy.yaml`): deklarierter Zweck, verbotene Dimensionen (politische Ausrichtung, Emotion einer Person, Biometrie), keine Personenprofile, k-Anonymitäts-Untergrenze bei der Aggregation, Anonymisierungspflicht für Cloud-Backends — Verstösse enden mit Exit 2
- **Eval-Harness**: Accuracy, Macro-F1, Cohen's Kappa, ECE, Tag-Slices und der Alternative Annotator Test (Calderon et al. 2025) in reinem Python
- **Kalibrierung**: `sentimentkit eval` gibt die ECE vor und nach Temperature-Scaling (kreuzvalidiert) und die zu setzende Temperatur aus; die Review-Schwellen bleiben in `policy.yaml`
- **Human Gate**: Konfidenzschwelle und Top-2-Margin leiten Urteile in `review_queue.csv`
- **Ausgaben**: `judgements.jsonl`, `run.json`, `review_queue.csv`, `report.md`
- **Claude-Skill** `sentiment-scan` in `skills/`

## Voraussetzungen

- Python 3.10+
- Das Lexikon-Backend braucht keinen Modell-Download und läuft auf einem Raspberry Pi 5 oder jedem CI-Container
- Das Encoder-Backend braucht das optionale Extra `encoder` (PyTorch, transformers) und lädt sein Modell einmal herunter: 436 MB für den deutschen Default, 1.1 GB für die mehrsprachige Alternative
- Das LLM-Scoring-Backend braucht das optionale Extra `llm` (httpx, Tokenizer ohne PyTorch) und einen Scoring-Server, den du selbst betreibst: SGLang (`/v1/score`, in der Praxis auf einer GPU) oder Ollama mit Logprobs-Unterstützung (CPU, Laptop, Raspberry Pi)

## Installation

```bash
git clone https://github.com/malkreide/sentimentkit.git
cd sentimentkit
pip install -e ".[dev]"
pip install -e ".[encoder]"   # optional: Encoder-Backend (PyTorch, transformers)
pip install -e ".[llm]"       # optional: LLM-Scoring-Backend (httpx, Tokenizer)
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

# Backends auf demselben Gold-Set vergleichen; das erste ist die Baseline
sentimentkit eval fixtures/de_synthetic_gold.jsonl --compare lexicon,encoder
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

## Stufe 1: Gold-Set aufbauen

Stufe 1 ersetzt das Fixture durch 300–500 reale Sätze aus öffentlichen Quellen, die je zwei Personen beurteilen. Beide Skripte arbeiten nur mit Regeln — kein Modell, kein Netzwerk.

1. **Quellen sammeln** als JSONL, ein Dokument pro Zeile mit `id`, `source`, `date`, `title`, `text`: Medienartikel über news-monitor-mcp, Gemeinderats-Voten über zurich-opendata-mcp. Personenfelder (`author`, `name` …) vorher entfernen, sonst bricht die Policy mit Exit 2 ab. Synthetisches Beispiel: `fixtures/sources_synthetic.jsonl`.
2. **Annotationsvorlage erzeugen.** Sätze werden getrennt (mit Reparatur deutscher Fehltrennungen wie «3. März» oder «12 Mio. Franken»), nur mit Schulbezug (`fixtures/keywords_schule.yaml`) und 5–60 Tokens behalten, von Dubletten und Beinahe-Dubletten bereinigt, reihum über die Quellen ausgewählt und gemischt.
   ```bash
   python scripts/build_gold_candidates.py quellen.jsonl --target 400
   ```
   Ergebnis: `fixtures/gold_v1_candidates.csv` mit `text_id, text, source, a1, a2, tags, note`. Regelbasierte Vorschläge wie `negation?` gelten erst, wenn eine annotierende Person das `?` entfernt.
3. **Annotieren** nach [docs/ANNOTATION.md](docs/ANNOTATION.md) — jede Person in einer eigenen Kopie, nur in der eigenen Spalte.
4. **Zum Gold-Set zusammenführen.** Die Kopien werden über `text_id` vereint; `gold` ist das Mehrheitslabel, bei Dissens `unclear` (eine optionale dritte Spalte `a3` schlichtet). Jede Zeile wird gegen `schema/gold.schema.json` geprüft; das Skript gibt Cohen's κ (Ziel ≥ 0.7) und die Tag-Slices des Stufe-1-Gates aus.
   ```bash
   python scripts/candidates_to_gold.py gold_v1_a1.csv gold_v1_a2.csv -o fixtures/gold_v1.jsonl
   ```
5. **Baseline messen:** `sentimentkit eval fixtures/gold_v1.jsonl --backend lexicon`

## Encoder-Backend (Stufe 2a)

`--backend encoder` lässt einen feinjustierten Transformer-Klassifikator lokal laufen. Der Softmax über seine drei Klassen ist die `distribution` des Urteils, die höchste Wahrscheinlichkeit seine `confidence`, P(positiv) − P(negativ) seine `intensity`. Texte verlassen den Rechner nie; nur die Gewichte werden einmal vom Hugging Face Hub geladen (oder aus einem lokalen Verzeichnis gelesen). `decide_many` verarbeitet echte Batches, und `analyze` sowie `eval` gruppieren ihre Fragen je Aspekt, damit der Encoder ganze Batches erhält. Aspekte werden wie im Lexikon-Backend auf die Teilsätze beschränkt, die sie nennen; ohne Treffer lautet die Antwort `not_mentioned`, ohne Modelldurchlauf.

| Modell | Sprachen | Lizenz | Gepinnte Revision |
|---|---|---|---|
| `oliverguhr/german-sentiment-bert` (Default) | Deutsch | MIT | `b1177ff59e30` |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` (Französisch, Italienisch) | 8, u. a. de, fr, it | auf der Model Card nicht deklariert | `f2f1202b1bde` |

Konfiguration über Umgebungsvariablen (oder Konstruktor-Argumente von `EncoderBackend`):

| Variable | Default | Bedeutung |
|---|---|---|
| `SENTIMENTKIT_ENCODER_MODEL` | `oliverguhr/german-sentiment-bert` | Hub-ID oder lokales Verzeichnis |
| `SENTIMENTKIT_ENCODER_REVISION` | gepinnter Commit der beiden Presets | Commit, Tag oder Branch |
| `SENTIMENTKIT_ENCODER_DEVICE` | `cpu` | `cpu`, `cuda`, `mps` |
| `SENTIMENTKIT_ENCODER_BATCH_SIZE` | `16` | Texte pro Modelldurchlauf |

`model_version` lautet `<modell>@<commit>`; ein Branch oder Tag wird auf den Commit aufgelöst, aus dem die Gewichte tatsächlich geladen wurden. Beide Presets sind gepinnt, damit ein erneuter Upload auf dem Hub die Ergebnisse nicht unbemerkt verändert. Jedes Preset erhält die Textbereinigung seiner Trainingsdaten (`germansentiment`: nur Buchstaben, Ziffern ausgeschrieben, Kleinschreibung; XLM-R: Platzhalter für Nutzernamen und Links). Die Evidenz listet die Klassenwahrscheinlichkeiten und Hinweise: `DIALECT?`, `LANG?:fr` (Sprache ausserhalb der Modellsprachen), `TRUNCATED?` (Eingabe länger als 512 Tokens). Nach dem ersten Download offline: `HF_HUB_OFFLINE=1`. Das Backend schaltet die Safetensors-Konvertierung von transformers im Hintergrund ab, die sonst den Hub kontaktieren und eine konvertierte Kopie aus einem offenen Pull Request im Modell-Repo herunterladen würde.

Gemessen auf dem Fixture-Set der Stufe 0 (30 synthetische deutsche Sätze; transformers 5.19, torch 2.14 CPU):

| Kennzahl | lexicon | encoder, german-sentiment-bert | encoder, twitter-xlm-roberta |
|---|---|---|---|
| Accuracy | 0.933 | 0.633 | 0.900 |
| Macro-F1 | 0.777 | 0.700 | 0.881 |
| Cohen's κ vs. Gold | 0.892 | 0.488 | 0.840 |
| ECE | 0.203 | 0.351 | 0.151 |
| Slice Ironie (n=2) | 0.00 | 0.00 | 1.00 |
| Slice Negation (n=7) | 1.00 | 0.86 | 0.57 |
| alt-test ω | 0.50 (bestanden) | 0.00 (nicht bestanden) | 1.00 (bestanden) |
| `eval.beats_baseline()` | — | nein (−0.077) | ja (+0.104) |

Diese Zahlen mit Vorsicht lesen:

- **Nicht das Gate der Stufe 2.** Das Fixture hat 30 Sätze und ist zusammen mit den Lexikon-Regeln entstanden, was das Lexikon begünstigt. Das Gate zählt auf dem Gold-Set der Stufe 1 (`fixtures/gold_v1.jsonl`), das es noch nicht gibt.
- **Der deutsche Default irrt mit hoher Sicherheit.** 10 seiner 11 Fehler haben eine Konfidenz ≥ 0.9; meist stuft er wertende, aber sachlich formulierte Sätze als `neutral` ein. Auf der synthetischen Befragung schickt er 0 von 144 Urteilen in die Review-Queue (Lexikon: 10). Unkalibriert umgeht er das Human Gate — vor der Kalibrierung nicht für Berichte verwenden (Gate der Stufe 2: ECE < 0.10).
- **Das mehrsprachige Modell** liegt auf diesem Fixture vorne, auch für Deutsch, und seine Fehler haben tiefe Konfidenz. Seine Model Card deklariert keine Lizenz; das vor einem produktiven Einsatz klären.

## LLM-Scoring-Backend (Stufe 2b)

`--backend llm-score` stellt einem lokalen LLM pro Text und Aspekt eine **Typed Question** und liest die Antwort an den Wahrscheinlichkeiten fester Ein-Token-Labels an der ersten Antwortposition ab. Das Modell schreibt nie Text.

| Label | Polarität | Angeboten |
|---|---|---|
| `A` | positive | immer |
| `B` | neutral | immer |
| `C` | negative | immer |
| `D` | unclear (gemischt, ironisch, nicht entscheidbar) | immer |
| `E` | not_mentioned | nur bei Aspektfragen |

- **Prompt**: der Text als abgegrenzte Daten, die Frage, die Optionen (deutscher Wortlaut, Version `v1`). Gerendert mit der Chat-Vorlage des Modells, Thinking ausgeschaltet, damit die Antwort das erste Token ist. Eine Änderung am Wortlaut braucht eine neue Prompt-Version; ein Test hält den Hash des Wortlauts fest.
- **Labels geprüft**: Jedes Label muss nach dem gerenderten Prompt genau ein eigenes Token ergeben, sonst verweigert das Backend die Bewertung.
- **Aspekte**: Ein Text, in dem kein Teilsatz einen Begriff des Aspekts nennt, erhält `not_mentioned` ohne Frage, dieselbe Regel wie im Lexikon- und im Encoder-Backend (siehe Halo-Effekt unten). Ein Text, der den Aspekt nennt, erhält die volle Frage samt E auf dem ganzen Text, damit Ironie und Kontrast lesbar bleiben.
- **Scoring**: SGLang `/v1/score` (Default; der Prompt geht als Token-IDs hinaus, das Modell liest also genau die geprüften Tokens) oder Ollama `/api/generate` mit `raw`, `logprobs` und `top_logprobs` 20 (Ausweichweg). Ollama läuft mit Temperatur 1 und ohne Repeat-Penalty: A–E kommen im Prompt vor, eine Penalty würde sie abwerten.
- **Ausgabe**: Der auf die Labels beschränkte Softmax ist die `distribution`, sein Maximum die `confidence`, P(positiv) − P(negativ) die `intensity`. Die Evidenz nennt die Optionswahrscheinlichkeiten und `label-mass` (wie viel Wahrscheinlichkeit überhaupt auf den Labels lag), dazu `LOW-LABEL-MASS?` unter 0.5, `NO-LABEL?` und `TRUNCATED?` (Text über 6000 Zeichen gekürzt). Jedes Atom trägt `prompt_hash`, den Hash des gerenderten Prompts.
- **Provenienz**: `model_version` lautet `<scorer>:<served model>+tok:<tokenizer-commit>+prompt:<version>[+T<temperatur>]`; bei Ollama enthält das Modell seinen Digest.

Konfiguration über Umgebungsvariablen (oder Konstruktor-Argumente von `LLMScoreBackend`):

| Variable | Default | Bedeutung |
|---|---|---|
| `SENTIMENTKIT_LLM_SCORER` | `sglang` | `sglang` oder `ollama` |
| `SENTIMENTKIT_LLM_BASE_URL` | `http://localhost:30000` (SGLang), `http://localhost:11434` (Ollama) | Scoring-Server |
| `SENTIMENTKIT_LLM_MODEL` | `Qwen/Qwen3-4B` (SGLang), `qwen3:4b` (Ollama) | Modellname auf diesem Server |
| `SENTIMENTKIT_LLM_TOKENIZER` | `Qwen/Qwen3-4B` | Hugging-Face-ID oder lokaler Pfad; rendert den Prompt und prüft die Labels |
| `SENTIMENTKIT_LLM_TOKENIZER_REVISION` | gepinnter Commit `1cfa9a720891` für den Default | Commit, Tag oder Branch |
| `SENTIMENTKIT_LLM_CALIBRATION_T` | `1` | Temperatur aus `sentimentkit eval` |
| `SENTIMENTKIT_LLM_TIMEOUT` | `120` | Sekunden pro Request (SGLang bewertet 16 Prompts pro Request) |

```bash
# SGLang auf einem GPU-Rechner, den du betreibst
python -m sglang.launch_server --model-path Qwen/Qwen3-4B --port 30000
sentimentkit eval fixtures/de_synthetic_gold.jsonl --compare lexicon,llm-score

# Ollama, z. B. auf einem Laptop oder einem Raspberry Pi
ollama pull qwen3:4b
SENTIMENTKIT_LLM_SCORER=ollama sentimentkit eval fixtures/de_synthetic_gold.jsonl --backend llm-score
```

**Texte gehen an den Scoring-Server.** Die Policy behandelt `llm-score` als lokales Backend; das stimmt nur, solange `SENTIMENTKIT_LLM_BASE_URL` auf einen Rechner zeigt, den du betreibst. Qwen3-4B steht unter Apache-2.0.

### Kalibrierung

Für Backends, deren Verteilung ein Softmax ist (`encoder`, `llm-score`), passt `sentimentkit eval` eine Temperatur T an, softmax(log p / T), und gibt die ECE vorher und nachher aus:

```
calibration: ECE 0.351 → 0.205 after temperature scaling (T=6.215, 5-fold cross-fitted, 29 of 30 judgements rescaled)
```

T wird auf den Log-Loss «Top-Label richtig» angepasst, also auf die Grösse, die ECE und das Review-Gate lesen. Ein Gold-Label ausserhalb des Antwortraums eines Backends (der Encoder kennt kein `unclear`) kann T so nicht ins Unendliche treiben. Die ECE nach dem Scaling ist 5-fach kreuzvalidiert: Kein Urteil wird mit einem T skaliert, das auf ihm selbst angepasst wurde. Scaling ändert nie, welche Option gewinnt; Accuracy und F1 bleiben gleich. Regelbasierte Antworten (das `not_mentioned` des Encoders aus dem Teilsatz-Scoping) behalten ihre Konfidenz. Für `llm-score` setzt du das angepasste T mit `SENTIMENTKIT_LLM_CALIBRATION_T`; es wird Teil von `model_version`. Die Review-Schwellen bleiben in `policy.yaml`.

Gemessen auf dem Fixture-Set der Stufe 0 (30 synthetische deutsche Sätze). Qwen3-4B lief in bf16 über transformers auf einer CPU hinter einem Stellvertreter-Server, der `/v1/score` und `/api/generate` so beantwortet, wie es SGLang 0.5.21 und die Ollama-API-Dokumentation festlegen. Das sind nicht SGLang oder Ollama selbst: Deren Kernels und Ollamas 4-Bit-Quantisierung verschieben die Zahlen etwas. Beide Scoring-Wege lieferten auf allen 30 Sätzen identische Vorhersagen.

| Kennzahl | lexicon | encoder, german-sentiment-bert | llm-score, Qwen3-4B |
|---|---|---|---|
| Accuracy | 0.933 | 0.633 | 0.900 |
| Macro-F1 | 0.777 | 0.700 | 0.860 |
| Cohen's κ vs. Gold | 0.892 | 0.488 | 0.836 |
| ECE | 0.203 | 0.351 | 0.096 |
| ECE nach Temperature-Scaling | — | 0.205 (T = 6.2) | 0.149 (T = 3.5) |
| Slice Ironie (n=2) | 0.00 | 0.00 | 1.00 |
| Slice Negation (n=7) | 1.00 | 0.86 | 0.86 |
| Fehler mit Konfidenz ≥ 0.9 | 0 von 2 | 10 von 11 | 3 von 3 |
| alt-test ω | 0.50 (bestanden) | 0.00 (nicht bestanden) | 1.00 (bestanden) |
| `eval.beats_baseline()` | — | nein (−0.077) | ja (+0.083) |

Diese Zahlen mit Vorsicht lesen:

- **Das Gate-Ergebnis hängt an einem Satz, und den entscheidet eine Regel.** Das Fixture hat einen einzigen `not_mentioned`-Satz; ihn beantwortet die Aspektregel, nicht das Modell. Ohne Regel antwortet Qwen3-4B mit `neutral`, die Klasse erreicht F1 = 0, der Macro-F1 fällt auf 0.592 und `beats_baseline()` sagt nein (−0.185). Auf 30 Sätzen mit einer Klasse aus einem Satz trägt Macro-F1 das Gate nicht. Der Entscheid gehört auf das Gold-Set der Stufe 1, mit vorab festgelegter Kennzahl.
- **Die Aspektregel antwortet auf einen gemessenen Halo-Effekt.** Zu jedem Aspekt befragt, übertrug Qwen3-4B die Gesamttonalität auf Aspekte, die der Text nie nennt: Aus «Die Betreuung im Hort ist chaotisch» wurde *Schulweg: negativ* mit Konfidenz 1.0. Von neun Aspektprüfungen waren 3 richtig, mit einer strengeren Frage 4, mit der Regel 9.
- **Das LLM erkennt Ironie** (2 von 2; Lexikon und Encoder 0 von 2). Seine drei Fehler: ein Kontrast als `neutral` gelesen («gut, aber … mangelhaft»), «kamen rechtzeitig» als positiv (Gold: neutral), «keine Beschwerden» als neutral (Gold: positiv).
- **Es irrt mit hoher Sicherheit.** Alle drei Fehler haben Konfidenz 1.0, das Review-Gate fängt sie also nicht. Die rohe ECE von 0.096 liegt nur deshalb unter der Schwelle der Stufe 2, weil 27 Treffer bei derselben 1.0 liegen. Kreuzvalidiertes Temperature-Scaling verschlechtert sie hier (0.149): Wenn Fehler so sicher sind wie Treffer, kann eine einzige Temperatur sie nicht trennen. `SENTIMENTKIT_LLM_CALIBRATION_T` nicht aus diesem Fixture setzen.
- Tempo auf dieser CPU ohne bf16-Unterstützung: rund 17 s pro Frage. Das sagt nichts über eine GPU oder Ollama auf einem Raspberry Pi; der Lauf auf dem Pi, den die Roadmap verlangt, ist noch offen.

## Verfügbare Tools

| Befehl | Beschreibung |
|---|---|
| `sentimentkit analyze INPUT --purpose P [--backend B] [--aspects A.yaml] [-o DIR]` | Jeden Text beurteilen; Atome, Run-Meta, Review-Queue und Report schreiben |
| `sentimentkit eval GOLD.jsonl [--backend B] [--json]` | Kennzahlen, Tag-Slices, alt-test, Kalibrierung (ECE vor/nach Temperature-Scaling), Fehlerliste |
| `sentimentkit eval GOLD.jsonl --compare B1,B2[,…] [--json]` | Jedes Backend auf demselben Gold-Set, nebeneinander, dazu `eval.beats_baseline()` gegen B1 |
| `sentimentkit backends` | Backends auflisten: verfügbar, Extra fehlt oder Roadmap-Stufe |
| `sentimentkit policy-check [--policy DATEI]` | Policy laden und Hash ausgeben |
| `sentimentkit validate judgements.jsonl` | Atome gegen das JSON-Schema prüfen |

Exit-Codes: `0` ok · `1` Fehler · `2` Policy-Verstoss.

## Konfiguration

| Datei | Zweck |
|---|---|
| `policy.yaml` | Governance-Regeln; wird in jeden Lauf gehasht |
| `lexicon/de_seed.tsv`, `lexicon/de_modifiers.yaml` | Baseline-Lexikon (eigene Arbeit, MIT) und deutsche Regeln |
| `fixtures/aspects_schule.yaml` | Beispiel-Typed-Questions für Schul-/Elternfeedback |
| `fixtures/keywords_schule.yaml` | Filter für den Schulbezug der Stufe-1-Kandidaten; dient zugleich als `aspect_terms` des Gold-Sets |
| `schema/*.schema.json` | Schemas für Urteil, Lauf und Gold-Set (Draft 2020-12) |
| `SENTIMENTKIT_ENCODER_*` | Modell, Revision, Gerät und Batch-Grösse des Encoder-Backends (siehe oben) |
| `SENTIMENTKIT_LLM_*` | Scorer, Server, Modell, Tokenizer und Kalibrierungstemperatur des LLM-Scoring-Backends (siehe oben) |

Eingabeformate: CSV/TSV (Spalte `text`), JSONL/JSON (`{"text": …}`), Markdown/TXT (ein Eintrag je Absatz).

## Projektstruktur

```
sentimentkit/
├── src/sentimentkit/
│   ├── models.py          # TextItem, Judgement (Evidenz-Atom), RunMeta
│   ├── policy.py          # Policy, PolicyViolation, Prüfungen
│   ├── preprocess.py      # Satzsplit, Tokens, Sprachschätzung, Decompound
│   ├── backends/          # base.py (Interface), lexicon.py, encoder.py, llm_score.py, Registry
│   ├── eval/              # metrics.py, alt_test.py, calibration.py, Gold-Set-Loader, Gates
│   ├── pipeline.py        # Policy → Backend → Atome → Gate
│   ├── gate.py            # Routing in die menschliche Prüfung
│   ├── report.py          # Aggregation mit k-Anonymitäts-Untergrenze, Markdown
│   ├── io.py              # Leser/Schreiber
│   └── cli.py
├── schema/                # JSON-Schemas
├── lexicon/               # Seed-Lexikon und Modifikatoren (Deutsch)
├── fixtures/              # synthetisches Gold-Set, Befragungs-CSV, Quelldokumente, Aspekte und Schlagwörter
├── scripts/               # build_gold_candidates.py, candidates_to_gold.py, validate_repo.py
├── skills/sentiment-scan/ # Claude-Skill
├── docs/                  # ROADMAP.md, ARCHITECTURE.md, ANNOTATION.md
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
