# Roadmap — Stufen mit Gates

Vorgehen wie in `dealkit`: jede Stufe hat ein messbares Gate. Ohne bestandenes
Gate beginnt die nächste Stufe nicht. Die Gates sind so formuliert, dass sie
sich mit den Mitteln des Repos prüfen lassen (`sentimentkit eval`, Tests, CI).

| Stufe | Inhalt | Gate |
|---|---|---|
| **0 — Skelett** (✅ 14.10.2026) | Paket, CLI, JSON-Schema (Evidenz-Atom, Run, Gold), `policy.yaml`, Lexikon-Backend mit Negation/Satzklammer/Komposita/Kontrast, Eval-Harness (Accuracy, Macro-F1, Kappa, ECE, alt-test), Review-Queue, Markdown-Report, synthetisches Gold-Set (30), Claude-Skill, CI | Baseline läuft auf dem Fixture-Set (Accuracy ≥ 0.85, Ironie dokumentiert bei 0); Policy-Verstoss bricht den Lauf mit Exit 2 ab; CI grün |
| **1 — Gold-Set & Baseline** | 300–500 Sätze aus öffentlichen Quellen: Medienartikel zum Schulamt (news-monitor-mcp) und Gemeinderats-Voten zu Schulthemen (zurich-opendata-mcp). Mindestens zwei Annotierende. Annotation in Argilla oder als CSV → `fixtures/gold_v1.jsonl` | Kappa Mensch–Mensch dokumentiert; Lexikon-F1 als Referenzwert in README; Tag-Slices (negation, irony, dialect, aspect) je ≥ 20 Instanzen |
| **2 — Lokale Backends** | `encoder` (german-sentiment-bert, twitter-xlm-roberta-base-sentiment), `llm-score` (Ollama/SGLang `/v1/score`, Qwen3, Typed Questions), optional `llm-cloud` (Claude API). Kalibrierung auf Hold-out. Lauf auf Raspberry Pi 5 nachgewiesen | `eval.beats_baseline()` für mindestens ein Backend (+0.05 Macro-F1); alt-test ω ≥ 0.5; ECE < 0.10 nach Kalibrierung |
| **3 — Schulamt-Pilot** | Offene Antworten einer bestehenden Befragung, anonymisiert, mit Bereich DQF. HTML-Report (Single-File, wie journeykit). 5 %-Stichprobe menschlich geprüft (`review_queue.csv` → `human_label`) | Fachbereich bestätigt Nutzen; Rechtsdienst bestätigt Scope; 0 Policy-Verstösse; Stichproben-Agreement ≥ 0.8 |
| **4 — Integration** | `sentiment-mcp` als Compute-Server (kein Datenbestand), Zweitmeinung in der news-monitor-Pipeline, journeykit/personakit als Konsumenten, Drift-Monitoring (promptfoo/Deepchecks), Re-Eval-Pflicht bei Modellwechsel im CI | mcp-audit bestanden; Drift-Alarm ausgelöst und dokumentiert |

## Offene Entscheide

- **Lexikon-Lizenz.** Stufe 0 nutzt ein eigenes Seed-Lexikon (MIT). SentiWS und GerVADER stehen unter CC BY-SA-NC — für sormena-Nutzung nicht ohne Weiteres verwendbar. Entscheid vor Stufe 2.
- **Erster produktiver Use Case.** Medien (öffentlich, sofort) für Stufe 1; Befragungen (höherer interner Wert) als Stufe 3. Reihenfolge ist Vorschlag, nicht Beschluss.
- **Name.** `sentimentkit` passt in die Reihe journeykit/personakit/dealkit; `tonekit` hätte mehr Abstand zum Begriff Emotionserkennung.
- **Decompounding.** Suffix-Match reicht für Stufe 0; CharSplit o. ä. erst, wenn das Gold-Set zeigt, dass Komposita ein messbares Problem sind.

## Prompts für die nächsten Claude-Code-Sessions

1. *Stufe 1:* «Lies docs/ROADMAP.md und docs/ARCHITECTURE.md. Baue `scripts/build_gold_candidates.py`, das aus einer JSONL mit Medienartikeln Sätze mit Schulbezug zieht, dedupliziert und als Annotationsvorlage (`fixtures/gold_v1_candidates.csv`, Spalten text_id, text, a1, a2, tags) schreibt. Keine Modelle, nur Regeln.»
2. *Stufe 2a:* «Implementiere `backends/encoder.py` nach dem Interface in `backends/base.py` mit `transformers` (optional-dependency `encoder`), Modell per Konfiguration, CPU-Default, Batch-Inferenz. Schreibe Tests mit einem gemockten Modell. Registriere es im Registry und hebe `NotAvailable` auf.»
3. *Stufe 2b:* «Implementiere `backends/llm_score.py`: Typed Question pro Aspekt, Single-Token-Labels (A/B/C/D/E), `/v1/score`-Aufruf gegen SGLang bzw. Logit-Scoring gegen Ollama, `prompt_hash` im Evidenz-Atom, Kalibrierung über `eval`. Kein Freitext.»
