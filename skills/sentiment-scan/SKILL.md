---
name: sentiment-scan
description: Tonalität und Aspekte in Texten mit sentimentkit bestimmen — Medienartikel, offene Befragungsantworten, Anfragen, Ratsdebatten, eigene Kommunikation. Verwende diesen Skill, wenn der User (1) Texte auf Stimmung, Tonalität oder Sentiment auswerten will, (2) offene Antworten aus Umfragen nach Thema × Tonalität zusammenfassen möchte, (3) eine Medienresonanz oder ein Lagebild mit Tonalität braucht, (4) ein Backend auf einem Gold-Set evaluieren oder gegen die Lexikon-Baseline vergleichen will, (5) Begriffe wie «Sentiment», «Stimmungsbild», «Tonalität», «Opinion Mining», «Elternfeedback auswerten» nennt. Nicht für Emotionserkennung von Personen, politische Ausrichtung oder Mitarbeitenden-/Schüler-Monitoring — das lehnt die Policy ab.
---

# sentiment-scan

Dieser Skill ruft das Python-Paket `sentimentkit` auf. Er beurteilt Texte
über Themen, nie Personen. Die Grenzen stehen in `policy.yaml` und werden vom
Werkzeug selbst erzwungen (Exit 2 bei Verstoss) — der Skill umgeht sie nicht.

## Voraussetzungen

```bash
pip install -e .            # im Repo malkreide/sentimentkit
sentimentkit backends       # zeigt, welche Backends in dieser Stufe verfügbar sind
```

## Ablauf

1. **Zweck klären** (Pflicht). Einer von: `media_resonance`, `survey_open_answers`,
   `inbox_triage`, `public_debate`, `own_communication`, `evidence_for_journeys`,
   `evaluation`. Passt der Zweck in keine Kategorie, nicht raten — nachfragen oder
   ablehnen.
2. **Daten prüfen.** Input ist CSV/TSV (Spalte `text`), JSONL (`{"text": …}`),
   JSON-Liste, Markdown oder TXT (Absätze). Spalten wie `author_id`, `email`,
   `name` müssen vorher entfernt werden; sonst bricht der Lauf ab.
   Cloud-Backends verlangen `--anonymized`.
3. **Aspekte definieren** (optional). YAML `{aspekt: [terme]}`; Vorlage:
   `fixtures/aspects_schule.yaml`.
4. **Lauf**
   ```bash
   sentimentkit analyze INPUT --purpose survey_open_answers \
     --aspects fixtures/aspects_schule.yaml --backend lexicon -o out/
   ```
   Ergebnis in `out/`: `judgements.jsonl` (Evidenz-Atome), `run.json`
   (Provenienz inkl. `policy_hash`), `review_queue.csv` (Human Gate),
   `report.md` (Aggregation ab n ≥ 20).
5. **Ergebnis einordnen.** Immer mitliefern: Backend und `model_version`,
   Anteil in der Review-Queue, bekannte Schwächen des Backends (Lexikon: Ironie,
   Dialekt). Zahlen unter der Aggregationsschwelle werden nicht berichtet —
   auch nicht mündlich.
6. **Evaluation** statt Bauchgefühl, wenn ein Backend gewählt werden soll:
   ```bash
   sentimentkit eval fixtures/de_synthetic_gold.jsonl --backend lexicon --json
   ```
   Macro-F1, Kappa, ECE, Tag-Slices, alt-test (ω ≥ 0.5 = Backend darf
   menschliche Annotation ersetzen, Calderon et al. 2025).

## Antwortformat

- Erst die Zahl, dann die Provenienz: «Tonalität *Betreuung*: 12 % negativ
  (n = 24, Backend lexicon seed:120, Run `…`)».
- Review-Queue nennen und anbieten, sie gemeinsam durchzugehen.
- Bei `unclear` oder Ironie-Verdacht: nicht nachträglich entscheiden, sondern
  als offen markieren.
- Schweizer Rechtschreibung (kein ß).

## Was der Skill verweigert

- Sentiment pro Person oder Absender, auch «nur intern».
- Politische Ausrichtung, Emotion oder psychischer Zustand einer Person.
- Monitoring einzelner Social-Media-Accounts, Mitarbeitenden- oder
  Schüler-Monitoring.
- Berichte unter der Aggregationsschwelle «ausnahmsweise».
