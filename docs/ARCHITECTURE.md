# Architektur

```
Input-Adapter ──► Vorverarbeitung ──► Backend (austauschbar) ──► Evidenz-Atome ──► Gate ──► Output
 CSV/TSV/JSONL      Sprache (de/fr/     lexicon   (Stufe 0)       Judgement:        Konfidenz   judgements.jsonl
 JSON/MD/TXT        it/en/gsw), Satz-   encoder   (Stufe 2)       polarity,         + Margin    run.json
                    split, Tokens,      llm-score (Stufe 2)       intensity,        → Review-   review_queue.csv
                    Decompounding       llm-cloud (Stufe 2)       confidence,       Queue       report.md
                                                                  distribution,
          policy.yaml ──► check_run / check_items ──────────────  backend, model_version,
                     └──► check_group_size (Aggregation) ──────  prompt_hash, evidence …
```

## Leitsatz

**LLMs denken und schreiben, ein Decision Model entscheidet, Code erzwingt.**
Sentiment pro Aspekt ist eine begrenzte, typisierte Entscheidung mit festem
Antwortraum. Deshalb ist das Backend-Interface (`backends/base.py`) als
*Typed Question* geformt: Text × Aspekt → Verteilung über feste Optionen,
immer mit Escape (`unclear`, `not_mentioned`). Lexikon-Scores, Encoder-Softmax
und Fixed-Answer-Scoring eines LLM werden so vergleichbar.

## Vier Entwurfsentscheide

1. **Ein Urteil = ein Evidenz-Atom mit Provenienz.** `schema/judgement.schema.json`.
   Jede Zahl im Report ist aus `judgements.jsonl` rekonstruierbar (Filter `run_id`,
   `aspect`). Dasselbe Muster wie in journeykit.
2. **Eval zuerst, Backends danach.** `eval/` liefert Accuracy, Macro-F1, Cohen's
   Kappa, ECE (Kalibrierung), Tag-Slices und den Alternative Annotator Test.
   `eval.beats_baseline()` ist das Gate der Stufe 2.
3. **Typed Questions statt Freitext-Prompt.** Aspekte sind ein YAML
   (`{aspekt: [terme]}`); pro Aspekt ein Urteil, Teilsatz-Scoping bei Kontrast.
4. **Governance als Code.** `policy.yaml` wird gehasht (`RunMeta.policy_hash`)
   und an zwei Stellen erzwungen: vor dem Backend (Zweck, Personenbezug,
   Anonymisierung, verbotene Dimensionen/Kontexte) und bei der Aggregation
   (k-Anonymitäts-Untergrenze). Verstoss = Exit 2.

## Deutsch-spezifische Regeln im Lexikon-Backend

| Phänomen | Regel | Beispiel |
|---|---|---|
| Satzgrenze | Kein Satzende nach Ordinalzahl (1–2 Ziffern) oder gängiger Abkürzung; «Art.», «Abs.», «Nr.», «Ziff.» nur vor einer Zahl (`preprocess.split_sentences`) | «am 3. März», «12 Mio. Franken», «z. B.» |
| Negation davor | Negator in den 5 Tokens vor dem Sentiment-Wort, rückwärts gesucht, Stopp an Teilsatzgrenze | «nicht gut», «kein Problem» |
| Satzklammer | Negator *nach* dem Sentiment-Wort bis Teilsatzende, nur `nicht`/`nöd`/`nid`/`ned` | «… überhaupt nicht zufrieden» |
| Kein Satzklammer-Flip bei Nomen | `nie`, `kein`, `ohne` negieren nach einem Nomen nicht das Nomen | «Problem wurde nie gelöst» bleibt negativ |
| Komposita | längster bekannter Suffix (≥ 4 Zeichen) | «Servicequalitätsproblem» → «problem» |
| Mehrwort | längste Lexikon-Phrase zuerst | «ernst genommen», «sehr gut» |
| Verstärker / Abschwächer | Faktor aus `de_modifiers.yaml` in den 2 Tokens davor | «sehr», «etwas» |
| `kaum` | Negator, nicht Abschwächer | «kaum ein Fortschritt» |
| Kontrast | Gewicht 0.5 vor, 1.5 nach `aber`/`jedoch`/`dagegen` … | «gut, aber mangelhaft» |
| Satzadverbien | negationsimmun | «leider nicht» bleibt «leider» |
| Dialekt | Markererkennung → `DIALECT?`, Konfidenz × 0.8 | «isch nöd guet» |
| Ironie | **nicht erkannt** — gemischte Signale werden `unclear` und gehen in die Review-Queue | «Toll, schon wieder Verspätung» |

Die Ironie-Instanzen im Gold-Set sind absichtlich drin: sie dokumentieren,
was die Baseline nicht kann, und sind der Massstab für Stufe 2.

## Konfidenz und Gate

`confidence` ist die Selbsteinschätzung des Backends, keine Trefferwahrschein-
lichkeit. Erst die Kalibrierung auf dem Gold-Set (ECE) sagt, ob 0.8 auch
«80 % richtig» bedeutet. Das Gate (`gate.py`) schickt Urteile in die
Review-Queue, wenn `confidence < review_threshold`, die Polarität `unclear`
ist oder die Top-2-Optionen näher als `review_margin` liegen.

## Was bewusst fehlt

- Keine Personenprofile, kein Sentiment pro Absender (Policy).
- Keine politische Ausrichtung, keine Emotion einer Person, nichts Biometrisches.
- Kein Social Listening einzelner Accounts; Scope sind aggregierte öffentliche
  Medien und eigene, anonymisierte Befragungen.
