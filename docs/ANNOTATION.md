# Annotationsanleitung — Gold-Set Stufe 1

Zwei Personen beurteilen unabhängig voneinander jeden Kandidatensatz: **Welchen
Eindruck vermittelt dieser Satz von der Volksschule?** Das Ergebnis ist
`fixtures/gold_v1.jsonl`, der Massstab, an dem sich ab Stufe 2 jedes Backend
messen muss. Ein unsauberes Gold-Set macht jede spätere Kennzahl wertlos —
diese Anleitung ist deshalb verbindlich, nicht Empfehlung.

Ziel: **Cohen's κ ≥ 0.7** zwischen den beiden Annotierenden.

## 1. Rahmen

- Beurteilt wird ein **Text über ein Thema**, nie eine Person: nicht die
  Haltung der Autorin oder des Redners, nicht die Gefühlslage von jemandem,
  nicht die politische Ausrichtung (`policy.yaml`).
- Beurteilt wird **nur der Satz**, ohne Artikel oder Votum drumherum. Die
  Vorlage ist absichtlich gemischt, damit kein Kontext mitliest.
- Massgeblich ist, was der Satz **ausdrückt oder klar nahelegt** — nicht die
  eigene Meinung zum Thema und nicht, ob die Aussage stimmt.

## 2. Ablauf

1. **Vorlage erzeugen**
   ```bash
   python scripts/build_gold_candidates.py quellen.jsonl --target 400
   ```
   Ergebnis: `fixtures/gold_v1_candidates.csv` mit den Spalten
   `text_id, text, source, a1, a2, tags, note`.
2. **Personennamen bereinigen**, bevor die Vorlage verteilt wird (Abschnitt 7).
3. **Pilot mit 50 Sätzen.** Beide annotieren, `candidates_to_gold.py` liefert
   κ, Differenzen gemeinsam besprechen, Anleitung schärfen. Die besprochenen
   Pilotsätze sind danach nicht mehr unabhängig: neu annotieren oder verwerfen.
4. **Hauptrunde.** Jede Person erhält eine eigene Kopie (`gold_v1_a1.csv`,
   `gold_v1_a2.csv`) und füllt nur die eigene Spalte. Kein Austausch über
   einzelne Sätze, bis beide fertig sind.
5. **Zusammenführen**
   ```bash
   python scripts/candidates_to_gold.py gold_v1_a1.csv gold_v1_a2.csv -o fixtures/gold_v1.jsonl
   ```
   Das Skript führt die Kopien über `text_id` zusammen, setzt `gold` auf das
   Mehrheitslabel bzw. `unclear` bei Dissens, prüft gegen
   `schema/gold.schema.json` und gibt κ, Labelverteilung und Tag-Slices aus.
6. **Schlichtung (empfohlen).** Bei Dissens setzt das Skript `unclear` — das
   vermischt «der Satz ist mehrdeutig» mit «zwei Personen waren sich uneinig».
   Besser: Eine dritte Person beurteilt nur die Zeilen mit Tag `dissent` in
   einer Spalte `a3` (eigene Datei, Spalten `text_id, text, a3`); die Datei wird
   als drittes Argument übergeben, dann entscheidet die Mehrheit.
7. **Baseline messen:** `sentimentkit eval fixtures/gold_v1.jsonl --backend lexicon`

## 3. Labels

| Label | Wann | Beispiel (synthetisch) |
|---|---|---|
| `positive` | Der Satz vermittelt ein günstiges Bild: Lob, Erfolg, Verbesserung, Zufriedenheit, gelöstes Problem | «Die neue Tagesschule wird von den Familien sehr geschätzt.» |
| `neutral` | Information ohne Wertung: Termine, Abläufe, Zahlen, Beschlüsse, offene Fragen | «Die Kreisschulbehörde tagt am Dienstagabend im Schulhaus.» |
| `negative` | Der Satz vermittelt ein ungünstiges Bild: Kritik, Mangel, Problem, Verschlechterung, Forderung nach Abhilfe | «Der Mangel an Lehrpersonen verschärft sich weiter.» |
| `unclear` | Der Ton lässt sich nicht bestimmen: positive und negative Wertung halten sich die Waage, oder Ironie, deren Absicht offen bleibt | «Einerseits lobt der Bericht die Schulen, andererseits kritisiert er sie scharf.» |
| `not_mentioned` | Der Satz kam durch den Wortfilter, handelt aber nicht von der Volksschule | «Die Betreuung im Alterszentrum wird ebenfalls ausgebaut.» |

Kurzformen beim Erfassen: `pos`, `neu`, `neg`, `unc`, `nm` (auch `positiv`,
`negativ`, `unklar`). Gross- und Kleinschreibung spielt keine Rolle.

`unclear` ist kein Verlegenheitslabel. Wer zwischen `neutral` und einer
Polarität schwankt, wählt `neutral` und schreibt den Zweifel in `note`.

## 4. Regeln für schwierige Fälle

### Sachlich formulierte Meldungen

Auch nüchtern geschriebene Sätze haben eine Valenz für die Volksschule. Ein
gemeldeter Mangel, eine Kürzung oder ein Vorfall ist `negative`, eine
Auszeichnung oder ein Ausbau `positive` — auch ohne wertende Wörter.
Reine Fakten ohne Richtung (Termine, Zuständigkeiten, Zahlen ohne Vergleich)
sind `neutral`.

- «Der Gemeinderat hat 12 Mio. Franken für die Erweiterung der Schule bewilligt.» → `positive`
- «Die Kreisschulbehörde behandelt das Geschäft am 3. März.» → `neutral`

### Ironie und Sarkasmus

- Es zählt die **gemeinte** Bedeutung, nicht die Oberfläche. Tag `irony`.
- «Toll, schon wieder eine Reform, die das Schulamt ohne die Lehrpersonen plant!» → `negative`, `irony`
- Ironie nur annehmen, wenn der Satz selbst sie zeigt: «toll, schon wieder»,
  demonstrative Anführungszeichen, offensichtliche Übertreibung. Bleibt die
  Absicht ohne Kontext offen → `unclear`, `irony` und ein Hinweis in `note`.

### Dialekt

- Mundart wird beurteilt wie Hochdeutsch. Tag `dialect` — auch für einzelne
  Mundart-Zitate in einem hochdeutschen Satz.
- «Mir finded d Tagesschule mega guet.» → `positive`, `dialect`
- Die Vorlage schlägt `dialect?` regelbasiert vor. Der Vorschlag irrt oft
  (z. B. bei «sind» in hochdeutschen Sätzen): prüfen, bei Hochdeutsch löschen.

### Gemischte Sätze

- **Eine Seite überwiegt** → deren Label. Nach «aber», «jedoch», «trotzdem»
  trägt meist der zweite Teil das Gewicht. Tag `contrast`.
  «Die Kindergärten sind gut ausgelastet, aber die Räume sind zu klein.» → `negative`, `contrast`
- **Gleichgewicht** → `unclear`, Tag `mixed`.
- **Mehrere Themen, nur eines betrifft die Schule** → es zählt der Ton
  gegenüber der Schule. Tag `aspect`.
  «Der Verkehr im Quartier ist ein Ärgernis, die neue Tagesschule dagegen ein Gewinn.» → `positive`, `aspect`

### Negation

- Die Polarität richtet sich nach der verneinten Aussage. Tag `negation`, wenn
  das Urteil an der Verneinung hängt.
- «Die Betreuung über Mittag ist nicht mehr überlastet.» → `positive`, `negation`
- «Die Kreisschulbehörde hat den Antrag ohne Gegenstimme angenommen.» → `neutral`;
  der Vorschlag `negation?` trifft hier nicht zu und wird gelöscht.

### Zitate, Forderungen, Fragen

- Zitate werden wie jeder andere Satz beurteilt. Wer spricht, spielt keine
  Rolle und wird nirgends erfasst.
- Forderungen legen meist einen Mangel nahe: «Die Betreuung braucht dringend
  mehr Ressourcen.» → `negative`.
- Offene Fragen ohne Wertung → `neutral`.

### Bruchstücke

Abgeschnittene oder falsch getrennte Sätze, Tabellenreste, Bildlegenden,
Satzfragmente ohne Aussage → Tag `exclude`. Die Zeile fällt beim
Zusammenführen weg.

## 5. Tags

Mehrere Tags mit Leerzeichen trennen. Tags sind Schnittkanten für die
Auswertung (`by_tag` im Eval), kein Teil des Labels.

| Tag | Bedeutung | Gesetzt von |
|---|---|---|
| `negation` | Urteil hängt an einer Verneinung | Annotierende (Vorschlag `negation?`) |
| `irony` | ironisch oder sarkastisch gemeint | Annotierende |
| `dialect` | Mundart, ganz oder teilweise | Annotierende (Vorschlag `dialect?`) |
| `aspect` | mehrere Themen, Ton gegenüber der Schule weicht ab | Annotierende |
| `contrast` | Kontrast mit «aber», «jedoch» … | Annotierende |
| `mixed` | positive und negative Wertung im Gleichgewicht | Annotierende |
| `exclude` | Zeile verwerfen (Bruchstück, Privatperson) | Annotierende |
| `headline` | Titel statt Satz (nur mit `--include-titles`) | Vorlage |
| `dissent` | Annotierende waren uneinig | `candidates_to_gold.py` |

**Vorschläge mit `?` bestätigen oder löschen.** Ein Tag wie `negation?` wird
erst übernommen, wenn das `?` entfernt ist; unbestätigte Vorschläge erreichen
das Gold-Set nie. Bestätigt eine der beiden Personen einen Tag, gilt er.

Das Gate der Stufe 1 verlangt je **mindestens 20 Instanzen** für `negation`,
`irony`, `dialect` und `aspect`. Die Ausgabe von `candidates_to_gold.py` zeigt
den Stand.

## 6. Qualitätsziel und Messung

- **Cohen's κ ≥ 0.7** zwischen a1 und a2 (ab 0.61 gilt die Übereinstimmung nach
  Landis & Koch als substanziell). Das Skript gibt κ und die rohe Übereinstimmung aus; beide
  Zahlen gehören in die Dokumentation, denn κ sinkt bei sehr ungleicher
  Labelverteilung (viele `neutral`) auch bei hoher Übereinstimmung.
- Liegt κ unter 0.7: Anleitung schärfen und betroffene Sätze neu
  annotieren — **nicht** Labels nachträglich angleichen. Ein geschöntes κ
  misst nichts mehr.
- κ wird im README festgehalten (Gate der Stufe 1).

## 7. Personen und Datenschutz

- **Privatpersonen** (Eltern, Schülerinnen und Schüler, Lehrpersonen mit
  Namen) → Tag `exclude`.
- **Amtsträgerinnen und Amtsträger** in ihrer Funktion: Namen in der Spalte
  `text` durch die Rolle ersetzen, z. B. «[Stadträtin]», «[Schulpräsident]».
  Das geschieht in der Vorlage, **bevor** die Kopien verteilt werden — weicht
  der Text zwischen zwei Kopien ab, bricht das Zusammenführen mit einem
  Konflikt ab.
- Annotierende erscheinen nur als `a1`, `a2`, `a3`. Die Zuordnung zu Personen
  wird ausserhalb des Repos geführt.
- In `note` keine Personendaten.
- Spalten wie `author`, `name`, `email` dürfen weder in der Quell-JSONL noch
  in der CSV stehen; beide Skripte brechen sonst mit Exit 2 ab.

## 8. Arbeiten mit Excel

- Die Vorlage ist UTF-8 mit BOM, damit Umlaute korrekt erscheinen.
- Excel mit Schweizer Einstellungen erwartet Semikolon: die Vorlage mit
  `--delimiter ";"` erzeugen oder über *Daten → Aus Text/CSV* importieren.
- Speichern als *CSV UTF-8*. Komma oder Semikolon erkennt
  `candidates_to_gold.py` selbst.
- `text_id` nie ändern; `text` nur zur Bereinigung von Namen (Abschnitt 7).
- `build_gold_candidates.py` überschreibt eine Vorlage, die bereits Labels
  enthält, nur mit `--force`.

## 9. Ablage und Rechte

- **Gemeinderatsprotokolle** sind amtliche Dokumente und nach Art. 5 URG nicht
  urheberrechtlich geschützt.
- **Medientexte** sind es in der Regel schon. Ob 300–500 Sätze aus
  Zeitungsartikeln in einem öffentlichen Repo unter MIT-Lizenz stehen dürfen,
  ist vor dem ersten Commit zu klären. Ausweg, falls nein: das Gold-Set mit
  Mediensätzen intern ablegen und im Repo nur Kennzahlen und den
  Gemeinderatsteil veröffentlichen.
- `CONTRIBUTING.md` verlangt bisher «synthetic fixtures only». Ein reales
  Gold-Set ist eine bewusste Ausnahme davon und braucht einen Entscheid.
