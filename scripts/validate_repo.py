#!/usr/bin/env python3
"""
validate_repo.py — Struktur- und Dokumentationsprüfung für ein Repo.

Meldet, ändert nichts. Auto-Fixes sind bewusst nicht implementiert:
bewusste Abweichungen (Selbstbezeichnung, präzisere Sektionstitel,
deutsche Synonyme) dürfen nicht "aufgeräumt" werden.

Aufruf:
    python3 validate_repo.py [REPO_PFAD]
    python3 validate_repo.py --json [REPO_PFAD]
    python3 validate_repo.py --fail-on=C8[,REGEL...] [REPO_PFAD]

Exit-Code: 0 = keine ERROR-Findings, 1 = mindestens ein ERROR.
`--fail-on` stuft WARN der genannten Regeln für diesen Lauf zu ERROR hoch —
für ein Repo, das eine Regel strenger nimmt als das Portfolio. Das ersetzt
den Heredoc, der dafür den JSON-Bericht ein zweites Mal las (SKILL.md 8.7).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

# --- Sektions-Vokabular ------------------------------------------------------
# Schluss-Sektionen in verbindlicher Reihenfolge (C1).
CLOSING_EN = ["contributing", "security", "license", "author"]
CLOSING_DE = ["contributing", "security", "license", "author"]

# Erlaubte Überschriften je logischer Sektion. Exakter Vergleich (E3),
# ergänzt um bewusst präzisere Titel (D2) und deutsche Synonyme (D1/D3).
HEADINGS = {
    "en": {
        "contributing": {"contributing", "contributing & support"},
        "security": {
            "security",
            "security & compliance",
            "security and compliance",
            "security & privacy",
            "security policy",
        },
        "license": {
            "license",
            "licence",
            "licenses",
            "licences",
            "software licence",
            "software license",
            "data licence",
            "data license",
            "license & data licence",
        },
        "author": {"author", "authors", "maintainer", "maintainers"},
    },
    "de": {
        "contributing": {"mitwirken", "mitmachen", "beitragen", "mitarbeit"},
        "security": {
            "sicherheit",
            "sicherheit & compliance",
            "sicherheit und compliance",
            "sicherheit & datenschutz",
            "sicherheitsrichtlinie",
        },
        "license": {
            "lizenz",
            "lizenzen",
            "software-lizenz",
            "softwarelizenz",
            "datenlizenz",
            "daten-lizenz",
            "lizenz & datenlizenz",
        },
        # D1: Selbstbezeichnungen sind keine Formatabweichung — alle Varianten gelten.
        "author": {
            "autor",
            "autorin",
            "autor·in",
            "autor:in",
            "autor*in",
            "autor/in",
            "autorin / autor",
            "autor / autorin",
            "autoren",
        },
    },
}

# Englische Sektionstitel, die in einer deutschen Datei ein Fehler sind (C6).
# 'Changelog' und 'Installation' fehlen bewusst: beides sind im Deutschen
# gebräuchliche Titel, technische Begriffe bleiben englisch.
EN_ONLY_IN_DE = {
    "contributing",
    "security",
    "license",
    "licence",
    "author",
    "overview",
    "features",
    "prerequisites",
    "usage",
    "configuration",
    "project structure",
    "available tools",
}

# C5/E7 — Emoji-Erkennung. Unicode kennt zwei Klassen, und der Unterschied ist
# genau der zwischen einem Emoji und einem Satzzeichen:
#
#   Emoji-Standarddarstellung   ⚡ ✨ 🗺   zaehlen fuer sich allein
#   Text-Standarddarstellung    ⚖ ❄ ✈ ↔  sind Typografie, bis ein
#                                          Variantenselektor VS16 (U+FE0F) folgt
#
# Die fruehere Fassung nahm ganze Bloecke pauschal — darunter U+2190–U+21FF,
# den kompletten Pfeilblock — und meldete deshalb `Zefix ↔ Amtsblatt` als
# Emoji. Das ist die zu breite Regel, vor der E4 warnt, nur trifft sie Pfeile
# statt Umlaute. Ebenfalls entfallen: U+24C2–U+1F251, ein einzelner Bereich,
# der nebenbei den gesamten CJK-Block verschluckte.
#
# Gegengeprueft an allen Ueberschriften des Portfolios: die dort vorkommenden
# Textdarstellungs-Emoji (⚖ ⚙ ⛰ ✈ ❄) tragen ausnahmslos VS16, die
# Default-Emoji (⚡ ✨) stehen ausnahmslos nackt. `scripts/test_emoji.py` haelt
# beide Seiten fest.
_EMOJI_PICTOGRAPH = "\U0001f000-\U0001faff"

# BMP-Zeichen mit Emoji_Presentation=Yes — vollstaendige Liste ausserhalb der
# Pictograph-Ebenen. Nicht als Bereich zusammenfassbar: sie liegen verstreut
# zwischen Zeichen, die Textdarstellung haben.
_EMOJI_DEFAULT = (
    "⌚⌛⏩-⏬⏰⏳◽◾☔☕"
    "♈-♓♿⚓⚡⚪⚫⚽⚾⛄⛅"
    "⛎⛔⛪⛲⛳⛵⛺⛽✅✊✋"
    "✨❌❎❓-❕❗➕-➗➰➿"
    "⬛⬜⭐⭕"
)

# Unsichtbare Steuerzeichen bewusst als Escape, nicht als Literal: im Quelltext
# waeren sie nicht zu sehen und beim naechsten Editieren still verloren.
_VS16 = "\ufe0f"  # Emoji-Darstellung erzwingen
_VS15 = "\ufe0e"  # Textdarstellung erzwingen
_ZWJ = "\u200d"  # verbindet Emoji-Sequenzen
_KEYCAP = "\u20e3"  # Keycap-Combiner (Ziffer + VS16 + U+20E3)

EMOJI_RE = re.compile(
    # Emoji; der Variantenselektor wird mitgenommen, sonst bliebe er in
    # normalise() als unsichtbarer Rest stehen und der Titelvergleich (E3)
    # scheiterte an einem Zeichen, das man nicht sieht.
    f"[{_EMOJI_PICTOGRAPH}{_EMOJI_DEFAULT}][{_VS16}{_VS15}]?"
    # Zeichen mit Textdarstellung — nur mit VS16 ein Emoji. Der Bereich
    # U+203C..U+3299 deckt Pfeile, technische Symbole und Dingbats ab;
    # ohne den Selektor greift hier nichts.
    f"|[\u203c-\u3299]{_VS16}"
    # Verwaiste Selektoren, ZWJ und Keycap-Combiner.
    f"|[{_VS16}{_ZWJ}{_KEYCAP}]"
)
MARKER_RE = re.compile(r"<!--\s*mcp-name:\s*([^\s>]+)\s*-->")
MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
HTML_IMG_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.IGNORECASE)
BADGE_HOSTS = (
    "shields.io",
    "badge.fury.io",
    "badgen.net",
    "codecov.io",
    "github.com/.*/workflows/.*badge",
    "img.shields.io",
)
ANCHOR_LINK_RE = re.compile(r"\]\(#([^)]+)\)")
BOLD_AUTHOR_RE = re.compile(
    r"^\*\*(author|autor|autorin|autor·in|maintainer)[^*]*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, level: str, rule: str, msg: str) -> None:
        self.items.append({"level": level, "rule": rule, "message": msg})

    error = lambda self, rule, msg: self.add("ERROR", rule, msg)  # noqa: E731
    warn = lambda self, rule, msg: self.add("WARN", rule, msg)  # noqa: E731
    info = lambda self, rule, msg: self.add("INFO", rule, msg)  # noqa: E731

    @property
    def failed(self) -> bool:
        return any(i["level"] == "ERROR" for i in self.items)


def strip_code_fences(text: str) -> str:
    """Entfernt ```-Blöcke, damit Kommentare im Code nicht als Überschrift zählen."""
    out, fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        out.append("" if fence else line)
    return "\n".join(out)


def headings(text: str) -> list[tuple[int, str]]:
    """(Ebene, Roh-Titel) für alle ATX-Überschriften ausserhalb von Code-Blöcken."""
    res = []
    for line in strip_code_fences(text).splitlines():
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*$", line)
        if m:
            res.append((len(m.group(1)), m.group(2).strip()))
    return res


def normalise(title: str) -> str:
    """Kleinschreibung ohne Emoji und Randzeichen — für den EXAKTEN Vergleich (E3)."""
    return EMOJI_RE.sub("", title).strip().strip("#").strip().lower()


def classify(title: str, lang: str) -> str | None:
    n = normalise(title)
    for key, variants in HEADINGS[lang].items():
        if n in variants:
            return key
    if lang == "de":
        # Englischer Titel in deutscher Datei: für die Reihenfolgeprüfung
        # trotzdem zuordnen — als Fehler meldet ihn separat C6.
        for key, variants in HEADINGS["en"].items():
            if n in variants:
                return key
    return None


def images(text: str) -> list[str]:
    """Bilder in BEIDEN Syntaxformen, Badges herausgefiltert (E2)."""
    found = MD_IMG_RE.findall(text) + HTML_IMG_RE.findall(text)
    return [s for s in found if not any(re.search(h, s) for h in BADGE_HOSTS)]


def check_readme(path: Path, lang: str, repo: Path, rep: Report) -> None:
    if not path.exists():
        rep.error("C1", f"{path.name} fehlt")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    hs = headings(text)

    # C1 — Reihenfolge der Schluss-Sektionen
    expected = CLOSING_EN if lang == "en" else CLOSING_DE
    tagged = [(lvl, classify(t, lang)) for lvl, t in hs]
    hits = [(lvl, c) for lvl, c in tagged if c in expected]

    # Für die *Existenz* zählt jede Ebene: eine vorhandene Sektion als fehlend
    # zu melden wäre schlimmer als eine tief verschachtelte durchzulassen.
    present = [c for _, c in hits]
    for want in expected:
        if want not in present:
            level = "WARN" if want == "contributing" else "ERROR"
            rep.add(level, "C1", f"{path.name}: Sektion '{want}' fehlt")

    # Für die *Reihenfolge* zählt nur der Schlussblock selbst. Zwei Filter,
    # beide aus konkreten Fehlmeldungen im Portfolio (E6):
    #
    # 1. Ebene — `### Security` als Unterpunkt von `## Safety & Limits` ist
    #    keine Dokumentsektion. Massgeblich ist die flachste Ebene, auf der
    #    überhaupt Schluss-Sektionen stehen; nicht hart `##`, damit ein README
    #    mit durchgehend tieferer Gliederung nicht als blockfrei gilt.
    # 2. Mehrfachnennung — ein Allowlist-Titel darf auch Inhaltssektion sein
    #    (`## Data License`, `## Security & Compliance`). Dann ist die LETZTE
    #    Nennung der Schlussblock. Vorher gewann die erste, und die Meldung
    #    zeigte auf eine Sektion mehrere hundert Zeilen weiter oben — der
    #    Schlussblock selbst war in allen beobachteten Fällen korrekt.
    #
    # Ein einfach genannter, falsch platzierter Schlussblock wird weiterhin
    # gemeldet: kollabiert werden ausschliesslich Duplikate.
    # `scripts/test_c1.py` haelt beide Haelften einzeln fest.
    if hits:
        doc_lvl = min(lvl for lvl, _ in hits)
        block = [c for lvl, c in hits if lvl == doc_lvl]
        last = {c: i for i, c in enumerate(block)}
        dedup = [c for i, c in enumerate(block) if last[c] == i]
        ranked = [expected.index(c) for c in dedup]
        if ranked != sorted(ranked):
            rep.error("C1", f"{path.name}: Schluss-Sektionen in falscher Reihenfolge: {dedup}")

    # C3 — Author als Überschrift, nicht als Fettdruck
    if BOLD_AUTHOR_RE.search(strip_code_fences(text)):
        rep.error("C3", f"{path.name}: Author/Autor als Fettdruck statt als Überschrift")

    # C5 + E4 — Emoji in Überschriften, mit Anker-Warnung
    anchors = set(ANCHOR_LINK_RE.findall(text))
    for lvl, title in hs:
        if EMOJI_RE.search(title):
            hint = ""
            slug = re.sub(r"[^a-z0-9\- ]", "", title.lower()).strip().replace(" ", "-")
            if any(slug in a or a in slug for a in anchors):
                hint = "  → ACHTUNG: Anker-Link zeigt auf diese Überschrift (E4)"
            rep.warn("C5", f"{path.name}: Emoji in Überschrift '{title}'{hint}")

    # C6 — englische Überschrift in deutscher Datei
    if lang == "de":
        for _, title in hs:
            if normalise(title) in EN_ONLY_IN_DE:
                rep.error(
                    "C6",
                    f"{path.name}: englische Überschrift '{title}' in deutscher Datei",
                )

    # C2 — vorhandene Dokumente müssen verlinkt sein
    for doc in ("SECURITY.md", "CONTRIBUTING.md"):
        if (repo / doc).exists() and doc not in text:
            rep.error("C2", f"{path.name}: {doc} existiert, wird aber nicht verlinkt")

    # Sprachumschalter
    other = "README.de.md" if lang == "en" else "README.md"
    if other not in text:
        rep.error("C1", f"{path.name}: Link auf {other} fehlt")

    # C4 — referenzierte Bilder müssen existieren
    for src in images(text):
        if src.startswith(("http://", "https://", "data:")):
            continue
        if not (repo / src.split("#")[0]).exists():
            rep.error("C4", f"{path.name}: Bild '{src}' referenziert, Datei fehlt")

    if lang == "de" and "ß" in text:
        rep.error(
            "DE",
            f"{path.name}: 'ß' gefunden — Schweizer Rechtschreibung verwendet 'ss'",
        )


def check_demo_parity(repo: Path, rep: Report) -> None:
    """C4 — Demo in beiden Sprachfassungen oder in keiner."""
    en, de = repo / "README.md", repo / "README.de.md"
    if not (en.exists() and de.exists()):
        return
    has = {}
    for p in (en, de):
        t = p.read_text(encoding="utf-8", errors="replace")
        has[p.name] = bool(images(t)) or any(normalise(x) == "demo" for _, x in headings(t))
    if has["README.md"] != has["README.de.md"]:
        missing = [k for k, v in has.items() if not v]
        rep.error("C4", f"Demo/Bild nur in einer Sprachfassung — fehlt in {missing[0]}")


def check_mcp_marker(repo: Path, rep: Report) -> None:
    """A1 — mcp-name-Marker in der Datei, die pyproject als readme deklariert.

    Lokale Anpassung (sentimentkit): nur für project_type mcp-server oder wenn
    eine server.json existiert. Eine Python-Library ohne Registry-Eintrag
    braucht keinen Marker. Kandidat für den Upstream in github-repo-skill.
    """
    pyproject = repo / "pyproject.toml"
    if not pyproject.exists():
        return
    project_type = repo_meta_field(repo, "project_type")
    if project_type and project_type != "mcp-server" and not (repo / "server.json").exists():
        rep.info("A1", f"project_type={project_type}: kein mcp-name-Marker erforderlich")
        return
    raw = pyproject.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'^\s*readme\s*=\s*["\']([^"\']+)["\']', raw, re.MULTILINE)
    readme_name = m.group(1) if m else "README.md"
    target = repo / readme_name
    if not target.exists():
        rep.error("A1", f"pyproject deklariert readme = {readme_name}, Datei fehlt")
        return
    markers = MARKER_RE.findall(target.read_text(encoding="utf-8", errors="replace"))
    rep.info("A1", f"mcp-name-Marker in {readme_name}: {len(markers)} → {markers}")
    if not markers:
        rep.error(
            "A1",
            f"{readme_name} hat keinen <!-- mcp-name: ... --> Marker. "
            "PyPI-Releases sind unveränderlich — vor dem Release ergänzen.",
        )
    elif len(markers) > 1:
        rep.warn("A1", f"{readme_name} hat {len(markers)} Marker — genau einer erwartet")

    sj = repo / "server.json"
    if sj.exists() and markers:
        try:
            name = json.loads(sj.read_text(encoding="utf-8")).get("name", "")
            if name and name != markers[0]:
                rep.error("A1", f"Marker '{markers[0]}' ≠ server.json name '{name}'")
        except json.JSONDecodeError as exc:
            rep.error("A2", f"server.json nicht parsebar: {exc}")


def check_server_json(repo: Path, rep: Report) -> None:
    """A2 — Registry lehnt description > 100 Zeichen mit 422 ab."""
    sj = repo / "server.json"
    if not sj.exists():
        return
    try:
        data = json.loads(sj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        rep.error("A2", f"server.json nicht parsebar: {exc}")
        return
    desc = data.get("description", "")
    if len(desc) > 100:
        rep.error(
            "A2",
            f"server.json description ist {len(desc)} Zeichen "
            f"(max. 100) — Registry antwortet sonst mit 422",
        )
    else:
        rep.info("A2", f"server.json description: {len(desc)}/100 Zeichen")

    pyproject = repo / "pyproject.toml"
    if pyproject.exists() and data.get("version"):
        raw = pyproject.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', raw, re.MULTILINE)
        if m and m.group(1) != data["version"]:
            rep.error(
                "A4",
                f"Versionsdrift: pyproject {m.group(1)} ≠ server.json {data['version']}",
            )


def check_ruff_config(repo: Path, rep: Report) -> None:
    """B1/B2 — jeder pyproject-Block braucht einen eigenen expliziten select."""
    for py in sorted(repo.rglob("pyproject.toml")):
        if any(part in {".venv", "venv", "node_modules", ".git"} for part in py.parts):
            continue
        raw = py.read_text(encoding="utf-8", errors="replace")
        rel = py.relative_to(repo)
        if "[tool.ruff" in raw or "ruff" in raw:
            if not re.search(r"\[tool\.ruff\.lint\][^\[]*select\s*=", raw, re.DOTALL):
                rep.error(
                    "B1",
                    f"{rel}: kein expliziter [tool.ruff.lint] select — "
                    "CI wird bei ruff-Updates ohne Codeänderung rot",
                )
        if re.search(r'ruff\s*>=\s*[\d.]+["\']', raw) and "<" not in raw.split("ruff")[1][:40]:
            rep.warn("B1", f"{rel}: ruff ohne Obergrenze gepinnt")


MCP_SPEC_TARGET = "2026-07-28"
RUFF_EXACT_RE = re.compile(r"ruff==\d+(?:\.\d+)*")


def repo_meta_field(repo: Path, key: str) -> str | None:
    """Wert eines Top-Level-Felds aus .github/repo-meta.yml — ohne YAML-Parser.

    Der Validator bleibt stdlib-only. Gelesen wird nur `key: wert` am
    Zeilenanfang, Kommentar und Anführungszeichen abgestreift.
    """
    meta = repo / ".github" / "repo-meta.yml"
    if not meta.exists():
        return None
    raw = meta.read_text(encoding="utf-8", errors="replace")
    m = re.search(rf"^{re.escape(key)}:[ \t]*([^#\n]*)", raw, re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip().strip("\"'") or None


def check_mcp_spec(repo: Path, rep: Report) -> None:
    """A5 — MCP-Server zielen auf Spec 2026-07-28 (references/mcp-spec.md).

    Geprüft wird die Deklaration im Intake, nicht der Draht: die
    Protokollversion gehört dem SDK, der Nachweis läuft über server/discover
    (mcp-spec.md, S4). Ein fehlendes Feld ist deshalb WARN — nicht gemessen
    ist nicht falsch —, ein anderer Wert ERROR.
    """
    is_mcp = (repo / "server.json").exists() or (
        repo_meta_field(repo, "project_type") == "mcp-server"
    )
    if not is_mcp:
        return
    declared = repo_meta_field(repo, "mcp_spec_version")
    if declared is None:
        rep.warn(
            "A5",
            "MCP-Server ohne mcp_spec_version in .github/repo-meta.yml — "
            f"Zielstand ist {MCP_SPEC_TARGET} (references/mcp-spec.md)",
        )
    elif declared != MCP_SPEC_TARGET:
        rep.error(
            "A5",
            f"mcp_spec_version ist {declared}, Zielstand ist {MCP_SPEC_TARGET} — "
            "Migration offen (references/mcp-spec.md)",
        )
    else:
        rep.info(
            "A5",
            f"mcp_spec_version: {declared} (deklariert — Nachweis am Draht: mcp-spec.md S4)",
        )


def _exact_ruff_pins(text: str) -> list[str]:
    """Exakte ruff-Pins in einem Text, Kommentarzeilen ausgenommen."""
    lines = [ln.split("#", 1)[0] for ln in text.splitlines()]
    return RUFF_EXACT_RE.findall("\n".join(lines))


def check_ruff_pin(repo: Path, rep: Report) -> None:
    """B1 — ein Formatgate braucht einen exakten Pin, und zwar an EINER Stelle.

    Gezählt werden die Stellen, an denen die Zahl steht: Workflow-Text,
    `requirements*.txt` und `constraints*.txt` im Wurzelverzeichnis und das
    `pyproject.toml`. Eine Stelle ist richtig, egal welche — die Vorlage
    schlägt `requirements-lint.txt` vor, aber ein exakter Pin im dev-Extra, in
    `requirements-dev.txt`, in einer Constraints-Datei oder als einziger im
    Workflow ist ebenfalls eine Quelle. Erst eine zweite Stelle ist eine
    Driftgelegenheit (WARN), abweichende Zahlen sind Drift (ERROR).

    Die erste Fassung dieser Regel warnte bei jedem `ruff==` im Workflow und
    kannte als Quelle nur `requirements-lint.txt` und `pyproject.toml`. Über
    das Portfolio gemessen waren damit vier von fünf Meldungen Fehlalarme:
    zwei Repos pinnten korrekt in `constraints.txt` bzw. `requirements-dev.txt`
    (ERROR «ohne exakten Pin»), zwei hatten die Zahl bewusst einzig im Workflow
    (WARN). Nur ein Repo trug die Zahl tatsächlich mehrfach.
    Die `rev` in `.pre-commit-config.yaml` zählt nicht mit: pre-commit kann
    keine Datei lesen, sie ist ein notwendiger Spiegel und kein Fehler.
    """
    wf_dir = repo / ".github" / "workflows"
    workflows = sorted(wf_dir.glob("*.y*ml")) if wf_dir.is_dir() else []
    texts = {w: w.read_text(encoding="utf-8", errors="replace") for w in workflows}
    if not any("ruff format --check" in t for t in texts.values()):
        return

    candidates = sorted(repo.glob("requirements*.txt")) + sorted(repo.glob("constraints*.txt"))
    candidates += list(texts)
    if (repo / "pyproject.toml").exists():
        candidates.append(repo / "pyproject.toml")

    places: list[tuple[str, list[str]]] = []
    for path in candidates:
        raw = texts.get(path) or path.read_text(encoding="utf-8", errors="replace")
        pins = _exact_ruff_pins(raw)
        if pins:
            places.append((path.relative_to(repo).as_posix(), sorted(set(pins))))

    if not places:
        rep.error(
            "B1",
            "Formatgate (ruff format --check) ohne exakten ruff-Pin — "
            "requirements-lint.txt mit ruff==X.Y.Z anlegen (8.1)",
        )
        return

    listing = ", ".join(f"{name} ({', '.join(pins)})" for name, pins in places)
    versions = {pin for _, pins in places for pin in pins}
    if len(versions) > 1:
        rep.error("B1", f"ruff-Pins widersprechen sich: {listing} (8.1)")
    elif len(places) > 1:
        rep.warn(
            "B1",
            f"ruff-Pin steht an {len(places)} Stellen: {listing} — auf eine "
            "Quelle zusammenführen, die übrigen lesen von dort (8.1)",
        )
    else:
        rep.info("B1", f"ruff exakt gepinnt, eine Quelle: {listing}")


def check_blind_assertions(repo: Path, rep: Report) -> None:
    """B3 — pytest.raises(Exception) besteht auch bei einem Tippfehler."""
    for py in sorted(repo.rglob("test_*.py")) + sorted(repo.rglob("*_test.py")):
        raw = py.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(raw.splitlines(), 1):
            if re.search(r"pytest\.raises\(\s*(Exception|BaseException)\s*[),]", line):
                rep.error(
                    "B3",
                    f"{py.relative_to(repo)}:{n}: blinde Assertion "
                    "pytest.raises(Exception) — konkrete Exception verwenden",
                )


def registered_tool_names(tree: ast.Module) -> list[str]:
    """Namen aller mit `@<x>.tool(...)` dekorierten Funktionen.

    Registriert ist das `name=`-Argument, wenn es gesetzt ist — sonst der
    Funktionsname. Genau diese Unterscheidung ist E1 (siehe review-rules.md).
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            target = call.func if call is not None else dec
            if not (isinstance(target, ast.Attribute) and target.attr == "tool"):
                continue
            name = node.name
            if call is not None:
                for kw in call.keywords:
                    if (
                        kw.arg == "name"
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    ):
                        name = kw.value.value
            names.append(name)
            break
    return names


def list_tool_names(repo: Path, rep: Report) -> None:
    """E1 — registrierte Namen ausgeben, nicht die Funktionsnamen.

    Über den AST statt über einen Regex auf dem Rohtext. Ein `@mcp.tool()`
    über einem `def` in einem Docstring-Beispiel ist Dokumentation, kein
    registriertes Tool — der frühere Regex konnte beides nicht unterscheiden
    und meldete in einem Skill-Repo ohne ein einziges Tool ein `my_tool` als
    «nicht im README dokumentiert».

    Nebeneffekt, gegengeprüft: Der Regex begrenzte die Dekorator-Argumente
    mit [^)]* und lief deshalb an jedem Dekorator vorbei, dessen Argumente
    selbst eine Klammer enthalten — etwa name=" ".join(...) oder eine
    Beschreibung mit einem Klammerausdruck darin. Zwei solche Tools blieben
    in einem Fixture unerkannt, während my_tool erfunden wurde.
    """
    registered: list[str] = []
    unparsed: list[str] = []
    for py in sorted(repo.rglob("*.py")):
        if any(p in {".venv", "venv", "build", "dist"} for p in py.parts):
            continue
        raw = py.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(raw)
        except SyntaxError:
            # Templates mit Platzhaltern, Fixtures, Py2-Reste: nicht parsebar
            # und praktisch nie ein echter Server. Gemeldet statt stillschweigend
            # übergangen — ein Pfad, den niemand prüft und niemand sieht, ist
            # derselbe Fehler eine Ebene höher.
            unparsed.append(str(py.relative_to(repo)))
            continue
        registered.extend(registered_tool_names(tree))
    if unparsed:
        rep.info("E1", f"Nicht parsebar, bei der Tool-Suche übersprungen: {sorted(unparsed)}")
    if registered:
        rep.info("E1", f"Registrierte Tool-Namen ({len(registered)}): {sorted(registered)}")
        # Alle Sprachfassungen, nicht nur README.md. Seit die Repos zweisprachig
        # sind, wäre ein nur im deutschen README dokumentiertes Tool sonst als
        # undokumentiert gemeldet worden — ein Fehlalarm derselben Klasse wie das
        # Docstring-Beispiel eine Funktion weiter oben.
        texts = {
            f.name: f.read_text(encoding="utf-8", errors="replace")
            for f in sorted(repo.glob("README*.md"))
        }
        if texts:
            unique = set(registered)
            missing = sorted(t for t in unique if not any(t in x for x in texts.values()))
            if missing:
                rep.warn("E1", f"Tools nicht im README dokumentiert: {missing}")
            # Getrennt gemeldet: «nirgends dokumentiert» und «nur in einer
            # Sprachfassung» sind verschiedene Fehler, und der zweite ist erst
            # sichtbar, wenn man beide Fassungen liest.
            for name, text in texts.items():
                gaps = sorted(t for t in unique - set(missing) if t not in text)
                if gaps:
                    rep.warn("E1", f"{name}: Tools fehlen in dieser Sprachfassung: {gaps}")


# C8 — Versionsanker. Die Quelle ist die oberste Release-Überschrift;
# `[Unreleased]` trägt keine Versionsnummer und wird vom Muster übersprungen.
RELEASE_HEADING_RE = re.compile(r"^## \[v?(?P<version>\d+\.\d+\.\d+)\]", re.MULTILINE)
# Zwei Schreibweisen im README, beide im Portfolio in Gebrauch: der Shields-Badge
# und die `**Version:**`-Zeile in einem Status-Abschnitt.
VERSION_ANCHORS = (
    (re.compile(r"badge/version-(\d+\.\d+\.\d+)-"), "Badge"),
    (
        re.compile(r"^\*\*Version:\*\*\s+v?(\d+\.\d+\.\d+)", re.MULTILINE),
        "Version-Zeile",
    ),
)
META_VERSION_RE = re.compile(r"^version:\s*v?(\d+\.\d+\.\d+)\s*$", re.MULTILINE)


def check_version_anchors(repo: Path, rep: Report) -> None:
    """C8 — Versionsangaben gegen die oberste Release-Überschrift im CHANGELOG.

    Die Version steht je nach Repo an bis zu vier Orten: im Badge jeder
    README-Sprachfassung, in einer `**Version:**`-Zeile, im `version`-Feld von
    `.github/repo-meta.yml` — und im CHANGELOG. Der CHANGELOG ist die Quelle,
    die übrigen folgen ihm.

    Zusammengehalten hat sie nichts, und entsprechend sind sie auseinander:
    Im Audit-Repo des Portfolios stand die Statuszeile drei Releases lang auf
    `v1.0.0`, in diesem Repo mussten beim v1.2.0-Release drei Anker von Hand
    nachgezogen werden.

    **WARN und nicht ERROR.** Eine veraltete Versionsangabe ist ein Doku-Mangel,
    kein Baufehler — und dieser Check läuft über ein gewachsenes Portfolio, in
    dem ein ERROR reihenweise blockieren würde, ohne dass etwas kaputt ist.

    Bewusst still, wo es nichts zu vergleichen gibt: ohne `CHANGELOG.md`, ohne
    Release-Überschrift (Repo vor dem ersten Release) oder ohne jeden Anker.
    Der letzte Fall wird als INFO gemeldet — «nichts gefunden» soll nicht wie
    «alles in Ordnung» aussehen.
    """
    changelog = repo / "CHANGELOG.md"
    if not changelog.is_file():
        return
    m = RELEASE_HEADING_RE.search(changelog.read_text(encoding="utf-8", errors="replace"))
    if m is None:
        return
    expected = m.group("version")

    found: list[tuple[str, str]] = []
    for readme in sorted(repo.glob("README*.md")):
        text = readme.read_text(encoding="utf-8", errors="replace")
        for pattern, label in VERSION_ANCHORS:
            found.extend((f"{readme.name} ({label})", v) for v in pattern.findall(text))
    meta = repo / ".github" / "repo-meta.yml"
    if meta.is_file():
        text = meta.read_text(encoding="utf-8", errors="replace")
        found.extend((".github/repo-meta.yml", v) for v in META_VERSION_RE.findall(text))

    if not found:
        rep.info("C8", f"Kein Versionsanker gefunden — CHANGELOG nennt {expected}")
        return
    stale = sorted({f"{where} → {v}" for where, v in found if v != expected})
    if stale:
        rep.warn("C8", f"Versionsangaben weichen vom CHANGELOG ({expected}) ab: {stale}")


def check_license_name(repo: Path, rep: Report) -> None:
    """C7 — LICENSE nennt den bürgerlichen Namen, nicht den GitHub-Handle."""
    lic = repo / "LICENSE"
    if not lic.exists():
        rep.error("C7", "LICENSE fehlt")
        return
    handle = git(repo, "config", "--get", "remote.origin.url") or ""
    m = re.search(r"[:/]([^/]+)/[^/]+?(?:\.git)?$", handle)
    if m:
        owner = m.group(1)
        for line in lic.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("copyright") and owner.lower() in line.lower():
                rep.warn(
                    "C7",
                    f"LICENSE nennt vermutlich den GitHub-Handle '{owner}' — "
                    "bürgerlichen Namen verwenden",
                )


def git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def check_branch(repo: Path, rep: Report) -> None:
    """E5 — der Default-Branch ist nicht immer main."""
    head = git(repo, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD")
    branch = head.rsplit("/", 1)[-1] if head else git(repo, "branch", "--show-current")
    if branch:
        rep.info("E5", f"Default-Branch: {branch}")
        if branch != "main":
            rep.warn(
                "E5",
                f"Default-Branch ist '{branch}', nicht 'main' — "
                "Push- und Workflow-Befehle anpassen",
            )


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    fail_on = {
        rule.strip()
        for a in sys.argv[1:]
        if a.startswith("--fail-on=")
        for rule in a.split("=", 1)[1].split(",")
        if rule.strip()
    }
    repo = Path(args[0] if args else ".").resolve()
    rep = Report()

    check_readme(repo / "README.md", "en", repo, rep)
    check_readme(repo / "README.de.md", "de", repo, rep)
    check_demo_parity(repo, rep)
    check_mcp_marker(repo, rep)
    check_server_json(repo, rep)
    check_mcp_spec(repo, rep)
    check_ruff_config(repo, rep)
    check_ruff_pin(repo, rep)
    check_blind_assertions(repo, rep)
    list_tool_names(repo, rep)
    check_license_name(repo, rep)
    check_version_anchors(repo, rep)
    check_branch(repo, rep)

    for item in rep.items:
        if item["level"] == "WARN" and item["rule"] in fail_on:
            item["level"] = "ERROR"
            item["message"] += " (--fail-on)"

    if as_json:
        print(json.dumps(rep.items, ensure_ascii=False, indent=2))
    else:
        icons = {"ERROR": "✗", "WARN": "!", "INFO": "·"}
        for level in ("ERROR", "WARN", "INFO"):
            for item in [i for i in rep.items if i["level"] == level]:
                print(f"{icons[level]} [{item['rule']}] {item['message']}")
        n_err = sum(1 for i in rep.items if i["level"] == "ERROR")
        n_warn = sum(1 for i in rep.items if i["level"] == "WARN")
        print(f"\n{repo.name}: {n_err} ERROR, {n_warn} WARN")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
