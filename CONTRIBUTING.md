# Contributing

Thanks for considering a contribution. The project is small and opinionated;
please read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.

## Ground rules

- **Policy first.** Changes that weaken `policy.yaml` or its enforcement are
  not accepted. New capabilities must still refuse person-level profiling.
- **Measure, don't argue.** A new backend or lexicon change needs a run of
  `sentimentkit eval fixtures/de_synthetic_gold.jsonl` before and after, and
  the numbers in the PR description. A backend becomes default only when
  `eval.beats_baseline()` says so on the stage-1 gold set.
- **Synthetic fixtures only.** No real parent, pupil or staff statements, no
  real names, no internal system names.
- **Swiss spelling** in German text (no `ß`).

## Setup

```bash
pip install -e ".[dev]"
pip install -r requirements-lint.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python scripts/validate_repo.py .
```

## Pull requests

- One topic per PR, Conventional Commits (`feat`, `fix`, `docs`, `refactor`,
  `test`, `chore`).
- Add a line under `[Unreleased]` in `CHANGELOG.md`.
- Keep README.md and README.de.md in sync.
