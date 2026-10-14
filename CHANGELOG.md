# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-14

### Added
- Stage-0 skeleton: Python package, CLI (`analyze`, `eval`, `backends`, `policy-check`, `validate`)
- Evidence-atom data model and JSON schemas (judgement, run, gold set)
- `policy.yaml` with checks before the backend and before aggregation; exit code 2 on violation
- Lexicon backend for German: negation (before and Satzklammer), compounds via suffix match, multiword entries, intensifiers/diminishers, contrast, negation-immune adverbs, Swiss-German flagging
- Backend registry with stage-2 placeholders (`encoder`, `llm-score`, `llm-cloud`)
- Eval harness: accuracy, macro-F1, Cohen's kappa, ECE, tag slices, Alternative Annotator Test (pure Python)
- Human gate and review queue; Markdown report with k-anonymity floor
- Synthetic German gold set (30), synthetic survey CSV (24), aspects example
- Claude skill `sentiment-scan`
- Bilingual README, ROADMAP, ARCHITECTURE, CI with pinned ruff and format gate
