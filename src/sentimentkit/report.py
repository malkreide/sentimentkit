"""Aggregation and the Markdown report.

Aggregation is where the k-anonymity floor is enforced: no group below
`policy.aggregation_floor` is ever rendered. The report cites the run id,
backend, model version and policy hash so every figure is traceable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from .models import Judgement, Polarity, RunMeta
from .policy import Policy, PolicyViolation


def aggregate(
    judgements: Iterable[Judgement],
    *,
    policy: Policy,
    by: str = "aspect",
    enforce_floor: bool = True,
) -> dict[str, dict[str, float | int]]:
    """Return {group: {n, positive, neutral, negative, unclear, not_mentioned,
    mean_intensity, review_rate}} with shares in [0, 1]."""
    groups: dict[str, list[Judgement]] = defaultdict(list)
    for j in judgements:
        key = getattr(j, by, None) if by != "aspect" else j.aspect
        groups[str(key)].append(j)

    out: dict[str, dict[str, float | int]] = {}
    for key, js in sorted(groups.items()):
        n = len(js)
        if enforce_floor:
            policy.check_group_size(key, n)
        counts = Counter(j.polarity for j in js)
        polar = [
            j for j in js if j.polarity in (Polarity.POSITIVE, Polarity.NEGATIVE, Polarity.NEUTRAL)
        ]
        out[key] = {
            "n": n,
            **{p.value: round(counts.get(p, 0) / n, 4) for p in Polarity},
            "mean_intensity": round(sum(j.intensity for j in polar) / len(polar), 4)
            if polar
            else 0.0,
            "review_rate": round(sum(1 for j in js if j.needs_review) / n, 4),
        }
    return out


def markdown_report(meta: RunMeta, judgements: list[Judgement], policy: Policy) -> str:
    lines = [
        "# sentimentkit report",
        "",
        f"- run_id: `{meta.run_id}`  ",
        f"- purpose: `{meta.purpose}`  ",
        f"- backend: `{meta.backend}` · model_version: `{meta.model_version}`  ",
        f"- policy_hash: `{meta.policy_hash}` · sentimentkit `{meta.sentimentkit_version}`  ",
        f"- items: {meta.n_items} · judgements: {meta.n_judgements} · review queue: {meta.n_review}  ",
        f"- aggregation floor: n ≥ {policy.aggregation_floor}",
        "",
        "## Tonalität nach Aspekt",
        "",
        "| Aspekt | n | positiv | neutral | negativ | unklar | nicht erwähnt | Ø Intensität | Review-Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    try:
        agg = aggregate(judgements, policy=policy)
    except PolicyViolation as exc:
        lines.append(f"| *(unterdrückt)* | | | | | | | | |\n\n> Policy: {exc}")
        return "\n".join(lines) + "\n"
    for aspect, row in agg.items():
        lines.append(
            f"| {aspect} | {row['n']} | {_pct(row['positive'])} | {_pct(row['neutral'])} | "
            f"{_pct(row['negative'])} | {_pct(row['unclear'])} | {_pct(row['not_mentioned'])} | "
            f"{row['mean_intensity']:+.2f} | {_pct(row['review_rate'])} |"
        )
    lines += [
        "",
        "## Hinweise",
        "",
        "- Jede Zahl ist aus Evidenz-Atomen (`judgements.jsonl`) rekonstruierbar; "
        "Filter: `run_id`, `aspect`.",
        "- Urteile mit `needs_review` sind in der Review-Queue und in den Anteilen enthalten.",
        "- Die Analyse beurteilt Texte über Themen, nicht Personen (policy.yaml).",
        "",
    ]
    return "\n".join(lines)


def _pct(x: float | int) -> str:
    return f"{100 * float(x):.0f} %"
