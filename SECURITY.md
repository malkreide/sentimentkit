# Security Policy

## Scope

sentimentkit processes text locally. Stage 0 has no network access at all;
later backends (`llm-cloud`) will send text to an external API only when the
run declares the input anonymised — the policy refuses otherwise.

## Data handling

- Never point the toolkit at person-level data. `policy.yaml` aborts runs whose
  input carries identifier columns, and refuses aggregates below the floor.
- Review queues and judgement files contain the judged sentences. Treat output
  directories with the same care as the input.
- Fixtures are synthetic. Do not add real survey answers, mails or posts to the
  repository.

## Reporting a vulnerability

Please open a private security advisory on GitHub
(`Security` → `Report a vulnerability`) or contact the author via the GitHub
profile. Do not open a public issue for security-relevant findings. You will
receive an acknowledgement within 7 days.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
