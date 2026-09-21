# Contributing (Human + Agent Workflow)

This repo is developed collaboratively by a human maintainer and AI coding agents (Codex, Antigravity, GitHub Copilot). The same rules apply to both.

## Branching & Commits
- Branch naming: `agent/<task-slug>` for agent-driven work, `dev/<topic>` for manual work.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`.
- Keep commits and PRs scoped to a single `TASKS.md` item.

## Before Opening a PR
1. `make lint` — ruff + mypy must pass.
2. `make test` — pytest must pass, including any new tests for the change.
3. Check off the corresponding item in `TASKS.md` with a one-line result note.
4. If the change affects the system design, update `docs/ARCHITECTURE.md`.

## PR Review
- Use `.github/PULL_REQUEST_TEMPLATE.md`.
- For agent-authored PRs, request a Copilot review first, then human review.
- Squash-merge to `main` once green.

## Reporting Issues
Open a GitHub issue with reproduction steps, expected vs actual behavior, and (for perf issues) the relevant `metrics.json`/profiler trace if available.
