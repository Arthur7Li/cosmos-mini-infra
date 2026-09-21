# Antigravity Agent Instructions

This file is read by Antigravity's agent manager alongside the root `AGENTS.md`. It restates the operating contract in Antigravity's expected format and adds guidance specific to how Antigravity should plan work in this repo.

## Read First
1. `/AGENTS.md` — global rules (branching, commits, testing, definition of done).
2. `/TASKS.md` — current backlog; always work top-down within the active phase.
3. `/docs/ARCHITECTURE.md` — target design; update it when you make a design decision.

## Planning Guidance for Antigravity
- Use the Manager/planning surface to decompose each `TASKS.md` item into an explicit file-change plan *before* editing, and share that plan in the task's PR description.
- Prefer Antigravity for tasks that span multiple packages (e.g. wiring `rl/rollout_worker.py` to `rl/trainer_worker.py` to `rl/weight_sync.py`) since it can hold repo-wide context; hand off narrow, single-file, boilerplate-heavy tasks (IPC scaffolding, DDP/FSDP wrapper skeletons) to Codex.
- After finishing a task, verify with the browser/artifact tools if generating plots or dashboards (`profiling/plots.py` output) rather than trusting logs alone.
- Always leave the repo in a state where `make lint && make test` passes before ending a session — do not leave partially-applied edits.

## Guardrails
- Do not modify `.github/workflows/ci.yml` to skip failing checks in order to get green — fix the underlying issue or mark the task `blocked:` in `TASKS.md`.
- Do not introduce new top-level dependencies without adding them to `pyproject.toml` and noting why in the relevant commit message.
