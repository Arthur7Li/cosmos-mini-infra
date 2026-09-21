# AGENTS.md — Operating Contract for Coding Agents (Codex, Antigravity, etc.)

This file is the root-level contract every agent (Codex CLI, Antigravity, GitHub Copilot, or any other agentic tool) must read before making changes in this repository. Subdirectories may contain their own `AGENTS.md` with more specific, scoped instructions — the deepest applicable file wins for conflicting guidance, but this file's global rules always apply.

## 1. Mission

Build `cosmos-mini-infra`: a small but architecturally correct training/RL-infra system (see `docs/ARCHITECTURE.md`). Prioritize **correctness and measurability** over model quality. Every feature must ship with a test and, where relevant, a benchmark.

## 2. Source of Truth

- `docs/ARCHITECTURE.md` — target system design, do not contradict it without updating it first.
- `TASKS.md` — the live backlog. Always pick up the next unchecked task in priority order unless the human explicitly assigns a different one.
- `configs/*.yaml` — experiment configuration; never hardcode values that belong in a config.

## 3. Workflow Rules

1. Before coding, restate the task from `TASKS.md` in 1-2 sentences and identify the acceptance criteria.
2. Work on a feature branch named `agent/<task-slug>`. Never commit directly to `main`.
3. Make small, reviewable commits using Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`).
4. Every new module needs: type hints, a docstring, and at least one test in `tests/`.
5. Run `make lint` and `make test` locally (or via the sandbox) before opening a PR. Do not open a PR with failing checks.
6. Update `TASKS.md` (check the box, add a one-line result note) and `docs/ARCHITECTURE.md` if the task changed the design.
7. Open a PR using `.github/PULL_REQUEST_TEMPLATE.md`. Keep PRs scoped to one task from `TASKS.md`.
8. Never commit secrets, API keys, or `.env` — only `.env.example`.
9. If a task is ambiguous or blocked, stop and add a `blocked:` comment on the task in `TASKS.md` instead of guessing silently.

## 4. Engineering Standards

- Python 3.11+, fully type-hinted, formatted/linted with `ruff`, checked with `mypy` (see `pyproject.toml`).
- No silent `except Exception: pass`. Fail loudly in training/RL loops — a swallowed exception in a rollout worker is worse than a crash.
- All randomness must be seedable (`--seed` flag / config field) for reproducibility.
- Any new distributed or IPC code must include a single-process fallback path so tests run without multi-GPU hardware.
- Benchmarks and profiling scripts write artifacts to `output/` (gitignored) plus a small `metrics.json` summary that CI can diff.

## 5. Agent-Specific Notes

- **Codex (CLI/cloud)**: operate primarily inside `train/`, `rl/`, and `profiling/` — these are the most mechanical/boilerplate-heavy (IPC plumbing, DDP/FSDP wrappers, profiler hooks). Prefer generating complete, runnable modules over partial snippets.
- **Antigravity**: use for architecture-level tasks that span multiple files — e.g. wiring the rollout worker and trainer worker together, or designing the weight-sync protocol — since it can hold the whole-repo plan in its "Manager" surface. Log significant design decisions back into `docs/ARCHITECTURE.md`.
- Both agents must re-read this file and `TASKS.md` at the start of every new session; do not rely on stale in-context memory of the plan.

## 6. Definition of Done (per task)

A task in `TASKS.md` is done only when: tests pass, lint/type-check pass, the task is checked off with a result note, and (if it changes behavior) the README or ARCHITECTURE doc reflects it.
