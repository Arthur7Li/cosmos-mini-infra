# AGENTS.md — profiling/

Scope: instrumentation and benchmark plotting.

- `profile_rl.py` wraps existing `rl/` entry points with `torch.profiler`; it must not duplicate rollout/trainer logic, only wrap it.
- Every benchmark run writes a `metrics.json` (throughput, GPU util, staleness, reward) into `output/<run_name>/` — never overwrite a previous run's directory.
- `plots.py` reads all `output/*/metrics.json` and produces comparison plots (matplotlib, no seaborn dependency) — keep it CLI-driven (`python -m profiling.plots --runs sync async`).
