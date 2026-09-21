# AGENTS.md — rl/

Scope: the disaggregated RL post-training loop. This is the core deliverable of the repo — prioritize correctness of weight versioning over raw performance.

- `rollout_worker.py` and `trainer_worker.py` must be launchable as independent processes (`python -m rl.rollout_worker ...`) as well as composable in-process for tests.
- `weight_sync.py` must tag every weight push with a monotonically increasing version number; the rollout worker must log which version it's acting on for every trajectory it emits. This is what makes the sync-vs-async benchmark meaningful.
- Do not let a rollout worker crash silently on a stale/missing weight file — raise, with a clear error naming the expected version.
- Any new IPC mechanism must be justified in `docs/ARCHITECTURE.md` under "Open Design Questions" before being adopted as the default.
