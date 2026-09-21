# Architecture

## Goal

Replicate, at small scale, the systems shape of large physical-AI/world-model training infrastructure: pretraining -> SFT -> RL post-training with disaggregated rollout/training and profiler-driven tuning.

## Components

```
                +-------------------+
                |   Toy Env (envs/) |
                +---------+---------+
                          |
                 trajectories (obs, act, rew)
                          v
   +----------------+          weights          +------------------+
   | Rollout Worker | <----------------------- |  Trainer Worker    |
   | (rl/rollout_)  |                           |  (rl/trainer_)     |
   +--------+-------+                           +---------+----------+
            |                                            ^
            |     trajectories (queue/IPC)               |
            +--------------------------------------------+

Weight sync modes (rl/weight_sync.py):
  - sync:  trainer blocks rollout, pushes weights every update
  - async: rollout runs N steps on stale weights before pulling fresh ones
```

## Training Stack

- `worldmodel/` — model + tokenizer, framework-agnostic of training loop.
- `train/sft.py` — supervised pretraining with AMP.
- `train/distributed.py` — DDP/FSDP behind one interface, with a single-process fallback for CI/local dev without multi-GPU.

## RL Infra

- `rl/rollout_worker.py` and `rl/trainer_worker.py` run as separate processes (or ranks), communicating trajectories and weights over an IPC channel (queue/shared memory/gRPC — decided in Phase 3, documented here once implemented).
- `rl/weight_sync.py` is the core contribution: it makes the sync/async tradeoff explicit and measurable, mirroring how large-scale RL post-training systems decouple experience collection from policy updates.

## Profiling

- `profiling/profile_rl.py` instruments both workers with `torch.profiler`.
- `profiling/plots.py` turns traces + `metrics.json` into comparison plots (throughput, GPU utilization, staleness vs. reward) across sync/async modes — see `docs/BENCHMARKS.md` for results once populated.

## Open Design Questions (agents: log decisions here as they're made)

- IPC mechanism for rollout<->trainer: queue (multiprocessing) vs ZeroMQ vs gRPC — decide in Phase 3 based on what's simplest to make async-correct.
- Whether FSDP is needed at this scale or DDP alone suffices — decide based on Phase 2 benchmark results.
