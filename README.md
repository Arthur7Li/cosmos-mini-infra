# cosmos-mini-infra

A small-scale, architecturally faithful replica of the systems patterns used in large physical-AI/world-model training stacks (e.g. NVIDIA Cosmos): distributed pretraining, RL post-training with a **disaggregated rollout/trainer split**, **weight synchronization** between the two, and **profiler-driven benchmarking**.

The goal is not model quality — it's demonstrating correct, measured systems engineering at small scale:

- Tiny transformer/ConvLSTM "world model" predicting next-frame/state in a toy simulated environment.
- Supervised pretraining + SFT with mixed precision (AMP/bf16).
- Multi-process **distributed training** via `torch.distributed` (DDP/FSDP), with measured compute/communication overlap.
- **RL post-training infra**: a rollout worker (collects trajectories from the current policy) and a trainer worker (PPO update), connected over IPC, running in both **synchronous** and **asynchronous** weight-sync modes.
- **Profiling & benchmarking**: `torch.profiler` traces, GPU utilization, throughput, and policy-staleness-vs-reward curves comparing sync vs async execution.

## Architecture

```text
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
```
For deep architectural details, benchmark results, and IPC pipeline notes, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Quickstart

```bash
make setup        # create venv + install deps
make lint          # ruff + mypy
make test          # pytest (unit + smoke)
make smoke-train   # tiny end-to-end SFT run
make rl-sync       # run disaggregated RL loop, synchronous weight sync
make rl-async      # run disaggregated RL loop, asynchronous weight sync
make profile       # produce benchmark plots in output/
```

## Repository Map

```
worldmodel/   tiny transformer/ConvLSTM model + tokenizer
envs/         toy physics simulation environment (gymnasium-style)
train/        SFT loop, DDP/FSDP wrapper, AMP
rl/           PPO trainer + rollout worker + weight-sync IPC
profiling/    torch.profiler hooks, benchmark scripts, plotting
configs/      YAML configs for sync vs async experiments
scripts/      environment verification, misc utilities
tests/        unit + smoke tests
docs/         architecture notes, benchmark write-ups
```

## Agentic Development

This repo is built with heavy use of AI coding agents (Codex, Antigravity). See [`AGENTS.md`](AGENTS.md) for the agent operating contract, [`TASKS.md`](TASKS.md) for the live task backlog agents should work from, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the target system design. Subsystem-scoped agent instructions live in each package's own `AGENTS.md`.

## Motivation

Built to demonstrate the systems concepts behind large-scale physical-AI/world-model training infrastructure — distributed parallelism, low-precision training, weight sync between training and inference, and disaggregated sync/async execution — at a scale runnable on 1-2 GPUs.
