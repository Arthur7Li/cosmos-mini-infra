# AGENTS.md — train/

Scope: SFT training loop (`sft.py`) and the distributed wrapper (`distributed.py`).

- `distributed.py` must expose one function, e.g. `wrap_model(model, cfg) -> nn.Module`, that internally picks DDP, FSDP, or a no-op single-process path based on `cfg.distributed.strategy`. Never branch on distributed strategy outside this module.
- All training loops must support running with `WORLD_SIZE=1` (no torchrun) for local dev/CI, and under `torchrun` for multi-process.
- Log step time, samples/sec, and (if CUDA available) `torch.cuda.utilization()` every N steps to `output/metrics.json` — `profiling/` scripts consume this.
- Use `torch.autocast` for AMP; never hand-roll fp16 casting.
