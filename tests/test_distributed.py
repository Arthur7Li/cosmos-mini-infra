"""Tests for the distributed utilities."""

from pytest import MonkeyPatch
from torch import nn

from train.distributed import cleanup_distributed, setup_distributed, wrap_model


def test_distributed_fallback(monkeypatch: MonkeyPatch) -> None:
    """Test that setup_distributed correctly falls back to single-process."""
    # Ensure environment variables are not set
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    is_dist, rank, local_rank, world_size = setup_distributed()

    assert not is_dist
    assert rank == 0
    assert local_rank == 0
    assert world_size == 1

    # Wrapping with none should return the same model
    model = nn.Linear(10, 10)
    wrapped = wrap_model(model, strategy="none", local_rank=0)
    assert wrapped is model

    # Cleanup should safely do nothing
    cleanup_distributed()
