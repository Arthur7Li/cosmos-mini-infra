"""Tests for the World Model architecture."""

import torch

from worldmodel.model import WorldModel, WorldModelConfig


def test_worldmodel_shape() -> None:
    """Test forward pass shape and basic properties."""
    config = WorldModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=2, max_seq_len=64)
    model = WorldModel(config)

    batch_size, seq_len = 2, 10
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx)
    assert loss is None
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.dim() == 0  # scalar


def test_worldmodel_overfit() -> None:
    """Test that the model can perfectly memorize a tiny batch (single step)."""
    torch.manual_seed(42)
    config = WorldModelConfig(vocab_size=10, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = WorldModel(config)

    # Tiny batch: memorize a single sequence
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    targets = torch.tensor([[2, 3, 4, 5, 6]])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Overfit loop
    for _ in range(50):
        optimizer.zero_grad()
        _, loss = model(idx, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()

    # Check if loss is near zero
    _, final_loss = model(idx, targets)
    assert final_loss is not None
    assert final_loss.item() < 0.1

    # Check argmax predictions
    logits, _ = model(idx)
    preds = logits.argmax(dim=-1)
    assert torch.equal(preds, targets)
