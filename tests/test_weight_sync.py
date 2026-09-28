"""Tests for weight_sync."""
import multiprocessing
import time
from typing import Any

from rl.weight_sync import AsyncWeightManager, SyncWeightManager


def _pusher(q_to_push: "multiprocessing.Queue[Any]") -> None:
    time.sleep(0.5)
    m = SyncWeightManager(q_to_push)
    m.push_weights({"w": 1})


def test_sync_weight_manager() -> None:
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue[Any] = ctx.Queue()
    manager = SyncWeightManager(q)

    weights: dict[str, Any] = {"layer1.weight": [1, 2, 3]}
    manager.push_weights(weights)

    received = manager.get_weights()
    assert received == weights


def test_sync_weight_manager_blocks() -> None:
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue[Any] = ctx.Queue()
    manager = SyncWeightManager(q)

    p = ctx.Process(target=_pusher, args=(q,))
    p.start()

    start_time = time.time()
    received = manager.get_weights()
    end_time = time.time()

    assert received == {"w": 1}
    assert end_time - start_time >= 0.4
    p.join()


def test_async_weight_manager_empty() -> None:
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue[Any] = ctx.Queue()
    manager = AsyncWeightManager(q)

    assert manager.get_weights() is None


def test_async_weight_manager_latest() -> None:
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue[Any] = ctx.Queue()
    manager = AsyncWeightManager(q)

    weights1: dict[str, Any] = {"w": 1}
    weights2: dict[str, Any] = {"w": 2}
    weights3: dict[str, Any] = {"w": 3}

    manager.push_weights(weights1)
    manager.push_weights(weights2)
    manager.push_weights(weights3)

    # Wait for the multiprocessing Queue background thread to flush
    time.sleep(0.1)

    received = manager.get_weights()
    assert received == weights3

    assert manager.get_weights() is None


def test_async_weight_manager_push_full() -> None:
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue[Any] = ctx.Queue(maxsize=1)
    manager = AsyncWeightManager(q)

    manager.push_weights({"w": 1})
    manager.push_weights({"w": 2})

    # Wait for the multiprocessing Queue background thread to flush
    time.sleep(0.1)

    received = manager.get_weights()
    assert received in [{"w": 1}, {"w": 2}]
