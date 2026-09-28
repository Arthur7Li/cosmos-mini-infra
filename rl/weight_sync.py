"""Weight synchronization managers for RL."""
import multiprocessing
import queue
from typing import Any


class WeightManager:
    """Base class for weight synchronization."""

    def __init__(self, weight_queue: "multiprocessing.Queue[Any]") -> None:
        self.queue = weight_queue

    def get_weights(self) -> Any:
        raise NotImplementedError

    def push_weights(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError

class SyncWeightManager(WeightManager):
    """Synchronous weight manager that blocks until weights are available."""

    def get_weights(self) -> dict[str, Any]:
        return self.queue.get(block=True)

    def push_weights(self, state_dict: dict[str, Any]) -> None:
        self.queue.put(state_dict)

class AsyncWeightManager(WeightManager):
    """Asynchronous weight manager that gets the latest available weights."""

    def get_weights(self) -> dict[str, Any] | None:
        latest_weights = None
        while True:
            try:
                latest_weights = self.queue.get_nowait()
            except queue.Empty:
                break
        return latest_weights

    def push_weights(self, state_dict: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(state_dict)
        except queue.Full:
            pass
