"""Smoke test: package imports and config files load. Keeps CI green from commit 1."""
from pathlib import Path

import yaml


def test_configs_load():
    for cfg in Path("configs").glob("*.yaml"):
        with open(cfg) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)


def test_packages_import():
    import envs  # noqa: F401
    import profiling  # noqa: F401
    import rl  # noqa: F401
    import train  # noqa: F401
    import worldmodel  # noqa: F401
