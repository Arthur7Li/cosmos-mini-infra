.PHONY: setup lint test smoke-train rl-sync rl-async profile

setup:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"

lint:
	ruff check .
	mypy . || true

test:
	pytest -q

smoke-train:
	python -m train.sft --config configs/sft.yaml --smoke

rl-sync:
	python -m rl.trainer_worker --config configs/rl_sync.yaml

rl-async:
	python -m rl.trainer_worker --config configs/rl_async.yaml

profile:
	python -m profiling.profile_rl --config configs/rl_async.yaml
	python -m profiling.plots
