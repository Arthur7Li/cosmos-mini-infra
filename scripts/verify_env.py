"""Sanity-check the dev environment. Agents should run this after `make setup`."""
import sys


def main() -> int:
    ok = True
    print(f"Python: {sys.version.split()[0]}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch: NOT INSTALLED")
        ok = False

    for pkg in ("gymnasium", "numpy", "yaml", "matplotlib", "pydantic"):
        try:
            __import__(pkg)
            print(f"{pkg}: ok")
        except ImportError:
            print(f"{pkg}: NOT INSTALLED")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
