from __future__ import annotations

import importlib.metadata


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    print("Environment check")
    for package in (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "huggingface-hub",
        "datasets",
        "accelerate",
        "lerobot",
    ):
        print(f"  {package}: {version(package)}")

    try:
        import torch

        print(f"  torch cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  cuda device count: {torch.cuda.device_count()}")
            print(f"  cuda device 0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"  torch cuda check failed: {exc}")


if __name__ == "__main__":
    main()
