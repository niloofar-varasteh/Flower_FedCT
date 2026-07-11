"""Download and verify datasets before Flower starts Ray client actors."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torchvision
import torchvision.transforms as transforms


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def load_dataset(name: str, download: bool = True) -> None:
    transform = transforms.ToTensor()

    if name == "CIFAR10":
        torchvision.datasets.CIFAR10(
            root=str(DATA_ROOT), train=True, download=download, transform=transform
        )
        torchvision.datasets.CIFAR10(
            root=str(DATA_ROOT), train=False, download=download, transform=transform
        )
        return

    if name == "FashionMNIST":
        torchvision.datasets.FashionMNIST(
            root=str(DATA_ROOT), train=True, download=download, transform=transform
        )
        torchvision.datasets.FashionMNIST(
            root=str(DATA_ROOT), train=False, download=download, transform=transform
        )
        return

    raise ValueError(f"Unsupported dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["CIFAR10", "FashionMNIST"])
    parser.add_argument(
        "--retry-clean",
        action="store_true",
        help="Remove this dataset cache and retry once if verification fails.",
    )
    args = parser.parse_args()

    try:
        load_dataset(args.dataset)
    except Exception:
        if not args.retry_clean:
            raise
        dataset_dir = DATA_ROOT / args.dataset
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        load_dataset(args.dataset)

    print(f"Dataset ready: {args.dataset} at {DATA_ROOT}")


if __name__ == "__main__":
    main()
