#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  echo "Error: Python not found. Activate/install your environment, or set PYTHON_BIN=/path/to/python."
  exit 1
fi

mkdir -p plots

plot_one() {
  local DATASET="$1"
  local ARCH="$2"

  case "$DATASET" in
    CIFAR10|FashionMNIST) ;;
    *) echo "Error: dataset must be CIFAR10 or FashionMNIST"; exit 1 ;;
  esac

  case "$ARCH" in
    LightCNN|ResNet18) ;;
    *) echo "Error: architecture must be LightCNN or ResNet18"; exit 1 ;;
  esac

  if ! compgen -G "logs/${DATASET}_${ARCH}_*/experiment.log" >/dev/null; then
    echo "Skipping ${DATASET} ${ARCH}: no FedCT log found"
    return 0
  fi
  if ! compgen -G "logs_fedavg/${DATASET}_${ARCH}_*/experiment.log" >/dev/null; then
    echo "Skipping ${DATASET} ${ARCH}: no FedAvg log found"
    return 0
  fi

  "${PYTHON_BIN}" compare_fedct_fedavg.py \
    --dataset "${DATASET}" \
    --arch "${ARCH}" \
    --metric-type train \
    --out "plots/${DATASET}_${ARCH}_train.png" \
    --title "${DATASET} ${ARCH}: FedCT vs FedAvg"

  "${PYTHON_BIN}" compare_fedct_fedavg.py \
    --dataset "${DATASET}" \
    --arch "${ARCH}" \
    --metric-type test \
    --out "plots/${DATASET}_${ARCH}_test.png" \
    --title "${DATASET} ${ARCH}: FedCT vs FedAvg"

  echo "Saved plots/${DATASET}_${ARCH}_train.png"
  echo "Saved plots/${DATASET}_${ARCH}_test.png"
}

if [ "$#" -eq 0 ]; then
  plot_one CIFAR10 LightCNN
  plot_one CIFAR10 ResNet18
  plot_one FashionMNIST LightCNN
  plot_one FashionMNIST ResNet18
elif [ "$#" -eq 2 ]; then
  plot_one "$1" "$2"
else
  echo "Usage:"
  echo "  bash plot_comparison.sh"
  echo "  bash plot_comparison.sh CIFAR10 LightCNN"
  echo "  bash plot_comparison.sh CIFAR10 ResNet18"
  echo "  bash plot_comparison.sh FashionMNIST LightCNN"
  echo "  bash plot_comparison.sh FashionMNIST ResNet18"
  exit 1
fi
