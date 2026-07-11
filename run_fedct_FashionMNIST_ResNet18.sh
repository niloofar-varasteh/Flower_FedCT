#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

if command -v flwr >/dev/null 2>&1; then
  FLWR_BIN="flwr"
elif [ -x ".venv/Scripts/flwr.exe" ]; then
  FLWR_BIN=".venv/Scripts/flwr.exe"
elif [ -x ".venv/bin/flwr" ]; then
  FLWR_BIN=".venv/bin/flwr"
else
  echo "Error: Flower CLI not found. Activate/install your environment, or set FLWR_BIN=/path/to/flwr."
  exit 1
fi

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

DATASET="FashionMNIST"
ARCH="ResNet18"
COMM_ROUNDS="${COMM_ROUNDS:-20}"
CLIENTS="${CLIENTS:-5}"
LOCAL_ROUNDS="${LOCAL_ROUNDS:-8}"
OPTIMIZER="${OPTIMIZER:-Adam}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UNLABELED_SIZE="${UNLABELED_SIZE:-100}"
PUBLIC_BATCH="${PUBLIC_BATCH:-32}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${DATASET}_${ARCH}_opt${OPTIMIZER}_lr${LR}_b${BATCH_SIZE}_R${COMM_ROUNDS}_L${LOCAL_ROUNDS}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

"${PYTHON_BIN}" prepare_dataset.py --dataset "${DATASET}" --retry-clean

"${FLWR_BIN}" run . \
  --federation-config "options.num-supernodes=${CLIENTS} options.backend.client-resources.num-cpus=1 options.backend.client-resources.num-gpus=0.2" \
  --run-config \
  "num-communication-rounds=${COMM_ROUNDS} \
  num-clients=${CLIENTS} \
  num-local-rounds=${LOCAL_ROUNDS} \
  unlabeled-size=${UNLABELED_SIZE} \
  dataset=\"${DATASET}\" \
  architecture=\"${ARCH}\" \
  optimizer=\"${OPTIMIZER}\" \
  learning-rate=${LR} \
  private-batch-size=${BATCH_SIZE} \
  public-batch-size=${PUBLIC_BATCH}" \
  2>&1 | tee "${LOG_DIR}/experiment.log"
