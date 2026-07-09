#!/bin/bash
set -e
set -o pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

################################################################################
# FedAvg Flower Runner
# - Runner style matches the old project script
# - FedAvg aggregation is Flower's built-in flwr.server.strategy.FedAvg
# - Model/data/evaluation are weak baseline code in baselines/fedavg_flower/app.py
################################################################################

NUM_COMMUNICATION_ROUNDS=20
NUM_CLIENTS=5
NUM_LOCAL_ROUNDS=8
DATASET="CIFAR10"
OPTIMIZER="Adam"
LEARNING_RATE=0.001
BATCH_SIZE=64
ARCH="LightCNN"
SEED=42
UNLABELED_SIZE=100

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

show_help() {
  echo "FedAvg Flower Runner (comparison baseline)"
  echo ""
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --communication-rounds <int>   Aggregation rounds (default: ${NUM_COMMUNICATION_ROUNDS})"
  echo "  -c, --clients <int>            Number of clients (default: ${NUM_CLIENTS})"
  echo "  --local-rounds <int>           Local epochs per aggregation (default: ${NUM_LOCAL_ROUNDS})"
  echo "  -d, --dataset <name>           CIFAR10 | FashionMNIST (default: ${DATASET})"
  echo "  -o, --optimizer <name>         SGD | Adam (default: ${OPTIMIZER})"
  echo "  --lr <float>                   Learning rate (default: ${LEARNING_RATE})"
  echo "  --batch-size <int>             Train batch size (default: ${BATCH_SIZE})"
  echo "  --arch <name>                  LightCNN | ResNet18 (default: ${ARCH})"
  echo "  --seed <int>                   Random seed (default: ${SEED})"
  echo "  --unlabeled-size <int>         Public U size excluded from private train (default: ${UNLABELED_SIZE})"
  echo "  --public-size <int>            Alias for --unlabeled-size, for old-script compatibility"
  echo "  -h, --help                     Show this help"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --communication-rounds) NUM_COMMUNICATION_ROUNDS="$2"; shift 2 ;;
    -c|--clients)           NUM_CLIENTS="$2"; shift 2 ;;
    --local-rounds)         NUM_LOCAL_ROUNDS="$2"; shift 2 ;;
    -d|--dataset)           DATASET="$2"; shift 2 ;;
    -o|--optimizer)         OPTIMIZER="$2"; shift 2 ;;
    --lr)                   LEARNING_RATE="$2"; shift 2 ;;
    --batch-size)           BATCH_SIZE="$2"; shift 2 ;;
    --arch)                 ARCH="$2"; shift 2 ;;
    --seed)                 SEED="$2"; shift 2 ;;
    --unlabeled-size)       UNLABELED_SIZE="$2"; shift 2 ;;
    --public-size)          UNLABELED_SIZE="$2"; shift 2 ;;
    -h|--help)              show_help ;;
    *) echo -e "${RED}Unknown option: $1${NC}"; show_help ;;
  esac
done

if [[ "$DATASET" != "CIFAR10" && "$DATASET" != "FashionMNIST" ]]; then
  echo -e "${RED}Error: dataset must be CIFAR10 or FashionMNIST${NC}"
  exit 1
fi
if [[ "$OPTIMIZER" != "SGD" && "$OPTIMIZER" != "Adam" ]]; then
  echo -e "${RED}Error: optimizer must be SGD or Adam${NC}"
  exit 1
fi

if command -v flwr >/dev/null 2>&1; then
  FLWR_BIN="$(command -v flwr)"
elif [ -x ".venv/Scripts/flwr.exe" ]; then
  FLWR_BIN="$(pwd)/.venv/Scripts/flwr.exe"
elif [ -x ".venv/bin/flwr" ]; then
  FLWR_BIN="$(pwd)/.venv/bin/flwr"
else
  echo -e "${RED}Error: Flower CLI not found. Activate/install your environment, or set FLWR_BIN=/path/to/flwr.${NC}"
  exit 1
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN="$(pwd)/.venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN="$(pwd)/.venv/bin/python"
else
  echo -e "${RED}Error: Python not found. Activate/install your environment, or set PYTHON_BIN=/path/to/python.${NC}"
  exit 1
fi

echo -e "${BLUE}================ FedAvg Config (Flower Built-in Strategy) ================${NC}"
echo -e "Dataset:                          ${GREEN}${DATASET}${NC}"
echo -e "Clients:                          ${GREEN}${NUM_CLIENTS}${NC}"
echo -e "Local epochs per aggregation:     ${GREEN}${NUM_LOCAL_ROUNDS}${NC}"
echo -e "Aggregation rounds (FedAvg):      ${GREEN}${NUM_COMMUNICATION_ROUNDS}${NC}"
echo -e "Batch size:                       ${GREEN}${BATCH_SIZE}${NC}"
echo -e "Optimizer/LR:                     ${GREEN}${OPTIMIZER}/${LEARNING_RATE}${NC}"
echo -e "Architecture:                     ${GREEN}${ARCH}${NC}"
echo -e "Seed:                             ${GREEN}${SEED}${NC}"
echo -e "Unlabeled/public U size:          ${GREEN}${UNLABELED_SIZE}${NC}"
echo -e "Flower app:                       ${GREEN}baselines/fedavg_flower${NC}"
echo -e "Baseline pipeline:                ${GREEN}weak ToTensor-only app.py${NC}"
echo -e "Logs dir:                         ${GREEN}logs_fedavg${NC}"
echo -e "${BLUE}==========================================================================${NC}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs_fedavg/${DATASET}_${ARCH}_opt${OPTIMIZER}_lr${LEARNING_RATE}_b${BATCH_SIZE}_R${NUM_COMMUNICATION_ROUNDS}_L${NUM_LOCAL_ROUNDS}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo -e "${YELLOW}Starting FedAvg training with Flower built-in FedAvg...${NC}"
echo -e "Saving logs to: ${LOG_DIR}/experiment.log"

"${PYTHON_BIN}" prepare_dataset.py --dataset "${DATASET}" --retry-clean

(
  cd baselines/fedavg_flower
  "${FLWR_BIN}" run . \
    --federation-config "options.num-supernodes=${NUM_CLIENTS}" \
    --run-config \
    "num-communication-rounds=${NUM_COMMUNICATION_ROUNDS} \
  num-clients=${NUM_CLIENTS} \
  num-local-rounds=${NUM_LOCAL_ROUNDS} \
  unlabeled-size=${UNLABELED_SIZE} \
  dataset=\"${DATASET}\" \
  architecture=\"${ARCH}\" \
  optimizer=\"${OPTIMIZER}\" \
  learning-rate=${LEARNING_RATE} \
  batch-size=${BATCH_SIZE} \
  seed=${SEED}" \
    2>&1
) | tee "${LOG_DIR}/experiment.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo -e "${GREEN}Experiment finished successfully!${NC}"
  echo -e "Results in: ${LOG_DIR}/experiment.log"
else
  echo -e "${RED}Experiment failed!${NC}"
  echo -e "Check logs at: ${LOG_DIR}/experiment.log"
  exit 1
fi
