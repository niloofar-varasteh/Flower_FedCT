#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

################################################################################
# Federated Averaging (FedAvg) Experiment Runner
# - Matches FedCT runner style/flags so comparisons are fair
# - Uses the SAME NN architecture as FedCT (default: resnet18)
# - Logs every LOCAL round (not only aggregations)
################################################################################

# -----------------------------
# Defaults (align with FedCT)
# -----------------------------
NUM_COMMUNICATION_ROUNDS=20  # FedAvg aggregation rounds
NUM_CLIENTS=5
NUM_LOCAL_ROUNDS=10          # local epochs per aggregation round
DATASET="CIFAR10"
OPTIMIZER="Adam"
LEARNING_RATE=0.001
BATCH_SIZE=64
ARCH="resnet18"               # <<< match FedCT architecture
SEED=42
LOG_PER_LOCAL=1               # <<< request per-local-round logging from Python

# -----------------------------
# Colors
# -----------------------------
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# -----------------------------
# Help
# -----------------------------
show_help() {
  echo "FedAvg Experiment Runner (aligned with FedCT)"
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
  echo "  --arch <name>                  resnet18 | resnet34 | cnn_small (default: ${ARCH})"
  echo "  --seed <int>                   Random seed (default: ${SEED})"
  echo "  --log-per-local <0|1>          Log accuracy each local round (default: ${LOG_PER_LOCAL})"
  echo "  -h, --help                     Show this help"
  exit 0
}

# -----------------------------
# Parse args
# -----------------------------
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
    --log-per-local)        LOG_PER_LOCAL="$2"; shift 2 ;;
    -h|--help)              show_help ;;
    *) echo -e "${RED}Unknown option: $1${NC}"; show_help ;;
  esac
done

# -----------------------------
# Validate
# -----------------------------
if [[ "$DATASET" != "CIFAR10" && "$DATASET" != "FashionMNIST" ]]; then
  echo -e "${RED}Error: dataset must be CIFAR10 or FashionMNIST${NC}"; exit 1
fi
if [[ "$OPTIMIZER" != "SGD" && "$OPTIMIZER" != "Adam" ]]; then
  echo -e "${RED}Error: optimizer must be SGD or Adam${NC}"; exit 1
fi

# -----------------------------
# Print config
# -----------------------------
echo -e "${BLUE}================ FedAvg Config (Aligned with FedCT) ================${NC}"
echo -e "Dataset:                          ${GREEN}${DATASET}${NC}"
echo -e "Clients:                          ${GREEN}${NUM_CLIENTS}${NC}"
echo -e "Local epochs per aggregation:     ${GREEN}${NUM_LOCAL_ROUNDS}${NC}"
echo -e "Aggregation rounds (FedAvg):      ${GREEN}${NUM_COMMUNICATION_ROUNDS}${NC}"
echo -e "Batch size:                       ${GREEN}${BATCH_SIZE}${NC}"
echo -e "Optimizer/LR:                     ${GREEN}${OPTIMIZER}/${LEARNING_RATE}${NC}"
echo -e "Architecture:                     ${GREEN}${ARCH}${NC}"
echo -e "Seed:                             ${GREEN}${SEED}${NC}"
echo -e "Log each LOCAL round (0/1):       ${GREEN}${LOG_PER_LOCAL}${NC}"
echo -e "Logs dir:                         ${GREEN}logs_fedavg${NC}"
echo -e "${BLUE}====================================================================${NC}"

# -----------------------------
# Prepare logs
# -----------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs_fedavg/${DATASET}_${ARCH}_opt${OPTIMIZER}_lr${LEARNING_RATE}_b${BATCH_SIZE}_R${NUM_COMMUNICATION_ROUNDS}_L${NUM_LOCAL_ROUNDS}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo -e "${YELLOW}Starting FedAvg training...${NC}"
echo -e "Saving logs to: ${LOG_DIR}/experiment.log"

# -----------------------------
# Run (direct Python)
# - We pass --arch and --log-per-local to ensure fair comparison and per-local plots
# -----------------------------
python run_fedavg.py \
  --communication-rounds ${NUM_COMMUNICATION_ROUNDS} \
  --clients ${NUM_CLIENTS} \
  --local-rounds ${NUM_LOCAL_ROUNDS} \
  --dataset ${DATASET} \
  --optimizer ${OPTIMIZER} \
  --lr ${LEARNING_RATE} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --arch ${ARCH} \
  --log-per-local ${LOG_PER_LOCAL} \
  2>&1 | tee "${LOG_DIR}/experiment.log"

# -----------------------------
# Done
# -----------------------------
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo -e "${GREEN}Experiment finished successfully!${NC}"
  echo -e "Results in: ${LOG_DIR}/experiment.log"
else
  echo -e "${RED}Experiment failed!${NC}"
  echo -e "Check logs at: ${LOG_DIR}/experiment.log"
  exit 1
fi