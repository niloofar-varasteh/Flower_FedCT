#!/bin/bash
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
################################################################################
# Federated Averaging (FedAvg) Experiment Runner
################################################################################

# پارامترهای پیش‌فرض
NUM_COMMUNICATION_ROUNDS=15
NUM_CLIENTS=5
NUM_LOCAL_ROUNDS=3
DATASET="CIFAR10"
OPTIMIZER="Adam"
LEARNING_RATE=0.001
BATCH_SIZE=64

# رنگ‌ها برای خروجی ترمینال
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# نمایش تنظیمات
echo -e "${BLUE}================ FedAvg Config ================${NC}"
echo -e "Dataset:            ${GREEN}${DATASET}${NC}"
echo -e "Clients:            ${GREEN}${NUM_CLIENTS}${NC}"
echo -e "Local epochs/round: ${GREEN}${NUM_LOCAL_ROUNDS}${NC}"
echo -e "FedAvg rounds:      ${GREEN}${NUM_COMMUNICATION_ROUNDS}${NC}"
echo -e "Batch size:         ${GREEN}${BATCH_SIZE}${NC}"
echo -e "Optimizer/LR:       ${GREEN}${OPTIMIZER}/${LEARNING_RATE}${NC}"
echo -e "Logs:               ${GREEN}logs_fedavg${NC}"
echo -e "${BLUE}==============================================${NC}"

# ساخت فولدر لاگ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs_fedavg/${DATASET}_opt${OPTIMIZER}_lr${LEARNING_RATE}_b${BATCH_SIZE}_R${NUM_COMMUNICATION_ROUNDS}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo -e "${YELLOW}Starting FedAvg training...${NC}"

# اجرای مستقیم پایتون (بدون Flower/Ray)
python run_fedavg.py \
  --communication-rounds ${NUM_COMMUNICATION_ROUNDS} \
  --clients ${NUM_CLIENTS} \
  --local-rounds ${NUM_LOCAL_ROUNDS} \
  --dataset ${DATASET} \
  --optimizer ${OPTIMIZER} \
  --lr ${LEARNING_RATE} \
  --batch-size ${BATCH_SIZE} \
  2>&1 | tee "${LOG_DIR}/experiment.log"

# بررسی موفقیت
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo -e "${GREEN}Experiment finished successfully!${NC}"
  echo -e "Results in: ${LOG_DIR}/experiment.log"
else
  echo -e "${RED}Experiment failed!${NC}"
  echo -e "Check logs at: ${LOG_DIR}/experiment.log"
fi
