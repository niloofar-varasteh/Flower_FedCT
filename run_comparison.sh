#!/bin/bash
################################################################################
# Run FedCT and FedAvg experiments with same parameters and compare results
################################################################################

set -e
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# Default parameters (same for both experiments)
COMM_ROUNDS=20
CLIENTS=5
LOCAL_ROUNDS=8
DATASET="CIFAR10"
Architecture="LightCNN"
OPTIMIZER="Adam"
LR=0.001
BATCH_SIZE=64

# FedCT-specific
UNLABELED_SIZE=100
PUBLIC_BATCH=32
Architecture="LightCNN"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}===============================================================${NC}"
echo -e "${BLUE}FedCT vs FedAvg Comparison Experiment${NC}"
echo -e "${BLUE}===============================================================${NC}"
echo -e "Parameters (same for both):"
echo -e "  Dataset:              ${GREEN}${DATASET}${NC}"
echo -e "  Communication rounds: ${GREEN}${COMM_ROUNDS}${NC}"
echo -e "  Clients:              ${GREEN}${CLIENTS}${NC}"
echo -e "  Local rounds:         ${GREEN}${LOCAL_ROUNDS}${NC}"
echo -e "  Optimizer:            ${GREEN}${OPTIMIZER}${NC}"
echo -e "  Learning rate:        ${GREEN}${LR}${NC}"
echo -e "  Batch size:           ${GREEN}${BATCH_SIZE}${NC}"
echo -e "${BLUE}===============================================================${NC}"
echo ""

# Run FedCT experiment
echo -e "${YELLOW}===============================================================${NC}"
echo -e "${YELLOW}Running FedCT Experiment...${NC}"
echo -e "${YELLOW}===============================================================${NC}"
bash run_experiment.sh \
    --communication-rounds ${COMM_ROUNDS} \
    --clients ${CLIENTS} \
    --local-rounds ${LOCAL_ROUNDS} \
    --dataset ${DATASET} \
    --optimizer ${OPTIMIZER} \
    --lr ${LR} \
    --private-batch ${BATCH_SIZE} \
    --public-batch ${PUBLIC_BATCH} \
    --unlabeled ${UNLABELED_SIZE}

if [ $? -ne 0 ]; then
    echo -e "${RED}FedCT experiment failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}FedCT experiment completed!${NC}"
echo ""

# Run FedAvg experiment
echo -e "${YELLOW}===============================================================${NC}"
echo -e "${YELLOW}Running FedAvg Experiment...${NC}"
echo -e "${YELLOW}===============================================================${NC}"
bash run_fedavg.sh \
    --communication-rounds ${COMM_ROUNDS} \
    --clients ${CLIENTS} \
    --local-rounds ${LOCAL_ROUNDS} \
    --dataset ${DATASET} \
    --optimizer ${OPTIMIZER} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --log-per-local 1

if [ $? -ne 0 ]; then
    echo -e "${RED}FedAvg experiment failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}FedAvg experiment completed!${NC}"
echo ""

# Generate comparison plot
echo -e "${YELLOW}===============================================================${NC}"
echo -e "${YELLOW}Generating Comparison Plot...${NC}"
echo -e "${YELLOW}===============================================================${NC}"
python compare_fedct_fedavg.py \
    --mode local \
    --out fedct_vs_fedavg_local.png \
    --title "FedCT vs FedAvg (Local Rounds)"

if [ $? -ne 0 ]; then
    echo -e "${RED}Comparison plot generation failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}===============================================================${NC}"
echo -e "${GREEN}Comparison Complete!${NC}"
echo -e "${GREEN}===============================================================${NC}"
echo -e "Results saved to: ${GREEN}fedct_vs_fedavg_local.png${NC}"
echo -e "Mode: ${GREEN}local rounds${NC} (showing all local training steps)"
echo ""