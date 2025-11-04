#!/bin/bash

################################################################################
# Federated Co-Training (FedCT) Experiment Runner
#
# This script runs FedCT experiments with clear, explicit parameters.
#
# Key Concepts:
#   - Num communication rounds: How many times clients communicate with server
#   - Num local rounds: How many passes through dataset before each communication
#   - Mixed batch training: Samples from BOTH private and public data each step
#
# Usage:
#   ./run_experiment.sh [OPTIONS]
#
# Options:
#   --communication-rounds <num>  Number of communication rounds (default: 20)
#   -c, --clients <num>           Number of clients (default: 5)
#   --local-rounds <num>          Num local rounds/passes through dataset (default: 5)
#   -u, --unlabeled <num>         Size of public unlabeled dataset (default: 500)
#   -d, --dataset <name>          Dataset: CIFAR10 or FashionMNIST (default: CIFAR10)
#   -o, --optimizer <name>        Optimizer: SGD or Adam (default: SGD)
#   --lr <rate>                   Learning rate (default: 0.01)
#   --private-batch <size>        Batch size for private data (default: 32)
#   --public-batch <size>         Batch size for public data (default: 32)
#   -h, --help                    Show this help message
################################################################################

# Default parameters (optimized for better performance)
NUM_COMMUNICATION_ROUNDS=15
NUM_CLIENTS=5
NUM_LOCAL_ROUNDS=3
UNLABELED_SIZE=100
DATASET="CIFAR10"
OPTIMIZER="Adam"
LEARNING_RATE=0.001
PRIVATE_BATCH_SIZE=64
PUBLIC_BATCH_SIZE=32

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display help
show_help() {
    echo "Federated Co-Training (FedCT) Experiment Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --communication-rounds <num>  Number of communication rounds (default: 15)"
    echo "  -c, --clients <num>           Number of clients (default: 5)"
    echo "  --local-rounds <num>          Number of local rounds (passes through dataset) before communication (default: 3)"
    echo "  -u, --unlabeled <num>         Size of public unlabeled dataset (default: 100)"
    echo "  -d, --dataset <name>          Dataset: CIFAR10 or FashionMNIST (default: CIFAR10)"
    echo "  -o, --optimizer <name>        Optimizer: SGD or Adam (default: Adam)"
    echo "  --lr <rate>                   Learning rate (default: 0.001)"
    echo "  --private-batch <size>        Batch size for private data (default: 64)"
    echo "  --public-batch <size>         Batch size for public pseudo-labeled data (default: 32)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Explanation:"
    echo "  - Communication rounds: Number of times clients communicate with server"
    echo "  - Local rounds: Number of times client goes through its entire dataset before communication"
    echo "  - In each training step, client samples:"
    echo "    * One batch from private data (size: private-batch)"
    echo "    * One batch from public pseudo-labeled data (size: public-batch)"
    echo ""
    echo "Examples:"
    echo "  $0                                                    # Use default settings (Adam, lr=0.001)"
    echo "  $0 --communication-rounds 20 --local-rounds 5         # More training rounds"
    echo "  $0 --dataset FashionMNIST                             # Try different dataset"
    echo "  $0 --optimizer SGD --lr 0.01                          # Use SGD instead of Adam"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --communication-rounds)
            NUM_COMMUNICATION_ROUNDS="$2"
            shift 2
            ;;
        -c|--clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --local-rounds)
            NUM_LOCAL_ROUNDS="$2"
            shift 2
            ;;
        -u|--unlabeled)
            UNLABELED_SIZE="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -o|--optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --private-batch)
            PRIVATE_BATCH_SIZE="$2"
            shift 2
            ;;
        --public-batch)
            PUBLIC_BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate dataset
if [[ "$DATASET" != "CIFAR10" && "$DATASET" != "FashionMNIST" ]]; then
    echo -e "${RED}Error: Invalid dataset '$DATASET'. Must be CIFAR10 or FashionMNIST${NC}"
    exit 1
fi

# Validate optimizer
if [[ "$OPTIMIZER" != "SGD" && "$OPTIMIZER" != "Adam" ]]; then
    echo -e "${RED}Error: Invalid optimizer '$OPTIMIZER'. Must be SGD or Adam${NC}"
    exit 1
fi

# Print experiment configuration
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}FedCT Experiment Configuration${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "Dataset:                       ${GREEN}${DATASET}${NC}"
echo -e "Num communication rounds:      ${GREEN}${NUM_COMMUNICATION_ROUNDS}${NC}"
echo -e "Number of clients:             ${GREEN}${NUM_CLIENTS}${NC}"
echo -e "Num local rounds:              ${GREEN}${NUM_LOCAL_ROUNDS}${NC} (passes through dataset)"
echo -e "Private batch size:            ${GREEN}${PRIVATE_BATCH_SIZE}${NC}"
echo -e "Public batch size:             ${GREEN}${PUBLIC_BATCH_SIZE}${NC}"
echo -e "Optimizer:                     ${GREEN}${OPTIMIZER}${NC}"
echo -e "Learning rate:                 ${GREEN}${LEARNING_RATE}${NC}"
echo -e "Unlabeled dataset size:        ${GREEN}${UNLABELED_SIZE}${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Create output directory for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${DATASET}_${OPTIMIZER}_lr${LEARNING_RATE}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo -e "${YELLOW}Starting experiment...${NC}"
echo -e "Logs will be saved to: ${LOG_DIR}"
echo ""

# Check if direct execution is requested
if [ "${USE_DIRECT_EXECUTION}" = "1" ]; then
    echo -e "${YELLOW}Using direct execution (bypassing Flower simulation)${NC}"
    echo ""
    
    # Run using direct execution Python script
    python run_direct.py \
        --communication-rounds ${NUM_COMMUNICATION_ROUNDS} \
        --clients ${NUM_CLIENTS} \
        --local-rounds ${NUM_LOCAL_ROUNDS} \
        --unlabeled ${UNLABELED_SIZE} \
        --dataset ${DATASET} \
        --optimizer ${OPTIMIZER} \
        --lr ${LEARNING_RATE} \
        --private-batch-size ${PRIVATE_BATCH_SIZE} \
        --public-batch-size ${PUBLIC_BATCH_SIZE} \
        2>&1 | tee "${LOG_DIR}/experiment.log"
else
    echo -e "${YELLOW}Using Flower framework${NC}"
    echo -e "${YELLOW}(If you encounter Ray errors, run: export USE_DIRECT_EXECUTION=1)${NC}"
    echo ""
    
    # Run using Flower framework
    flwr run . --run-config "num-communication-rounds=${NUM_COMMUNICATION_ROUNDS} num-clients=${NUM_CLIENTS} num-local-rounds=${NUM_LOCAL_ROUNDS} unlabeled-size=${UNLABELED_SIZE} dataset=\"${DATASET}\" optimizer=\"${OPTIMIZER}\" learning-rate=${LEARNING_RATE} private-batch-size=${PRIVATE_BATCH_SIZE} public-batch-size=${PUBLIC_BATCH_SIZE}" \
        2>&1 | tee "${LOG_DIR}/experiment.log"
fi

# Check if experiment completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}Experiment completed successfully!${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo -e "Results saved to: ${LOG_DIR}"
    echo ""
    
    # Extract and display final results
    if [ -f "${LOG_DIR}/experiment.log" ]; then
        echo -e "${BLUE}Final Results Summary:${NC}"
        echo -e "${BLUE}=================================${NC}"
        
        # Extract average test accuracy from last round
        LAST_ROUND_ACC=$(grep -oP "Average Test Acc:\s+\K[0-9.]+(?=\s+±)" "${LOG_DIR}/experiment.log" | tail -1)
        if [ ! -z "$LAST_ROUND_ACC" ]; then
            echo -e "Average Test Accuracy (final): ${GREEN}${LAST_ROUND_ACC}${NC}"
        fi
        
        # Extract agreement statistics from last round
        LAST_AGREEMENT=$(grep -oP "Mean Agreement:\s+\K[0-9.]+" "${LOG_DIR}/experiment.log" | tail -1)
        if [ ! -z "$LAST_AGREEMENT" ]; then
            echo -e "Mean Agreement (final):        ${GREEN}${LAST_AGREEMENT}${NC}"
        fi
        
        echo -e "${BLUE}=================================${NC}"
    fi
else
    echo ""
    echo -e "${RED}=================================${NC}"
    echo -e "${RED}Experiment failed!${NC}"
    echo -e "${RED}=================================${NC}"
    echo -e "Check logs at: ${LOG_DIR}/experiment.log"
    exit 1
fi

