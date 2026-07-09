#!/bin/bash
set -e

bash run_fedavg.sh \
  --dataset CIFAR10 \
  --arch LightCNN \
  --communication-rounds "${COMM_ROUNDS:-20}" \
  --clients "${CLIENTS:-5}" \
  --local-rounds "${LOCAL_ROUNDS:-8}" \
  --optimizer "${OPTIMIZER:-Adam}" \
  --lr "${LR:-0.001}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --unlabeled-size "${UNLABELED_SIZE:-100}" \
  --seed "${SEED:-42}"
