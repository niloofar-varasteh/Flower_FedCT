# Flower FedCT vs FedAvg

This project runs FedCT and FedAvg through Flower only.

The core Python application is:

```text
fedcot/client_app.py
fedcot/server_app.py
fedcot/task.py
```

`run_direct.py` and `run_fedavg.py` are intentionally removed. FedAvg is called through Flower's built-in `flwr.server.strategy.FedAvg`.

## Flower Entry Points

The Flower app is configured in `pyproject.toml`:

```text
serverapp = "fedcot.server_app:app"
clientapp = "fedcot.client_app:app"
```

The run config key `algorithm` selects the method:

```text
algorithm = "fedct"
algorithm = "fedavg"
```

## Run FedCT

```bash
bash run_fedct_CIFAR10_LightCNN.sh
bash run_fedct_CIFAR10_ResNet18.sh
bash run_fedct_FashionMNIST_LightCNN.sh
bash run_fedct_FashionMNIST_ResNet18.sh
```

## Run FedAvg

```bash
bash run_fedavg_CIFAR10_LightCNN.sh
bash run_fedavg_CIFAR10_ResNet18.sh
bash run_fedavg_FashionMNIST_LightCNN.sh
bash run_fedavg_FashionMNIST_ResNet18.sh
```

All eight scripts use the same default fair-comparison settings:

```text
COMM_ROUNDS=20
CLIENTS=5
LOCAL_ROUNDS=8
OPTIMIZER=Adam
LR=0.001
BATCH_SIZE=64
PUBLIC_SIZE / UNLABELED_SIZE=100
```

Override defaults from the shell when needed:

```bash
COMM_ROUNDS=5 LOCAL_ROUNDS=2 bash run_fedct_CIFAR10_LightCNN.sh
COMM_ROUNDS=5 LOCAL_ROUNDS=2 bash run_fedavg_CIFAR10_LightCNN.sh
```

## Plot Comparisons

After running a matching FedCT/FedAvg pair, generate train and test plots:

```bash
bash plot_comparison.sh CIFAR10 LightCNN
```

Change `--dataset` and `--arch` for the other three combinations.
