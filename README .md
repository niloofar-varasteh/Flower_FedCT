# Federated Co-Training (FedCT)

A Flower-based implementation of FedCT where clients collaboratively train models by sharing predictions (not model weights) on a public unlabeled dataset.

## 🚀 Quick Start

```bash
# Default run
./run_experiment.sh

# Custom configuration
./run_experiment.sh --clients 5 --communication-rounds 10 --local-rounds 3

# If Ray/Pydantic issues
export USE_DIRECT_EXECUTION=1
./run_experiment.sh
```

---

## 📦 Installation

```bash
# Create environment
conda create -n fedcot python=3.9
conda activate fedcot

# Install dependencies
pip install torch torchvision
pip install "numpy<2.0"
pip install "pydantic>=1.10.0,<2.0.0"
pip install "ray>=2.7.0,<2.10.0"
pip install "pyarrow<15.0.0"
pip install "flwr[simulation]>=1.8.0"
```

**Troubleshooting:** If Ray/Pydantic errors occur, use direct execution: `export USE_DIRECT_EXECUTION=1`

---

## 🎯 How It Works

**FedCT** enables collaborative federated learning without sharing private data or model weights:

1. **Private Data**: Each client has labeled data (not shared)
2. **Public Data**: All clients access shared unlabeled dataset
3. **Predictions**: Clients share predictions (hard labels only)
4. **Voting**: Server creates consensus labels via majority voting
5. **Training**: Clients train on private + pseudo-labeled public data

### Training Flow

```
Cycle 1 (First):
  Local Rounds 1-N: Train on PRIVATE data only
  Communication: Share predictions → Create consensus labels

Cycle 2+:
  Local Rounds 1-N: Train on PRIVATE + PSEUDO-LABELED data
  Communication: Share predictions → Update consensus labels
```

**Mixed Batch Training**: After first communication, each training step samples from both private data (ground truth) and public data (consensus labels).

---

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--communication-rounds` | 15 | Number of communication rounds |
| `--local-rounds` | 3 | Training passes through dataset before communication |
| `--clients` | 5 | Number of clients |
| `--unlabeled` | 100 | Size of public unlabeled dataset |
| `--dataset` | CIFAR10 | CIFAR10 or FashionMNIST |
| `--optimizer` | Adam | SGD or Adam |
| `--lr` | 0.001 | Learning rate |
| `--private-batch` | 64 | Batch size for private data |
| `--public-batch` | 32 | Batch size for public data |

---

## 📚 Examples

**Quick test (fast):**
```bash
./run_experiment.sh --clients 3 --communication-rounds 5 --local-rounds 2
```

**Balanced (recommended):**
```bash
./run_experiment.sh --communication-rounds 10 --local-rounds 2
```

**FashionMNIST:**
```bash
./run_experiment.sh --dataset FashionMNIST
```

**Use less-busy GPU:**
```bash
CUDA_VISIBLE_DEVICES=4 ./run_experiment.sh
```

---

## 📊 Expected Output

**Communication Cycle:**
```
================================================================================
📊 COMMUNICATION CYCLE 1/10 (Flower Round 1)
================================================================================
→ Clients will do 3 local training rounds
→ Then share predictions on public dataset
================================================================================
✓ Local Round 1/3 [Flower Round 1]: Train Loss: 1.93, Train Acc: 0.28, Test Acc: 0.34
✓ Local Round 2/3 [Flower Round 2]: Train Loss: 1.87, Train Acc: 0.31, Test Acc: 0.36
✓ Local Round 3/3 [Flower Round 3]: Train Loss: 1.82, Train Acc: 0.34, Test Acc: 0.38

================================================================================
🗳️  COMMUNICATION ROUND 1/10 - MAJORITY VOTING
================================================================================
Consensus Labels (L̄_t) - 100 samples
Agreement Statistics:
  Mean Agreement: 0.750
  Unanimous: 34/100
  Labels Changed: 20/100

Client Performance Summary:
  Average Train Acc: 0.59 ± 0.01
  Average Test Acc:  0.43 ± 0.02
================================================================================
```

**Key Metrics:**
- **Agreement**: Higher = models converging
- **Unanimous**: More = stronger consensus
- **Labels Changed**: Models refining predictions

---

## 📁 Project Structure

```
Flower_FedCT/
├── fedcot/
│   ├── client_app.py      # Flower client (training & predictions)
│   ├── server_app.py      # Flower server (voting & coordination)
│   └── task.py            # ML logic (model, data, training)
├── run_experiment.sh      # Main runner
├── run_direct.py          # Direct execution (Ray fallback)
├── pyproject.toml         # Flower configuration
└── requirements.txt       # Dependencies
```

---

## 🔬 Architecture

### Model
**ImprovedCNN (ResNet-18 style)**:
- 8 residual blocks with skip connections
- 11.2M parameters
- Target accuracy: ~70-80% on CIFAR-10

### Two Execution Modes

| Feature | Flower Simulation | Direct Execution |
|---------|-------------------|------------------|
| Algorithm | ✅ FedCT | ✅ FedCT |
| Results | Identical | Identical |
| Use Case | Production/Research | Debugging/Fallback |
| Setup | Flower + Ray | Pure Python |

**Both produce identical results!**

---

## 💡 Tips

**Speed up training:**
```bash
# Reduce rounds
./run_experiment.sh --local-rounds 2 --communication-rounds 10

# Use direct execution (faster startup)
export USE_DIRECT_EXECUTION=1
./run_experiment.sh

# Use less-busy GPU
CUDA_VISIBLE_DEVICES=4 ./run_experiment.sh
```

**Improve accuracy:**
```bash
# More training
./run_experiment.sh --communication-rounds 20 --local-rounds 5

# Try different optimizer
./run_experiment.sh --optimizer SGD --lr 0.01
```

**GPU memory issues:**
```bash
# Smaller batches
./run_experiment.sh --private-batch 32 --public-batch 16
```

---

## 📝 Citation

```bibtex
@inproceedings{abourayya2025little,
  title={Little is enough: Boosting privacy by sharing only hard labels in federated semi-supervised learning},
  author={Abourayya, Amr and Kleesiek, Jens and Rao, Kanishka and Ayday, Erman and Rao, Bharat and Webb, Geoffrey I and Kamp, Michael},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={15},
  pages={15293--15301},
  year={2025}
}
```

---

## 📄 License

MIT License

---

**Happy Federated Learning with FedCT!** 🌸🚀
