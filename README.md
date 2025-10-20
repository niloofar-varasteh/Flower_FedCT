# Federated Deep Co-Training (FedDCT)

A **Flower-based implementation** of Federated Deep Co-Training (FedDCT), a semi-supervised federated learning approach where clients collaboratively train models by **sharing predictions only** (not model weights) on a public unlabeled dataset.

---

## 🚀 Quick Start

**Run with Flower framework (recommended):**
```bash
./run_experiment.sh --clients 5 --communication-rounds 10 --local-rounds 2
```

**Run with direct execution (if Ray/Pydantic issues):**
```bash
export USE_DIRECT_EXECUTION=1
./run_experiment.sh --clients 5 --communication-rounds 10 --local-rounds 2
```

**Both produce identical results!** See [Running Experiments](#-running-experiments) for details.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Running Experiments](#-running-experiments)
- [Parameters](#️-parameters)
- [Examples](#-examples)
- [Output](#-output)
- [Project Structure](#-project-structure)
- [Implementation Architecture](#-implementation-architecture)
- [Technical Details](#-how-the-method-works-technical-details)

---

## 🎯 Overview

**FedDCT (Federated Deep Co-Training)** is a semi-supervised federated learning method that:

- ✅ Trains on **private labeled data** at each client
- ✅ Leverages **public unlabeled data** shared across all clients
- ✅ Clients share **hard predictions** (labels), not model weights
- ✅ Server performs **majority voting** to create consensus labels
- ✅ Clients train on both private data and pseudo-labeled public data

### Key Features

- **Privacy-Preserving**: Clients only share predictions, not raw data or model weights
- **Semi-Supervised**: Utilizes both labeled (private) and unlabeled (public) data
- **Collaborative**: All clients benefit from collective knowledge via majority voting
- **Flexible**: Supports multiple datasets, optimizers, and hyperparameters

---

## 🔧 How It Works

### High-Level Overview

FedDCT enables multiple clients to collaboratively train models **without sharing private data or model weights**. Instead:

1. 🗂️ Each client has **private labeled data** (not shared)
2. 📊 All clients access the same **public unlabeled dataset** (shared)
3. 🤝 Clients share **predictions only** (hard labels, not probabilities)
4. 🗳️ Server creates **consensus labels** via majority voting
5. 🔄 Clients train on **private data + pseudo-labeled public data**

### Training Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ INITIALIZATION                                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Split dataset:                                              │
│     - Sample public unlabeled set (shared by all clients)      │
│     - Divide remaining data among clients (private sets)       │
│  2. Each client initializes its own independent model          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ COMMUNICATION CYCLE 1 (No Pseudo-Labels Yet)                   │
├─────────────────────────────────────────────────────────────────┤
│  Local Round 1/N:                                               │
│    → All clients train on PRIVATE DATA ONLY                     │
│    → Training completed (metrics not shared)                    │
│                                                                 │
│  Local Round 2/N:                                               │
│    → All clients train on PRIVATE DATA ONLY                     │
│    → Training completed (metrics not shared)                    │
│  ...                                                            │
│                                                                 │
│  Local Round N/N:                                               │
│    → All clients train on PRIVATE DATA ONLY                     │
│    → Training completed                                         │
│                                                                 │
│  🗳️  COMMUNICATION ROUND 1:                                      │
│    1. Each client predicts labels on public dataset            │
│    2. Clients send predictions to server                       │
│    3. Server performs majority voting → consensus labels       │
│    4. Server sends consensus labels to all clients             │
│    5. Clients report Train/Test metrics                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ COMMUNICATION CYCLE 2+ (With Pseudo-Labels)                    │
├─────────────────────────────────────────────────────────────────┤
│  Local Round 1/N:                                               │
│    → All clients train on PRIVATE + PSEUDO-LABELED PUBLIC      │
│    → Each step: sample from both datasets                       │
│    → Training completed (metrics not shared)                    │
│                                                                 │
│  Local Round 2/N:                                               │
│    → All clients train on PRIVATE + PSEUDO-LABELED PUBLIC      │
│    → Each step: sample from both datasets                       │
│    → Training completed (metrics not shared)                    │
│  ...                                                            │
│                                                                 │
│  Local Round N/N:                                               │
│    → All clients train on PRIVATE + PSEUDO-LABELED PUBLIC      │
│    → Training completed                                         │
│                                                                 │
│  🗳️  COMMUNICATION ROUND:                                        │
│    1. Clients predict on public dataset                        │
│    2. Server performs majority voting → UPDATE labels          │
│    3. Server sends updated consensus labels                    │
│    4. Clients report metrics                                   │
└─────────────────────────────────────────────────────────────────┘

Repeat: Local Training → Communication → Local Training → ...
```

### Mixed Batch Training (Key Innovation)

**After the first communication round**, clients train on **both** private and pseudo-labeled public data in each training step:

```python
# Each training iteration (after first communication):

# Step 1: Sample a batch from private labeled data
private_batch = sample_from_private_data(batch_size=32)      # 32 private samples (ground truth labels)

# Step 2: Sample a batch from public pseudo-labeled data
public_batch = sample_from_public_data(batch_size=32)        # 32 public samples (consensus labels)

# Step 3: Concatenate batches
combined_batch = torch.cat([private_batch, public_batch])    # 64 total samples

# Step 4: Train on combined batch
loss = criterion(model(combined_batch), labels)
loss.backward()
optimizer.step()
```

**Why This Works:**
- **Private data**: Provides reliable ground truth labels (but limited quantity)
- **Public data**: Provides additional training signal from consensus (larger quantity, noisy but helpful)
- **Mixed training**: Model learns from both sources simultaneously
- **Iterative refinement**: As models improve, consensus labels become more accurate

### Example Timeline

**Configuration**: `--communication-rounds 3 --local-rounds 2`

```
Flower Round 1: Cycle 1, Local 1/2 → Train on PRIVATE only
Flower Round 2: Cycle 1, Local 2/2 → Train on PRIVATE only → COMMUNICATE → Get consensus labels

Flower Round 3: Cycle 2, Local 1/2 → Train on PRIVATE + PUBLIC (pseudo-labeled)
Flower Round 4: Cycle 2, Local 2/2 → Train on PRIVATE + PUBLIC → COMMUNICATE → Update labels

Flower Round 5: Cycle 3, Local 1/2 → Train on PRIVATE + PUBLIC (updated labels)
Flower Round 6: Cycle 3, Local 2/2 → Train on PRIVATE + PUBLIC → COMMUNICATE → Final labels

Total: 6 Flower server rounds = 3 communication rounds × 2 local rounds
```

---

## 📦 Installation

### Requirements

- Python 3.9+
- PyTorch
- torchvision
- Flower (flwr)
- NumPy
- Ray (for Flower simulation)

### Recommended Setup

```bash
# Clone the repository
cd Flower_FedCT

# Create a clean environment (recommended)
conda create -n fedcot python=3.9
conda activate fedcot

# Install dependencies with compatible versions
pip install torch torchvision
pip install "numpy<2.0"
pip install "pydantic>=1.10.0,<2.0.0"
pip install "ray>=2.7.0,<2.10.0"
pip install "pyarrow<15.0.0"
pip install "flwr[simulation]>=1.8.0"
```

### Quick Setup (if above works)

```bash
# Or simply install from requirements.txt
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

You have **two options** to run FedDCT experiments:

### Option 1: Flower Simulation (Recommended)

**Uses the Flower framework with Ray for distributed simulation.**

```bash
./run_experiment.sh --clients 5 --communication-rounds 10 --local-rounds 2
```

**Advantages:**
- ✅ Official Flower framework integration
- ✅ Proper client-server architecture
- ✅ Scalable to distributed systems
- ✅ Compatible with all Flower tools

**When to use:** For production, research papers, or when you want the full Flower experience.

### Option 2: Direct Execution (Fallback)

**Bypasses Flower/Ray, runs everything in a single Python process.**

**Method A: Using environment variable**
```bash
export USE_DIRECT_EXECUTION=1
./run_experiment.sh --clients 5 --communication-rounds 10 --local-rounds 2
```

**Method B: Direct Python script**
```bash
python run_direct.py --clients 5 --communication-rounds 10 --local-rounds 2
```

**Advantages:**
- ✅ No Ray/Pydantic compatibility issues
- ✅ Simpler debugging
- ✅ Same algorithm, same results

**When to use:** 
- If you encounter Ray/Pydantic errors
- For quick debugging
- When running on Windows (Ray has limited Windows support)

---

## ⚠️ Troubleshooting Installation

### Issue 1: Ray/Pydantic Compatibility

**Symptoms:**
```
TypeError: issubclass() arg 1 must be a class
AttributeError: _ARRAY_API not found
```

**Solution:** Use direct execution
```bash
export USE_DIRECT_EXECUTION=1
./run_experiment.sh
```

Or install compatible versions:
```bash
pip uninstall ray pydantic numpy pyarrow
pip install "numpy<2.0" "pydantic>=1.10.0,<2.0.0" "ray>=2.7.0,<2.10.0" "pyarrow<15.0.0"
```

### Issue 2: NumPy Version Conflicts

**Symptoms:**
```
ImportError: numpy.core.multiarray failed to import
```

**Solution:**
```bash
pip install "numpy<2.0"
```

### Issue 3: Ray Won't Start

**Symptoms:**
```
RayActorError: The actor died unexpectedly
```

**Solution:** Use direct execution (see Option 2 above)

---

## 🎮 Quick Start Examples

### Run with Default Settings

**Using Flower (Recommended):**
```bash
./run_experiment.sh
```

**Using Direct Execution:**
```bash
export USE_DIRECT_EXECUTION=1
./run_experiment.sh
```

Default configuration:
- **Communication rounds**: 10
- **Clients**: 5
- **Local rounds**: 2
- **Unlabeled size**: 100
- **Dataset**: CIFAR10
- **Optimizer**: SGD
- **Learning rate**: 0.01
- **Batch sizes**: 32 (both private and public)

### Custom Configuration

**Flower simulation:**
```bash
./run_experiment.sh \
    --communication-rounds 20 \
    --clients 10 \
    --local-rounds 5 \
    --unlabeled 500 \
    --dataset CIFAR10 \
    --optimizer SGD \
    --lr 0.01 \
    --private-batch 32 \
    --public-batch 32
```

**Direct execution:**
```bash
python run_direct.py \
    --communication-rounds 20 \
    --clients 10 \
    --local-rounds 5 \
    --unlabeled 500 \
    --dataset CIFAR10 \
    --optimizer SGD \
    --lr 0.01 \
    --private-batch 32 \
    --public-batch 32
```

**Both methods produce identical results!**

---

## ⚙️ Parameters

### Core Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--communication-rounds` | int | Number of communication rounds | 10 |
| `--local-rounds` | int | Number of local training rounds (passes through dataset) before each communication | 2 |
| `-c, --clients` | int | Number of clients | 5 |
| `-u, --unlabeled` | int | Size of public unlabeled dataset | 100 |

### Data & Model Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `-d, --dataset` | str | Dataset: `CIFAR10` or `FashionMNIST` | CIFAR10 |
| `--private-batch` | int | Batch size for private data sampling | 32 |
| `--public-batch` | int | Batch size for public pseudo-labeled data sampling | 32 |

### Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `-o, --optimizer` | str | Optimizer: `SGD` or `Adam` | SGD |
| `--lr` | float | Learning rate | 0.01 |

### Parameter Explanations

#### `num_communication_rounds`
**What it controls:** How many times clients communicate with the server

**Example:** `--communication-rounds 20`
- Clients will share predictions 20 times
- Server performs majority voting 20 times
- Each communication happens after `num_local_rounds` of training

#### `num_local_rounds`
**What it controls:** How many times each client goes through its entire dataset before communicating

**Example:** `--local-rounds 5`
- Client processes all its data 5 times
- 1 local round = 1 complete pass through the client's dataset
- After 5 local rounds, client communicates with server

#### Total Training
```
Total passes through dataset = num_communication_rounds × num_local_rounds
Example: 20 × 5 = 100 total epochs
```

---

## 📚 Examples

### Example 1: Quick Test (Fast)
```bash
./run_experiment.sh \
    --communication-rounds 2 \
    --clients 2 \
    --local-rounds 1 \
    --unlabeled 50
```
- **Purpose**: Quick sanity check
- **Time**: ~2-3 minutes
- **Total training**: 2 × 1 = 2 epochs per client

### Example 2: Standard Experiment
```bash
./run_experiment.sh \
    --communication-rounds 20 \
    --clients 5 \
    --local-rounds 5 \
    --unlabeled 500
```
- **Purpose**: Standard experimental setup
- **Time**: ~30-40 minutes
- **Total training**: 20 × 5 = 100 epochs per client

### Example 3: FashionMNIST with Adam
```bash
./run_experiment.sh \
    --dataset FashionMNIST \
    --optimizer Adam \
    --lr 0.001 \
    --communication-rounds 15 \
    --local-rounds 10
```
- **Purpose**: Different dataset and optimizer
- **Total training**: 15 × 10 = 150 epochs per client

### Example 4: Large Batches (GPU Required)
```bash
./run_experiment.sh \
    --private-batch 128 \
    --public-batch 128 \
    --communication-rounds 10 \
    --local-rounds 5
```
- **Purpose**: Faster training with large batches
- **Requirement**: GPU with sufficient memory

### Example 5: Frequent Communication
```bash
./run_experiment.sh \
    --communication-rounds 50 \
    --local-rounds 2 \
    --unlabeled 1000
```
- **Purpose**: More frequent collaboration
- **Effect**: Clients share knowledge more often

---

## 📊 Output

### Expected Output Format

Both Flower simulation and direct execution produce similar output:

#### Initial Configuration

```
=================================
FedDCT Experiment Configuration
=================================
Dataset:                       CIFAR10
Num communication rounds:      3
Number of clients:             5
Num local rounds:              2 (passes through dataset)
Private batch size:            32
Public batch size:             32
Optimizer:                     SGD
Learning rate:                 0.01
Unlabeled dataset size:        20
=================================

Using Flower framework
(If you encounter Ray errors, run: export USE_DIRECT_EXECUTION=1)

Loading project configuration... 
Success
INFO :      Starting Flower ServerApp, config: num_rounds=6, no round_timeout
```

#### Communication Cycle Output

```
================================================================================
📊 COMMUNICATION CYCLE 1/3
================================================================================
→ Clients will do 2 local training rounds
→ Then share predictions on public dataset
================================================================================

✓ Local Round 1/2: Training completed (metrics shared only at communication rounds)
✓ Local Round 2/2: Training completed (metrics shared only at communication rounds)

================================================================================
🗳️  COMMUNICATION ROUND 1/3 - MAJORITY VOTING
================================================================================

--------------------------------------------------------------------------------
MAJORITY VOTING RESULTS:
--------------------------------------------------------------------------------
Consensus Labels (L̄_t) - 20 samples:
  [5, 8, 8, 4, 5, 5, 8, 8, 4, 5, 0, 3, 8, 7, 9, 5, 3, 6, 7, 5]

Agreement Statistics:
  Mean Agreement: 0.640
  Min Agreement:  0.400
  Max Agreement:  1.000
  Unanimous:      3/20
  Labels Changed: 0/20

Client Performance Summary:
  Client 1: Train Loss=1.9421, Train Acc=0.2721, Test Loss=1.8174, Test Acc=0.3391
  Client 2: Train Loss=1.8992, Train Acc=0.2923, Test Loss=1.6797, Test Acc=0.3587
  Client 3: Train Loss=1.9285, Train Acc=0.2838, Test Loss=1.6449, Test Acc=0.3807
  Client 4: Train Loss=1.9354, Train Acc=0.2789, Test Loss=1.8039, Test Acc=0.3266
  Client 5: Train Loss=1.9421, Train Acc=0.2815, Test Loss=1.7262, Test Acc=0.3130

  Average Train Loss: 1.9295 ± 0.0160
  Average Train Acc:  0.2817 ± 0.0066
  Average Test Loss:  1.7344 ± 0.0675
  Average Test Acc:   0.3436 ± 0.0239
================================================================================

================================================================================
📊 COMMUNICATION CYCLE 2/3
================================================================================
→ Clients will do 2 local training rounds
→ Then share predictions on public dataset
================================================================================

✓ Local Round 1/2: Training completed (metrics shared only at communication rounds)
✓ Local Round 2/2: Training completed (metrics shared only at communication rounds)

================================================================================
🗳️  COMMUNICATION ROUND 2/3 - MAJORITY VOTING
================================================================================

Consensus Labels (L̄_t) - 20 samples:
  [5, 8, 8, 4, 5, 6, 8, 8, 6, 5, 9, 3, 8, 7, 9, 5, 3, 6, 7, 5]

Agreement Statistics:
  Mean Agreement: 0.750
  Min Agreement:  0.400
  Max Agreement:  1.000
  Unanimous:      4/20
  Labels Changed: 3/20

Client Performance Summary:
  Client 1: Train Loss=1.1639, Train Acc=0.5918, Test Loss=1.5818, Test Acc=0.4276
  Client 2: Train Loss=1.1796, Train Acc=0.5883, Test Loss=1.5282, Test Acc=0.4370
  Client 3: Train Loss=1.1614, Train Acc=0.5912, Test Loss=1.6510, Test Acc=0.4191
  Client 4: Train Loss=1.1661, Train Acc=0.5876, Test Loss=1.4920, Test Acc=0.4544
  Client 5: Train Loss=1.1676, Train Acc=0.5880, Test Loss=1.6618, Test Acc=0.3853

  Average Train Loss: 1.1677 ± 0.0063
  Average Train Acc:  0.5894 ± 0.0018
  Average Test Loss:  1.5830 ± 0.0665
  Average Test Acc:   0.4247 ± 0.0229
================================================================================

...

=================================
Experiment completed successfully!
=================================
Results saved to: logs/CIFAR10_SGD_lr0.01_20251020_135031

Final Results Summary:
=================================
Average Test Accuracy (final): 0.4337
Mean Agreement (final):        0.780
=================================
```

### Understanding the Output

#### During Local Rounds

```
✓ Local Round 1/2: Training completed (metrics shared only at communication rounds)
```
- Clients train on their private data (+ public pseudo-labeled after 1st communication)
- No metrics are shared (privacy-preserving)
- Just confirms training finished

#### At Communication Rounds

```
🗳️  COMMUNICATION ROUND 1/3 - MAJORITY VOTING
```
- Clients share predictions on public dataset
- Server performs majority voting
- Consensus labels are created/updated
- **Full metrics are reported:**
  - **Train Loss/Acc**: Measured on each client's private labeled data
  - **Test Loss/Acc**: Measured on the full test set (same for all clients)
  - Individual client metrics + averages

#### Agreement Statistics

```
Agreement Statistics:
  Mean Agreement: 0.750    ← Average agreement across all samples
  Min Agreement:  0.400    ← Worst agreement (40% of clients agreed)
  Max Agreement:  1.000    ← Best agreement (100% of clients agreed)
  Unanimous:      4/20     ← Samples where all clients agree
  Labels Changed: 3/20     ← Labels that changed from previous round
```

**What it means:**
- **Higher agreement** → Models are converging to similar predictions
- **More unanimous** → Stronger consensus on pseudo-labels
- **Labels changing** → Models are refining their understanding

### Observing Co-Training Progress

Watch these metrics improve over communication rounds:

| Metric | Round 1 | Round 2 | Round 3 | **Trend** |
|--------|---------|---------|---------|-----------|
| **Train Acc** | 28.17% | 58.94% | 59.51% | ⬆️ Improving |
| **Test Acc** | 34.36% | 42.47% | 43.37% | ⬆️ Improving |
| **Agreement** | 0.640 | 0.750 | 0.780 | ⬆️ Increasing |
| **Unanimous** | 3/20 | 4/20 | 6/20 | ⬆️ More consensus |

**This shows the co-training is working!** Models are learning from both private and pseudo-labeled data.

---

## 📁 Project Structure

```
Flower_FedCT/
├── fedcot/
│   ├── __init__.py
│   ├── client_app.py       # Flower client implementation
│   ├── server_app.py       # Flower server implementation
│   └── task.py             # ML logic (model, data, training)
├── run_direct.py           # Direct execution script (bypasses Ray)
├── run_experiment.sh       # Main experiment runner
├── pyproject.toml          # Flower configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Key Files

- **`run_experiment.sh`**: Bash script to run experiments with custom parameters using Flower
- **`fedcot/client_app.py`**: Flower client implementation (handles local training and predictions)
- **`fedcot/server_app.py`**: Flower server implementation (handles majority voting and coordination)
- **`fedcot/task.py`**: Contains model definition, data loading, training, and evaluation functions
- **`pyproject.toml`**: Flower configuration and dependencies
- **`run_direct.py`**: Alternative direct execution script (bypasses Flower, for debugging only)

---

## 🌸 Implementation Architecture

### Two Execution Modes

This project implements FedDCT in **two equivalent ways**:

| Feature | Flower Simulation | Direct Execution |
|---------|-------------------|------------------|
| **Algorithm** | ✅ Same FedDCT | ✅ Same FedDCT |
| **Results** | ✅ Identical | ✅ Identical |
| **Framework** | Uses Flower + Ray | Pure Python |
| **Architecture** | Client-Server (distributed) | Single process |
| **Use Case** | Production, research, scaling | Debugging, fallback |
| **Setup** | Requires Ray/Pydantic | Simple installation |

**Both modes implement the exact same algorithm and produce identical results!**

---

### Flower Simulation Architecture

**Uses the Flower (flwr) framework** for federated learning:

```
┌─────────────────────────────────────────────────┐
│                  run_experiment.sh              │
│                        ↓                         │
│                   flwr run .                     │
└─────────────────────────────────────────────────┘
                         ↓
         ┌───────────────┴───────────────┐
         ↓                               ↓
┌─────────────────┐            ┌─────────────────┐
│  Server App     │←──────────→│   Client App    │
│  server_app.py  │   Flower   │  client_app.py  │
│                 │   Protocol │                 │
│  • Coordination │            │  • Train model  │
│  • Voting       │            │  • Predictions  │
│  • Consensus    │            │  • Metrics      │
└─────────────────┘            └─────────────────┘
         ↓                               ↓
   pyproject.toml              task.py (ML logic)
   Configuration               Model, Data, Train
```

**Components:**

1. **Server App** (`fedcot/server_app.py`):
   - Coordinates all clients via Flower's Strategy
   - Tracks local rounds and communication cycles
   - Performs majority voting on predictions
   - Distributes consensus labels to clients
   - Aggregates and reports metrics

2. **Client App** (`fedcot/client_app.py`):
   - Receives configuration from server
   - Loads private and public datasets
   - Trains model locally with mixed batch sampling
   - Makes predictions on public dataset
   - Reports metrics back to server

3. **Task Module** (`fedcot/task.py`):
   - Model definition (`SimpleCNN`)
   - Data loading and splitting
   - Training and evaluation functions
   - Prediction and pseudo-labeling logic

4. **Configuration** (`pyproject.toml`):
   - Flower app configuration
   - Default hyperparameters
   - Can be overridden via `--run-config`

5. **Runner Script** (`run_experiment.sh`):
   - Parses command-line arguments
   - Calls `flwr run` with proper config
   - Logs all output

**Flower Benefits:**
- ✅ Standardized federated learning API
- ✅ Easy to scale to real distributed systems
- ✅ Compatible with Flower ecosystem tools
- ✅ Proper client-server separation

---

### Direct Execution Architecture

**Pure Python implementation** without Flower/Ray:

```
┌─────────────────────────────────────────────────┐
│         run_direct.py OR run_experiment.sh      │
│         (with USE_DIRECT_EXECUTION=1)           │
└─────────────────────────────────────────────────┘
                         ↓
         ┌───────────────┴───────────────┐
         ↓                               ↓
┌─────────────────┐            ┌─────────────────┐
│  Simulated      │            │  Simulated      │
│  Server Logic   │            │  Client Logic   │
│                 │            │                 │
│  • For loop     │            │  • For loop     │
│  • Voting       │            │  • Train model  │
│  • Consensus    │            │  • Predictions  │
└─────────────────┘            └─────────────────┘
         ↓                               ↓
            task.py (same ML logic)
            Model, Data, Train
```

**Components:**

1. **Direct Runner** (`run_direct.py`):
   - Single Python script
   - Simulates server logic (for loops)
   - Simulates client logic (for loops)
   - Uses same `task.py` functions
   - Produces identical results

2. **Task Module** (`fedcot/task.py`):
   - **Shared with Flower mode!**
   - Same model, same training, same data
   - Ensures consistency

**Direct Execution Benefits:**
- ✅ No Ray/Pydantic compatibility issues
- ✅ Simpler debugging (single process)
- ✅ Works on all platforms (including Windows)
- ✅ Same results as Flower mode

---

## 🔬 How the Method Works (Technical Details)

### 1. Data Splitting

```python
# CIFAR10 has 50,000 training samples
total_samples = 50,000
unlabeled_size = 500
num_clients = 5

# Split:
public_unlabeled = randomly_sample(500)           # Shared by all
remaining = 50,000 - 500 = 49,500
private_per_client = 49,500 / 5 = 9,900          # Private to each client

# Test set (10,000 samples) is shared by all clients for evaluation
```

### 2. Training Process

**First Communication Cycle** (no consensus labels yet):
```
Local Round 1: All 5 clients train on private data only
Local Round 2: All 5 clients train on private data only
...
Local Round N: All 5 clients train on private data only
→ Communication Round 1: Share predictions, create consensus labels
```

**Subsequent Communication Cycles** (with consensus labels):
```
Local Round N+1: All clients train on private data + pseudo-labeled public data
Local Round N+2: All clients train on private data + pseudo-labeled public data
...
→ Communication Round 2: Share predictions, update consensus labels
```

### 3. Majority Voting

For each sample in the public unlabeled dataset:
```python
# Example: 5 clients predict labels for sample X
client_predictions = [3, 3, 3, 5, 8]

# Majority voting
label_counts = {3: 3, 5: 1, 8: 1}
consensus_label = 3  # Most frequent label

# Agreement ratio
agreement = 3/5 = 0.6  # 60% of clients agree
```

### 4. Mixed Batch Training

```python
for epoch in range(num_local_rounds):
    for private_batch in private_loader:  # Iterate through private data
        # Sample a batch from public pseudo-labeled data
        public_batch = next(public_loader)
        
        # Combine batches
        combined_batch = torch.cat([private_batch, public_batch])
        
        # Train on combined batch
        loss = compute_loss(model(combined_batch))
        loss.backward()
        optimizer.step()
```

---

## 🎓 Understanding the Parameters

### Scenario 1: More Collaboration
```bash
# Clients communicate frequently
--communication-rounds 50 --local-rounds 2
```
- **Effect**: Consensus labels updated frequently
- **Use case**: When you want clients to collaborate closely

### Scenario 2: More Local Training
```bash
# Clients train more before communicating
--communication-rounds 10 --local-rounds 10
```
- **Effect**: More local learning between communications
- **Use case**: When communication is expensive or slow

### Scenario 3: Balanced
```bash
# Balance between local training and collaboration
--communication-rounds 20 --local-rounds 5
```
- **Effect**: Good balance
- **Use case**: Standard experimental setup

---

## 💡 Tips

### For Best Results

1. **Start Small**: Test with 1-2 communication rounds and 2 clients first
2. **Batch Size**: Adjust based on your GPU memory
   - Small GPU: `--private-batch 16 --public-batch 16`
   - Large GPU: `--private-batch 128 --public-batch 128`
3. **Learning Rate**:
   - SGD: 0.01 (default)
   - Adam: 0.001 (recommended)
4. **Unlabeled Size**: 
   - Too small (<100): Limited benefit from co-training
   - Too large (>5000): Slower training
   - Sweet spot: 500-1000 samples

### Troubleshooting

**Out of Memory**:
```bash
# Reduce batch sizes
--private-batch 16 --public-batch 16
```

**Training too slow**:
```bash
# Increase batch sizes (if you have GPU memory)
--private-batch 64 --public-batch 64

# Or reduce data size for testing
--unlabeled 100
```

**Low accuracy**:
```bash
# More training
--communication-rounds 30 --local-rounds 10

# Try Adam optimizer
--optimizer Adam --lr 0.001
```

---

## 🎯 Summary: Flower vs Direct Execution

### Which One Should You Use?

| Scenario | Recommendation | Command |
|----------|---------------|---------|
| **Production/Research** | ✅ **Flower Simulation** | `./run_experiment.sh` |
| **Publishing Papers** | ✅ **Flower Simulation** | `./run_experiment.sh` |
| **Ray/Pydantic Errors** | ✅ **Direct Execution** | `export USE_DIRECT_EXECUTION=1` |
| **Quick Debugging** | ✅ **Direct Execution** | `python run_direct.py` |
| **Windows Users** | ✅ **Direct Execution** | `python run_direct.py` |
| **Scaling to Real Systems** | ✅ **Flower Simulation** | Deploy Flower apps |

### Key Points

1. **Both implement the exact same FedDCT algorithm**
2. **Both produce identical results**
3. **Both share the same core ML logic** (`fedcot/task.py`)
4. **Flower is recommended** for official deployments
5. **Direct execution is a reliable fallback** for compatibility issues

### Running Both Modes

**Test Flower:**
```bash
./run_experiment.sh --clients 2 --communication-rounds 2 --local-rounds 1
```

**Test Direct:**
```bash
export USE_DIRECT_EXECUTION=1
./run_experiment.sh --clients 2 --communication-rounds 2 --local-rounds 1
```

Compare the results - they should be identical!

---

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{fedcot2025,
  title={Federated Deep Co-Training with Flower},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Flower_FedCT}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Federated Learning with FedDCT!** 🌸🚀
