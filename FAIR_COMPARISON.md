# Fair Comparison Between FedCT and FedAvg

## The Problem

The original comparison between FedCT and FedAvg was **unfair** because:

### FedCT Data Usage:
- Takes 100 samples from the training set as "public unlabeled" dataset (U)
- Remaining 49,900 samples are split among 5 clients = **9,980 samples/client** (private labeled data)
- All clients also train on the **100 pseudo-labeled samples** (shared across all clients)
- **Total effective training data per client: ~10,080 samples**

### FedAvg Data Usage (Original - UNFAIR):
- All 50,000 samples split among 5 clients = **10,000 samples/client**
- **Does NOT use the 100 "public" samples**
- **Total training data per client: 10,000 samples**

### The Unfairness:
FedCT gets to leverage the 100 public samples with **pseudo-labels** (learned through majority voting and improved over time), while FedAvg doesn't use them at all. This gives FedCT an unfair advantage.

## The Solution: Fair Comparison Mode

We modified FedAvg to support a **fair comparison mode** where:

### FedAvg (Fair Mode - with `--public-size 100`):
- Uses the **same 100 public samples** (with **TRUE labels**, not pseudo-labels)
- Uses the **same 49,900 private samples** split among 5 clients = **9,980 samples/client**
- **Total training data per client: ~10,080 samples** (same as FedCT)

### Why This Is Fair:
1. **Same data split**: Both methods use the exact same train/test split (seed=42)
2. **Same total data**: Both methods train on the same 10,080 samples per client
3. **Fair advantage comparison**: 
   - FedCT uses 100 samples with **pseudo-labels** (learned, may be noisy)
   - FedAvg uses 100 samples with **TRUE labels** (ground truth)
   - This tests whether FedCT's co-training approach can overcome noisy pseudo-labels

## How to Use Fair Comparison

### Run FedAvg in Fair Mode:
```bash
# Fair comparison (uses same data split as FedCT)
bash run_fedavg.sh --public-size 100

# Or with custom parameters
bash run_fedavg.sh \
  --communication-rounds 10 \
  --clients 5 \
  --local-rounds 3 \
  --dataset CIFAR10 \
  --optimizer Adam \
  --lr 0.001 \
  --batch-size 64 \
  --arch LightCNN \
  --public-size 100
```

### Run Standard FedAvg (No Fair Mode):
```bash
# Standard mode (uses all data, no public split)
bash run_fedavg.sh --public-size 0
```

### Run FedCT:
```bash
# FedCT with 100 unlabeled samples
bash run_experiment.sh \
  --communication-rounds 10 \
  --clients 5 \
  --local-rounds 3 \
  --unlabeled 100
```

### Compare Results:
```bash
# Plot comparison per local round
python compare_fedct_fedavg.py --mode local --out comparison_local.png

# Plot comparison per communication round
python compare_fedct_fedavg.py --mode comm --out comparison_comm.png
```

## Expected Results

With fair comparison:
- If **FedCT performs better**, it demonstrates that co-training with pseudo-labels is effective
- If **FedAvg performs better**, it shows that true labels are more valuable than co-training benefits
- If **similar performance**, it suggests pseudo-labels are nearly as good as true labels in this setting

## Technical Details

### Data Split (seed=42, CIFAR-10):
- **Public indices**: 100 samples randomly selected with `np.random.choice(50000, size=100, replace=False)`
- **Private indices**: Remaining 49,900 samples
- **Client split**: Private data divided equally among clients (IID split)

### Key Difference:
- **FedCT**: Public samples get pseudo-labels through majority voting (Algorithm 1 in paper)
- **FedAvg**: Public samples keep their true labels

### Code Changes:
1. `run_fedavg.py`: Added `iid_partition_with_public()` function and `--public-size` argument
2. `run_fedavg.sh`: Added `--public-size` parameter (default=100 for fair comparison)
3. `compare_fedct_fedavg.py`: Fixed argument name bug (`fedcot_log` vs `fedct_log`)

## Quick Start

Run a fair comparison experiment:

```bash
# 1. Run FedCT
bash run_experiment.sh --communication-rounds 10 --local-rounds 3 --unlabeled 100

# 2. Run FedAvg (fair mode)
bash run_fedavg.sh --communication-rounds 10 --local-rounds 3 --public-size 100

# 3. Compare results
python compare_fedct_fedavg.py --mode local --title "Fair Comparison: FedCT vs FedAvg"
```

This ensures both methods are evaluated on **exactly the same data distribution** with the only difference being how the 100 public samples are labeled (pseudo vs true).

