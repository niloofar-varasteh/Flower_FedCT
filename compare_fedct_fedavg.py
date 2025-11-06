import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- utils ----------
def latest_log(root: str):
    cands = sorted(Path(root).rglob("experiment.log"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def parse_fedavg(path: Path):
    rounds, losses, accs = [], [], []
    pat = re.compile(r"Round\s+(\d+)\s*[-–]\s*Loss:\s*([0-9.]+)\s*[-–]\s*Accuracy:\s*([0-9.]+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line.strip())
        if m:
            rounds.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            accs.append(float(m.group(3)))
    print(f"Parsed {len(rounds)} FedAvg rounds from {path}")
    return rounds, losses, accs

def parse_fedct(path: Path):
    srounds, test_losses, test_accs = [], [], []
    cycle_rounds, cycle_accs = [], []
    pat = re.compile(
        r"Local Round\s+(\d+)/(\d+)\s+\[Flower Round\s+(\d+)\]:.*?Test Loss:\s+([0-9.]+),\s+Test Acc:\s+([0-9.]+)"
    )
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            a, b, R = int(m.group(1)), int(m.group(2)), int(m.group(3))
            tl, ta = float(m.group(4)), float(m.group(5))
            srounds.append(R)
            test_losses.append(tl)
            test_accs.append(ta)
            if a == b:
                cycle_rounds.append(R)
                cycle_accs.append(ta)
    print(f"Parsed {len(cycle_rounds)} FedCT communication rounds from {path}")
    return srounds, test_losses, test_accs, cycle_rounds, cycle_accs

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fedavg-log", type=str, help="Path to FedAvg experiment.log (optional)")
    ap.add_argument("--fedct-log", type=str, help="Path to FedCT experiment.log (optional)")
    ap.add_argument("--out", type=str, default="fedct_vs_fedavg.png")
    ap.add_argument("--title", type=str, default="FedCT vs FedAvg (aligned per communication round)")
    args = ap.parse_args()

    fav_path = Path(args.fedavg_log) if args.fedavg_log else latest_log("logs_fedavg")
    fct_path = Path(args.fedct_log)  if args.fedct_log  else latest_log("logs")

    if not fav_path or not fav_path.exists():
        raise SystemExit("FedAvg log not found. Pass --fedavg-log or put logs under logs_fedavg/**/experiment.log")
    if not fct_path or not fct_path.exists():
        raise SystemExit("FedCT log not found. Pass --fedct-log or put logs under logs/**/experiment.log")

    print(f"Using FedAvg log: {fav_path}")
    print(f"Using FedCT  log: {fct_path}")

    fav_r, fav_l, fav_a = parse_fedavg(fav_path)
    fct_r, fct_l, fct_a, fct_cR, fct_cA = parse_fedct(fct_path)

    if not fav_r:
        raise SystemExit("No FedAvg rounds parsed. Check FedAvg log format.")
    if not fct_r:
        raise SystemExit("No FedCT rounds parsed. Check FedCT log format.")

    # Map Flower-round -> (loss, acc) for quick lookup
    fct_loss_by_round = {r: l for r, l in zip(fct_r, fct_l)}

    # Use end-of-cycle points for fair comparison (one per communication round)
    if fct_cR:
        fct_r_use = list(range(1, len(fct_cR) + 1))
        fct_a_use = fct_cA
        fct_l_use = [fct_loss_by_round.get(cr, fct_l[-1]) for cr in fct_cR]
    else:
        # Fallback: use all rounds (less ideal)
        fct_r_use, fct_l_use, fct_a_use = fct_r, fct_l, fct_a

    fav_r_use = list(range(1, len(fav_r) + 1))

    # Plot
    plt.figure(figsize=(10, 4.5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(fct_r_use, fct_l_use, "o-", label="FedCT (comm rounds)")
    ax1.plot(fav_r_use, fav_l, "o-", label="FedAvg (comm rounds)")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(fct_r_use, fct_a_use, "o-", label="FedCT (comm rounds)")
    ax2.plot(fav_r_use, fav_a, "o-", label="FedAvg (comm rounds)")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(args.title, y=1.05)
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved plot -> {args.out}")

if __name__ == "__main__":
    main()
