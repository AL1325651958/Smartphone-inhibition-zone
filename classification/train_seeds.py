"""Train several seeds and ensemble them on the shared test split."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
PY = sys.executable
DATA = CNN / "dataset_augmented224"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="small")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    (CNN / "logs").mkdir(exist_ok=True)
    for s in args.seeds:
        name = f"ens_{args.variant}_s{s}"
        if (CNN / "runs" / name / "results.json").exists():
            print(f"{name}: already done, skipping", flush=True)
            continue
        cmd = [PY, str(CNN / "train_disknet.py"), "--run-name", name,
               "--variant", args.variant, "--epochs", str(args.epochs),
               "--batch", "32", "--img-size", "224", "--data", str(DATA),
               "--ema", "0.95", "--seed", str(s)]
        print(f"\n########## {name} ##########", flush=True)
        t0 = time.time()
        with open(CNN / "logs" / f"{name}.log", "w", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1)
            for line in p.stdout:
                f.write(line)
                if "TEST acc" in line or "selected-epoch" in line:
                    print(line.rstrip(), flush=True)
            p.wait()
        print(f"########## {name} rc={p.returncode} "
              f"{(time.time()-t0)/60:.1f} min ##########", flush=True)
    print("SEEDS DONE", flush=True)


if __name__ == "__main__":
    main()
