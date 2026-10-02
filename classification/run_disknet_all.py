"""Run all DiskNet variants on the same split/epochs for a fair comparison."""
import json
import subprocess
import sys
import time
from pathlib import Path

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
PY = sys.executable
VARIANTS = ["tiny", "small", "base"]
EPOCHS = 80
IMG = 224
DATA = CNN / "dataset_augmented224"


def main():
    (CNN / "logs").mkdir(exist_ok=True)
    for v in VARIANTS:
        name = f"disknet224_{v}"
        log = CNN / "logs" / f"{name}.log"
        cmd = [PY, str(CNN / "train_disknet.py"), "--run-name", name,
               "--variant", v, "--epochs", str(EPOCHS), "--batch", "32",
               "--img-size", str(IMG), "--data", str(DATA),
               "--ema", "0.95"]
        print(f"\n########## {name} ##########", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1)
            for line in p.stdout:
                f.write(line)
                if line.startswith("  ep") and int(line.split()[1].split("/")[0]) % 20 == 0:
                    print(line.rstrip(), flush=True)
                if "TEST:" in line or "params:" in line:
                    print(line.rstrip(), flush=True)
            p.wait()
        print(f"########## {name} rc={p.returncode} "
              f"{(time.time()-t0)/60:.1f} min ##########", flush=True)

    print("\n=== summary (224px, 80 epochs) ===")
    for v in VARIANTS:
        rp = CNN / "runs" / f"disknet224_{v}" / "results.json"
        if rp.exists():
            r = json.loads(rp.read_text(encoding="utf-8"))
            t, va = r["test"], r["val"]
            print(f"  DiskNet-{v:5} params {r['params']/1e6:.2f}M  "
                  f"TEST acc {t['accuracy']:.4f} mF1 {t['macro_f1']:.4f}  "
                  f"| val acc {va['accuracy']:.4f}")
    print("\nreference PillCNN @112 (same split): TEST acc 0.750  mF1 0.757")
    print("DONE")


if __name__ == "__main__":
    main()
