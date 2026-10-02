"""Run the full experiment suite sequentially on the single GPU.

  orig     : the dataset's own train/val split (per-image, no augmented-copy
             leakage; the 60 source pills are shared -> practical deployment case)
  group    : 5-fold cross-validation grouped by source pill (leave-pills-out,
             the strict generalisation bound for unseen disks)
  control  : grouped CV with a heavy blur that erases the printed drug code
             (isolates label-reading from agar-texture cues)
  group1   : grouped hold-out ablation with mixup/CutMix enabled
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# recipe validated on the grouped hold-out pilot (80.5% leave-pills-out, 25 epochs)
BASE = ["--img-size", "160", "--workers", "0", "--tta", "1", "--batch", "64",
        "--lr", "2e-3", "--wd", "1e-4", "--drop", "0.10", "--head-drop", "0.30",
        "--mixup", "0"]

RUNS = [
    ("orig", ["--epochs", "25", *BASE]),
    ("group", ["--epochs", "25", "--folds", "3", *BASE]),
    ("control", ["--epochs", "25", "--folds", "3", "--blur", "9", *BASE]),
    ("group1", ["--epochs", "30", "--wd", "5e-4", "--drop", "0.15",
                "--head-drop", "0.40", "--mixup", "0.4", "--mix-prob", "0.5",
                "--img-size", "160", "--workers", "0", "--tta", "1", "--batch", "64"]),
]


def is_done(name):
    """A protocol is complete once results.json exists (written at the very end)."""
    return (HERE / "runs" / name / "results.json").exists()


def main():
    logdir = HERE / "logs"
    logdir.mkdir(exist_ok=True)
    overall = time.time()
    for name, extra in RUNS:
        if is_done(name):
            print(f"########## {name}: already complete, skipping ##########", flush=True)
            continue
        log = logdir / f"{name}.log"
        cmd = [PY, str(HERE / "train.py"), "--protocol", name, *extra]
        print(f"\n########## {name} ##########\n{' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n===== invocation {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in p.stdout:
                f.write(line)
                f.flush()
                print(line, end="", flush=True)
            p.wait()
        print(f"########## {name} rc={p.returncode} in {(time.time()-t0)/60:.1f} min ##########",
              flush=True)
    missing = [n for n, _ in RUNS if not is_done(n)]
    print(f"\nALL DONE in {(time.time()-overall)/60:.1f} min; still missing: {missing}",
          flush=True)


if __name__ == "__main__":
    main()
