# Antibiotic-disk CNN classifier (药片分类 / antibiotic-abbreviation recognition)

Seven-class convolutional neural network that recognises the antibiotic
abbreviation printed on a disk-diffusion disk, from the greyscale crops in
`../Antibacterial zone mask_Class/dataset`.

Classes: `CRO` (ceftriaxone), `DA` (clindamycin), `E` (erythromycin),
`LEV` (levofloxacin), `LZD` (linezolid), `P` (penicillin), `VA` (vancomycin).

## Quick start

```powershell
$env:PYTHONPATH="..\_tools\pylibs"     # torchvision (installed locally)

python build_manifest.py                # index the dataset + leakage audit
python train.py --protocol orig         # shipped split (the 100% number)
python run_all.py                       # orig + grouped CV + blurred control
python predict.py --folds 3             # out-of-fold ensemble predictions
python make_report.py                   # figures + tables
python make_figures.py                  # manuscript Figure 6
python infer.py --image some_crop.jpg   # classify a disk crop
python infer.py --plate plate.jpg --out annotated.png
```

`run_all.py` runs the whole suite in order and skips protocols that already
finished, so it can simply be re-launched after an interruption. Training also
resumes from a per-epoch state file, and cross-validation reuses completed folds.

## Data

| | value |
|---|---|
| images | 9,060 (7,246 train / 1,814 val as shipped) |
| classes | 7, balanced (965–1,089 images per class) |
| distinct source pills | 60 |
| images per source pill | ~151 |
| crop size | 50–297 px square, median 146 px |
| colour | greyscale stored as RGB (`L` mode for 60 files) |
| training resolution | 160 x 160, replicated to 3 channels |

The images are already augmented (the `aug_0000_*` prefixes) and already
enhanced, so the pipeline is: square-pad, resize, then augment online.

### Important: the shipped split leaks

All 60 source pills appear in **both** the shipped `train` and `val` folders
(the split is over augmented copies of the *same* pills, not over pills).
Verified by MD5 (no identical file crosses the split) and by augmentation
index. This is why the shipped split reports 100 % accuracy.

`build_manifest.py` therefore adds a `split_group` column that assigns each
whole pill to one side, and `--protocol group` performs a k-fold
cross-validation in which no source pill is shared between training and
validation. Quote the grouped numbers for claims about *unseen* disks.

## Evaluation protocols

| protocol | meaning |
|---|---|
| `orig` | shipped train/val folders — per-image, pills shared (deployment-style estimate) |
| `group` | 3-fold CV, pills confined to one fold — strict generalisation bound |
| `control` | grouped folds with a 9 px blur that erases the printed code — separates label reading from agar-texture cues |
| `group1` | single grouped hold-out; used only for recipe comparisons, excluded from the paper summary |

`make_report.py` and `make_figures.py` only consume `orig`, `group` and
`control` (see `PROTOCOLS` in `make_report.py`); other run directories under
`runs/` are ablations and stay out of the summary.

## Results (final configuration)

| protocol | n | accuracy | macro-F1 | balanced acc. | macro spec. |
|---|---|---|---|---|---|
| shipped split | 1,814 | 1.000 | 1.000 | 1.000 | 1.000 |
| grouped 3-fold CV | 9,060 | 0.808 | 0.815 | 0.810 | 0.968 |
| grouped CV, blurred label | 9,060 | 0.172 | 0.162 | 0.173 | 0.862 |

Per-fold macro-F1 under grouped CV: 0.745, 0.684, 0.997 (mean 0.809 +/- 0.136).
The near-perfect third fold and the large spread are both consequences of having
only 60 independent source disks (about 8-9 per antibiotic).

Confidence-based triage from the out-of-fold ensemble: 83.3% accuracy at
confidence >= 0.70 (93.2% of crops retained), 90.5% at >= 0.95 (65.1%).

## Model

Compact ResNet (`PillCNN`) trained from scratch, 2.85 M parameters:
5x5 stem convolution with stride 2, 3x3 convolution, max-pool, then four
residual stages (32/64/128/256 channels, two basic blocks each) with GroupNorm,
SiLU and squeeze-excitation; global average pooling and a dropout head.

Training: AdamW (lr 2e-3, cosine schedule with warmup, weight decay 5e-4),
label smoothing 0.05, class-balanced sampling, EMA weights, dropout 0.15 in the
stages and 0.40 before the classifier, and random-resized-crop + affine + flip +
colour-jitter + Gaussian-blur augmentation with random-erasing cutout.

Recipe provenance (`compare_recipes.py`, `logs/`): on the same 15-disk hold-out,
weight decay 5e-4 + dropout 0.15/0.40 gave 0.896 accuracy versus 0.854 for
weight decay 1e-4 + dropout 0.10/0.30; across all three folds it improved every
fold (pooled accuracy 0.7945 -> 0.8081). Mixup/CutMix did **not** help
(0.880 versus 0.896 on the same hold-out) and is therefore disabled.

## Notes on the environment

* `torchvision` is installed into `..\_tools\pylibs` — set `PYTHONPATH` to it.
* Keep `--workers 0`. With worker processes every PIL image is pickled through
  a pipe each epoch (574 ms/batch versus 152 ms/batch in-process).
* Training writes a per-epoch `*.state.pt` and resumes automatically, so a
  killed run continues instead of restarting.
* Image decoding is cached in RAM once per run (`--cache-px 192`, ~334 MB).

## Outputs

| path | contents |
|---|---|
| `runs/<protocol>/results.json` | metrics, per-class metrics, confusion matrices |
| `runs/<protocol>/history.json` | per-epoch training curves |
| `runs/<protocol>/fold*.pt` | fold checkpoints (used as an out-of-fold ensemble) |
| `reports/` | summary tables, confusion matrices, reliability, error gallery |
| `figures/` | manuscript Figure 6 and the protocol-comparison supplement |
