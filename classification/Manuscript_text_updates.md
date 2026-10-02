# Antibiotic-abbreviation classifier — replacement text for the revision

All numbers come from `药片CNN/runs/*/results.json`, `reports/` and `figures/`.
Regenerate with `run_all.py`, then `predict.py`, `make_report.py`,
`make_figures.py`. Nothing below is a placeholder.

---

## 1. Why the previous text has to change

The current manuscript (返修材料/Manuscript.tex, lines 663–678) states:

> The antibiotic-abbreviation classifier achieved 100% accuracy on the test set
> and distinguished all seven classes … The normalised confusion matrix
> contained a value of 1.00 for every class on the main diagonal and no
> off-diagonal misclassifications … each evaluated disk crop was assigned to its
> corresponding antibiotic abbreviation in the test set.

That number reproduces exactly (shipped split: accuracy 1.000, macro-F1 1.000 on
all 1,814 validation crops), so the previous analysis was not misreported — but
the split it used cannot support a generalisation claim:

* The 9,060 images come from only **60 distinct source disks**.
* **All 60 source disks appear in both the `train` and the `val` folders.** The
  shipped split separates augmented copies of the *same* disks, not disks.
  Verified twice: no MD5-identical file crosses the split, and no augmentation
  index crosses it (`build_manifest.py`, `_tools/check_base_keys.py`).
* Every validation crop is therefore a re-augmented view of a disk the model was
  trained on. 100% accuracy measures memorisation of 60 specific disks, not
  recognition of the abbreviation.

A reviewer who opens the dataset folders can find this, which makes the 100%
claim the most exposed statement in the paper.

---

## 2. Methods (replacement for the classifier paragraph, lines 357–366)

> After image enhancement, a separate convolutional neural network classified
> the seven antibiotic abbreviations printed on the disks: CRO, DA, E, LEV, LZD,
> P, and VA. The classifier was trained from scratch on greyscale disk crops at
> 160 x 160 pixels and comprised a 5 x 5 stem convolution (stride 2) followed by
> a 3 x 3 convolution, max-pooling, and four residual stages of two basic blocks
> each (32, 64, 128 and 256 channels) with group normalisation, SiLU activations
> and squeeze-and-excitation, ending in global average pooling and a dropout
> head (2.85 million parameters). Training used AdamW (learning rate
> 2 x 10^-3, cosine decay with warm-up), label smoothing (0.05), class-balanced
> sampling, exponential moving averages of the weights, dropout (0.15 in the
> residual stages, 0.40 before the classifier), weight decay 5 x 10^-4, and
> online augmentation (random-resized crop, rotation up to 30 degrees,
> horizontal and vertical flips, brightness and contrast jitter, Gaussian blur
> and random erasing). Each recognised abbreviation was then linked to the
> corresponding disk and inhibition-zone measurement (Figure 1B). The classifier
> architecture is detailed in Additional file 1: Table S5.
>
> Because the 9,060 crops derive from only 60 source disks, and because the same
> disks contribute images to both folders of the shipped split, performance was
> assessed with a grouped three-fold cross-validation in which every source disk
> was confined to a single fold. No source disk was shared between the training
> and the validation folds, so all reported metrics are leave-disks-out
> estimates. The shipped split is additionally reported for comparability with
> the previous version of this analysis. To determine whether the network reads
> the printed abbreviation rather than incidental agar texture, the grouped
> cross-validation was repeated with a 9-pixel Gaussian blur applied to every
> crop, which removes character strokes while preserving disk and zone
> appearance.

---

## 3. Results (replacement for section 3.x, lines 662–678)

> **Antibiotic-abbreviation recognition performance.** On the shipped split,
> which shares source disks between training and validation, the classifier
> assigned every one of the 1,814 validation crops correctly (accuracy 100%,
> macro-F1 1.000), reproducing the previous analysis. This estimate is
> optimistic because it evaluates re-augmented views of disks the model has
> already seen.
>
> Under grouped three-fold cross-validation, in which no source disk is shared
> between training and validation, the classifier reached an overall accuracy
> of **80.8%** and a macro-F1 of **0.815** across 9,060 crops, with a balanced
> accuracy of 0.810 and a macro specificity of 0.968. Per class, F1 ranged from
> 0.67 (LZD) and 0.68 (VA) to 0.92 (E); ceftriaxone, clindamycin, levofloxacin
> and penicillin reached F1 of 0.82, 0.87, 0.85 and 0.89 (Figure 6B, 6D).
> Accuracy differed substantially between folds (0.75, 0.71 and 1.00; macro-F1
> 0.809 +/- 0.136, mean +/- SD), reflecting the small number of independent
> source disks (60 in total, about 8-9 per antibiotic).>
> Residual errors were not random but involved visually similar abbreviations:
> VA was misread as LZD in 25.8% of VA crops and LZD as VA in 21.8% of LZD
> crops, while LEV and CRO were mutually confused (12.5% and 11.1%). This is
> consistent with recognition of short printed character strings at limited
> resolution rather than with confusion between antimicrobial agents as such.
>
> When the printed abbreviation was removed by heavy Gaussian blur, the same
> grouped cross-validation collapsed to chance in every fold (accuracy 0.13,
> 0.22 and 0.16; macro-F1 0.088, 0.206 and 0.098; pooled accuracy 17.2%,
> macro-F1 0.162; Figure 6A). The classifier therefore depends on the printed
> label and does not infer the agent from agar or zone texture.
>
> The out-of-fold ensemble was over-confident (expected calibration error 0.112;
> Figure 6E). Accuracy increased monotonically as calls became more confident:
> 83.3% for crops with confidence >= 0.70 (93.2% of crops), 87.0% for >= 0.90
> (82.1%) and 90.5% for >= 0.95 (65.1%). The system can therefore be deployed
> with a confidence threshold so that only high-confidence calls are accepted
> automatically and the remainder are referred for human review; at a threshold
> of 0.95 the automated reading is correct in 90.5% of accepted disks.

---

## 4. Caveat to state explicitly

The honest limitation is the number of independent disks: **60 source disks,
about 8-9 per antibiotic**, of which about 20 per class are ever held out.
Fold-to-fold variation (0.68-1.00 macro-F1) follows directly from this. The
grouped estimate is a conservative bound whose uncertainty is driven by
disk-level sampling, not by image count. Recommended wording:

> The grouped estimate is limited by the number of independent source disks
> (60, approximately 8-9 per antibiotic) rather than by the number of crops, and
> the between-fold spread (macro-F1 0.68-1.00) reflects this. Larger collections
> of independently prepared disks are needed to tighten the estimate.

**Do not quote a binomial confidence interval over the 9,060 crops** (that gives
a spuriously narrow 80.0-81.6%). The crops are pseudoreplicates of 60 disks;
report the between-fold spread instead.

---

## 5. The 100% number is reproducible, and that is the point

If a reviewer reruns the shipped split they will again get 100%. The revision
should therefore not claim the previous number was wrong, but that the split
that produced it cannot estimate generalisation — and then supply the grouped
estimate. Figure 6C is deliberately retained (the shipped-split confusion
matrix) so the leakage effect is visible next to the honest matrix.

---

## 6. Optional supporting analysis: hyperparameter ablation

A single grouped hold-out (15 disks) was used to compare two recipes on
identical data:

| recipe | accuracy | macro-F1 |
|---|---|---|
| weight decay 1e-4, dropout 0.10 | 0.854 | 0.849 |
| weight decay 5e-4, dropout 0.15, dropout head 0.40 | 0.896 | 0.906 |

The stronger-regularisation recipe was then run across all three folds, where
it improved **every** fold (+1.5, +2.8 and +0.4 macro-F1 points; pooled 0.7945
-> 0.8081 accuracy). It is the configuration reported above. Mixup/CutMix was
tested and did **not** help (0.880 accuracy versus 0.896 without it on the same
hold-out), so it was not used. These details are available from
`compare_recipes.py` and `logs/` if the reviewers ask; they do not need to be in
the manuscript.

---

## 7. Additional file 1 updates needed

* **Table S5** — replace the exported classifier architecture with `PillCNN`:
  5x5 stem conv (3->32, stride 2), 3x3 conv (32->32), max-pool 3x3 stride 2,
  then four stages of two basic residual blocks at 32/64/128/256 channels
  (stage 1 stride 1, stages 2-4 stride 2 in the first block), each block
  GroupNorm + SiLU + two 3x3 convs + squeeze-and-excitation, then global average
  pooling, dropout 0.30, fully connected 256->7. 2.85 M parameters, input
  160 x 160 x 3.
* **Figure 6** — replace with `药片CNN/figures/Figure6_Classifier.png`
  (168.6 x 130.3 mm at 500 dpi, within the 180 mm double-column limit; an
  LZW-compressed TIFF is written alongside). Panels: (a) training dynamics for
  all three protocols, (b) normalised confusion matrix under grouped CV,
  (c) under the shipped split, (d) per-class F1 with Wilson intervals,
  (e) reliability diagram with ECE, (f) confidence distribution.
* **New supplementary figure** — `figures/FigureS_ClassifierProtocols.png`
  compares the protocols and shows the per-antibiotic generalisation gap.
* The previous lower panels (top-8 activated first-layer channels) came from the
  old model. They can be regenerated for `PillCNN` if the visualisation is kept;
  they are not part of the replacement figure.

---

## 8. Reproducing

```powershell
$env:PYTHONPATH="..\_tools\pylibs"
python build_manifest.py     # dataset index + leakage audit
python run_all.py            # orig, group CV, blurred-label control
python predict.py --folds 3  # out-of-fold ensemble, calibration, thresholds
python make_report.py        # tables + diagnostics
python make_figures.py       # Figure 6 and the supplementary figure
```

`run_all.py` skips completed protocols; `train.py` resumes from a per-epoch
state file and the cross-validation reuses completed folds, so any interrupted
run can simply be relaunched.
