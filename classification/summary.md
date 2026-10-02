# Antibiotic disk classification - CNN results

Seven-class classification of single antibiotic disks cropped from disk-diffusion plates (CRO/DA/E/LEV/LZD/P/VA). Model: compact ResNet-style CNN trained from scratch on 160x160 grayscale crops (2.85 M parameters).

## Metrics by evaluation protocol

| Protocol | Evaluation | n | Accuracy | 95% CI | Macro-F1 | Balanced acc. | Macro spec. |
|---|---|---|---|---|---|---|---|
| orig | Dataset split (per-image, pills shared) | 1814 | 1.0000 | [0.998, 1.000] | 1.0000 | 1.0000 | 1.0000 |
| group | Grouped 3-fold CV (leave-pills-out) | 9060 | 0.8081 | [0.800, 0.816] | 0.8146 | 0.8098 | 0.9678 |
| control | Grouped CV, label blurred (texture control) | 9060 | 0.1715 | [0.164, 0.179] | 0.1616 | 0.1733 | 0.8624 |

## Protocol definitions

- **orig** - the dataset's own `train`/`val` folders. No augmented copy of a validation image appears in training (verified by MD5 and by augmentation index), but all 60 source pills are shared, so this estimates accuracy on new images of familiar disks and reproduces the 100% previously reported.
- **group** - 3-fold cross-validation in which every source pill is confined to a single fold (leave-pills-out). This is the strict generalisation bound and is the number to quote for *unseen* disks.
- **control** - the grouped folds re-run with a 9 px Gaussian blur that erases the printed drug code, isolating how much of the accuracy comes from reading the label.

![orig confusion](confusion_orig_norm.png)

![group confusion](confusion_group_norm.png)

![control confusion](confusion_control_norm.png)
