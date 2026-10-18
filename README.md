# LagSleep

Implementation of **Decision-aware marginalization of clock-offset uncertainty for sleep staging with incomplete EEG and EOG**. The model combines offset-conditioned stage probabilities and uses their weighted Jensen–Shannon disagreement to choose multimodal fusion or an explicitly trained EEG-only path.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Obtain Sleep Cassette recordings from [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/) or ISRUC-Sleep subgroup I. Export each aligned night as a NumPy archive with:

| Key | Shape | Meaning |
| --- | --- | --- |
| `eeg`, `eog` | `(T,)` each | Synchronized, equally sampled signals with the same origin |
| `fs` | scalar | Sampling frequency, default pipeline setting 100 Hz |
| `labels` | `(epochs,)` | One label for each 30-second EEG epoch, starting at recording time zero |

Map wake, N1, N2, N3 and REM to `0,1,2,3,4`; merge original stages 3 and 4 into N3. Encode movement and unscored epochs as `-1`. Fix EEG/EOG derivations and annotator selection before export. Resample both channels together; do not shift labels during EOG perturbations.

Create `data/manifest.json` with paths relative to the manifest:

```json
[
  {"subject": "SC400", "split": "train", "file": "raw/SC4001.npz"},
  {"subject": "SC400", "split": "train", "file": "raw/SC4002.npz"},
  {"subject": "SC401", "split": "dev", "file": "raw/SC4011.npz"},
  {"subject": "SC402", "split": "test", "file": "raw/SC4021.npz"}
]
```

Use your complete subject-disjoint partition. Both nights of one subject must stay together. ISRUC can be evaluated with a frozen source checkpoint by exporting a separate target test partition with the same channel and class conventions.



Preparation drops unscored epochs and epochs without the required real EOG context. It retains ten seconds on either side of each EOG epoch so injected shifts and candidate corrections can be jointly cropped without padding. Constant or nonfinite EOG marks the retained epoch as technically ineligible and invokes EEG fallback; nonfinite values are sanitized only for safe tensor construction. The fixed candidate set is `[-5,-3,-2,-1,-0.5,0,0.5,1,2,3,5]` seconds. Positive injected shifts read later EOG samples; the correct compensating candidate has the opposite sign. EEG labels stay fixed.

```bash
lagsleep prepare --manifest data/manifest.json --fs 100 --output data/prepared
```

## Training and inference

```bash
lagsleep train --data data/prepared --output runs/train --seed 42
```

## Evaluation

```bash
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

## Comparisons and ablations

```bash
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --shifts 0 --output runs/aligned
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --shifts 3 --output runs/offset3
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --shifts 0 --missing 0.4 --output runs/missing40
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --method eeg --output runs/eeg
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --method hard --output runs/hard
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --method entropy --output runs/entropy
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --method feature --output runs/feature
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --method mixture --output runs/mixture
lagsleep train --data data/prepared --seed 42 --no-shift-loss --output runs/no_shift_loss
lagsleep train --data data/prepared --seed 42 --no-missing-loss --output runs/no_missing_loss
lagsleep train --data data/prepared --seed 42 --no-shift-augmentation --output runs/modality_dropout
```

Evaluate each fitted training control with its own checkpoint.

## Outputs

Training writes `model.pt` with weights, model dimensions, both frozen gate thresholds, sampling rate and fitting-subject identities. Evaluation writes:

- `metrics.json`: all-epoch and transition-neighborhood macro-F1 for each offset/missingness pair, plus fallback frequency.
- `predictions.npz`: stage probabilities, disagreement, offset entropy, subjects and references.
- `confusion.csv`: the five-class confusion matrix for the first requested condition.
- `offset_curve.png`: transition macro-F1 against imposed clock offset.

Macro-F1 averages all five declared classes with zero contribution for an undefined class F1. An empty transition subset produces `null`. Fallback returns an EEG prediction for the epoch, so its frequency is distinct from prediction coverage. Missing EOG directly selects the EEG-only output.
