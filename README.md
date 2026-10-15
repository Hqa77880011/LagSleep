# LagSleep

Implementation of **Decision-aware marginalization of clock-offset uncertainty for sleep staging with incomplete EEG and EOG**. The model combines offset-conditioned stage probabilities and uses their weighted Jensen–Shannon disagreement to choose multimodal fusion or an explicitly trained EEG-only path.

## Install

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
```

Commands default to CPU. Pass `--device cuda` with a compatible PyTorch GPU installation.

## Prepare nights

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

```bash
lagsleep prepare --manifest data/manifest.json --fs 100 --output data/prepared
lagsleep train --data data/prepared --output runs/train
lagsleep evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

Preparation drops unscored epochs and epochs without the required real EOG context. It retains ten seconds on either side of each EOG epoch so injected shifts and candidate corrections can be jointly cropped without padding. Constant or nonfinite EOG marks the retained epoch as technically ineligible and invokes EEG fallback; nonfinite values are sanitized only for safe tensor construction. The fixed candidate set is `[-5,-3,-2,-1,-0.5,0,0.5,1,2,3,5]` seconds. Positive injected shifts read later EOG samples; the correct compensating candidate has the opposite sign. EEG labels stay fixed.

## Conditions, controls and ablations

```bash
lagsleep evaluate --shifts 0 --output runs/aligned
lagsleep evaluate --shifts 3 --output runs/offset3
lagsleep evaluate --shifts 0 --missing 0.4 --output runs/missing40
lagsleep evaluate --method eeg --output runs/eeg
lagsleep evaluate --method hard --output runs/hard
lagsleep evaluate --method entropy --output runs/entropy
lagsleep evaluate --method feature --output runs/feature
lagsleep evaluate --method mixture --output runs/mixture
lagsleep train --no-shift-loss --output runs/no_shift_loss
lagsleep train --no-missing-loss --output runs/no_missing_loss
lagsleep train --no-shift-augmentation --output runs/modality_dropout
```

Evaluate each fitted ablation with its checkpoint. `hard` selects the maximum-weight offset, `entropy` applies an offset-entropy gate, `feature` averages offset features before classification, and `mixture` retains probability averaging without disagreement fallback. All use the same trained encoders and heads. These are mechanism controls, not replacements labeled as DeepSleepNet, XSleepNet or MMTSleepNet.

`--epochs`, `--lr`, `--batch-size`, `--width` and `--seed` control fitting. `--temperature` controls offset-weight softmax. `--dev-shift` fixes the development condition for choosing both gate thresholds, default three seconds; test labels never select thresholds. `--transition-radius` defines evaluation neighborhoods in epochs, default one. `--missing` controls label-independent EOG absence; its default is zero. Aligned, shifted and missing conditions can therefore be reported separately. Offsets outside ±5 seconds are rejected.

## Outputs

Training writes `model.pt` with weights, model dimensions, both frozen gate thresholds, sampling rate and fitting-subject identities. Evaluation writes:

- `metrics.json`: all-epoch and transition-neighborhood macro-F1 for each offset/missingness pair, plus fallback frequency.
- `predictions.npz`: stage probabilities, disagreement, offset entropy, subjects and references.
- `confusion.csv`: the five-class confusion matrix for the first requested condition.
- `offset_curve.png`: transition macro-F1 against imposed clock offset.

Macro-F1 averages all five declared classes with zero contribution for an undefined class F1. An empty transition subset produces `null`. Fallback returns an EEG prediction for the epoch, so its frequency is distinct from prediction coverage. Missing EOG directly selects the EEG-only output.

## Implementation choices

Each modality uses a compact temporal convolutional encoder; a shared fusion head evaluates every offset. Mixture loss supervises complete stage distributions, and the missing-path objective separately supervises the EEG head. Training includes 20% label-independent EOG dropout and random shifts from the fixed candidate set. Removing missing-path training also removes dropout supervision of that head.

Shift supervision is restricted to training epochs within transition neighborhoods whose maximum absolute EEG/EOG correlation is at least 0.25 and whose best-to-second-best offset-correlation margin is at least 0.02. This is an explicit information criterion; it is not a claim that physiological phase and acquisition offset coincide. The identifiability mask never enters the deployed gate. Both signal encoders normalize only the selected epoch. The lightweight architecture and the fixed criterion supply concrete choices where the manuscript does not specify layers or identifiability thresholds.
