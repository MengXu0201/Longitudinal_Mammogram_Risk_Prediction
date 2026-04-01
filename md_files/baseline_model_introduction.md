# Baseline Model Introduction (No Alignment)

This document introduces the baseline risk prediction setup launched by:
- `scripts/train_risk_NoAlign.sh`

and summarizes the related code in:
- `src/train/main_train_risk_prediction.py`
- `src/train/train_risk_prediction.py`
- `src/models/model_combined_alignment_risk.py`
- `src/models/model_risk_prediction.py`
- `src/models/model_feat_alignment.py`
- `src/dataloaders/risk_prediction/dataset_embed.py`
- `src/preprocessing/split_dataset.py`
- `src/utils/utils.py`

## 1. What this baseline is

The baseline is a **two-timepoint mammogram risk model without alignment**:
- Input: a pair of adjacent exams from the same patient-side-view
  - `previous_image`
  - `current_image`
  - `time_gap` (year difference, capped at 5)
- Backbone: shared `ResNet18Encoder` (ImageNet pretrained)
- Head: cumulative-probability survival-style predictor for 6 outputs (`0..5` indices for a 5-year horizon + censor slot)
- Training target: masked BCE over multi-horizon targets for fused/current/prior branches

In code, this baseline corresponds to:
- `RiskModelNoAlignment` in `src/models/model_combined_alignment_risk.py`
- Used in `train_val_jointly(...)` when `no_feat_alignment == "True"` in `src/train/train_risk_prediction.py`

## 2. Entry script and run configuration

From `scripts/train_risk_NoAlign.sh`:
- CSV file:
  - `/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Alignment/output_csv/EMBED_combined_cases_with_followup_with_split.csv`
- Image root:
  - `/mnt/cv_data/users/mengxu/EMBED_Dataset_Split`
- Output directory:
  - `/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Alignment/output_models/risk_prediction_no_alignment`
- Training ID:
  - `embed_balanced_split_no_alignment`
- Dataset:
  - `EMBED`

Script-level flags:
- `--use_scheduler True`
- `--accumulation_steps 1`
- `--augmentations False`
- `--use_img_alignment False`
- `--no_feat_alignment True`
- `--use_reg_loss False`
- `--seed 2023`

Key defaults from `main_train_risk_prediction.py` (unless overridden in script):
- `learning_rate=1e-4`
- `weight_decay=1e-5`
- `num_epochs=100`
- `batch_size=12`
- `num_workers=4`
- `patience_lr_scheduler=5`
- `patience=15`
- `lr_decay=0.5`
- `pin_memory=True`

## 3. Data pipeline and split behavior

### 3.1 Split source
The training CSV includes a `split_group` column (`train/val/test`) generated earlier in preprocessing.

### 3.2 Image split folders
`BreastCancerRiskDataset` reads from:
- `<data_root>/train`
- `<data_root>/val`
- `<data_root>/test`

The helper `split_data_and_copy_images_risk_embed(...)` in `src/preprocessing/split_dataset.py` copies images to these folders according to CSV `split_group` at patient level.

### 3.3 Filename convention
Expected PNG name pattern in `dataset_embed.py`:
- `patient_id_laterality_view_studydate_last4_status.png`
- status is one of:
  - `pos_cancer`
  - `pos_nocancer`
  - `neg_nocancer`

### 3.4 Pair construction
For each `(patient_id, laterality, view)`:
1. sort exams by date
2. create adjacent pairs `(exam_{t-1}, exam_t)`

So one patient-side-view with exams `[e1,e2,e3]` yields pairs:
- `(e1,e2)`
- `(e2,e3)`

## 4. Input preprocessing inside dataset

In `dataset_embed.py`:
- Image is loaded from PNG
- Rescaled to 16-bit-like dynamic range (`imgunit16`)
- Normalized as:
  - `(img - 7047.99) / 12005.5`
- Converted to tensor and returned as single-channel (model expands to 3 channels later)
- `time_gap = min(abs(year_cur - year_prev), 5)`

Optional augmentations (disabled in this script) would use Kornia:
- random rotation
- random affine

## 5. Targets and censoring formulation

`dataset_embed.py` creates:
- `target` for current exam (length 6)
- `target_prior` for prior exam (length 6)
- `y_mask`, `y_mask_prior` for valid supervision horizon
- `event_observed` (1 cancer within horizon, else 0)
- `event_times` (event/censor index)

Logic summary:
- If cancer occurs within horizon: set target from event index onward to 1 (except reserved final censor slot)
- If no cancer in horizon: set last element (censor slot) to 1
- Mask (`y_mask`) enables loss only up to available follow-up/event time

This allows learning from censored trajectories and different follow-up lengths.

## 6. Baseline model architecture

### 6.1 Encoder
`ResNet18Encoder` (`model_feat_alignment.py`):
- torchvision ResNet-18 with ImageNet weights
- removes avgpool + FC
- outputs feature map (not class logits)

### 6.2 No-alignment model
`RiskModelNoAlignment` (`model_combined_alignment_risk.py`):
1. Expand grayscale to 3 channels by repetition
2. Encode current/prior via shared ResNet18 encoder
3. Send features to `TemporalRiskPredictionWithCumulativeProbLayer_no_alignment`

### 6.3 Risk head
`TemporalRiskPredictionWithCumulativeProbLayer_no_alignment` (`model_risk_prediction.py`):
- Global average pool current and prior features
- Build three prediction branches:
  - fused: `[f_cur, f_pri]`
  - current-only
  - prior-only
- Each branch uses `CumulativeProbabilityLayer`
- Final output applies sigmoid, producing probabilities per year slot

`CumulativeProbabilityLayer`:
- predicts hazards and base hazard
- applies upper-triangular masking to accumulate over time
- outputs cumulative risk-like sequence over horizon

## 7. Training objective and optimization

In `train_val_jointly(...)` (`train_risk_prediction.py`):

### 7.1 Loss
Risk loss = sum of masked BCE across branches:
- `BCE(pred_fused, target, y_mask)`
- `BCE(pred_cur, target, y_mask)`
- `BCE(pred_pri, target_prior, y_mask_prior)`

Implemented by `get_risk_loss_BCE(...)` in `utils.py`.

For no-alignment baseline:
- total loss = risk loss only
- alignment and deformation regularization terms are not used

### 7.2 Optimizer and scheduler
- Optimizer: Adam
- LR: `1e-4` (default)
- Weight decay: `1e-5` (default)
- Scheduler: `ReduceLROnPlateau(mode='max', factor=0.5, patience=5)` when enabled

### 7.3 Early stopping and checkpoints
- Monitor validation C-index
- Save best model when C-index improves:
  - `best_model_risk_prediction_id-<id>.pth`
- Early stop after `patience=15` non-improving epochs
- Save final epoch model:
  - `model_risk_prediction_training_id_<id>_last_epoch.pth`

## 8. Evaluation signals logged during training

Training:
- Training loss
- Training risk loss
- Training C-index

Validation:
- Validation risk loss
- Validation C-index
- Year-specific AUC (`Year 1` ... `Year 5`)

Logging backends:
- file logger (`train_risk_prediction_training_id_<id>.log`)
- Weights & Biases (`EMBED_Risk_Prediction` project)

## 9. End-to-end baseline flow

1. `train_risk_NoAlign.sh` calls `python -m src.train.main_train_risk_prediction`
2. Main script builds EMBED train/val datasets and dataloaders
3. Because `use_img_alignment=False`, training uses `train_val_jointly`
4. Because baseline flag is true, model class is `RiskModelNoAlignment`
5. Train with masked BCE risk objective and monitor C-index/AUC
6. Save best/early-stop/final checkpoints in output directory

## 10. Practical note

`main_train_risk_prediction.py` currently references `args.no_feat_Alignment` (capital `A`) when calling `train_val_jointly`, while the parser defines `--no_feat_alignment` (lowercase `a`).

If not already patched locally, this mismatch can raise an `AttributeError` at runtime. The intended baseline behavior is clearly the no-alignment path configured by `--no_feat_alignment True`.
