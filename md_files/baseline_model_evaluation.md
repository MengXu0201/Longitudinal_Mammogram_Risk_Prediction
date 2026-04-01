# Baseline Model Evaluation (No Alignment)

This document explains how baseline evaluation is implemented for:
- `scripts/test_risk_NoAlign.sh`

and related files:
- `src/evaluate/main_test_risk_prediction.py`
- `src/evaluate/test_risk_prediction.py`
- `src/utils/c_index.py`
- `src/utils/utils.py`
- `src/dataloaders/risk_prediction/dataset_embed.py`

## 1. Evaluation entry point and flow

From `scripts/test_risk_NoAlign.sh`, evaluation runs:

```bash
python3 -m src.evaluate.main_test_risk_prediction \
  --csv_file ...EMBED_combined_cases_with_followup_with_split.csv \
  --data_root .../EMBED_Dataset_Split \
  --path_out_dir .../output_models/risk_prediction_no_alignment \
  --path_test_folder .../output_test_results/risk_prediction_no_alignment \
  --id_training embed_balanced_split_no_alignment \
  --num_epoch 22 \
  --batch_size 12 \
  --use_img_alignment False \
  --no_feat_alignment True \
  --early_stop True \
  --dataset EMBED \
  --seed 2023
```

Because `--use_img_alignment False`, `main_test_risk_prediction.py` dispatches to:
- `test_jointly_feat_alignment_risk(...)`

Because `--no_feat_alignment True`, that function loads:
- `RiskModelNoAlignment`

Model outputs used for metrics:
- `pred_fused` (shape: `N x 6`), where 6 slots correspond to year-indexed risk outputs.

Data labels used for metrics:
- `event_times`
- `event_observed`
- `density`

Results are saved to:
- `results.json`
- test log file in `path_test_folder`

## 2. What metrics are reported

For baseline no-alignment evaluation, reported metrics are:

1. C-index (IPCW/Uno-style) + 95% CI
2. Yearly AUC (Year 1 to Year 5) + 95% CI
3. AUC by density category (A/B/C/D)
4. C-index by density category (A/B/C/D)

`NJD` is only computed when deformation fields are produced (alignment models), so it is typically absent for no-alignment baseline.

## 3. Metric definitions, toy examples, and calculation details

## 3.1 C-index (IPCW, time-dependent)

### Definition
C-index measures ranking quality for time-to-event prediction:
- Higher value means patients with earlier events tend to receive higher risk (or lower predicted survival) than patients with later/censored outcomes.
- Around `0.5` means random ranking.
- Closer to `1.0` means strong ranking.

This code uses an IPCW-weighted variant (Uno-style idea), where each comparable pair is weighted by inverse censoring probability.

### How this code calculates it
In `c_index.py`:
1. Build censoring distribution `G(t)` using Kaplan-Meier (`get_censoring_dist`).
2. Convert model risk to score via `predicted_scores = 1 - predictions`.
3. For each comparable pair, compute concordance/ties using scores at relevant time index.
4. Weight pair contributions by `1 / G(t)^2`.
5. Return:

`C-index = (weighted_correct + 0.5 * weighted_ties) / weighted_pairs`

### Toy example
Suppose at a specific horizon we compare three patients:
- A: event earlier
- B: censored later
- C: event later

If model ranks risk as `A > C > B`, most comparable pairs are concordant.
If it ranks `B > A > C`, many pairs are discordant.
So C-index decreases.

## 3.2 Yearly AUC (1Y..5Y)

### Definition
For each follow-up year, AUC evaluates binary discrimination:
- positive class = event has occurred by that year
- negative class = event not yet occurred and still under follow-up at that year

### How this code constructs labels per year
In `compute_auc_x_year_auc(...)`:
- for each follow-up index `f in {0,1,2,3,4}`:
  - include sample if:
    - `gold == 1 and censor_time <= f` (event happened by year `f`), or
    - `censor_time >= f` (still at risk / event-free through year `f`)
  - label included sample as:
    - `1` if event by `f`
    - `0` otherwise
  - score is `prob_arr[f]`
  - compute ROC AUC from these `(score, label)` pairs.

### Toy example
At Year 2 (index `f=1`), assume 4 samples:
- P1: event at year 1 -> include, label 1, score `p[1]=0.90`
- P2: event at year 4 -> include as not yet event by year 2, label 0, score `0.40`
- P3: censored at year 3 -> include, label 0, score `0.20`
- P4: censored before year 2 -> excluded

AUC is then ROC-AUC over included points `(0.90,1), (0.40,0), (0.20,0)`.

## 3.3 AUC by density category

### Definition
Same yearly AUC definition, but computed separately within each density subgroup:
- A, B, C, D

### How calculated
In `bootstrap_auc_by_density(...)`:
1. Filter records for one density group.
2. Bootstrap-resample cancer and non-cancer separately (class-balanced resampling).
3. Recompute yearly AUC on each bootstrap sample.
4. Report mean and 95% percentile CI per year for that density.

### Toy example
If density C subgroup has higher separation than density A:
- C may show Year-3 AUC ~0.78
- A may show Year-3 AUC ~0.64

This can reveal subgroup-specific model behavior.

## 3.4 C-index by density category

### Definition
Same IPCW C-index, computed within each density subgroup.

### How calculated
In `bootstrap_c_index_by_density(...)`:
1. Filter to one density group.
2. Bootstrap-resample cancer/non-cancer indices within that group.
3. Compute IPCW C-index for each bootstrap sample.
4. Return subgroup mean and 95% CI.

### Toy example
If density D has few samples, C-index CI may be wide:
- Mean 0.70, CI (0.55, 0.84)

Wider CI reflects greater uncertainty.

## 3.5 95% confidence intervals (bootstrap)

### Definition
A 95% CI gives a plausible range for the metric under sampling variability.

### How calculated here
For each metric:
1. Generate many bootstrap samples (default 1000).
2. Recompute metric per sample.
3. Use percentile interval:
- lower = 2.5th percentile
- upper = 97.5th percentile

### Toy example
If bootstrap C-index values cluster around 0.69:
- 2.5th percentile = 0.64
- 97.5th percentile = 0.73
- report: `0.69 (95% CI: 0.64–0.73)`

## 4. Important implementation notes for baseline

1. The test script selects model checkpoint by `--early_stop True`, so evaluation uses:
- `early_stopping_risk_prediction_id-<id>.pth`

2. `--num_epoch 22` does not control inference iterations; it only influences model file selection logic in `main_test_risk_prediction.py`.

3. For baseline no-alignment, deformation metrics are not produced.

## 5. Logic check: potential issues in evaluation code

Below are the main issues I found.

1. Non-deterministic bootstrap (primary reason for slightly different repeated results)
- `main_test_risk_prediction.py` sets `random.seed` and torch seeds, but **does not set `np.random.seed`**.
- Bootstrap functions (`bootstrap_auc`, `bootstrap_c_index`, etc.) use NumPy / sklearn resampling randomness.
- Result: each run resamples differently, so means/CIs differ slightly.

2. Additional reproducibility gap on GPU
- No explicit deterministic backend settings in test script (`torch.backends.cudnn.deterministic=True`, `benchmark=False`).
- Even with same model/data, tiny numerical differences can appear on GPU kernels.

3. Fragile AUC failure handling
- `compute_auc_x_year_auc` returns string `"NA"` when ROC-AUC fails.
- Later bootstrap aggregation uses `np.percentile`/`np.mean`, which expect numeric values.
- If failures occur frequently in a subgroup/year, this can break or produce inconsistent behavior.

4. Model selection logic can be confusing
- If `early_stop=True`, code always picks early-stop checkpoint regardless of `num_epoch`.
- This is not wrong, but easy to misinterpret when comparing runs.

5. Related (non-baseline path) argument mismatch
- `main_test_risk_prediction.py` expects `--use_img_feat_alignment_way1`
- `scripts/test_risk_ImgFeatAlign.sh` uses `--use_img_feat_alignment`
- Not affecting baseline no-align run, but this is a real inconsistency in related evaluation scripts.

## 6. Why you got slightly different metrics in two runs

Most likely cause:
- bootstrap randomness is not fully seeded (`numpy` seed missing), so C-index/AUC means and CIs vary a bit from run to run.

Secondary causes:
- non-deterministic GPU execution order / floating-point accumulation.

This is expected if strict determinism is not enforced.

## 7. Recommended fixes for stable repeatability

1. In `main_test_risk_prediction.py`, add:
- `np.random.seed(args.seed)`

2. Set deterministic PyTorch backend in test mode:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

3. Pass fixed `random_state` to sklearn `resample(...)` (or derive from loop index + seed).

4. Replace `"NA"` outputs in AUC with `np.nan`, and aggregate with nan-safe stats.

With these changes, repeated runs should be much closer (or identical, depending on hardware/backend).
