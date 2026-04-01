# Longitudinal Mammogram Alignment: Notebook Logic Walkthrough

This document explains the code logic of the three notebooks, in this order:

1. `Preprocess_EMBED_POSITIVE_GROUP_FINAL.ipynb`
2. `Preprocess_EMBED_NEGATIVE_GROUP_FINAL.ipynb`
3. `Generate_CSV_File_Risk_prediction.ipynb`

## 1) `Preprocess_EMBED_POSITIVE_GROUP_FINAL.ipynb`

### Goal
Build a **positive cohort** CSV for risk modeling, where each row is an image/exam from patients who eventually develop cancer, and include time-to-cancer labels.

### Main pipeline

1. Load and sort source tables.
- Reads reduced clinical and metadata CSVs.
- Converts date columns to datetime.
- Sorts by patient/date and writes sorted copies.

2. Define cancer-positive records.
- A record is treated as cancer-related if:
  - `asses == "K"` OR
  - `path_severity` in `[0, 1]`.
- Saves this subset as `POSITIVE_GROUP_cancer_img.csv`.

Toy example:
- If an exam has `asses='B'` but `path_severity=1`, it is still included as cancer-positive.

3. Compute each patient’s latest diagnosis date.
- For every `empi_anon`, take max `study_date_anon` among cancer rows.
- Merge this date back as `latest_diagnosed_date`.

Why this matters:
- Later, all candidate prior screenings are filtered to be on or before this latest diagnosis date.

4. Build pre-diagnosis screening candidates.
- Starts from metadata (`screenings_df`), merges `tissueden` from clinical by patient+exam.
- Renames cancer `study_date_anon` to `diagnosed_date_anon`.
- For each cancer row, finds screening images with:
  - same patient,
  - same laterality (`ImageLateralityFinal` == cancer `side`),
  - date <= latest diagnosis date.
- Labels these as `Non-Cancer` and cancer rows as `Cancer`, then concatenates.
- Forces rows where `diagnosed_date_anon == study_date_anon` to label `Cancer`.

Toy example:
- Patient P1 has left-side cancer diagnosed 2020-06-01.
- Left breast exams from 2018, 2019, 2020-06-01 are kept.
- 2018/2019 rows become `Non-Cancer`; 2020-06-01 is `Cancer`.

5. Keep longitudinally useful patients and standard views.
- Keeps only patients with >1 unique exam (`acc_anon`).
- Keeps only `ViewPosition` in `['CC', 'MLO']`.
- Drops duplicate `anon_dicom_path`.

6. Remove undesired protocol variants and extra columns.
- Excludes many protocol strings (`SCC`, `SMLO`, `CEDM`, etc.).
- Drops demographic/unused columns.
- Saves intermediate CSV: `POSITIVE_GROUP_with_screenings_NEW_2D_CView.csv`.

7. Compute `Time_to_Cancer_Years` and bin groups.
- Computes year-only difference:
  - `diagnosed_year - study_year`.
- Keeps range 0..5 years.
- Keeps only `FinalImageType == '2D'`.
- Again removes patients with only one exam.
- Creates bins for non-cancer rows:
  - `BC 1Y`, `BC 2Y`, ..., `BC 5Y`.
- Saves: `POSITIVE_GROUP_with_screenings_time_to_cancer_NEW.csv`.

Toy example:
- Study date 2017-11-10, diagnosis 2020-01-05 -> year diff = `2020-2017 = 3`, assigned `BC 3Y`.
- Note: this is not exact day-based years; it is year-component subtraction.

8. Build final positive output schema.
- Converts relative DICOM path to absolute path under EMBED root.
- Renames columns:
  - `empi_anon -> patient_id`
  - `acc_anon -> exam_id`
  - `ViewPosition -> view`
  - `tissueden -> density`
- Drops many task-irrelevant columns.
- Saves final: `POSITIVE_GROUP_FINAL.csv`.

9. Validation checks.
- Prints non-null density counts.
- Compares final density values against full clinical table by patient/exam.
- Shows mismatch rows if any.

### Outputs produced
- `POSITIVE_GROUP_cancer_img.csv`
- `POSITIVE_GROUP_with_screenings_NEW_2D_CView.csv`
- `POSITIVE_GROUP_with_screenings_time_to_cancer_NEW.csv`
- `POSITIVE_GROUP_FINAL.csv`

---

## 2) `Preprocess_EMBED_NEGATIVE_GROUP_FINAL.ipynb`

### Goal
Build a **negative cohort** CSV with strict filtering so rows are likely cancer-free cases with sufficient negative follow-up.

### Main pipeline

1. Load full clinical/metadata tables and select key columns.
- Converts `study_date_anon` to datetime.
- Keeps only 2D + standard views (`MLO`, `CC`) in metadata (`meta_2d`).

2. Focus on screening exams.
- `screening_magview = magview` rows with `desc` containing "screen".
- Derives exam-level laterality (`B`, `L`, `R`) from text in `desc`.
- Fills missing `side` as bilateral (`B`).

3. Expand bilateral `side='B'` into left/right rows.
- Duplicates bilateral rows so one becomes `L`, one becomes `R`.
- This ensures side-specific logic can run.

Toy example:
- One bilateral screening exam row -> two rows: left and right.

4. Build contralateral negatives for incomplete bilateral capture.
- Checks bilateral exams where only one side appears.
- Synthesizes opposite-side rows and marks them benign:
  - set `asses='N'`, `path_severity=NaN` for synthetic opposite side.
- Concatenates original + synthetic rows.
- Saves helper file: `EMBED_OpenData_magview_with_controlateral.csv`.

5. Create candidate benign groups.
- `b0`: rows with `asses == 'A'` (incomplete; often needing diagnostic follow-up).
- `b12`: rows with `asses in ['B','N']` (benign/negative).
- `diag_magview`: diagnostic exams (`desc` contains `diag`).

6. For BIRADS-0, require benign diagnostic follow-up within 3 months.
- Merge `b0` with diagnostic exams on patient.
- Keep same side (or diagnostic side bilateral/unknown).
- Keep follow-up days 0..90.
- Keep only benign follow-up diagnoses (`asses_dx in ['N','B']`) -> `b0_12dx`.

7. Define positive reference group for overlap removal.
- `b0_3456dx` identifies clearly positive exams (`asses='K'` or `path_severity in [0,1]`).
- Later used to remove any negative examples that collide with positive exam+side.

8. Build negative candidate and require >1 year benign follow-up.
- `neg_group = b12 + b0_12dx`.
- Merge `neg_group` with `b12` on patient to find future benign same-side follow-up.
- Compute `delta_date_1yrfu`; keep >360 days.
- Keep first follow-up exam per (`acc_anon`, `side`).
- Exclude any biopsy evidence (`path_severity` must be NaN).

Toy example:
- If a screening was 2018-01-01 and benign follow-up exists 2019-03-01 (424 days), it passes 1-year follow-up rule.

9. Attach image metadata and ensure side match.
- Merge with `meta_2d` on patient/exam/date.
- Keep rows where clinical `side == ImageLateralityFinal`.
- Drop duplicate DICOM paths.

10. Remove overlap with positive exams.
- Creates positive image table (`pos_group_images`).
- Builds key `acc_anon + side`.
- Removes any negative row whose key appears in positive set.

11. Remove overlap with final positive patient set.
- Loads `POSITIVE_GROUP_FINAL.csv`.
- Excludes any negative rows with patient IDs appearing in positive final set.
- Prints overlap before/after as sanity check.

12. Additional exclusions.
- Remove patients with `BreastImplantPresent == 'YES'`.
- Remove special protocols (SCC/SMLO/etc.; allows null ProtocolName).

13. Require long follow-up window.
- For each patient, compute follow-up duration as:
  - `max(study_year) - min(study_year)`.
- Keep patients with duration >= 5 years.

Note:
- Uses year difference, not exact day delta.

14. Convert paths and standardize final schema.
- Converts relative DICOM to absolute EMBED path.
- Saves temporary `NEGATIVE_GROUP_FINAL_temp.csv` with selected columns.

15. Reduce to one image per laterality/view combination per exam.
- Group by patient+exam.
- Within each group:
  - count `ProtocolName` frequency,
  - sort by `ViewPosition` + protocol frequency,
  - keep one row per (`ImageLateralityFinal`, `ViewPosition`).
- Renames columns:
  - `empi_anon -> patient_id`
  - `acc_anon -> exam_id`
  - `ViewPosition -> view`
  - `anon_dicom_path -> file_path_dcm`
  - `tissueden -> density`
- Saves final: `NEGATIVE_GROUP_FINAL.csv`.

16. Validation checks.
- Density completeness counts.
- Random-patient clinical-vs-final density comparison.
- Full mismatch report across all final rows.

### Outputs produced
- `EMBED_OpenData_magview_with_controlateral.csv`
- `NEGATIVE_GROUP_FINAL_temp.csv`
- `NEGATIVE_GROUP_FINAL.csv`

---

## 3) `Generate_CSV_File_Risk_prediction.ipynb`

### Goal
Combine positive and negative final cohorts, compute follow-up labels, and produce patient-level train/val/test splits stratified by positive/negative status.

### Main pipeline

1. Load final positive and negative CSVs.
- Positive parses `study_date_anon` and `diagnosed_date_anon`.
- Negative parses `study_date_anon`.
- Checks patient overlap between cohorts.

2. Compute follow-up duration field `years_last_followup`.

Positive logic:
- Per row: `(diagnosed_date_anon - study_date_anon).days / 365.25`, cast to int.

Negative logic:
- For each patient, find latest exam date.
- For each row: `(latest_exam_date - study_date_anon).days / 365.25`, cast to int.

Toy example:
- Negative patient exams in 2015, 2017, 2020.
- `years_last_followup` becomes about 5, 3, 0 (after int cast).

3. Concatenate cohorts.
- `all_cases = positive_cases + negative_cases`.
- Saves `EMBED_combined_cases_with_followup.csv`.

4. Build patient-level stratification target (binary).
- Defines exam-level positivity as `Label.notna()`.
  - `Label` present => positive exam.
  - `Label` missing => negative exam.
- Aggregates to patient-level:
  - `is_positive_patient = max(is_positive)` across all exams.
  - If a patient has at least one positive exam, patient is positive.

Toy example:
- Patient A has exam labels `[NaN, NaN, Cancer]` -> `is_positive_patient = 1`.
- Patient B has `[NaN, NaN]` -> `is_positive_patient = 0`.

5. Stratified patient split.
- Uses `train_test_split` with `stratify` on `is_positive_patient`.
- Target split ratio train/val/test = 0.5/0.2/0.3.
  - first split: 50% train, 50% temp
  - second split temp into 40% val, 60% test (=> 20/30 overall)

6. Push split labels back to image-level rows.
- Assigns `split_group` in full dataframe by patient membership.
- Asserts no unassigned rows.
- Adds leakage safety check:
  - each patient must appear in exactly one split.

7. Post-split checks (first summary block).
- Prints counts by split.
- Prints positive/negative patient distribution per split.
- Prints positive patient ratio per split.

8. Additional split-diagnostics block (`df_check` copy; no change to saved dataset).
- Recomputes helper flags:
  - `is_positive` from `Label.notna()`
  - `is_cancer_exam` from `Label == 'Cancer'`
- Reports:
  - patient-level and exam-level split counts/ratios
  - patient-level and exam-level positive/negative distributions
  - exam-level cancer ratio by split
- If `Time_to_Cancer_Years` exists:
  - bins into `1Y`..`5Y`, `Missing_TTC`, `Other`
  - prints count and normalized crosstabs by split
  - computes horizon availability indicators (`>=1Y` ... `>=5Y`) and prints counts/ratios
- Computes exams-per-patient statistics by split (mean/std/median/min/max).

9. Save final output.
- Saves final:
  - `EMBED_combined_cases_with_followup_with_split.csv`.

### Outputs produced
- `EMBED_combined_cases_with_followup.csv`
- `EMBED_combined_cases_with_followup_with_split.csv`

---

## End-to-end dependency map

1. Positive notebook creates `POSITIVE_GROUP_FINAL.csv`.
2. Negative notebook creates `NEGATIVE_GROUP_FINAL.csv` and explicitly excludes positive-patient overlap using positive final output.
3. Risk-prediction notebook consumes both finals, computes follow-up, then creates split-ready combined CSV.

## Practical notes

- Time differences are often converted using integer years (or year subtraction), which is simple and reproducible but coarse near year boundaries.
- Multiple steps intentionally remove overlap between positive and negative sets at patient or exam-side level to avoid leakage.
- Final split is patient-level stratified, which helps prevent train/test leakage from the same patient appearing in different splits.
