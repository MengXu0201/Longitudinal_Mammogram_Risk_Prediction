#!/bin/bash

CSV_FILE_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/EMBED_combined_cases_with_followup_with_split.csv"
DATA_ROOT_PATH="/mnt/cv_data/users/mengxu/EMBED_Split_Cropped_PNG_Uniform_Orientation"
OUTPUT_DIR_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_models/risk_prediction_no_alignment_uniform_orientation_cropped_moco_init"
TEST_FOLDER_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_test_results/risk_prediction_no_alignment_uniform_orientation_cropped_moco_init"
TRAINING_ID="embed_balanced_split_no_alignment_uniform_orientation_cropped_moco_init"
DATASET="EMBED"

mkdir -p "$OUTPUT_DIR_PATH"
mkdir -p "$TEST_FOLDER_PATH"

python3 -m src.evaluate.main_test_risk_prediction \
--csv_file "$CSV_FILE_PATH" \
--data_root "$DATA_ROOT_PATH" \
--path_out_dir "$OUTPUT_DIR_PATH" \
--path_test_folder "$TEST_FOLDER_PATH" \
--id_training "$TRAINING_ID" \
--num_epoch 0 \
--batch_size 12 \
--use_img_alignment "False" \
--no_feat_alignment "True" \
--early_stop "False" \
--dataset "$DATASET" \
--seed 2023
