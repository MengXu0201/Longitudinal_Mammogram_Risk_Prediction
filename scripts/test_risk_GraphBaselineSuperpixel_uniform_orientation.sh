#!/bin/bash

CSV_FILE_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/EMBED_combined_cases_with_followup_with_split.csv"
DATA_ROOT_PATH="/mnt/cv_data/users/mengxu/EMBED_Split_Cropped_PNG_Uniform_Orientation"
CACHE_ROOT_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_graph_cache/superpixel_uniform_orientation"
FEATURE_CACHE_ROOT_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_feature_cache/resnet18_no_alignment_uniform_orientation_cropped_early_stop"
OUTPUT_DIR_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_models/risk_prediction_graph_baseline_superpixel_uniform_orientation_cropped"
TEST_FOLDER_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_test_results/risk_prediction_graph_baseline_superpixel_uniform_orientation_cropped"
TRAINING_ID="embed_balanced_split_graph_baseline_superpixel_uniform_orientation_cropped"
DATASET="EMBED"

mkdir -p "$OUTPUT_DIR_PATH"
mkdir -p "$TEST_FOLDER_PATH"

python3 -m src.evaluate.main_test_risk_prediction_graph_superpixel \
--csv_file "$CSV_FILE_PATH" \
--data_root "$DATA_ROOT_PATH" \
--cache_root "$CACHE_ROOT_PATH" \
--feature_cache_root "$FEATURE_CACHE_ROOT_PATH" \
--path_out_dir "$OUTPUT_DIR_PATH" \
--path_test_folder "$TEST_FOLDER_PATH" \
--id_training "$TRAINING_ID" \
--num_epoch 99 \
--batch_size 12 \
--early_stop "False" \
--dataset "$DATASET" \
--seed 2023
