#!/bin/bash

# Define placeholder variables for paths
CSV_FILE_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/EMBED_combined_cases_with_followup_with_split.csv"
DATA_ROOT_PATH="/mnt/cv_data/users/mengxu/EMBED_Dataset_Split_Uniform_Orientation"
OUTPUT_DIR_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_models/risk_prediction_no_alignment_uniform_orientation" # saved model .pth
TEST_FOLDER_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_test_results/risk_prediction_no_alignment_uniform_orientation" # where to save the test results (logs, predictions, and metrics)
TRAINING_ID="embed_balanced_split_no_alignment_uniform_orientation"
DATASET="EMBED"  # “CSAW” or "EMBED"

# Create directory if it doesn't exist
mkdir -p "$OUTPUT_DIR_PATH"
mkdir -p "$TEST_FOLDER_PATH"

# Run the Python script with the specified arguments
python3  -m src.evaluate.main_test_risk_prediction \
--csv_file "$CSV_FILE_PATH"  \
--data_root "$DATA_ROOT_PATH"  \
--path_out_dir "$OUTPUT_DIR_PATH" \
--path_test_folder "$TEST_FOLDER_PATH" \
--id_training "$TRAINING_ID" \
--num_epoch 22 \
--batch_size 12 \
--use_img_alignment "False" \
--no_feat_alignment "True" \
--early_stop "True" \
--dataset "$DATASET" \
--seed 2023 
