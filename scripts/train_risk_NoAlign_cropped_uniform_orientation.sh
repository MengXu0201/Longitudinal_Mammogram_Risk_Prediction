#!/bin/bash

# Define placeholder variables for paths
CSV_FILE_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/EMBED_combined_cases_with_followup_with_split.csv"
DATA_ROOT_PATH="/mnt/cv_data/users/mengxu/EMBED_Split_Cropped_PNG_Uniform_Orientation" 
OUTPUT_DIR_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_models/risk_prediction_no_alignment_uniform_orientation_cropped"
TRAINING_ID="embed_balanced_split_no_alignment_uniform_orientation_cropped" 
DATASET="EMBED"  # “CSAW” or "EMBED"

# Create directory if it doesn't exist
mkdir -p "$OUTPUT_DIR_PATH"

# Run the Python script with the specified arguments
WANDB_MODE=disabled python3 -m src.train.main_train_risk_prediction \
--csv_file "$CSV_FILE_PATH"  \
--data_root "$DATA_ROOT_PATH" \
--path_out_dir "$OUTPUT_DIR_PATH" \
--id_training "$TRAINING_ID" \
--use_scheduler "True" \
--accumulation_steps 1 \
--augmentations "False" \
--use_img_alignment "False" \
--no_feat_alignment "True" \
--use_reg_loss "False" \
--dataset "$DATASET" \
--seed 2023 \
