#!/bin/bash

CSV_FILE_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/EMBED_combined_cases_with_followup_with_split.csv"
DATA_ROOT_PATH="/mnt/cv_data/users/mengxu/EMBED_Dataset_Split_Uniform_Orientation"
OUTPUT_DIR_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_models/risk_prediction_graph_baseline_masked_grid_uniform_orientation"
TRAINING_ID="embed_balanced_split_graph_baseline_masked_grid_uniform_orientation"
DATASET="EMBED"

mkdir -p "$OUTPUT_DIR_PATH"

WANDB_MODE=disabled python3 -m src.train.main_train_risk_prediction_graph \
--csv_file "$CSV_FILE_PATH" \
--data_root "$DATA_ROOT_PATH" \
--path_out_dir "$OUTPUT_DIR_PATH" \
--id_training "$TRAINING_ID" \
--use_scheduler "True" \
--accumulation_steps 1 \
--augmentations "False" \
--dataset "$DATASET" \
--seed 2023
