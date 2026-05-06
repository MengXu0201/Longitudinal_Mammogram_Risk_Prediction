#!/bin/bash

CSV_FILE_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/EMBED_combined_cases_with_followup_with_split.csv"
DATA_ROOT_PATH="/mnt/cv_data/users/mengxu/EMBED_Split_Cropped_PNG_Uniform_Orientation"
OUTPUT_DIR_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_models/pretraining/moco_embed_uniform_orientation_cropped"
LOG_DIR_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/logs/pretraining"
TRAINING_ID="embed_moco_uniform_orientation_cropped"
DATASET="EMBED"

mkdir -p "$OUTPUT_DIR_PATH"
mkdir -p "$LOG_DIR_PATH"

WANDB_MODE=disabled python3 -m src.train.main_train_moco_pretraining \
--csv_file "$CSV_FILE_PATH" \
--data_root "$DATA_ROOT_PATH" \
--path_out_dir "$OUTPUT_DIR_PATH" \
--log_dir "$LOG_DIR_PATH" \
--id_training "$TRAINING_ID" \
--dataset "$DATASET" \
--augmentations "True" \
--use_scheduler "True" \
--save_debug_augs "True" \
--batch_size 16 \
--num_workers 4 \
--learning_rate 1e-4 \
--weight_decay 1e-5 \
--num_epochs 100 \
--queue_size 16384 \
--momentum 0.999 \
--temperature 0.07 \
--projection_dim 128 \
--projector_hidden_dim 512 \
--seed 2023

