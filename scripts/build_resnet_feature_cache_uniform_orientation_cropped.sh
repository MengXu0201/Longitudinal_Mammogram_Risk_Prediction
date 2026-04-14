#!/bin/bash

DATA_ROOT_PATH="/mnt/cv_data/users/mengxu/EMBED_Split_Cropped_PNG_Uniform_Orientation"
CACHE_ROOT_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_feature_cache/resnet18_no_alignment_uniform_orientation_cropped_early_stop"
CHECKPOINT_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_models/risk_prediction_no_alignment_uniform_orientation_cropped/early_stopping_risk_prediction_id-embed_balanced_split_no_alignment_uniform_orientation_cropped.pth"

mkdir -p "$CACHE_ROOT_PATH"

python3 -m src.preprocessing.build_resnet_feature_cache \
--data_root "$DATA_ROOT_PATH" \
--cache_root "$CACHE_ROOT_PATH" \
--checkpoint_path "$CHECKPOINT_PATH" \
--splits train val test \
--batch_size 16 \
--feature_dtype float16 \
--seed 2023
