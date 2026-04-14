#!/bin/bash

DATA_ROOT_PATH="/mnt/cv_data/users/mengxu/EMBED_Split_Cropped_PNG_Uniform_Orientation"
FEATURE_CACHE_ROOT_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_feature_cache/resnet18_no_alignment_uniform_orientation_cropped_early_stop"
CACHE_ROOT_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_graph_cache/masked_grid_uniform_orientation"

mkdir -p "$CACHE_ROOT_PATH"

python3 -m src.preprocessing.build_masked_grid_graph_cache \
--data_root "$DATA_ROOT_PATH" \
--feature_cache_root "$FEATURE_CACHE_ROOT_PATH" \
--cache_root "$CACHE_ROOT_PATH" \
--splits train val test \
--similarity_threshold 0.70 \
--k_min 2 \
--k_max 16 \
--min_area_ratio 0.05
