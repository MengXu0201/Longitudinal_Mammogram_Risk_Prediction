#!/bin/bash

DATA_ROOT_PATH="/mnt/cv_data/users/mengxu/EMBED_Dataset_Split_Uniform_Orientation"
CACHE_ROOT_PATH="/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_graph_cache/superpixel_uniform_orientation"

mkdir -p "$CACHE_ROOT_PATH"

python3 -m src.preprocessing.build_superpixel_graph_cache \
--data_root "$DATA_ROOT_PATH" \
--cache_root "$CACHE_ROOT_PATH" \
--splits train val test \
--num_superpixels 64 \
--compactness 10.0 \
--sigma 1.0 \
--min_superpixel_area 64 \
--feature_height 32 \
--feature_width 16
