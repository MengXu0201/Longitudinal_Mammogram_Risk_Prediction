import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from skimage.segmentation import slic
except ImportError:  # pragma: no cover - depends on user env
    slic = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build offline superpixel graph metadata cache for the uniform-orientation dataset.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing train/val/test PNG folders.")
    parser.add_argument("--cache_root", type=str, required=True, help="Output root for cached graph metadata.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to process.")
    parser.add_argument("--num_superpixels", type=int, default=64, help="Target number of superpixels per image.")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness.")
    parser.add_argument("--sigma", type=float, default=1.0, help="SLIC sigma.")
    parser.add_argument("--min_superpixel_area", type=int, default=64, help="Minimum valid superpixel area in pixels.")
    parser.add_argument("--feature_height", type=int, default=32, help="Encoder feature height.")
    parser.add_argument("--feature_width", type=int, default=16, help="Encoder feature width.")
    return parser.parse_args()


def _largest_connected_component(mask):
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels <= 1:
        return mask_uint8.astype(bool)

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest_idx


def _build_breast_mask(image):
    image = image.astype(np.float32)
    mask = image > (float(image.min()) + 1e-6)
    mask = _largest_connected_component(mask)
    return mask


def _compute_superpixels(image, breast_mask, num_superpixels, compactness, sigma):
    if slic is None:
        raise ImportError(
            "scikit-image is required for the superpixel cache builder. "
            "Please install scikit-image in the training environment."
        )

    image_float = image.astype(np.float32)
    image_min = float(image_float.min())
    image_max = float(image_float.max())
    if image_max > image_min:
        image_float = (image_float - image_min) / (image_max - image_min)
    else:
        image_float = np.zeros_like(image_float, dtype=np.float32)

    labels = slic(
        image_float,
        n_segments=num_superpixels,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        mask=breast_mask.astype(bool),
        channel_axis=None,
    )
    labels = labels.astype(np.int32)
    labels[~breast_mask] = -1
    return labels


def _filter_and_reindex_labels(label_map, breast_mask, min_superpixel_area):
    valid_ids = [int(x) for x in np.unique(label_map) if x >= 0]
    kept_ids = []
    for label_id in valid_ids:
        area = int(np.sum(label_map == label_id))
        if area >= min_superpixel_area:
            kept_ids.append(label_id)

    reindexed = np.full_like(label_map, fill_value=-1)
    old_to_new = {}
    for new_id, old_id in enumerate(kept_ids):
        old_to_new[old_id] = new_id
        reindexed[label_map == old_id] = new_id

    reindexed[~breast_mask] = -1
    return reindexed, old_to_new


def _build_feature_grid_pooling(label_map, feature_height, feature_width):
    image_height, image_width = label_map.shape
    unique_ids = [int(x) for x in np.unique(label_map) if x >= 0]
    num_nodes = len(unique_ids)

    if num_nodes == 0:
        raise ValueError("No valid superpixels remained after filtering.")

    pooling = np.zeros((num_nodes, feature_height * feature_width), dtype=np.float32)
    feature_label_map = np.full((feature_height, feature_width), fill_value=-1, dtype=np.int32)

    row_edges = np.linspace(0, image_height, feature_height + 1, dtype=int)
    col_edges = np.linspace(0, image_width, feature_width + 1, dtype=int)

    for feat_row in range(feature_height):
        row_start = row_edges[feat_row]
        row_end = row_edges[feat_row + 1]
        for feat_col in range(feature_width):
            col_start = col_edges[feat_col]
            col_end = col_edges[feat_col + 1]
            patch_labels = label_map[row_start:row_end, col_start:col_end]
            patch_labels = patch_labels[patch_labels >= 0]
            if patch_labels.size == 0:
                continue

            counts = np.bincount(patch_labels)
            node_id = int(np.argmax(counts))
            flat_idx = feat_row * feature_width + feat_col
            pooling[node_id, flat_idx] = 1.0
            feature_label_map[feat_row, feat_col] = node_id

    valid_row_sum = pooling.sum(axis=1)
    keep_node_mask = valid_row_sum > 0
    if not np.all(keep_node_mask):
        old_to_new = {}
        new_idx = 0
        for old_idx, keep in enumerate(keep_node_mask.tolist()):
            if keep:
                old_to_new[old_idx] = new_idx
                new_idx += 1
            else:
                old_to_new[old_idx] = -1

        pooling = pooling[keep_node_mask]

        remapped_feature_label_map = np.full_like(feature_label_map, fill_value=-1)
        for old_idx, new_idx in old_to_new.items():
            if new_idx >= 0:
                remapped_feature_label_map[feature_label_map == old_idx] = new_idx
        feature_label_map = remapped_feature_label_map

    row_sum = pooling.sum(axis=1, keepdims=True)
    pooling = pooling / np.clip(row_sum, a_min=1.0, a_max=None)
    return pooling, feature_label_map


def _build_superpixel_adjacency(feature_label_map, num_nodes):
    adjacency = np.eye(num_nodes, dtype=np.float32)
    neighbors = [(0, 1), (1, 0), (1, 1), (1, -1)]
    height, width = feature_label_map.shape

    for row in range(height):
        for col in range(width):
            src = int(feature_label_map[row, col])
            if src < 0:
                continue
            for d_row, d_col in neighbors:
                nbr_row = row + d_row
                nbr_col = col + d_col
                if nbr_row < 0 or nbr_row >= height or nbr_col < 0 or nbr_col >= width:
                    continue
                dst = int(feature_label_map[nbr_row, nbr_col])
                if dst < 0 or dst == src:
                    continue
                adjacency[src, dst] = 1.0
                adjacency[dst, src] = 1.0

    edge_rows, edge_cols = np.where(adjacency > 0)
    edge_index = np.stack([edge_rows, edge_cols], axis=0).astype(np.int64)
    return adjacency, edge_index


def _compute_node_stats(label_map, feature_label_map, num_nodes):
    node_area_pixels = np.zeros((num_nodes,), dtype=np.float32)
    node_centroids_image = np.zeros((num_nodes, 2), dtype=np.float32)
    node_centroids_feature = np.zeros((num_nodes, 2), dtype=np.float32)

    for node_id in range(num_nodes):
        image_positions = np.argwhere(label_map == node_id)
        if image_positions.size > 0:
            centroid_yx = image_positions.mean(axis=0)
            height, width = label_map.shape
            node_centroids_image[node_id] = np.array(
                [centroid_yx[1] / max(width - 1, 1), centroid_yx[0] / max(height - 1, 1)],
                dtype=np.float32,
            )
            node_area_pixels[node_id] = float(image_positions.shape[0])

        feature_positions = np.argwhere(feature_label_map == node_id)
        if feature_positions.size > 0:
            centroid_yx = feature_positions.mean(axis=0)
            feat_h, feat_w = feature_label_map.shape
            node_centroids_feature[node_id] = np.array(
                [centroid_yx[1] / max(feat_w - 1, 1), centroid_yx[0] / max(feat_h - 1, 1)],
                dtype=np.float32,
            )
        else:
            node_centroids_feature[node_id] = node_centroids_image[node_id]

    return node_area_pixels, node_centroids_image, node_centroids_feature


def build_superpixel_metadata(image_path, num_superpixels, compactness, sigma, min_superpixel_area, feature_height, feature_width):
    image = np.array(Image.open(image_path)).astype(np.float32)
    breast_mask = _build_breast_mask(image)
    label_map = _compute_superpixels(image, breast_mask, num_superpixels, compactness, sigma)
    label_map, _ = _filter_and_reindex_labels(label_map, breast_mask, min_superpixel_area)

    pooling_matrix, feature_label_map = _build_feature_grid_pooling(
        label_map, feature_height=feature_height, feature_width=feature_width
    )
    num_nodes = int(pooling_matrix.shape[0])
    adjacency, edge_index = _build_superpixel_adjacency(feature_label_map, num_nodes=num_nodes)
    node_area_pixels, node_centroids_image, node_centroids_feature = _compute_node_stats(
        label_map, feature_label_map, num_nodes=num_nodes
    )

    metadata = {
        "image_id": os.path.basename(image_path),
        "image_shape": tuple(image.shape),
        "feature_grid_shape": (feature_height, feature_width),
        "num_nodes": num_nodes,
        "feature_label_map": torch.from_numpy(feature_label_map.astype(np.int16)),
        "pooling_matrix": torch.from_numpy(pooling_matrix.astype(np.float32)),
        "adjacency": torch.from_numpy(adjacency.astype(np.float32)),
        "edge_index": torch.from_numpy(edge_index.astype(np.int64)),
        "node_area_pixels": torch.from_numpy(node_area_pixels.astype(np.float32)),
        "node_centroids_image": torch.from_numpy(node_centroids_image.astype(np.float32)),
        "node_centroids_feature": torch.from_numpy(node_centroids_feature.astype(np.float32)),
        "cache_config": {
            "num_superpixels": int(num_superpixels),
            "compactness": float(compactness),
            "sigma": float(sigma),
            "min_superpixel_area": int(min_superpixel_area),
            "feature_height": int(feature_height),
            "feature_width": int(feature_width),
        },
    }
    return metadata


def build_cache_for_split(data_root, cache_root, split, num_superpixels, compactness, sigma, min_superpixel_area, feature_height, feature_width):
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    out_dir = os.path.join(cache_root, split)
    os.makedirs(out_dir, exist_ok=True)

    summary = defaultdict(int)

    for filename in sorted(os.listdir(split_dir)):
        if not filename.lower().endswith(".png"):
            continue

        image_path = os.path.join(split_dir, filename)
        metadata = build_superpixel_metadata(
            image_path=image_path,
            num_superpixels=num_superpixels,
            compactness=compactness,
            sigma=sigma,
            min_superpixel_area=min_superpixel_area,
            feature_height=feature_height,
            feature_width=feature_width,
        )

        output_name = f"{os.path.splitext(filename)[0]}.superpixel_graph.pt"
        output_path = os.path.join(out_dir, output_name)
        torch.save(metadata, output_path)
        summary["num_images"] += 1

    print(f"[INFO] Finished split={split}, cached {summary['num_images']} images to {out_dir}")


def main():
    args = parse_arguments()

    os.makedirs(args.cache_root, exist_ok=True)

    for split in args.splits:
        build_cache_for_split(
            data_root=args.data_root,
            cache_root=args.cache_root,
            split=split,
            num_superpixels=args.num_superpixels,
            compactness=args.compactness,
            sigma=args.sigma,
            min_superpixel_area=args.min_superpixel_area,
            feature_height=args.feature_height,
            feature_width=args.feature_width,
        )


if __name__ == "__main__":
    main()
