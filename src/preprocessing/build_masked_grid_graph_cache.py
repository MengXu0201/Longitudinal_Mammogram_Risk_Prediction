import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders.risk_prediction.graph_utils import _foreground_mask_from_image


DEFAULT_DATA_ROOT = "/mnt/cv_data/users/mengxu/EMBED_Split_Cropped_PNG_Uniform_Orientation"
DEFAULT_FEATURE_CACHE_ROOT = (
    "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/"
    "output_feature_cache/resnet18_no_alignment_uniform_orientation_cropped_early_stop"
)
DEFAULT_CACHE_ROOT = (
    "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/"
    "output_graph_cache/masked_grid_uniform_orientation"
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build offline masked-grid graph metadata cache.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing train/val/test PNG folders.",
    )
    parser.add_argument(
        "--feature_cache_root",
        type=str,
        default=DEFAULT_FEATURE_CACHE_ROOT,
        help="Root directory of cached ResNet feature maps.",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=DEFAULT_CACHE_ROOT,
        help="Output root for masked-grid graph metadata.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to process.")
    parser.add_argument("--similarity_threshold", type=float, default=0.70, help="Cosine similarity threshold for edges.")
    parser.add_argument("--k_min", type=int, default=2, help="Minimum fallback neighbors per node.")
    parser.add_argument("--k_max", type=int, default=16, help="Maximum neighbors per node after thresholding.")
    parser.add_argument("--min_area_ratio", type=float, default=0.05, help="Minimum foreground area ratio per feature cell.")
    parser.add_argument("--threshold_eps", type=float, default=1e-6, help="Foreground threshold epsilon above image minimum.")
    parser.add_argument("--max_images_per_split", type=int, default=None, help="Optional debugging limit.")
    return parser.parse_args()


def _feature_cache_path(feature_cache_root, split, filename):
    stem, _ = os.path.splitext(filename)
    return os.path.join(feature_cache_root, split, f"{stem}.resnet18_feature.pt")


def _metadata_path(cache_root, split, filename):
    stem, _ = os.path.splitext(filename)
    return os.path.join(cache_root, split, f"{stem}.masked_grid_graph.pt")


def _load_feature_map(feature_cache_root, split, filename):
    path = _feature_cache_path(feature_cache_root, split, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature cache not found for {filename}: {path}")

    payload = torch.load(path, map_location="cpu")
    feature_map = payload["feature_map"] if isinstance(payload, dict) else payload
    return feature_map.to(dtype=torch.float32), path


def _load_image_for_mask(image_path):
    image = np.array(Image.open(image_path))
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image at {image_path}, got shape {image.shape}")
    return torch.from_numpy(image.astype(np.float32))


def _node_coords_from_positions(valid_positions, feat_h, feat_w, dtype=torch.float32):
    coords = valid_positions.to(dtype=dtype)
    coords[:, 0] = coords[:, 0] / max(feat_h - 1, 1)
    coords[:, 1] = coords[:, 1] / max(feat_w - 1, 1)
    return coords[:, [1, 0]]


def _build_similarity_adjacency(node_features, similarity_threshold, k_min, k_max):
    num_nodes = int(node_features.shape[0])
    if num_nodes == 0:
        return torch.zeros((0, 0), dtype=torch.float32), torch.zeros((2, 0), dtype=torch.long), {
            "avg_degree": 0.0,
            "isolated_before_fallback": 0,
        }

    if num_nodes == 1:
        adjacency = torch.ones((1, 1), dtype=torch.float32)
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        return adjacency, edge_index, {
            "avg_degree": 1.0,
            "isolated_before_fallback": 0,
        }

    features = F.normalize(node_features.to(dtype=torch.float32), p=2, dim=1)
    similarity = torch.matmul(features, features.T)
    similarity.fill_diagonal_(-float("inf"))

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    isolated_before_fallback = 0
    max_neighbors = max(1, min(int(k_max), num_nodes - 1))
    min_neighbors = max(0, min(int(k_min), max_neighbors))

    for node_idx in range(num_nodes):
        row = similarity[node_idx]
        threshold_candidates = torch.nonzero(row >= similarity_threshold, as_tuple=False).flatten()

        if threshold_candidates.numel() == 0:
            isolated_before_fallback += 1

        if threshold_candidates.numel() > 0:
            candidate_scores = row[threshold_candidates]
            order = torch.argsort(candidate_scores, descending=True)
            selected = threshold_candidates[order[:max_neighbors]]
        else:
            selected = torch.empty((0,), dtype=torch.long)

        if selected.numel() < min_neighbors:
            fallback_count = min_neighbors - selected.numel()
            fallback_order = torch.argsort(row, descending=True)
            fallback = []
            selected_set = set(int(x) for x in selected.tolist())
            for neighbor_idx in fallback_order.tolist():
                if neighbor_idx == node_idx or neighbor_idx in selected_set:
                    continue
                fallback.append(neighbor_idx)
                selected_set.add(neighbor_idx)
                if len(fallback) >= fallback_count:
                    break
            if fallback:
                selected = torch.cat([selected, torch.tensor(fallback, dtype=torch.long)])

        for neighbor_idx in selected.tolist():
            raw_weight = float(similarity[node_idx, neighbor_idx])
            weight = max(raw_weight, 1e-6)
            adjacency[node_idx, neighbor_idx] = max(float(adjacency[node_idx, neighbor_idx]), weight)
            adjacency[neighbor_idx, node_idx] = max(float(adjacency[neighbor_idx, node_idx]), weight)

    adjacency.fill_diagonal_(1.0)
    edge_rows, edge_cols = torch.nonzero(adjacency > 0, as_tuple=True)
    edge_index = torch.stack([edge_rows, edge_cols], dim=0).to(dtype=torch.long)
    avg_degree = float((adjacency > 0).sum(dim=1).float().mean().item())

    return adjacency, edge_index, {
        "avg_degree": avg_degree,
        "isolated_before_fallback": int(isolated_before_fallback),
    }


def build_masked_grid_metadata(
    image_path,
    feature_cache_root,
    split,
    cache_root,
    similarity_threshold,
    k_min,
    k_max,
    min_area_ratio,
    threshold_eps,
):
    filename = os.path.basename(image_path)
    feature_map, feature_cache_path = _load_feature_map(feature_cache_root, split, filename)
    _, feat_h, feat_w = feature_map.shape

    image_2d = _load_image_for_mask(image_path)
    valid_grid_mask = _foreground_mask_from_image(
        image_2d=image_2d,
        feature_size=(feat_h, feat_w),
        threshold_eps=threshold_eps,
        min_area_ratio=min_area_ratio,
    )

    valid_positions = torch.nonzero(valid_grid_mask, as_tuple=False).to(dtype=torch.long)
    if valid_positions.shape[0] == 0:
        valid_grid_mask = torch.ones((feat_h, feat_w), dtype=torch.bool)
        valid_positions = torch.nonzero(valid_grid_mask, as_tuple=False).to(dtype=torch.long)

    node_features = feature_map.permute(1, 2, 0)[valid_positions[:, 0], valid_positions[:, 1]]
    node_coords = _node_coords_from_positions(valid_positions, feat_h, feat_w, dtype=torch.float32)
    adjacency, edge_index, graph_stats = _build_similarity_adjacency(
        node_features=node_features,
        similarity_threshold=similarity_threshold,
        k_min=k_min,
        k_max=k_max,
    )

    metadata = {
        "image_id": filename,
        "split": split,
        "source_image_root": os.path.dirname(os.path.dirname(image_path)),
        "feature_cache_root": feature_cache_root,
        "feature_cache_path": feature_cache_path,
        "image_shape": tuple(image_2d.shape),
        "feature_map_shape": tuple(feature_map.shape),
        "num_nodes": int(valid_positions.shape[0]),
        "valid_grid_mask": valid_grid_mask.to(dtype=torch.bool),
        "valid_grid_positions": valid_positions,
        "node_coords": node_coords,
        "adjacency": adjacency,
        "edge_index": edge_index,
        "edge_weight": adjacency[edge_index[0], edge_index[1]],
        "graph_stats": graph_stats,
        "cache_config": {
            "similarity_threshold": float(similarity_threshold),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "min_area_ratio": float(min_area_ratio),
            "threshold_eps": float(threshold_eps),
            "edge_rule": "cosine_similarity_threshold",
        },
    }
    return metadata


def build_cache_for_split(
    data_root,
    feature_cache_root,
    cache_root,
    split,
    similarity_threshold,
    k_min,
    k_max,
    min_area_ratio,
    threshold_eps,
    max_images_per_split=None,
):
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    out_dir = os.path.join(cache_root, split)
    os.makedirs(out_dir, exist_ok=True)

    filenames = sorted(filename for filename in os.listdir(split_dir) if filename.lower().endswith(".png"))
    if max_images_per_split is not None:
        filenames = filenames[:max_images_per_split]

    summary = defaultdict(float)

    for filename in filenames:
        metadata = build_masked_grid_metadata(
            image_path=os.path.join(split_dir, filename),
            feature_cache_root=feature_cache_root,
            split=split,
            cache_root=cache_root,
            similarity_threshold=similarity_threshold,
            k_min=k_min,
            k_max=k_max,
            min_area_ratio=min_area_ratio,
            threshold_eps=threshold_eps,
        )
        torch.save(metadata, _metadata_path(cache_root, split, filename))
        summary["num_images"] += 1
        summary["num_nodes"] += metadata["num_nodes"]
        summary["avg_degree"] += metadata["graph_stats"]["avg_degree"]
        summary["isolated_before_fallback"] += metadata["graph_stats"]["isolated_before_fallback"]

    num_images = max(int(summary["num_images"]), 1)
    print(
        f"[INFO] Finished split={split}, cached {int(summary['num_images'])} images to {out_dir} | "
        f"avg_nodes={summary['num_nodes'] / num_images:.2f} | "
        f"avg_degree={summary['avg_degree'] / num_images:.2f} | "
        f"isolated_before_fallback={int(summary['isolated_before_fallback'])}"
    )


def main():
    args = parse_arguments()
    os.makedirs(args.cache_root, exist_ok=True)

    for split in args.splits:
        build_cache_for_split(
            data_root=args.data_root,
            feature_cache_root=args.feature_cache_root,
            cache_root=args.cache_root,
            split=split,
            similarity_threshold=args.similarity_threshold,
            k_min=args.k_min,
            k_max=args.k_max,
            min_area_ratio=args.min_area_ratio,
            threshold_eps=args.threshold_eps,
            max_images_per_split=args.max_images_per_split,
        )


if __name__ == "__main__":
    main()
