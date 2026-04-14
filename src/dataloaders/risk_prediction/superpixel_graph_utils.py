import os

import torch


def _metadata_path(cache_root, split, image_id):
    stem, _ = os.path.splitext(image_id)
    return os.path.join(cache_root, split, f"{stem}.superpixel_graph.pt")


def _feature_cache_path(feature_cache_root, split, image_id):
    stem, _ = os.path.splitext(image_id)
    return os.path.join(feature_cache_root, split, f"{stem}.resnet18_feature.pt")


def _load_single_metadata(cache_root, split, image_id):
    path = _metadata_path(cache_root, split, image_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Superpixel graph metadata not found for {image_id}: {path}")
    return torch.load(path, map_location="cpu")


def _load_feature_map(feature_cache_root, split, image_id, device):
    path = _feature_cache_path(feature_cache_root, split, image_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ResNet feature cache not found for {image_id}: {path}")

    payload = torch.load(path, map_location="cpu")
    feature_map = payload["feature_map"] if isinstance(payload, dict) else payload
    return feature_map.to(device=device, dtype=torch.float32)


def _pool_superpixel_features(feature_map, pooling_matrix):
    feature_flat = feature_map.flatten(1).transpose(0, 1)
    return torch.matmul(pooling_matrix, feature_flat)


def build_cached_superpixel_graph_batch(cache_root, feature_cache_root, split, image_ids, device):
    """
    Load cached ResNet feature maps and superpixel graph metadata.

    Returns:
        dict with
        - node_features: (B, Nmax, C)
        - node_coords: (B, Nmax, 2)
        - node_mask: (B, Nmax)
        - adjacency: (B, Nmax, Nmax)
        - pooling_matrix: (B, Nmax, Hf*Wf)
        - node_area_pixels: (B, Nmax, 1)
    """
    feature_maps = [_load_feature_map(feature_cache_root, split, image_id, device=device) for image_id in image_ids]
    metadatas = [_load_single_metadata(cache_root, split, image_id) for image_id in image_ids]

    channels = int(feature_maps[0].shape[0])
    feat_h = int(feature_maps[0].shape[1])
    feat_w = int(feature_maps[0].shape[2])
    max_nodes = max(int(metadata["num_nodes"]) for metadata in metadatas)
    device = feature_maps[0].device
    dtype = feature_maps[0].dtype

    node_features_batch = torch.zeros((len(image_ids), max_nodes, channels), dtype=dtype, device=device)
    node_coords_batch = torch.zeros((len(image_ids), max_nodes, 2), dtype=dtype, device=device)
    node_mask_batch = torch.zeros((len(image_ids), max_nodes), dtype=torch.bool, device=device)
    adjacency_batch = torch.zeros((len(image_ids), max_nodes, max_nodes), dtype=dtype, device=device)
    pooling_matrix_batch = torch.zeros((len(image_ids), max_nodes, feat_h * feat_w), dtype=dtype, device=device)
    node_area_pixels_batch = torch.zeros((len(image_ids), max_nodes, 1), dtype=dtype, device=device)

    for batch_idx, (feature_map, metadata) in enumerate(zip(feature_maps, metadatas)):
        num_nodes = int(metadata["num_nodes"])
        if num_nodes == 0:
            raise ValueError(f"Cached superpixel graph has zero nodes for {image_ids[batch_idx]}")

        pooling_matrix = metadata["pooling_matrix"].to(device=device, dtype=dtype)
        pooled_features = _pool_superpixel_features(feature_map, pooling_matrix)

        node_features_batch[batch_idx, :num_nodes] = pooled_features
        node_coords_batch[batch_idx, :num_nodes] = metadata["node_centroids_feature"].to(device=device, dtype=dtype)
        node_mask_batch[batch_idx, :num_nodes] = True
        adjacency_batch[batch_idx, :num_nodes, :num_nodes] = metadata["adjacency"].to(device=device, dtype=dtype)
        pooling_matrix_batch[batch_idx, :num_nodes] = pooling_matrix
        node_area = metadata["node_area_pixels"].to(device=device, dtype=dtype).unsqueeze(-1)
        node_area_pixels_batch[batch_idx, :num_nodes] = node_area

    return {
        "node_features": node_features_batch,
        "node_coords": node_coords_batch,
        "node_mask": node_mask_batch,
        "adjacency": adjacency_batch,
        "pooling_matrix": pooling_matrix_batch,
        "node_area_pixels": node_area_pixels_batch,
    }
