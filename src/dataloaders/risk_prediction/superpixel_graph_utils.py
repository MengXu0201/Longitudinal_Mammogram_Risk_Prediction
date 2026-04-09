import os

import torch


def _metadata_path(cache_root, split, image_id):
    stem, _ = os.path.splitext(image_id)
    return os.path.join(cache_root, split, f"{stem}.superpixel_graph.pt")


def _load_single_metadata(cache_root, split, image_id):
    path = _metadata_path(cache_root, split, image_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Superpixel graph metadata not found for {image_id}: {path}")
    return torch.load(path, map_location="cpu")


def build_superpixel_graph_batch(cache_root, split, image_ids, feature_maps):
    """
    Load cached graph metadata and pool online CNN features into superpixel nodes.

    Returns:
        dict with
        - node_features: (B, Nmax, C)
        - node_coords: (B, Nmax, 2)
        - node_mask: (B, Nmax)
        - adjacency: (B, Nmax, Nmax)
        - pooling_matrix: (B, Nmax, Hf*Wf)
    """
    batch_size, channels, feat_h, feat_w = feature_maps.shape
    feat_flat = feature_maps.flatten(2).transpose(1, 2)  # (B, Hf*Wf, C)

    metadatas = [_load_single_metadata(cache_root, split, image_id) for image_id in image_ids]
    max_nodes = max(int(metadata["num_nodes"]) for metadata in metadatas)
    device = feature_maps.device

    node_features = feature_maps.new_zeros((batch_size, max_nodes, channels))
    node_coords = feature_maps.new_zeros((batch_size, max_nodes, 2))
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
    adjacency = feature_maps.new_zeros((batch_size, max_nodes, max_nodes))
    pooling_matrix = feature_maps.new_zeros((batch_size, max_nodes, feat_h * feat_w))
    node_area_pixels = feature_maps.new_zeros((batch_size, max_nodes, 1))

    for batch_idx, metadata in enumerate(metadatas):
        num_nodes = int(metadata["num_nodes"])
        pool = metadata["pooling_matrix"].to(device=device, dtype=feature_maps.dtype)
        pooled_features = torch.matmul(pool, feat_flat[batch_idx])  # (N, C)

        node_features[batch_idx, :num_nodes] = pooled_features
        node_coords[batch_idx, :num_nodes] = metadata["node_centroids_feature"].to(
            device=device, dtype=feature_maps.dtype
        )
        node_mask[batch_idx, :num_nodes] = True
        adjacency[batch_idx, :num_nodes, :num_nodes] = metadata["adjacency"].to(
            device=device, dtype=feature_maps.dtype
        )
        pooling_matrix[batch_idx, :num_nodes] = pool
        area = metadata["node_area_pixels"].to(device=device, dtype=feature_maps.dtype).unsqueeze(-1)
        node_area_pixels[batch_idx, :num_nodes] = area

    return {
        "node_features": node_features,
        "node_coords": node_coords,
        "node_mask": node_mask,
        "adjacency": adjacency,
        "pooling_matrix": pooling_matrix,
        "node_area_pixels": node_area_pixels,
    }
