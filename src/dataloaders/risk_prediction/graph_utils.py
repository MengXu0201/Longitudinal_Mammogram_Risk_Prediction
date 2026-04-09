from collections import deque

import torch
import torch.nn.functional as F


def _largest_connected_component(mask_2d):
    """
    Keep only the largest connected component on a low-resolution boolean mask.

    The graph baseline only needs breast support at feature-map resolution, so
    this low-resolution connected-component cleanup is enough to remove isolated
    background islands without touching the original dataset PNGs.
    """
    device = mask_2d.device
    mask_cpu = mask_2d.detach().to(torch.bool).cpu()
    height, width = mask_cpu.shape
    visited = torch.zeros((height, width), dtype=torch.bool)

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    best_component = []

    for row in range(height):
        for col in range(width):
            if not mask_cpu[row, col] or visited[row, col]:
                continue

            queue = deque([(row, col)])
            visited[row, col] = True
            current_component = []

            while queue:
                cur_row, cur_col = queue.popleft()
                current_component.append((cur_row, cur_col))

                for d_row, d_col in neighbors:
                    nxt_row = cur_row + d_row
                    nxt_col = cur_col + d_col
                    if nxt_row < 0 or nxt_row >= height or nxt_col < 0 or nxt_col >= width:
                        continue
                    if visited[nxt_row, nxt_col] or not mask_cpu[nxt_row, nxt_col]:
                        continue

                    visited[nxt_row, nxt_col] = True
                    queue.append((nxt_row, nxt_col))

            if len(current_component) > len(best_component):
                best_component = current_component

    if not best_component:
        return mask_2d

    out = torch.zeros((height, width), dtype=torch.bool)
    for row, col in best_component:
        out[row, col] = True

    return out.to(device=device)


def _foreground_mask_from_image(image_2d, feature_size, threshold_eps=1e-6, min_area_ratio=0.05):
    """
    Build a breast-area mask directly from the already processed PNG tensor.

    The uniform-orientation PNGs are saved with background near the minimum
    intensity of the image. We threshold relative to the per-image minimum,
    then downsample to feature-map resolution and keep the largest component.
    """
    image_min = image_2d.amin()
    mask_full = image_2d > (image_min + threshold_eps)
    mask_lowres = F.interpolate(
        mask_full[None, None].float(),
        size=feature_size,
        mode="area",
    ).squeeze(0).squeeze(0)
    mask_lowres = mask_lowres > min_area_ratio

    if torch.count_nonzero(mask_lowres) == 0:
        mask_lowres = F.interpolate(
            mask_full[None, None].float(),
            size=feature_size,
            mode="nearest",
        ).squeeze(0).squeeze(0) > 0

    if torch.count_nonzero(mask_lowres) == 0:
        mask_lowres = torch.ones(feature_size, dtype=torch.bool, device=image_2d.device)

    return _largest_connected_component(mask_lowres)


def _build_adjacency_from_mask(mask_2d, connectivity=8):
    """
    Build a dense adjacency matrix for the valid masked-grid nodes.
    """
    valid_positions = torch.nonzero(mask_2d, as_tuple=False)
    num_nodes = valid_positions.shape[0]
    device = mask_2d.device

    if num_nodes == 0:
        return (
            torch.zeros((0, 2), dtype=torch.long, device=device),
            torch.zeros((0, 0), dtype=torch.float32, device=device),
            {},
        )

    node_lookup = {}
    for node_idx, (row, col) in enumerate(valid_positions.tolist()):
        node_lookup[(row, col)] = node_idx

    adjacency = torch.eye(num_nodes, dtype=torch.float32, device=device)

    if connectivity == 4:
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    for node_idx, (row, col) in enumerate(valid_positions.tolist()):
        for d_row, d_col in neighbor_offsets:
            nbr_key = (row + d_row, col + d_col)
            nbr_idx = node_lookup.get(nbr_key)
            if nbr_idx is not None:
                adjacency[node_idx, nbr_idx] = 1.0
                adjacency[nbr_idx, node_idx] = 1.0

    return valid_positions.to(device=device), adjacency, node_lookup


def build_masked_grid_graph_batch(
    images,
    feature_maps,
    connectivity=8,
    threshold_eps=1e-6,
    min_area_ratio=0.05,
):
    """
    Convert a batch of images + CNN feature maps into a dense padded graph batch.

    Returns:
        dict with
        - node_features: (B, Nmax, C)
        - node_coords: (B, Nmax, 2)
        - node_mask: (B, Nmax)
        - adjacency: (B, Nmax, Nmax)
        - valid_grid_mask: (B, Hf, Wf)
    """
    batch_size, channels, feat_h, feat_w = feature_maps.shape
    if images.dim() != 4 or images.shape[1] != 1:
        raise ValueError("Expected grayscale images with shape (B, 1, H, W).")

    sample_graphs = []
    max_nodes = 0

    for batch_idx in range(batch_size):
        image_2d = images[batch_idx, 0]
        feature_map = feature_maps[batch_idx]

        valid_grid_mask = _foreground_mask_from_image(
            image_2d=image_2d,
            feature_size=(feat_h, feat_w),
            threshold_eps=threshold_eps,
            min_area_ratio=min_area_ratio,
        )

        valid_positions, adjacency, _ = _build_adjacency_from_mask(
            valid_grid_mask, connectivity=connectivity
        )

        if valid_positions.shape[0] == 0:
            valid_grid_mask = torch.ones((feat_h, feat_w), dtype=torch.bool, device=feature_map.device)
            valid_positions, adjacency, _ = _build_adjacency_from_mask(
                valid_grid_mask, connectivity=connectivity
            )

        node_features = feature_map.permute(1, 2, 0)[valid_grid_mask]

        coords = valid_positions.to(dtype=feature_map.dtype)
        coords[:, 0] = coords[:, 0] / max(feat_h - 1, 1)
        coords[:, 1] = coords[:, 1] / max(feat_w - 1, 1)
        coords = coords[:, [1, 0]]

        sample_graphs.append(
            {
                "node_features": node_features,
                "node_coords": coords,
                "adjacency": adjacency,
                "valid_grid_mask": valid_grid_mask,
            }
        )
        max_nodes = max(max_nodes, node_features.shape[0])

    node_features_batch = feature_maps.new_zeros((batch_size, max_nodes, channels))
    node_coords_batch = feature_maps.new_zeros((batch_size, max_nodes, 2))
    node_mask_batch = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=feature_maps.device)
    adjacency_batch = feature_maps.new_zeros((batch_size, max_nodes, max_nodes))
    valid_grid_masks = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.bool, device=feature_maps.device)

    for batch_idx, graph in enumerate(sample_graphs):
        num_nodes = graph["node_features"].shape[0]
        node_features_batch[batch_idx, :num_nodes] = graph["node_features"]
        node_coords_batch[batch_idx, :num_nodes] = graph["node_coords"]
        node_mask_batch[batch_idx, :num_nodes] = True
        adjacency_batch[batch_idx, :num_nodes, :num_nodes] = graph["adjacency"]
        valid_grid_masks[batch_idx] = graph["valid_grid_mask"]

    return {
        "node_features": node_features_batch,
        "node_coords": node_coords_batch,
        "node_mask": node_mask_batch,
        "adjacency": adjacency_batch,
        "valid_grid_mask": valid_grid_masks,
    }
