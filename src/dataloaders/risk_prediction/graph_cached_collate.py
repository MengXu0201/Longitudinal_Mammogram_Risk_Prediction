import torch

from src.dataloaders.risk_prediction.graph_utils import build_cached_masked_grid_graph_batch
from src.dataloaders.risk_prediction.superpixel_graph_utils import build_cached_superpixel_graph_batch


def _stack_float(batch, key):
    return torch.as_tensor([sample[key] for sample in batch], dtype=torch.float32)


def _stack_long(batch, key):
    return torch.as_tensor([sample[key] for sample in batch], dtype=torch.long)


def _stack_array(batch, key):
    return torch.stack(
        [torch.as_tensor(sample[key], dtype=torch.float32) for sample in batch],
        dim=0,
    )


class MaskedGridGraphCollator:
    """
    Collate graph risk samples and load cached masked-grid graph tensors.
    """

    def __init__(self, cache_root, feature_cache_root, split):
        self.cache_root = cache_root
        self.feature_cache_root = feature_cache_root
        self.split = split

    def __call__(self, batch):
        current_image_ids = [sample["current_image_id"] for sample in batch]
        previous_image_ids = [sample["previous_image_id"] for sample in batch]

        current_graph = build_cached_masked_grid_graph_batch(
            cache_root=self.cache_root,
            feature_cache_root=self.feature_cache_root,
            split=self.split,
            image_ids=current_image_ids,
            device="cpu",
        )
        previous_graph = build_cached_masked_grid_graph_batch(
            cache_root=self.cache_root,
            feature_cache_root=self.feature_cache_root,
            split=self.split,
            image_ids=previous_image_ids,
            device="cpu",
        )

        return {
            "current_graph": current_graph,
            "previous_graph": previous_graph,
            "current_image_id": current_image_ids,
            "previous_image_id": previous_image_ids,
            "event_observed": _stack_float(batch, "event_observed"),
            "event_times": _stack_float(batch, "event_times"),
            "time_gap": _stack_long(batch, "time_gap"),
            "y_mask": _stack_array(batch, "y_mask"),
            "y_mask_prior": _stack_array(batch, "y_mask_prior"),
            "target": _stack_array(batch, "target"),
            "target_prior": _stack_array(batch, "target_prior"),
            "density": [sample["density"] for sample in batch],
        }


class SuperpixelGraphCollator:
    """
    Collate graph risk samples and load cached superpixel graph tensors.
    """

    def __init__(self, cache_root, feature_cache_root, split, node_feature_mode="superpixel_pool"):
        self.cache_root = cache_root
        self.feature_cache_root = feature_cache_root
        self.split = split
        self.node_feature_mode = node_feature_mode

    def __call__(self, batch):
        current_image_ids = [sample["current_image_id"] for sample in batch]
        previous_image_ids = [sample["previous_image_id"] for sample in batch]

        current_graph = build_cached_superpixel_graph_batch(
            cache_root=self.cache_root,
            feature_cache_root=self.feature_cache_root,
            split=self.split,
            image_ids=current_image_ids,
            device="cpu",
            node_feature_mode=self.node_feature_mode,
        )
        previous_graph = build_cached_superpixel_graph_batch(
            cache_root=self.cache_root,
            feature_cache_root=self.feature_cache_root,
            split=self.split,
            image_ids=previous_image_ids,
            device="cpu",
            node_feature_mode=self.node_feature_mode,
        )

        return {
            "current_graph": current_graph,
            "previous_graph": previous_graph,
            "current_image_id": current_image_ids,
            "previous_image_id": previous_image_ids,
            "event_observed": _stack_float(batch, "event_observed"),
            "event_times": _stack_float(batch, "event_times"),
            "time_gap": _stack_long(batch, "time_gap"),
            "y_mask": _stack_array(batch, "y_mask"),
            "y_mask_prior": _stack_array(batch, "y_mask_prior"),
            "target": _stack_array(batch, "target"),
            "target_prior": _stack_array(batch, "target_prior"),
            "density": [sample["density"] for sample in batch],
        }
