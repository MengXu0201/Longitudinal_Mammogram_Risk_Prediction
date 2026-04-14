
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_combined_alignment_risk import RiskModelNoAlignment


DEFAULT_DATA_ROOT = "/mnt/cv_data/users/mengxu/EMBED_Split_Cropped_PNG_Uniform_Orientation"
DEFAULT_CACHE_ROOT = (
    "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/"
    "output_feature_cache/resnet18_no_alignment_uniform_orientation_cropped_early_stop"
)
DEFAULT_CHECKPOINT_PATH = (
    "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/"
    "output_models/risk_prediction_no_alignment_uniform_orientation_cropped/"
    "early_stopping_risk_prediction_id-embed_balanced_split_no_alignment_uniform_orientation_cropped.pth"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build a shared offline ResNet18 feature-map cache for graph baselines."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing train/val/test PNG folders.",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=DEFAULT_CACHE_ROOT,
        help="Output root for cached ResNet feature maps.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the trained no-alignment risk checkpoint.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to process.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction.")
    parser.add_argument("--num_years", type=int, default=5, help="Number of years used by the checkpoint risk head.")
    parser.add_argument(
        "--feature_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for cached feature maps.",
    )
    parser.add_argument(
        "--max_images_per_split",
        type=int,
        default=None,
        help="Optional debugging limit. Omit for full cache generation.",
    )
    parser.add_argument("--seed", type=int, default=2023, help="Random seed for deterministic CUDA settings.")
    return parser.parse_args()


def imgunit16(img):
    """
    Match the risk dataset preprocessing by rescaling intensities to [0, 65535].
    """
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()

    if img_max == img_min:
        return np.zeros_like(img, dtype=np.float32)

    return (img - img_min) / (img_max - img_min) * 65535


def load_image_tensor(image_path):
    image = np.array(Image.open(image_path))

    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image at {image_path}, got shape {image.shape}")

    image = imgunit16(image)
    image = (image - 7047.99) / 12005.5
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(torch.float32)
    return image_tensor


def load_frozen_encoder(checkpoint_path, device, num_years):
    model = RiskModelNoAlignment(num_years=num_years)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model.encoder


def output_path_for_image(cache_root, split, filename):
    stem, _ = os.path.splitext(filename)
    return os.path.join(cache_root, split, f"{stem}.resnet18_feature.pt")


def save_cache_config(args, device):
    os.makedirs(args.cache_root, exist_ok=True)
    config = {
        "cache_type": "resnet18_feature_map",
        "source_image_root": args.data_root,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_type": "early_stopping" if "early_stopping" in os.path.basename(args.checkpoint_path) else "unspecified",
        "model_family": "RiskModelNoAlignment",
        "encoder": "ResNet18Encoder",
        "feature_dtype": args.feature_dtype,
        "expected_feature_shape": [512, 32, 16],
        "image_normalization": {
            "intensity_rescale": "per-image min-max to [0, 65535]",
            "mean": 7047.99,
            "std": 12005.5,
        },
        "splits": args.splits,
        "batch_size": args.batch_size,
        "device": str(device),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    config_path = os.path.join(args.cache_root, "cache_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"[INFO] Saved cache config to {config_path}")


def cache_split_features(data_root, cache_root, split, encoder, device, batch_size, feature_dtype, max_images_per_split=None):
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    out_dir = os.path.join(cache_root, split)
    os.makedirs(out_dir, exist_ok=True)

    filenames = sorted(filename for filename in os.listdir(split_dir) if filename.lower().endswith(".png"))
    if max_images_per_split is not None:
        filenames = filenames[:max_images_per_split]

    storage_dtype = torch.float16 if feature_dtype == "float16" else torch.float32
    total_cached = 0

    for start_idx in range(0, len(filenames), batch_size):
        batch_filenames = filenames[start_idx:start_idx + batch_size]
        image_tensors = []

        for filename in batch_filenames:
            image_path = os.path.join(split_dir, filename)
            image_tensors.append(load_image_tensor(image_path))

        batch = torch.stack(image_tensors, dim=0).to(device)
        batch = batch.repeat(1, 3, 1, 1)

        with torch.inference_mode():
            feature_maps = encoder(batch).detach().cpu().to(storage_dtype)

        for filename, feature_map in zip(batch_filenames, feature_maps):
            feature_map = feature_map.contiguous().clone()
            feature_payload = {
                "image_id": filename,
                "split": split,
                "source_image_root": data_root,
                "feature_shape": tuple(feature_map.shape),
                "feature_dtype": feature_dtype,
                "feature_map": feature_map,
            }
            torch.save(feature_payload, output_path_for_image(cache_root, split, filename))
            total_cached += 1

    print(f"[INFO] Finished split={split}, cached {total_cached} feature maps to {out_dir}")


def main():
    args = parse_arguments()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.cache_root, exist_ok=True)

    encoder = load_frozen_encoder(
        checkpoint_path=args.checkpoint_path,
        device=device,
        num_years=args.num_years,
    )
    save_cache_config(args, device)

    for split in args.splits:
        cache_split_features(
            data_root=args.data_root,
            cache_root=args.cache_root,
            split=split,
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
            feature_dtype=args.feature_dtype,
            max_images_per_split=args.max_images_per_split,
        )


if __name__ == "__main__":
    main()
