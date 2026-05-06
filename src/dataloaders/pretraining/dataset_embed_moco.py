import os
import re
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.dataloaders.risk_prediction.dataset_embed import imgunit16


class EMBEDMoCoPretrainingDataset(Dataset):
    """
    Self-supervised EMBED dataset for MoCo pretraining on individual images.

    Each sample returns two independently augmented views of the same image.
    """

    FILENAME_PATTERN = re.compile(
        r"^(\d+)_([A-Z]+)_([A-Z0-9]+)_(\d{4}-\d{2}-\d{2})_([A-Za-z0-9]+)_(pos_cancer|pos_nocancer|neg_nocancer)\.png$"
    )

    def __init__(self, image_dir, mode="train", transform_q=None, transform_k=None):
        self.data_dir = image_dir
        self.mode = mode
        self.transform_q = transform_q
        self.transform_k = transform_k if transform_k is not None else transform_q
        self.samples = self._load_image_metadata()

    def _load_image_metadata(self):
        split_dir = os.path.join(self.data_dir, self.mode)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Could not find split directory: {split_dir}")

        samples = []
        for filename in sorted(os.listdir(split_dir)):
            match = self.FILENAME_PATTERN.match(filename)
            if not match:
                continue

            samples.append(
                {
                    "filename": filename,
                    "patient_id": match.group(1),
                    "laterality": match.group(2),
                    "view": match.group(3),
                    "study_date": datetime.strptime(match.group(4), "%Y-%m-%d"),
                    "status_tag": match.group(6),
                    "path": os.path.join(split_dir, filename),
                }
            )

        if not samples:
            raise ValueError(f"No valid PNG images found under: {split_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_normalized_image(self, image_path):
        image_pil = Image.open(image_path)
        image_np = imgunit16(np.array(image_pil))
        image_np = (image_np - 7047.99) / 12005.5
        return torch.from_numpy(image_np).unsqueeze(0).to(torch.float32)

    def _apply_transform(self, image, transform):
        if transform is None:
            return image.clone()

        transformed = transform(image.unsqueeze(0))
        if transformed.dim() == 4:
            transformed = transformed.squeeze(0)
        return transformed.to(torch.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self._load_normalized_image(sample["path"])

        view_q = self._apply_transform(image, self.transform_q)
        view_k = self._apply_transform(image, self.transform_k)

        return {
            "image": image,
            "view_q": view_q,
            "view_k": view_k,
            "image_id": sample["filename"],
            "patient_id": sample["patient_id"],
            "laterality": sample["laterality"],
            "view": sample["view"],
            "status_tag": sample["status_tag"],
        }
