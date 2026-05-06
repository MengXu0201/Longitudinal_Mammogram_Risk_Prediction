import argparse
import os
import random

import kornia.augmentation as K
import torch
from torch.utils.data import DataLoader

from src.dataloaders.pretraining.dataset_embed_moco import EMBEDMoCoPretrainingDataset
from src.train.train_moco_pretraining import train_moco_pretraining


class MammoMoCoAugmentations(torch.nn.Module):
    """
    Conservative augmentation policy for cropped, uniform-orientation mammograms.
    """

    def __init__(self):
        super().__init__()
        self.geometry = torch.nn.Sequential(
            K.RandomRotation(degrees=10.0, p=0.7),
            K.RandomAffine(degrees=0.0, translate=(0.05, 0.05), scale=(0.95, 1.05), p=0.7),
        )
        self.blur = K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=0.3)

    def forward(self, x):
        x = self.geometry(x)
        x = self.blur(x)

        if torch.rand(1).item() < 0.3:
            brightness = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.95, 1.05)
            x = x * brightness

        if torch.rand(1).item() < 0.3:
            contrast = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.95, 1.05)
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * contrast + mean

        if torch.rand(1).item() < 0.3:
            x = x + torch.randn_like(x) * 0.02

        return x


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training config for MoCo mammogram pretraining")

    parser.add_argument("--csv_file", type=str, default="", help="Optional CSV path for bookkeeping")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of dataset images")
    parser.add_argument("--path_out_dir", type=str, required=True, help="Output directory for saving models")
    parser.add_argument("--log_dir", type=str, default="", help="Optional directory for saving log files")
    parser.add_argument("--id_training", type=str, required=True, help="Unique training run ID")
    parser.add_argument("--dataset", type=str, default="EMBED", help="Dataset name for bookkeeping")

    parser.add_argument("--augmentations", type=str, required=True, help="Enable augmentation if 'True'")
    parser.add_argument("--use_scheduler", type=str, required=True, help="Use learning rate scheduler if 'True'")
    parser.add_argument("--save_debug_augs", type=str, default="True", help="Save debug augmentation examples if 'True'")

    parser.add_argument("--patience_lr_scheduler", default=5, type=int, help="Patience epochs for LR scheduler")
    parser.add_argument("--lr_decay", default=0.5, type=float, help="Learning rate decay factor")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--pin_memory", default=True, type=bool, help="Use pin_memory in DataLoader")

    parser.add_argument("--queue_size", default=16384, type=int, help="MoCo queue size")
    parser.add_argument("--momentum", default=0.999, type=float, help="Momentum coefficient for key encoder")
    parser.add_argument("--temperature", default=0.07, type=float, help="InfoNCE temperature")
    parser.add_argument("--projection_dim", default=128, type=int, help="Projection head output dimension")
    parser.add_argument("--projector_hidden_dim", default=512, type=int, help="Projection head hidden dimension")

    parser.add_argument("--seed", default=2023, type=int, help="Random seed for reproducibility")
    return parser.parse_args()


def build_augmentation_pipeline(enable_augmentations):
    if enable_augmentations != "True":
        return None

    return MammoMoCoAugmentations()


def main():
    args = parse_arguments()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transforms_train = build_augmentation_pipeline(args.augmentations)
    print(f"Train augmentations: {transforms_train}")

    train_dataset = EMBEDMoCoPretrainingDataset(
        image_dir=args.data_root,
        mode="train",
        transform_q=transforms_train,
        transform_k=transforms_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
        pin_memory=args.pin_memory,
    )

    model_path = os.path.join(
        args.path_out_dir,
        f"model_moco_pretraining_training_id-{args.id_training}_last_epoch.pth",
    )
    log_root = args.log_dir if args.log_dir else args.path_out_dir
    os.makedirs(args.path_out_dir, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    log_path = os.path.join(
        log_root,
        f"train_moco_pretraining_training_id-{args.id_training}.log",
    )

    print(f"Training on device: {device}")
    print(f"Model will be saved to: {model_path}")
    print(f"Log will be saved to: {log_path}")

    train_moco_pretraining(
        train_loader=train_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        path_logger=log_path,
        path_model=model_path,
        id=args.id_training,
        use_scheduler=args.use_scheduler,
        path_out_dir=args.path_out_dir,
        patience_lr_scheduler=args.patience_lr_scheduler,
        lr_decay=args.lr_decay,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        save_debug_augs=args.save_debug_augs,
        run_config=vars(args),
    )


if __name__ == "__main__":
    main()
