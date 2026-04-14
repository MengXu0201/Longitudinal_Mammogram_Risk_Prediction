import argparse
import os
import random

import torch
from torch.utils.data import DataLoader

from src.dataloaders.risk_prediction.dataset_embed_graph import BreastCancerRiskGraphDataset
from src.dataloaders.risk_prediction.graph_cached_collate import SuperpixelGraphCollator
from src.train.train_risk_prediction_graph_superpixel import train_val_graph_superpixel_baseline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training config for superpixel graph breast cancer risk prediction")

    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with dataset info")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of dataset images")
    parser.add_argument("--cache_root", type=str, required=True, help="Root directory of cached graph metadata")
    parser.add_argument("--feature_cache_root", type=str, required=True, help="Root directory of cached frozen ResNet feature maps")
    parser.add_argument("--path_out_dir", type=str, required=True, help="Output directory for saving models and logs")
    parser.add_argument("--id_training", type=str, required=True, help="Unique training run ID")
    parser.add_argument("--dataset", type=str, default="EMBED", help="Dataset to use (EMBED or CSAW)")

    parser.add_argument("--augmentations", type=str, required=True, help="Enable data augmentation if 'True'")
    parser.add_argument("--use_scheduler", type=str, required=True, help="Use learning rate scheduler if 'True'")

    parser.add_argument("--patience_lr_scheduler", default=5, type=int, help="Patience epochs for LR scheduler")
    parser.add_argument("--patience", default=15, type=int, help="Patience epochs for early stopping")
    parser.add_argument("--accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--lr_decay", default=0.5, type=float, help="Learning rate decay factor")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")

    parser.add_argument("--batch_size", default=12, type=int, help="Batch size for training and validation")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--schuffle", default=True, type=bool, help="Shuffle training data")
    parser.add_argument("--pin_memory", default=True, type=bool, help="Use pin_memory in DataLoader")

    parser.add_argument("--hidden_dim", default=512, type=int, help="Graph hidden dimension")
    parser.add_argument("--num_graph_layers", default=2, type=int, help="Number of graph message-passing layers")

    parser.add_argument("--seed", default=2023, type=int, help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_arguments()

    os.makedirs(args.path_out_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.augmentations == "True":
        raise ValueError("Graph cached training does not support online image augmentations.")

    if args.dataset.upper() != "EMBED":
        raise ValueError("Superpixel cached graph dataset is currently implemented for EMBED only.")

    train_dataset = BreastCancerRiskGraphDataset(args.csv_file, args.data_root, "train")
    validation_dataset = BreastCancerRiskGraphDataset(args.csv_file, args.data_root, "val")

    train_collate = SuperpixelGraphCollator(
        cache_root=args.cache_root,
        feature_cache_root=args.feature_cache_root,
        split="train",
    )
    validation_collate = SuperpixelGraphCollator(
        cache_root=args.cache_root,
        feature_cache_root=args.feature_cache_root,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.schuffle,
        pin_memory=args.pin_memory,
        collate_fn=train_collate,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
        collate_fn=validation_collate,
    )

    model_path = f"model_risk_prediction_training_id_{args.id_training}_last_epoch.pth"
    log_path = f"train_risk_prediction_training_id_{args.id_training}.log"
    path_out_model = os.path.join(args.path_out_dir, model_path)
    path_logger = os.path.join(args.path_out_dir, log_path)

    print(f"Training on device: {device}")
    print(f"Model will be saved to: {path_out_model}")
    print(f"Log will be saved to: {path_logger}")

    train_val_graph_superpixel_baseline(
        train_loader=train_loader,
        valid_loader=validation_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        path_loggger=path_logger,
        path_model=path_out_model,
        id_training=args.id_training,
        use_scheduler=args.use_scheduler,
        out_dir=args.path_out_dir,
        accumulation_steps=args.accumulation_steps,
        patience_lr_scheduler=args.patience_lr_scheduler,
        patience=args.patience,
        lr_decay=args.lr_decay,
        hidden_dim=args.hidden_dim,
        num_graph_layers=args.num_graph_layers,
    )


if __name__ == "__main__":
    main()
