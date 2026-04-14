import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataloaders.risk_prediction.dataset_embed_graph import BreastCancerRiskGraphDataset
from src.dataloaders.risk_prediction.graph_cached_collate import SuperpixelGraphCollator
from src.evaluate.test_risk_prediction_graph_superpixel import test_graph_superpixel_baseline_risk


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test script for superpixel graph breast cancer risk prediction.")

    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with data info.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data root.")
    parser.add_argument("--cache_root", type=str, required=True, help="Path to graph metadata cache root.")
    parser.add_argument("--feature_cache_root", type=str, required=True, help="Path to cached frozen ResNet feature maps.")
    parser.add_argument("--path_out_dir", type=str, required=True, help="Directory for saved models.")
    parser.add_argument("--path_test_folder", type=str, required=True, help="Output folder for test results.")

    parser.add_argument("--id_training", type=str, required=True, help="ID of training run.")
    parser.add_argument("--num_epoch", type=int, required=True, help="Number of epochs (used to decide model path).")
    parser.add_argument("--dataset", type=str, choices=["CSAW", "EMBED"], required=True, help="Dataset to use.")
    parser.add_argument("--early_stop", type=str, default="False", help="Use early stopping model instead of best or last.")

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--shuffle", default=False, type=bool)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--seed", default=2023, type=int)

    return parser.parse_args()


def main():
    args = parse_arguments()

    os.makedirs(args.path_test_folder, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset != "EMBED":
        raise ValueError("Superpixel cached graph evaluation is currently implemented for EMBED only.")

    print("Creating graph test dataset...")
    test_dataset = BreastCancerRiskGraphDataset(args.csv_file, args.data_root, "test")
    test_collate = SuperpixelGraphCollator(
        cache_root=args.cache_root,
        feature_cache_root=args.feature_cache_root,
        split="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        collate_fn=test_collate,
    )

    if args.early_stop == "True":
        model_filename = f"early_stopping_risk_prediction_id-{args.id_training}.pth"
    elif args.num_epoch == 99:
        model_filename = f"model_risk_prediction_training_id_{args.id_training}_last_epoch.pth"
    else:
        model_filename = f"best_model_risk_prediction_id-{args.id_training}.pth"

    path_model = os.path.join(args.path_out_dir, model_filename)
    path_logger = os.path.join(args.path_test_folder, f"test_risk_prediction_training_id_{args.id_training}.log")

    print(f"Model path: {path_model}")
    print(f"Logger path: {path_logger}")

    test_graph_superpixel_baseline_risk(
        test_loader=test_loader,
        device=device,
        path_model=path_model,
        out_dir=args.path_test_folder,
        path_logger=path_logger,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
