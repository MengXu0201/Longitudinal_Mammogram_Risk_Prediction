import json
import os
import time

import torch
import wandb
from torchvision.utils import save_image
from tqdm import tqdm

from src.models.model_moco_pretraining import MoCoPretrainingModel
from src.utils.utils import create_logger


def _save_json_config(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _save_debug_augmentation_examples(output_dir, batch, max_examples=4):
    os.makedirs(output_dir, exist_ok=True)

    originals = batch["image"].detach().cpu()
    queries = batch["view_q"].detach().cpu()
    keys = batch["view_k"].detach().cpu()
    image_ids = batch["image_id"]

    num_examples = min(max_examples, originals.size(0))
    for idx in range(num_examples):
        stem = os.path.splitext(os.path.basename(image_ids[idx]))[0]
        save_image(
            originals[idx],
            os.path.join(output_dir, f"{idx:02d}_{stem}_original.png"),
            normalize=True,
            scale_each=True,
        )
        save_image(
            queries[idx],
            os.path.join(output_dir, f"{idx:02d}_{stem}_query.png"),
            normalize=True,
            scale_each=True,
        )
        save_image(
            keys[idx],
            os.path.join(output_dir, f"{idx:02d}_{stem}_key.png"),
            normalize=True,
            scale_each=True,
        )


def _build_checkpoint_payload(model, optimizer, scheduler, epoch, avg_loss, run_config):
    return {
        "epoch": epoch,
        "train_loss": avg_loss,
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.get_encoder_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "queue_ptr": model.queue_ptr.detach().cpu().clone(),
        "config": run_config,
    }


def train_moco_pretraining(
    train_loader,
    device,
    learning_rate,
    weight_decay,
    num_epochs,
    path_logger,
    path_model,
    id,
    use_scheduler,
    path_out_dir,
    patience_lr_scheduler,
    lr_decay,
    queue_size,
    momentum,
    temperature,
    projection_dim,
    projector_hidden_dim,
    save_debug_augs,
    run_config,
):
    print("[INFO] Training MoCo pretraining model...")
    start_time = time.time()

    os.makedirs(path_out_dir, exist_ok=True)
    logger = create_logger(path_logger)
    logger.info(f"Number of Pretraining Epochs: {num_epochs}")

    model = MoCoPretrainingModel(
        projection_dim=projection_dim,
        projector_hidden_dim=projector_hidden_dim,
        queue_size=queue_size,
        momentum=momentum,
        temperature=temperature,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler == "True":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_decay,
            patience=patience_lr_scheduler,
        )
        print("Scheduler initialized:", scheduler)

    wandb.init(
        project="EMBED_MoCo_Pretraining",
        config={
            "optimizer": optimizer.__class__.__name__,
            "architecture": "MoCoPretrainingModel",
            "dataset": "EMBED",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "queue_size": queue_size,
            "momentum": momentum,
            "temperature": temperature,
            "projection_dim": projection_dim,
        },
    )
    wandb.define_metric("epoch", hidden=True)
    for metric in [
        "Train Loss",
        "Positive Similarity",
        "Negative Similarity",
        "Learning Rate",
    ]:
        wandb.define_metric(metric, step_metric="epoch")

    config_path = os.path.join(path_out_dir, f"config_moco_pretraining_training_id-{id}.json")
    _save_json_config(config_path, run_config)

    best_loss = float("inf")
    debug_saved = False

    for epoch in tqdm(range(num_epochs), desc="MoCo Epochs"):
        print(f"[INFO] Epoch: {epoch}")
        logger.info(f"##### Epoch: {epoch} #####")

        model.train()
        running_loss = 0.0
        running_positive_similarity = 0.0
        running_negative_similarity = 0.0
        num_batches = 0

        for batch in train_loader:
            torch.cuda.empty_cache()
            num_batches += 1

            if save_debug_augs == "True" and not debug_saved:
                debug_dir = os.path.join(path_out_dir, "debug_aug_examples")
                _save_debug_augmentation_examples(debug_dir, batch)
                debug_saved = True

            view_q = batch["view_q"].to(device, dtype=torch.float32)
            view_k = batch["view_k"].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(view_q, view_k)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_positive_similarity += outputs["positive_similarity"].item()
            running_negative_similarity += outputs["negative_similarity"].item()

        avg_loss = running_loss / max(num_batches, 1)
        avg_positive_similarity = running_positive_similarity / max(num_batches, 1)
        avg_negative_similarity = running_negative_similarity / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step(avg_loss)

        logger.info(f"Train Loss: {avg_loss:.6f}")
        logger.info(f"Positive Similarity: {avg_positive_similarity:.6f}")
        logger.info(f"Negative Similarity: {avg_negative_similarity:.6f}")
        logger.info(f"Learning Rate: {current_lr:.8f}")

        wandb.log(
            {
                "epoch": epoch,
                "Train Loss": avg_loss,
                "Positive Similarity": avg_positive_similarity,
                "Negative Similarity": avg_negative_similarity,
                "Learning Rate": current_lr,
            }
        )

        checkpoint_payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            avg_loss=avg_loss,
            run_config=run_config,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(
                path_out_dir, f"best_model_moco_pretraining_training_id-{id}.pth"
            )
            encoder_path = os.path.join(
                path_out_dir, f"encoder_only_moco_pretraining_training_id-{id}.pth"
            )
            torch.save(checkpoint_payload, best_model_path)
            torch.save(model.get_encoder_state_dict(), encoder_path)

        print(
            f"Epoch {epoch} | Train Loss: {avg_loss:.6f} | "
            f"Positive Similarity: {avg_positive_similarity:.6f} | "
            f"Negative Similarity: {avg_negative_similarity:.6f}"
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time taken to pretrain the model: {elapsed_time:.2f}s")
    print(f"[INFO] Total time taken to pretrain the model: {elapsed_time:.2f}s")

    final_payload = _build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=num_epochs - 1,
        avg_loss=avg_loss,
        run_config=run_config,
    )
    torch.save(final_payload, path_model)

    artifact = wandb.Artifact("moco_pretraining_model", type="model")
    artifact.add_file(path_model)
    artifact.add_file(config_path)
    wandb.log_artifact(artifact)
    wandb.finish()

