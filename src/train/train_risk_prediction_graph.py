import os
import time

import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.models.model_graph_risk import GraphBaselineMaskedGridRiskModel
from src.utils.c_index import get_censoring_dist
from src.utils.utils import (compute_auc_x_year_auc, concordance_index_ipcw,
                             create_logger, get_masked_binary_accuracy,
                             get_risk_loss_BCE)


def _move_graph_to_device(graph_batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in graph_batch.items()
    }


def train_val_graph_baseline(
    train_loader,
    valid_loader,
    device,
    learning_rate,
    weight_decay,
    num_epochs,
    path_loggger,
    path_model,
    id_training,
    use_scheduler,
    out_dir,
    accumulation_steps,
    patience_lr_scheduler,
    patience,
    lr_decay,
    hidden_dim=512,
    num_graph_layers=2,
):
    print("[INFO] Training the masked-grid graph baseline...")
    start_time = time.time()

    logger = create_logger(path_loggger)
    logger.info(f"Number of Training Epochs: {num_epochs}")

    model_risk = GraphBaselineMaskedGridRiskModel(
        hidden_dim=hidden_dim,
        num_graph_layers=num_graph_layers,
        num_years=5,
    ).to(device)

    optimizer = torch.optim.Adam(
        model_risk.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = None
    if use_scheduler == "True":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_decay,
            patience=patience_lr_scheduler,
        )
        print("Scheduler Initialized:", scheduler)

    wandb.init(
        project="EMBED_Risk_Prediction_Graph",
        config={
            "optimizer": optimizer.__class__.__name__,
            "architecture": "GraphBaselineMaskedGridRiskModel",
            "dataset": "EMBED",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "model": model_risk.__class__.__name__,
        },
    )

    wandb.define_metric("epoch", hidden=True)
    for metric in [
        "Training Loss",
        "Training Accuracy",
        "Training Risk Loss",
        "Training C-index",
        "Validation Loss",
        "Validation Risk Loss",
        "Validation C-index",
        "Year 1 AUC",
        "Year 2 AUC",
        "Year 3 AUC",
        "Year 4 AUC",
        "Year 5 AUC",
    ]:
        wandb.define_metric(metric, step_metric="epoch")

    best_c_index = 0
    patience_counter = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"[INFO] Epoch: {epoch}")
        logger.info(f"##### Epoch: {epoch} #####")

        model_risk.train()
        running_loss = 0.0
        running_risk_loss = 0.0
        counter = 0
        all_preds, all_times, all_events = [], [], []
        train_accuracy_sum = 0.0

        for idx, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            counter += 1

            current_graph = _move_graph_to_device(batch["current_graph"], device)
            previous_graph = _move_graph_to_device(batch["previous_graph"], device)
            time_gap = batch["time_gap"].to(device)
            event_times_batch = batch["event_times"].to(device, dtype=torch.float32)
            event_observed_batch = batch["event_observed"]
            target = batch["target"]
            target_prior = batch["target_prior"]
            y_mask = batch["y_mask"]
            y_mask_prior = batch["y_mask_prior"]

            outputs = model_risk(current_graph, previous_graph, time_gap)
            pred = outputs["risk_prediction"]

            risk_loss_fused = get_risk_loss_BCE(pred["pred_fused"], target, y_mask)
            risk_loss_cur = get_risk_loss_BCE(pred["pred_cur"], target, y_mask)
            risk_loss_pri = get_risk_loss_BCE(pred["pred_pri"], target_prior, y_mask_prior)
            risk_loss = risk_loss_fused + risk_loss_cur + risk_loss_pri

            total_loss = risk_loss / accumulation_steps
            total_loss.backward()

            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += total_loss.item()
            running_risk_loss += risk_loss.item()
            train_accuracy_sum += get_masked_binary_accuracy(pred["pred_fused"], target, y_mask)
            all_preds.append(pred["pred_fused"].detach().cpu().numpy())
            all_times.append(event_times_batch.cpu().numpy())
            all_events.append(event_observed_batch.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        times = np.concatenate(all_times, axis=0)
        events = np.concatenate(all_events, axis=0)
        censoring = get_censoring_dist(times, events)
        c_index = concordance_index_ipcw(times, preds, events, censoring)

        avg_loss = running_loss / counter
        avg_risk = running_risk_loss / counter
        avg_accuracy = train_accuracy_sum / counter

        wandb.log(
            {
                "epoch": epoch,
                "Training C-index": c_index,
                "Training Loss": avg_loss,
                "Training Accuracy": avg_accuracy,
                "Training Risk Loss": avg_risk,
            }
        )

        logger.info(f"Training Loss: {avg_loss:.4f}")
        logger.info(f"Training Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Training Risk Loss: {avg_risk:.4f}")
        logger.info(f"Training C-index: {c_index:.4f}")

        print(f"[Epoch {epoch}] Total Loss: {avg_loss:.4f} | Risk Loss: {avg_risk:.4f}")

        with torch.inference_mode():
            model_risk.eval()
            valid_loss_total = 0.0
            risk_loss_total = 0.0

            predictions, event_observed, event_times = [], [], []
            num_batches = len(valid_loader)

            for batch in valid_loader:
                torch.cuda.empty_cache()

                current_graph = _move_graph_to_device(batch["current_graph"], device)
                previous_graph = _move_graph_to_device(batch["previous_graph"], device)
                time_gap = batch["time_gap"].to(device)
                event_times_batch = batch["event_times"]
                event_observed_batch = batch["event_observed"]
                y_mask = batch["y_mask"]
                y_mask_prior = batch["y_mask_prior"]
                target = batch["target"]
                target_prior = batch["target_prior"]

                outputs = model_risk(current_graph, previous_graph, time_gap)
                risk_preds = outputs["risk_prediction"]

                loss_fused = get_risk_loss_BCE(risk_preds["pred_fused"], target, y_mask)
                loss_cur = get_risk_loss_BCE(risk_preds["pred_cur"], target, y_mask)
                loss_pri = get_risk_loss_BCE(risk_preds["pred_pri"], target_prior, y_mask_prior)
                risk_loss = loss_fused + loss_cur + loss_pri

                risk_loss_total += risk_loss.item()
                valid_loss_total += risk_loss.item()

                predictions.append(risk_preds["pred_fused"].cpu().numpy())
                event_observed.append(event_observed_batch.cpu().numpy())
                event_times.append(event_times_batch.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            event_times = np.concatenate(event_times, axis=0)
            event_observed = np.concatenate(event_observed, axis=0)

            aucs = compute_auc_x_year_auc(predictions, event_times, event_observed)
            for year, auc in aucs.items():
                print(f"Year {year + 1}: AUC = {auc:.4f}")
                logger.info(f"Year {year + 1} AUC: {auc:.4f}")
                wandb.log({f"Year {year + 1} AUC": auc, "epoch": epoch})

            censoring_dist = get_censoring_dist(times, events)
            c_index_val = concordance_index_ipcw(
                event_times,
                predictions,
                event_observed,
                censoring_dist,
            )

            if use_scheduler == "True":
                scheduler.step(c_index_val)

            if c_index_val > best_c_index:
                best_c_index = c_index_val
                patience_counter = 0
                best_model_path = os.path.join(out_dir, f"best_model_risk_prediction_id-{id_training}.pth")
                torch.save(model_risk.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_path = os.path.join(out_dir, f"early_stopping_risk_prediction_id-{id_training}.pth")
                    torch.save(model_risk.state_dict(), early_stop_path)
                    print("Early stopping triggered.")
                    break

            valid_loss_avg = valid_loss_total / num_batches
            risk_loss_avg = risk_loss_total / num_batches

            print(
                f"[Validation] Total Loss: {valid_loss_avg:.4f} | "
                f"Risk Loss: {risk_loss_avg:.4f} | "
                f"C-index: {c_index_val:.4f}"
            )
            logger.info(
                f"[Validation] Total Loss: {valid_loss_avg:.4f} | "
                f"Risk Loss: {risk_loss_avg:.4f} | "
                f"C-index: {c_index_val:.4f}"
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "Validation Loss": valid_loss_avg,
                    "Validation Risk Loss": risk_loss_avg,
                    "Validation C-index": c_index_val,
                }
            )

    torch.save(model_risk.state_dict(), path_model)
    total_training_time = (time.time() - start_time) / 60.0
    print(f"Training completed in {total_training_time:.2f} minutes.")
    logger.info(f"Training completed in {total_training_time:.2f} minutes.")
    wandb.finish()
