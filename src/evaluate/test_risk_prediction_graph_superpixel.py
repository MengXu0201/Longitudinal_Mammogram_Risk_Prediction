import json
import os
import pprint

import numpy as np
import torch

from src.models.model_graph_superpixel_risk import GraphBaselineSuperpixelRiskModel
from src.utils.c_index import get_censoring_dist
from src.utils.utils import (
    bootstrap_auc,
    bootstrap_auc_by_density,
    bootstrap_c_index,
    bootstrap_c_index_by_density,
    create_logger,
)


def _to_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _save_results_json(results, out_dir, logger=None, filename="results.json"):
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, filename)

    with open(results_path, "w", encoding="utf-8") as file_handle:
        json.dump(_to_json_serializable(results), file_handle, indent=4)

    print(f"[INFO] Results saved to: {results_path}")
    if logger is not None:
        logger.info(f"[INFO] Results JSON saved to: {results_path}")


def _move_graph_to_device(graph_batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in graph_batch.items()
    }


def test_graph_superpixel_baseline_risk(
    test_loader,
    device,
    path_model,
    out_dir,
    path_logger,
    seed=None,
):
    logger = create_logger(path_logger)
    print("[INFO] Loading trained superpixel graph risk model...")

    model_risk = GraphBaselineSuperpixelRiskModel(num_years=5)
    model_risk.load_state_dict(torch.load(path_model, map_location=device))
    model_risk.to(device).eval()

    print("[INFO] Evaluating on test dataset...")

    predictions, event_times, event_observed, density_categories = [], [], [], []

    with torch.inference_mode():
        for batch in test_loader:
            torch.cuda.empty_cache()

            current_graph = _move_graph_to_device(batch["current_graph"], device)
            previous_graph = _move_graph_to_device(batch["previous_graph"], device)
            time_gap = batch["time_gap"].to(device)
            event_time = batch["event_times"].to(device, dtype=torch.float32)
            event_obs = batch["event_observed"].to(device, dtype=torch.float32)
            density = batch["density"]

            output = model_risk(current_graph, previous_graph, time_gap)
            risk_pred = output["risk_prediction"]["pred_fused"]

            predictions.append(risk_pred.cpu().numpy())
            event_times.append(event_time.cpu().numpy())
            event_observed.append(event_obs.cpu().numpy())
            density_categories.append(density)

    print("[INFO] Calculating metrics...")

    predictions = np.concatenate(predictions, axis=0)
    event_times = np.concatenate(event_times, axis=0)
    event_observed = np.concatenate(event_observed, axis=0)
    density_categories = np.concatenate(density_categories, axis=0)

    censoring_dist = get_censoring_dist(event_times, event_observed)

    mean_c_index, c_index_ci = bootstrap_c_index(
        event_times, predictions, event_observed, censoring_dist, random_state=seed
    )
    auc_summary = bootstrap_auc(
        event_times,
        predictions,
        event_observed,
        random_state=None if seed is None else seed + 1000,
    )
    auc_by_density = bootstrap_auc_by_density(
        event_times,
        predictions,
        event_observed,
        density_categories,
        random_state=None if seed is None else seed + 2000,
    )
    c_index_by_density = bootstrap_c_index_by_density(
        event_times,
        predictions,
        event_observed,
        density_categories,
        censoring_dist,
        random_state=None if seed is None else seed + 3000,
    )

    auc_formatted = {
        f"{year}": {"Mean": mean_auc, "95% CI": ci}
        for year, (mean_auc, ci) in auc_summary.items()
    }

    results = {
        "C-index": {"Mean": mean_c_index, "95% CI": c_index_ci},
        "Yearly AUCs": auc_formatted,
        "AUC by density categories": auc_by_density,
        "C index by density categories": c_index_by_density,
    }

    logger.info(f"[RESULTS] Evaluation Summary:\n{results}")
    _save_results_json(results, out_dir, logger=logger, filename="results.json")

    pprint.pprint(results, sort_dicts=False)
    return results
