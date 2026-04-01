import json
import os
import pprint

from src.models.MammoRegNet import MammoRegNet
from src.models.model_combined_alignment_risk import (
    CombinedAlignmentRiskModel, CombinedImgAlignmentRiskModel,
    CombinedImgAlignmentRiskModel_downsample_img_deformation_field,
    RiskModelNoAlignment)
from src.utils.c_index import *
from src.utils.utils import *


def _to_json_serializable(obj):
    """
    Recursively convert numpy / torch-friendly values into JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def _save_results_json(results, out_dir, logger=None, filename="results.json"):
    """
    Save results dict to a JSON file under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, filename)

    clean_results = _to_json_serializable(results)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=4)

    print(f"[INFO] Results saved to: {results_path}")
    if logger is not None:
        logger.info(f"[INFO] Results JSON saved to: {results_path}")


def test_jointly_feat_alignment_risk(
        test_loader,
        device,
        path_model,
        out_dir,
        path_logger,
        no_feat_Alignment,
        seed=None,
):
    """
    Evaluate a risk prediction model (with or without feature alignment) on the test set.

    Args:
        test_loader: DataLoader for the test dataset.
        device: CUDA or CPU device.
        path_model: Path to the saved model.
        out_dir: Directory to store deformation field visualizations and results.json.
        path_logger: Path for the log file.
        no_feat_Alignment: "True" to disable feature alignment.

    Returns:
        Dictionary of evaluation metrics including C-index, AUC, and optionally NJD.
    """
    logger = create_logger(path_logger)
    print("[INFO] Loading trained risk model...")

    model_cls = RiskModelNoAlignment if no_feat_Alignment == "True" else CombinedAlignmentRiskModel
    model_risk = model_cls(num_years=5)
    model_risk.load_state_dict(torch.load(path_model, map_location=device))
    model_risk.to(device).eval()

    print("[INFO] Evaluating on test dataset...")

    predictions, event_times, event_observed, density_categories = [], [], [], []
    njd_values, test_running_njd_value = [], 0.0
    counter = 0

    with torch.inference_mode():
        for batch in test_loader:
            torch.cuda.empty_cache()

            img_curr = batch["current_image"].to(device, dtype=torch.float32)
            img_prev = batch["previous_image"].to(device, dtype=torch.float32)
            time_gap = batch["time_gap"].to(device)
            event_time = batch["event_times"].to(device, dtype=torch.float32)
            event_obs = batch["event_observed"].to(device, dtype=torch.float32)
            density = batch["density"]

            output = model_risk(img_curr, img_prev, time_gap)
            risk_pred = output["risk_prediction"]["pred_fused"]

            predictions.append(risk_pred.cpu().numpy())
            event_times.append(event_time.cpu().numpy())
            event_observed.append(event_obs.cpu().numpy())
            density_categories.append(density)

            # Only alignment-based models provide deformation_field
            if "deformation_field" in output:
                flow = output["deformation_field"]

                for i in range(flow.size(0)):
                    counter += 1
                    flow_np = flow[i].unsqueeze(0).detach().cpu().permute(0, 2, 3, 1).numpy()
                    njd = NJD_percentage(flow_np)
                    jac_det = NJD().get_Ja(flow_np)

                    njd_values.append(njd.item())
                    test_running_njd_value += njd.item()

                    plot_deformation_field(
                        batch["current_image_id"][i],
                        batch["previous_image_id"][i],
                        njd,
                        out_dir,
                        flow_np.squeeze(),
                        jac_det,
                    )

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

    if counter > 0 and len(njd_values) > 0:
        mean_njd = test_running_njd_value / counter
        njd_ci = bootstrap_confidence_interval(
            np.array(njd_values),
            random_state=None if seed is None else seed + 4000,
        )
        results["NJD"] = {"Mean": mean_njd, "95% CI": njd_ci}

    logger.info(f"[RESULTS] Evaluation Summary:\n{results}")

    _save_results_json(results, out_dir, logger=logger, filename="results.json")

    pprint.pprint(results, sort_dicts=False)
    return results


def test_img_alignment_risk_pred_combined_train(
    test_loader,
    device,
    path_model,
    out_dir,
    path_logger,
    use_img_feat_alignment,
    dataset,
    seed=None,
):
    """
    Evaluate the trained image-level (optionally feature-enhanced) alignment + risk model.

    Args:
        test_loader: PyTorch DataLoader for the test set
        device: Device for computation ('cuda' or 'cpu')
        path_model: Path to the saved risk prediction model
        out_dir: Directory to save visualization outputs and results.json
        path_logger: Path for saving logs
        use_img_feat_alignment: If "True", use image-feature combined model
        dataset: 'CSAW' or 'EMBED' (for loading appropriate registration model)

    Returns:
        results: Dictionary containing C-index, AUCs, NJD, etc.
    """
    logger = create_logger(path_logger)
    print("[INFO] Loading trained registration and risk models...")

    model_reg = MammoRegNet()
    if dataset == "CSAW":
        path_reg = "/storage/CsawCC/NICE-Trans_train_results/model_registration_training_id_125_last_epoch.pth"
    else:
        path_reg = "/storage/EMBED/NICE-Trans_train_results_embed/model_registration_training_id_2_last_epoch.pth"

    model_reg.load_state_dict(torch.load(path_reg, map_location=device))
    model_reg.to(device).eval()

    if use_img_feat_alignment == "True":
        model_risk = CombinedImgAlignmentRiskModel_downsample_img_deformation_field(
            num_years=5, registration_model=model_reg
        )
    else:
        model_risk = CombinedImgAlignmentRiskModel(
            num_years=5, registration_model=model_reg
        )

    model_risk.load_state_dict(torch.load(path_model, map_location=device))
    model_risk.to(device).eval()

    print("[INFO] Evaluating on test dataset...")

    predictions, event_times, event_observed, density_categories = [], [], [], []
    njd_values, test_running_njd_value, counter = [], 0.0, 0

    with torch.inference_mode():
        for batch in test_loader:
            torch.cuda.empty_cache()

            img_curr = batch["current_image"].to(device, dtype=torch.float32)
            img_prev = batch["previous_image"].to(device, dtype=torch.float32)
            time_gap = batch["time_gap"].to(device)

            outputs = model_risk(img_curr, img_prev, time_gap)
            risk_pred = outputs["risk_prediction"]["pred_fused"]

            predictions.append(risk_pred.cpu().numpy())
            event_observed.append(batch["event_observed"].cpu().numpy())
            event_times.append(batch["event_times"].cpu().numpy())
            density_categories.append(batch["density"])

            deformation_field = outputs["deformation_field"]
            for i in range(deformation_field.shape[0]):
                counter += 1

                df_ds = F.interpolate(
                    deformation_field[i].unsqueeze(0),
                    size=(32, 16),
                    mode="bilinear",
                    align_corners=True,
                ).detach().cpu()

                df_ds[0, 0, :, :] *= (16 / 512)
                df_ds[0, 1, :, :] *= (32 / 1024)
                df_ds = df_ds.permute(0, 2, 3, 1).numpy()

                njd = NJD_percentage(df_ds)
                jac_det = NJD().get_Ja(df_ds)
                test_running_njd_value += njd
                njd_values.append(njd)

                plot_deformation_field(
                    batch["current_image_id"][i],
                    batch["previous_image_id"][i],
                    njd,
                    out_dir,
                    df_ds.squeeze(),
                    jac_det,
                )

            del outputs

    print("[INFO] Computing evaluation metrics...")

    njd_test = test_running_njd_value / counter
    njd_ci = bootstrap_confidence_interval(
        np.array(njd_values),
        random_state=None if seed is None else seed + 4000,
    )

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

    auc_bootstrap_formatted = {
        f"{year}": {"Mean": mean_auc, "95% CI": ci}
        for year, (mean_auc, ci) in auc_summary.items()
    }

    results = {
        "C-index": {"Mean": mean_c_index, "95% CI": c_index_ci},
        "Yearly AUCs": auc_bootstrap_formatted,
        "AUC by density categories": auc_by_density,
        "C index by density categories": c_index_by_density,
        "NJD": {"Mean": njd_test, "95% CI": njd_ci},
    }

    logger.info(f"[RESULTS] Evaluation Summary:\n{results}")

    _save_results_json(results, out_dir, logger=logger, filename="results.json")

    pprint.pprint(results, sort_dicts=False)
    return results
