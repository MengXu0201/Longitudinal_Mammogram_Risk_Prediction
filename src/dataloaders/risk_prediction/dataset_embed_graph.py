import numpy as np
import pandas as pd

from src.dataloaders.risk_prediction.dataset_embed import BreastCancerRiskDataset


class BreastCancerRiskGraphDataset(BreastCancerRiskDataset):
    """
    EMBED graph risk dataset that reuses the CNN dataset pair/label logic without
    loading image tensors. Graph construction is delegated to method-specific collators.
    """

    def __getitem__(self, idx):
        if not self.fixed_pairs:
            raise ValueError("No valid fixed pairs generated.")

        if idx < 0 or idx >= len(self.fixed_pairs):
            raise ValueError(
                f"Index {idx} is out of range. Valid range: 0 to {len(self.fixed_pairs) - 1}"
            )

        _, previous_image_info, current_image_info = self.fixed_pairs[idx]

        current_image_file = current_image_info["filename"]
        previous_image_file = previous_image_info["filename"]

        current_date = current_image_info["study_date"]
        previous_date = previous_image_info["study_date"]

        time_gap = abs(current_date.year - previous_date.year)
        time_gap = min(time_gap, self.n_years)

        patient_id, image_laterality, view = current_image_file.split("_")[:3]
        patient_id_prior, image_laterality_prior, view_prior = previous_image_file.split("_")[:3]

        study_date = pd.to_datetime(current_date)
        study_date_prior = pd.to_datetime(previous_date)

        matching_row = self.data[
            (self.data["patient_id"] == patient_id)
            & (self.data["ImageLateralityFinal"] == image_laterality)
            & (self.data["view"] == view)
            & (self.data["study_date_anon"] == study_date)
        ]

        matching_row_prior = self.data[
            (self.data["patient_id"] == patient_id_prior)
            & (self.data["ImageLateralityFinal"] == image_laterality_prior)
            & (self.data["view"] == view_prior)
            & (self.data["study_date_anon"] == study_date_prior)
        ]

        if matching_row.empty:
            raise ValueError(f"No matching CSV row found for current image: {current_image_file}")

        if matching_row_prior.empty:
            raise ValueError(f"No matching CSV row found for previous image: {previous_image_file}")

        matching_row = matching_row.iloc[0]
        matching_row_prior = matching_row_prior.iloc[0]

        years_last_followup_prior = matching_row_prior["years_last_followup"]
        years_last_followup = matching_row["years_last_followup"]

        time_to_cancer = matching_row["Time_to_Cancer_Years"]
        time_to_cancer_prior_img = matching_row_prior["Time_to_Cancer_Years"]

        density = self.map_density(matching_row["density"])

        if pd.isna(time_to_cancer):
            time_to_cancer = self.n_years + 1
        elif time_to_cancer == 0:
            time_to_cancer = 1

        if pd.isna(time_to_cancer_prior_img):
            time_to_cancer_prior_img = self.n_years + 1
        elif time_to_cancer_prior_img == 0:
            time_to_cancer_prior_img = 1

        years_last_followup = int(years_last_followup)
        years_last_followup_prior = int(years_last_followup_prior)
        time_to_cancer = int(time_to_cancer) - 1
        time_to_cancer_prior_img = int(time_to_cancer_prior_img) - 1

        any_cancer = time_to_cancer < self.n_years

        target = np.zeros(self.n_years + 1)
        target_prior = np.zeros(self.n_years + 1)

        if any_cancer:
            time_at_event = int(time_to_cancer)
            time_at_event_prior = int(time_to_cancer_prior_img)
            event_observed = 1

            target[time_at_event:] = 1
            target_prior[time_at_event_prior:] = 1
            target[-1] = 0
            target_prior[-1] = 0
        else:
            if years_last_followup == 0:
                years_last_followup = 1
            if years_last_followup_prior == 0:
                years_last_followup_prior = 1

            time_at_event = int(min(years_last_followup, self.n_years)) - 1
            time_at_event_prior = int(min(years_last_followup_prior, self.n_years)) - 1
            event_observed = 0

            target[-1] = 1
            target_prior[-1] = 1

        y_mask = np.array(
            [1] * (time_at_event + 1) + [0] * ((self.n_years + 1) - (time_at_event + 1))
        )
        y_mask_prior = np.array(
            [1] * (time_at_event_prior + 1)
            + [0] * ((self.n_years + 1) - (time_at_event_prior + 1))
        )

        y_mask = y_mask[: self.n_years + 1]
        y_mask_prior = y_mask_prior[: self.n_years + 1]

        return {
            "current_image_id": current_image_file,
            "previous_image_id": previous_image_file,
            "event_observed": event_observed,
            "event_times": time_at_event,
            "time_gap": time_gap,
            "y_mask": y_mask,
            "y_mask_prior": y_mask_prior,
            "target": target,
            "target_prior": target_prior,
            "density": density,
        }
