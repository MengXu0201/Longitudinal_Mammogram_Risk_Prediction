import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def imgunit16(img):
    """
    Rescale image intensities to the 16-bit range [0, 65535].

    This function is applied to each loaded PNG image before normalization.
    """
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()

    # Avoid division by zero for constant-value images
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.float32)

    mammogram_scaled = (img - img_min) / (img_max - img_min) * 65535
    return mammogram_scaled


class BreastCancerRiskDataset(Dataset):
    """
    Breast cancer risk dataset using fixed adjacent temporal pairs.

    Each sample consists of:
    - a current image
    - the immediately previous image from the same patient, laterality, and view
    - survival/risk targets for both current and previous exams

    Expected PNG filename format:
        patient_id_laterality_view_study_date_last4_status.png

    Example:
        12345678_L_CC_2019-05-12_4821_pos_cancer.png
    """

    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        """
        Args:
            csv_file (str): Path to the CSV file containing exam-level metadata.
            image_dir (str): Root directory containing split folders such as
                             train/, val/, and test/.
            mode (str): Dataset split to load, e.g. "train", "val", or "test".
            transforms (callable, optional): Optional transform applied to image tensors.
            n_years (int): Prediction horizon in years. Default is 5.
        """
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode

        # Standardize key CSV columns once during initialization
        self.data["patient_id"] = self.data["patient_id"].astype(str).str.strip()
        self.data["ImageLateralityFinal"] = self.data["ImageLateralityFinal"].astype(str).str.strip()
        self.data["view"] = self.data["view"].astype(str).str.strip()
        self.data["study_date_anon"] = pd.to_datetime(self.data["study_date_anon"])

        # Load images and group them by (patient_id, laterality, view)
        self.image_data = self._load_image_data()

        # Precompute all fixed adjacent temporal pairs
        # Each element is:
        # ((patient_id, laterality, view), previous_image_info, current_image_info)
        self.fixed_pairs = []

        for group_key, view_images in self.image_data.items():
            if len(view_images) > 1:
                # Sort images by full study date so we can build adjacent pairs
                view_images = sorted(view_images, key=lambda x: x["study_date"])

                # Create all adjacent pairs:
                # (img1, img0), (img2, img1), ..., (imgN, imgN-1)
                for i in range(1, len(view_images)):
                    previous_image = view_images[i - 1]
                    current_image = view_images[i]
                    self.fixed_pairs.append((group_key, previous_image, current_image))

        self.num_elements = len(self.fixed_pairs)

    def map_density(self, value):
        """
        Map numeric density values to BI-RADS density categories.
        """
        if value == 1:
            return "A"
        elif value == 2:
            return "B"
        elif value == 3:
            return "C"
        elif value == 4:
            return "D"
        else:
            return "NA"

    def _load_image_data(self):
        """
        Load PNG image metadata from the selected split folder and group images by:
            (patient_id, laterality, view)

        Expected filename format:
            patient_id_laterality_view_study_date_last4_status.png

        Example:
            12345678_L_CC_2019-05-12_4821_pos_cancer.png
        """
        image_data = defaultdict(list)

        split_dir = os.path.join(self.data_dir, self.mode)

        # Regex for the new filename format:
        # 1: patient_id
        # 2: laterality
        # 3: view
        # 4: study_date
        # 5: last_4_numbers
        # 6: status_tag
        pattern = re.compile(
            r"^(\d+)_([A-Z]+)_([A-Z0-9]+)_(\d{4}-\d{2}-\d{2})_([A-Za-z0-9]+)_(pos_cancer|pos_nocancer|neg_nocancer)\.png$"
        )

        for filename in os.listdir(split_dir):
            match = pattern.match(filename)
            if not match:
                continue

            patient_id = match.group(1)
            laterality = match.group(2)
            view = match.group(3)
            study_date = datetime.strptime(match.group(4), "%Y-%m-%d")
            last_4_numbers = match.group(5)
            status_tag = match.group(6)

            group_key = (patient_id, laterality, view)

            image_data[group_key].append(
                {
                    "filename": filename,
                    "study_date": study_date,
                    "last_4_numbers": last_4_numbers,
                    "status_tag": status_tag,
                }
            )

        return image_data

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        """
        Return one fixed adjacent temporal pair.

        The pair is constructed from two consecutive exams of the same:
        - patient
        - laterality
        - view

        Returns:
            dict containing images, labels, masks, event information, and metadata.
        """
        if not self.fixed_pairs:
            raise ValueError("No valid fixed pairs generated.")

        if idx < 0 or idx >= len(self.fixed_pairs):
            raise ValueError(
                f"Index {idx} is out of range. Valid range: 0 to {len(self.fixed_pairs) - 1}"
            )

        # Access one fixed adjacent pair
        group_key, previous_image_info, current_image_info = self.fixed_pairs[idx]
        patient_id_group, laterality_group, view_group = group_key

        current_image_file = current_image_info["filename"]
        previous_image_file = previous_image_info["filename"]

        current_date = current_image_info["study_date"]
        previous_date = previous_image_info["study_date"]

        # Time gap between adjacent exams, capped at n_years
        time_gap = abs(current_date.year - previous_date.year)
        time_gap = min(time_gap, self.n_years)

        current_image_path = os.path.join(self.data_dir, self.mode, current_image_file)
        previous_image_path = os.path.join(self.data_dir, self.mode, previous_image_file)

        # Load current and previous images
        current_image_pil = Image.open(current_image_path)
        previous_image_pil = Image.open(previous_image_path)

        # Convert to numpy, rescale to 16-bit, and normalize
        current_image_np = np.array(current_image_pil)
        current_image_np = imgunit16(current_image_np)
        current_image_np = (current_image_np - 7047.99) / 12005.5
        current_image = torch.from_numpy(current_image_np).unsqueeze(0).to(torch.float16)

        previous_image_np = np.array(previous_image_pil)
        previous_image_np = imgunit16(previous_image_np)
        previous_image_np = (previous_image_np - 7047.99) / 12005.5
        previous_image = torch.from_numpy(previous_image_np).unsqueeze(0).to(torch.float16)

        # Apply optional transforms
        if self.transform is not None:
            current_image = self.transform(current_image.to(torch.float32))
            previous_image = self.transform(previous_image.to(torch.float32))

            # If transforms keep channel dimension as [1, H, W], remove the leading channel
            if current_image.dim() == 3 and current_image.shape[0] == 1:
                current_image = current_image.squeeze(0)
            if previous_image.dim() == 3 and previous_image.shape[0] == 1:
                previous_image = previous_image.squeeze(0)

        # Parse metadata from current filename
        parts = current_image_file.split("_")
        patient_id = parts[0]
        image_laterality = parts[1]
        view = parts[2]

        # Parse metadata from prior filename
        parts_prior = previous_image_file.split("_")
        patient_id_prior = parts_prior[0]
        image_laterality_prior = parts_prior[1]
        view_prior = parts_prior[2]

        # Convert exam dates to pandas datetime for CSV matching
        study_date = pd.to_datetime(current_date)
        study_date_prior = pd.to_datetime(previous_date)

        # Match current exam row in CSV
        matching_row = self.data[
            (self.data["patient_id"] == patient_id)
            & (self.data["ImageLateralityFinal"] == image_laterality)
            & (self.data["view"] == view)
            & (self.data["study_date_anon"] == study_date)
        ]

        # Match previous exam row in CSV
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

        # If multiple rows match, use the first one
        matching_row = matching_row.iloc[0]
        matching_row_prior = matching_row_prior.iloc[0]

        # Extract clinical metadata
        years_last_followup_prior = matching_row_prior["years_last_followup"]
        years_last_followup = matching_row["years_last_followup"]

        time_to_cancer = matching_row["Time_to_Cancer_Years"]
        time_to_cancer_prior_img = matching_row_prior["Time_to_Cancer_Years"]

        
        density = matching_row["density"]
        density = self.map_density(density)

        # Handle cancer timing edge cases
        # If time_to_cancer == 0, move it to 1 so that after subtracting 1
        # it maps to index 0 rather than -1
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
        time_to_cancer = int(time_to_cancer)
        time_to_cancer_prior_img = int(time_to_cancer_prior_img)

        # Convert to zero-based event time indices
        time_to_cancer = time_to_cancer - 1
        time_to_cancer_prior_img = time_to_cancer_prior_img - 1

        any_cancer = time_to_cancer < self.n_years

        # Target length is n_years + 1
        # For n_years = 5, target length is 6
        target = np.zeros(self.n_years + 1)
        target_prior = np.zeros(self.n_years + 1)

        if any_cancer:
            # Cancer occurs within the prediction window
            time_at_event = int(time_to_cancer)
            time_at_event_prior = int(time_to_cancer_prior_img)
            event_observed = 1

            # Mark the event year and all later years as 1
            target[time_at_event:] = 1
            target_prior[time_at_event_prior:] = 1

            # Last entry is reserved for censored/non-cancer cases
            target[-1] = 0
            target_prior[-1] = 0
        else:
            # No cancer observed within the prediction window
            if years_last_followup == 0:
                years_last_followup = 1
            if years_last_followup_prior == 0:
                years_last_followup_prior = 1

            time_at_event = int(min(years_last_followup, self.n_years)) - 1
            time_at_event_prior = int(min(years_last_followup_prior, self.n_years)) - 1
            event_observed = 0

            # Mark the last entry as censored/non-cancer
            target[-1] = 1
            target_prior[-1] = 1

        # Mask valid supervision time points
        y_mask = np.array(
            [1] * (time_at_event + 1) + [0] * ((self.n_years + 1) - (time_at_event + 1))
        )
        y_mask_prior = np.array(
            [1] * (time_at_event_prior + 1)
            + [0] * ((self.n_years + 1) - (time_at_event_prior + 1))
        )

        y_mask = y_mask[: self.n_years + 1]
        y_mask_prior = y_mask_prior[: self.n_years + 1]

        event_times = time_at_event

        data = {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": current_image_file,
            "previous_image_id": previous_image_file,
            "event_observed": event_observed,
            "event_times": event_times,
            "time_gap": time_gap,
            "y_mask": y_mask,
            "y_mask_prior": y_mask_prior,
            "target": target,
            "target_prior": target_prior,
            "density": density,
        }

        return data
    


# # The following main method is only for testing and debugging the dataset implementation. 
# if __name__ == "__main__":

#     csv_file = "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Alignment/output_csv/EMBED_combined_cases_with_followup_with_split.csv"
#     image_dir = "/mnt/cv_data/users/mengxu/EMBED_Dataset_Split"
#     mode = "train"

#     dataset = BreastCancerRiskDataset(
#         csv_file=csv_file,
#         image_dir=image_dir,
#         mode=mode,
#         transforms=None,
#         n_years=5,
#     )

#     print(f"Dataset mode: {mode}")
#     print(f"Total fixed pairs: {len(dataset)}")
#     print("-" * 80)

#     num_samples_to_check = 5

#     for i in range(min(num_samples_to_check, len(dataset))):
#         sample = dataset[i]

#         print(f"Sample index: {i}")
#         print(f"Current image ID:  {sample['current_image_id']}")
#         print(f"Previous image ID: {sample['previous_image_id']}")
#         print(f"Time gap: {sample['time_gap']}")
#         print(f"Event observed: {sample['event_observed']}")
#         print(f"Event times: {sample['event_times']}")
#         # print(f"Density: {sample['density']}")

#         print(f"Target: {sample['target']}")
#         print(f"Target prior: {sample['target_prior']}")
#         print(f"y_mask: {sample['y_mask']}")
#         print(f"y_mask_prior: {sample['y_mask_prior']}")

#         print(f"Current image shape: {sample['current_image'].shape}")
#         print(f"Previous image shape: {sample['previous_image'].shape}")
#         print("-" * 80)


# The following main method is to check the number of pairs each patient contributes 
# and the distribution of pair contributions across patients.

if __name__ == "__main__":

    csv_file = "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Alignment/output_csv/EMBED_combined_cases_with_followup_with_split.csv"
    image_dir = "/mnt/cv_data/users/mengxu/EMBED_Dataset_Split"
    mode = "train"

    dataset = BreastCancerRiskDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        mode=mode,
        transforms=None,
        n_years=5,
    )

    print(f"Dataset mode: {mode}")
    print(f"Total fixed pairs: {len(dataset)}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Count how many fixed pairs each patient contributes
    # -------------------------------------------------------------------------
    # dataset.fixed_pairs contains tuples of:
    # ((patient_id, laterality, view), previous_image_info, current_image_info)
    patient_pair_count = defaultdict(int)

    for group_key, previous_image_info, current_image_info in dataset.fixed_pairs:
        patient_id, laterality, view = group_key
        patient_pair_count[patient_id] += 1

    # Convert to DataFrame for easier inspection
    patient_pair_count_df = pd.DataFrame(
        sorted(patient_pair_count.items(), key=lambda x: (x[1], x[0])),
        columns=["patient_id", "num_pairs"],
    )

    print("First 20 patients sorted by number of contributed pairs (ascending):")
    print(patient_pair_count_df.head(20).to_string(index=False))
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Step 2: Build the distribution table:
    #   # of pairs  ->  # of patients
    # -------------------------------------------------------------------------
    pair_distribution_df = (
        patient_pair_count_df.groupby("num_pairs")
        .size()
        .reset_index(name="num_patients")
        .sort_values("num_pairs", ascending=True)
        .rename(columns={"num_pairs": "# of pairs", "num_patients": "# of patients"})
    )

    print("Distribution of patient-level pair contributions:")
    print(pair_distribution_df.to_string(index=False))
    print("-" * 80)

    # Optional: sanity checks
    total_patients = len(patient_pair_count_df)
    total_pairs_from_patients = patient_pair_count_df["num_pairs"].sum()

    print(f"Total patients contributing at least one pair: {total_patients}")
    print(f"Total pairs summed across patients: {total_pairs_from_patients}")
    print(f"Total fixed pairs in dataset: {len(dataset)}")

    if total_pairs_from_patients == len(dataset):
        print("Sanity check passed: summed patient-level pairs match dataset length.")
    else:
        print("Sanity check failed: summed patient-level pairs do NOT match dataset length.")