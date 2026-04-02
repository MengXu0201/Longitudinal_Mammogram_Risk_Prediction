import os

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers import apply_windowing


def decompress_dicom(file_path):
    """
    Decompress a DICOM file if it contains PixelData.
    """
    try:
        dataset = pydicom.dcmread(file_path)

        if "PixelData" not in dataset:
            print(f"Skipping {file_path}: No Pixel Data found.")
            return None

        dataset.decompress()
        print(f"Decompressed {file_path}")
        return dataset

    except (pydicom.errors.InvalidDicomError, AttributeError, KeyError) as e:
        print(f"Skipping {file_path}: Invalid DICOM or missing metadata. Error: {e}")
        return None
    except Exception as e:
        print(f"Skipping {file_path}: Unexpected error - {e}")
        return None


def check_and_flip_dicom(dicom_img):
    """
    Determine whether the DICOM image should be flipped horizontally based on orientation and laterality.
    """
    laterality = dicom_img.ImageLaterality
    if laterality is None:
        raise ValueError("Image Laterality (0020,0062) is missing.")

    patient_orientation = dicom_img.PatientOrientation
    print("Patient orientation", patient_orientation[0], patient_orientation[1], "laterality", laterality)

    flipHorz = False

    if patient_orientation[0] == "P":
        flipHorz = True

    if laterality == "R":
        if patient_orientation[0] == "A" and patient_orientation[1] in ["FL", "L"]:
            flipHorz = True

    return flipHorz


def resize_with_alignment(image, target_width, target_height, align="L"):
    """
    Resize image to target dimensions with aspect ratio maintained, aligned left or right.
    """
    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    result = np.zeros((target_height, target_width), dtype=np.uint8)

    y_offset = (target_height - new_height) // 2
    x_offset = 0 if align == "L" else target_width - new_width

    result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return result


# 
def preprocess_mammogram_with_largest_contour(csv_file, output_path, positive_group):
    """
    Process mammography DICOM files from a CSV, extract breast area using contour detection,
    resize and save them as 16-bit PNGs.
    Args:
        csv_file (str): Path to the input CSV file containing DICOM paths and metadata.
        output_path (str): Directory where processed PNG images will be saved.
        positive_group (bool): Flag indicating whether the CSV corresponds to the positive group (True) or negative group (False).
    """
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        if positive_group:
            patient_id = row["patient_id"]
            study_date = row["study_date_anon"]
            laterality = row["ImageLateralityFinal"]
            dicom_path = row["file_path_dcm"]
            view = row["view"]
            label = row["Label"]

            # For cancer patients, we want to differentiate between each of their exams with and without cancer
            if label == "Cancer":
                status_tag = "pos_cancer"
            elif label == "Non-Cancer":
                status_tag = "pos_nocancer"
            else:
                print(f"Skipping {dicom_path}: unexpected Label value {label}")
                continue

        else:
            patient_id = row["patient_id"]
            study_date = row["study_date_anon"]
            laterality = row["ImageLateralityFinal"]
            dicom_path = row["file_path_dcm"]
            view = row["view"]
            # note that there does not exist "neg_cancer" in the negative group, so we can directly assign the tag here
            status_tag = "neg_nocancer" 

        if not os.path.exists(dicom_path):
            print(f"Skipping {dicom_path}, file not found.")
            continue

        dicom_img = decompress_dicom(dicom_path)
        if dicom_img is None:
            continue

        img_array = dicom_img.pixel_array.astype(float)

        if check_and_flip_dicom(dicom_img):
            img_array = np.fliplr(img_array)

        img = apply_windowing(img_array, dicom_img)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        image = img.astype(np.uint8)

        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        binary_image = binary_image.astype(np.uint8)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"No contours found in {dicom_path}, skipping.")
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        breast_only = cv2.bitwise_and(image, mask)

        resized_image = resize_with_alignment(breast_only, 512, 1024, laterality)
        resized_image = (resized_image / resized_image.max()) * 65535
        final_image = Image.fromarray(np.uint16(resized_image))

        study_date_str = pd.to_datetime(study_date).strftime("%Y-%m-%d")
        last_4_numbers = os.path.basename(dicom_path).split(".")[-2][-4:]

        filename = f"{patient_id}_{laterality}_{view}_{study_date_str}_{last_4_numbers}_{status_tag}.png"

        png_path = os.path.join(output_path, filename)
        final_image.save(png_path)

        print(f"Processed {os.path.basename(dicom_path)} and saved as {png_path}")


# Newly added main method to process both positive and negative groups based on the provided CSV files. 
if __name__ == "__main__":

    # ========================   CSV files ======================== 
    positive_csv = "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/POSITIVE_GROUP_FINAL.csv"
    negative_csv = "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/NEGATIVE_GROUP_FINAL.csv"

    # print the number of unique patients, cases, and images in each CSV file before processing

    df_pos = pd.read_csv(positive_csv)
    df_neg = pd.read_csv(negative_csv)
    
    def get_stats(df):
        print(f"# Patients: {df.patient_id.nunique()}\n")
        print(f"# Cases: {df.exam_id.nunique()}\n")
        print(f"# Images: {df.file_path_dcm.nunique()}\n")  

    print("Positive group stats:")
    get_stats(df_pos)
    print("Negative group stats:")
    get_stats(df_neg)

    # ========================  Output directories (same for both groups) ======================== 
    output_dir = "/mnt/cv_data/users/mengxu/EMBED_PNG_FILES"

    # Create directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    print("Processing POSITIVE group...")
    preprocess_mammogram_with_largest_contour(
        csv_file=positive_csv,
        output_path=output_dir,
        positive_group=True
    )

    print("Processing NEGATIVE group...")
    preprocess_mammogram_with_largest_contour(
        csv_file=negative_csv,
        output_path=output_dir,
        positive_group=False
    )

    print("All preprocessing finished.")
