import os

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers import apply_windowing


def decompress_dicom(file_path):
    """
    Read a DICOM file if it contains PixelData.
    Actual decoding will happen later when pixel_array is accessed.
    """
    try:
        dataset = pydicom.dcmread(file_path)

        if "PixelData" not in dataset:
            print(f"Skipping {file_path}: No Pixel Data found.")
            return None

        return dataset

    except (pydicom.errors.InvalidDicomError, AttributeError, KeyError) as e:
        print(f"Skipping {file_path}: Invalid DICOM or missing metadata. Error: {e}")
        return None
    except Exception as e:
        print(f"Skipping {file_path}: Unexpected error - {e}")
        return None


def check_and_flip_dicom(dicom_img):
    """
    Follow EMBED official notebook logic for horizontal flipping.

    Logic:
    1. If PatientOrientation exists:
       - For CC / MLO / ML views:
         orientation[0] == 'P' -> flip horizontally
         otherwise -> do not flip

    2. If PatientOrientation is missing:
       - For right breast (R) and CC / MLO / ML -> flip horizontally
       - otherwise -> do not flip

    Returns:
        bool: whether to flip horizontally
    """
    laterality = getattr(dicom_img, "ImageLaterality", None)
    orientation = getattr(dicom_img, "PatientOrientation", None)
    view = getattr(dicom_img, "ViewPosition", None)

    if laterality is None:
        raise ValueError("Image Laterality (0020,0062) is missing.")

    flip_horz = False

    if view in ["CC", "MLO", "ML"]:
        if orientation is not None and len(orientation) >= 1:
            flip_horz = (orientation[0] == "P")
        else:
            flip_horz = (laterality == "R")
    else:
        # Unexpected view: keep as-is
        flip_horz = False

    print(
        f"[Metadata Flip] view={view}, laterality={laterality}, "
        f"orientation={orientation}, flip_horz={flip_horz}"
    )
    return flip_horz


def force_face_right(image):
    """
    Force all mammograms to face RIGHT:
    chest wall on the LEFT, breast pointing RIGHT.

    Heuristic:
    - The nipple side usually has more black background.
    - The chest wall side is denser/brighter and closer to the image edge.
    - Compare black pixels near the left and right borders.
    - If the left border has more black background, then nipple is on the left,
      so flip the image.
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape

    border_w = max(1, w // 4)
    left_black = (image[:, :border_w] < 5).sum()
    right_black = (image[:, -border_w:] < 5).sum()

    if left_black > right_black:
        image = np.fliplr(image)
        print(
            f"[Direction Fix] left_black={left_black}, "
            f"right_black={right_black}, action=flip"
        )
    else:
        print(
            f"[Direction Fix] left_black={left_black}, "
            f"right_black={right_black}, action=keep"
        )

    return image


def resize_with_alignment(image, target_width, target_height, align="L"):
    """
    Resize image to target dimensions with aspect ratio maintained,
    aligned left or right.
    """
    original_height, original_width = image.shape[:2]

    if original_height == 0 or original_width == 0:
        raise ValueError("Input image has invalid shape.")

    scale = min(target_width / original_width, target_height / original_height)
    new_width = max(1, int(original_width * scale))
    new_height = max(1, int(original_height * scale))

    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    result = np.zeros((target_height, target_width), dtype=np.uint8)

    y_offset = (target_height - new_height) // 2
    x_offset = 0 if align == "L" else target_width - new_width

    result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return result


def extract_largest_contour_region(image):
    """
    Extract the largest connected breast region using contour detection.
    """
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    breast_only = cv2.bitwise_and(image, mask)

    return breast_only


def normalize_to_uint8_after_windowing(img):
    """
    Normalize windowed image to uint8 [0, 255].
    """
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()

    if img_max <= img_min:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - img_min) / (img_max - img_min) * 255.0
    return img.astype(np.uint8)


def convert_uint8_to_uint16(image):
    """
    Convert uint8 image [0, 255] to uint16 [0, 65535].
    """
    if image is None:
        return None

    image = image.astype(np.float32)
    max_val = image.max()

    if max_val <= 0:
        return np.zeros_like(image, dtype=np.uint16)

    image = (image / max_val) * 65535.0
    return image.astype(np.uint16)


def preprocess_mammogram_with_largest_contour(csv_file, output_path, positive_group):
    """
    Process mammography DICOM files from a CSV, extract breast area using contour detection,
    resize and save them as 16-bit PNGs.

    Final standardization target:
    - All images face RIGHT
    - Chest wall on the LEFT
    - All images are LEFT-aligned
    """
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        if positive_group:
            patient_id = row["patient_id"]
            study_date = row["study_date_anon"]
            laterality_csv = row["ImageLateralityFinal"]
            dicom_path = row["file_path_dcm"]
            view = row["view"]
            label = row["Label"]

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
            laterality_csv = row["ImageLateralityFinal"]
            dicom_path = row["file_path_dcm"]
            view = row["view"]
            status_tag = "neg_nocancer"

        if not os.path.exists(dicom_path):
            print(f"Skipping {dicom_path}: file not found.")
            continue

        dicom_img = decompress_dicom(dicom_path)
        if dicom_img is None:
            continue

        try:
            img_array = dicom_img.pixel_array.astype(np.float32)
        except Exception as e:
            print(f"Skipping {dicom_path}: cannot read pixel_array. Error: {e}")
            continue

        # Step 1: EMBED-style metadata flip
        try:
            if check_and_flip_dicom(dicom_img):
                img_array = np.fliplr(img_array)
        except Exception as e:
            print(f"Warning: metadata flip failed for {dicom_path}. Error: {e}")

        # Step 2: windowing + uint8 normalization
        try:
            img_windowed = apply_windowing(img_array, dicom_img)
        except Exception as e:
            print(f"Skipping {dicom_path}: apply_windowing failed. Error: {e}")
            continue

        image = normalize_to_uint8_after_windowing(img_windowed)

        # Step 3: force all images to face RIGHT
        image = force_face_right(image)

        # Step 4: extract largest breast contour
        breast_only = extract_largest_contour_region(image)
        if breast_only is None:
            print(f"No contours found in {dicom_path}, skipping.")
            continue

        # Step 5: resize and unify placement
        # All images are LEFT-aligned
        resized_image = resize_with_alignment(
            breast_only, target_width=512, target_height=1024, align="L"
        )

        # Step 6: save as uint16 PNG
        resized_image_uint16 = convert_uint8_to_uint16(resized_image)
        final_image = Image.fromarray(resized_image_uint16)

        study_date_str = pd.to_datetime(study_date).strftime("%Y-%m-%d")

        try:
            last_4_numbers = os.path.basename(dicom_path).split(".")[-2][-4:]
        except Exception:
            last_4_numbers = "unkn"

        filename = (
            f"{patient_id}_{laterality_csv}_{view}_{study_date_str}_"
            f"{last_4_numbers}_{status_tag}.png"
        )

        png_path = os.path.join(output_path, filename)
        final_image.save(png_path)

        print(f"Processed {os.path.basename(dicom_path)} and saved as {png_path}")


if __name__ == "__main__":

    # ======================== CSV files ========================
    positive_csv = "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/POSITIVE_GROUP_FINAL.csv"
    negative_csv = "/mnt/cv_data/users/mengxu/Longitudinal_Mammogram_Risk_Prediction/output_csv/NEGATIVE_GROUP_FINAL.csv"

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

    # ======================== Output directory ========================
    output_dir = "/mnt/cv_data/users/mengxu/EMBED_PNG_Uniform_Orientation"

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