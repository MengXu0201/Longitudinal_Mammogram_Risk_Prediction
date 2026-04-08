from pathlib import Path

import cv2
import numpy as np


SOURCE_ROOT = Path("/mnt/cv_data/users/mengxu/EMBED_Dataset_Split_Uniform_Orientation")
OUTPUT_ROOT = Path("/mnt/cv_data/users/mengxu/EMBED_Split_Cropped_PNG_Uniform_Orientation")
TARGET_HEIGHT = 1024
TARGET_WIDTH = 512
VALID_EXTENSIONS = {".png"}


def ensure_output_directories(output_root):
    """
    Create the output root and the train/val/test subdirectories.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        (output_root / split_name).mkdir(parents=True, exist_ok=True)


def load_grayscale_image(image_path):
    """
    Load an image without changing its bit depth and return a single grayscale channel.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype not in (np.uint8, np.uint16):
        image = image.astype(np.uint16)

    return image


def normalize_to_uint8(image):
    """
    Convert an image to uint8 for thresholding while preserving contrast.
    """
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())

    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = (image - min_value) / (max_value - min_value)
    return (normalized * 255).astype(np.uint8)


def find_breast_bounding_box(image):
    """
    Find the smallest bounding box of the breast region.

    EMBED_Dataset_Split images were already produced with the breast isolated on a zero
    background, so the most faithful crop is the bounding box of all nonzero pixels.
    Otsu thresholding is retained only as a fallback in case an image does not follow
    that convention.
    """
    nonzero_points = cv2.findNonZero((image > 0).astype(np.uint8))

    if nonzero_points is not None:
        x, y, width, height = cv2.boundingRect(nonzero_points)
        if width > 0 and height > 0:
            return x, y, width, height

    image_uint8 = normalize_to_uint8(image)
    _, binary_mask = cv2.threshold(
        image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0, 0, image.shape[1], image.shape[0]

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest_contour)

    if width <= 0 or height <= 0:
        return 0, 0, image.shape[1], image.shape[0]

    return x, y, width, height


def resize_with_zero_padding(image, target_height, target_width):
    """
    Resize an image to fit within the target canvas while preserving aspect ratio,
    then zero-pad the remaining area.

    The uniform-orientation dataset already standardizes all views to the same facing
    direction with the chest wall on the left, so the cropped output should keep every
    image left-aligned regardless of laterality.
    """
    source_height, source_width = image.shape[:2]

    if source_height == 0 or source_width == 0:
        raise ValueError("Cannot resize an empty image.")

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )

    canvas = np.zeros((target_height, target_width), dtype=image.dtype)
    y_offset = (target_height - resized_height) // 2
    x_offset = 0

    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return canvas


def preserve_embed_bit_depth(image):
    """
    Keep output images in the same 16-bit style used by preprocess_img_embed.py.
    """
    if image.dtype == np.uint16:
        return image

    if image.dtype == np.uint8:
        return image.astype(np.uint16) * 257

    image = np.clip(image, 0, 65535)
    return image.astype(np.uint16)


def crop_and_resize_image(image):
    """
    Crop an image to the breast bounding box and resize it onto a padded 1024 x 512 canvas.
    """
    x, y, width, height = find_breast_bounding_box(image)
    cropped = image[y:y + height, x:x + width]

    if cropped.size == 0:
        cropped = image

    resized = resize_with_zero_padding(cropped, TARGET_HEIGHT, TARGET_WIDTH)
    return preserve_embed_bit_depth(resized)


def process_split_directory(source_split_dir, output_split_dir):
    """
    Process every supported image in a split directory and mirror the relative path
    in the output directory.
    """
    processed_count = 0
    skipped_count = 0

    for image_path in sorted(source_split_dir.rglob("*")):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            skipped_count += 1
            print(f"Skipping unsupported file: {image_path}")
            continue

        relative_path = image_path.relative_to(source_split_dir)
        output_path = (output_split_dir / relative_path).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            image = load_grayscale_image(image_path)
            processed_image = crop_and_resize_image(image)

            if not cv2.imwrite(str(output_path), processed_image):
                raise ValueError(f"Failed to write image: {output_path}")

            processed_count += 1
            print(f"Processed {image_path} -> {output_path}")
        except Exception as exc:
            skipped_count += 1
            print(f"Skipping {image_path}: {exc}")

    return processed_count, skipped_count


def process_embed_split_dataset(source_root=SOURCE_ROOT, output_root=OUTPUT_ROOT):
    """
    Process the full EMBED split dataset into a new cropped directory tree while leaving
    the source dataset untouched.
    """
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")

    ensure_output_directories(output_root)

    total_processed = 0
    total_skipped = 0

    for split_name in ("train", "val", "test"):
        source_split_dir = source_root / split_name
        output_split_dir = output_root / split_name

        if not source_split_dir.exists():
            print(f"Skipping missing split directory: {source_split_dir}")
            continue

        print(f"Processing split: {split_name}")
        processed_count, skipped_count = process_split_directory(
            source_split_dir, output_split_dir
        )

        total_processed += processed_count
        total_skipped += skipped_count

        print(
            f"Finished {split_name}: processed={processed_count}, skipped={skipped_count}"
        )

    print(
        f"All done. Total processed={total_processed}, total skipped={total_skipped}"
    )


if __name__ == "__main__":
    process_embed_split_dataset()
