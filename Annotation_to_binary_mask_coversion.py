
import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def find_image(images_dir, filename):
    """Find an image by its exported filename."""
    direct_path = images_dir / filename

    if direct_path.is_file():
        return direct_path

    matches = list(images_dir.rglob(Path(filename).name))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Multiple files found for {filename}. "
            "Ensure image filenames are unique."
        )

    raise FileNotFoundError(
        f"Image not found: {filename} under {images_dir}"
    )


def load_image_size(image_path):
    """Return image width and height."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    return width, height


def polygon_to_mask(mask, polygon):
    """Rasterize one polygon into the existing mask."""
    points = np.asarray(polygon, dtype=np.float32)

    if points.size < 6 or points.size % 2 != 0:
        print("[WARNING] Invalid polygon skipped.")
        return

    points = points.reshape(-1, 2)

    if not np.isfinite(points).all():
        print("[WARNING] Polygon contains invalid coordinates.")
        return

    points = np.rint(points).astype(np.int32)

    cv2.fillPoly(mask, [points], color=255)


def save_mask(mask, output_path):
    """Save a single-channel mask with only 0 and 255."""
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), binary_mask)

    if not success:
        raise IOError(f"Failed to save mask: {output_path}")


def convert_coco(
    annotation_file,
    images_dir,
    output_dir,
    target_label
):
    """Convert COCO polygon annotations into binary PNG masks."""

    with open(annotation_file, "r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = {
        category["id"]: category["name"]
        for category in coco.get("categories", [])
    }

    target_category_ids = {
        category_id
        for category_id, name in categories.items()
        if name == target_label
    }

    if not target_category_ids:
        raise ValueError(
            f"Label '{target_label}' not found in COCO categories. "
            f"Available labels: {list(categories.values())}"
        )

    annotations_by_image = defaultdict(list)

    for annotation in coco.get("annotations", []):
        if annotation.get("category_id") in target_category_ids:
            annotations_by_image[annotation["image_id"]].append(
                annotation
            )

    processed = 0
    empty_masks = 0

    for image_info in coco.get("images", []):
        filename = image_info["file_name"]

        image_path = find_image(images_dir, filename)

        width, height = load_image_size(image_path)

        mask = np.zeros((height, width), dtype=np.uint8)

        annotations = annotations_by_image.get(
            image_info["id"], []
        )

        for annotation in annotations:
            segmentation = annotation.get("segmentation", [])

            # COCO polygon segmentation is a list of coordinate arrays.
            if isinstance(segmentation, list):

                for polygon in segmentation:
                    polygon_to_mask(mask, polygon)

            # COCO RLE requires a separate decoder.
            elif isinstance(segmentation, dict):
                raise ValueError(
                    f"RLE segmentation encountered for {filename}. "
                    "This script supports polygon segmentation only."
                )

        output_path = output_dir / (
            Path(filename).stem + ".png"
        )

        save_mask(mask, output_path)

        processed += 1

        if not np.any(mask):
            empty_masks += 1
            print(
                f"[EMPTY] {filename}: no '{target_label}' "
                "pixels found."
            )
        else:
            print(f"[OK] {filename} -> {output_path.name}")

    print("\nCOCO conversion completed.")
    print(f"Images processed: {processed}")
    print(f"Empty masks: {empty_masks}")
    print(f"Output directory: {output_dir}")


def convert_cvat_xml(
    annotation_file,
    images_dir,
    output_dir,
    target_label
):
    """Convert CVAT XML polygon annotations into binary PNG masks."""

    tree = ET.parse(annotation_file)
    root = tree.getroot()

    processed = 0
    empty_masks = 0

    for image_element in root.findall(".//image"):

        filename = image_element.get("name")

        if not filename:
            print("[WARNING] Image entry without a filename skipped.")
            continue

        image_path = find_image(images_dir, filename)

        width, height = load_image_size(image_path)

        mask = np.zeros((height, width), dtype=np.uint8)

        polygons = image_element.findall("polygon")

        for polygon_element in polygons:

            label = polygon_element.get("label")

            if label != target_label:
                continue

            points_text = polygon_element.get("points", "")

            try:
                points = []

                for coordinate_pair in points_text.split(";"):
                    coordinate_pair = coordinate_pair.strip()

                    if not coordinate_pair:
                        continue

                    x, y = coordinate_pair.split(",")

                    points.extend([float(x), float(y)])

                polygon_to_mask(mask, points)

            except (ValueError, TypeError):
                print(
                    f"[WARNING] Invalid polygon in {filename}; "
                    "skipping it."
                )

        output_path = output_dir / (
            Path(filename).stem + ".png"
        )

        save_mask(mask, output_path)

        processed += 1

        if not np.any(mask):
            empty_masks += 1
            print(
                f"[EMPTY] {filename}: no '{target_label}' "
                "pixels found."
            )
        else:
            print(f"[OK] {filename} -> {output_path.name}")

    print("\nCVAT XML conversion completed.")
    print(f"Images processed: {processed}")
    print(f"Empty masks: {empty_masks}")
    print(f"Output directory: {output_dir}")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert CVAT polygon or COCO polygon annotations "
            "to binary PNG masks."
        )
    )

    parser.add_argument(
        "--format",
        required=True,
        choices=["coco", "cvat-xml"],
        help="Annotation format."
    )

    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Path to the annotation JSON or XML file."
    )

    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Directory containing the original images."
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to save the binary masks."
    )

    parser.add_argument(
        "--label",
        default="target",
        help="Target label name. Default: target."
    )

    args = parser.parse_args()

    if not args.annotations.is_file():
        parser.error(
            f"Annotation file does not exist: {args.annotations}"
        )

    if not args.images.is_dir():
        parser.error(
            f"Image directory does not exist: {args.images}"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    if args.format == "coco":

        convert_coco(
            args.annotations,
            args.images,
            args.output,
            args.label
        )

    elif args.format == "cvat-xml":

        convert_cvat_xml(
            args.annotations,
            args.images,
            args.output,
            args.label
        )


if __name__ == "__main__":
    main()
