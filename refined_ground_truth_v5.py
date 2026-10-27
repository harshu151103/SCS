
import cv2
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset/custom_data")
IMAGE_DIR = DATASET_DIR / "images"
GT_DIR = DATASET_DIR / "masks"

PRED_DIR = Path("Result/custom_data")
COMPARISON_DIR = Path("Comparison/custom_data")

OUTPUT_DIR = Path("dataset/custom_data_refined")
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_MASK_DIR = OUTPUT_DIR / "masks"
LOG_PATH = OUTPUT_DIR / "refinement_log.csv"

IOU_THRESHOLD = 0.80

EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

for directory in [OUTPUT_IMAGE_DIR, OUTPUT_MASK_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE MANAGEMENT
# ============================================================

def index_files(directory):
    files = {}

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in files:
                raise ValueError(
                    f"Duplicate filename stem: {path.stem} in {directory}"
                )
            files[path.stem] = path

    return files


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read: {path}")

    return image


def binary_mask(mask):
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def load_decisions():
    """Load persistent decisions from earlier runs."""
    if not LOG_PATH.exists():
        return {}

    df = pd.read_csv(LOG_PATH)
    decisions = {}

    if "filename" not in df.columns or "decision" not in df.columns:
        raise ValueError(
            "Existing CSV has an incompatible format. "
            "Back it up before changing the log schema."
        )

    for record in df.to_dict("records"):
        decisions[record["filename"]] = record

    return decisions


def write_log(decisions):
    """Rewrite the CSV from the current source of truth."""
    if not decisions:
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        return

    df = pd.DataFrame(list(decisions.values()))

    if "filename" in df.columns:
        df = df.drop_duplicates(subset=["filename"], keep="last")
        df = df.sort_values("filename")

    df.to_csv(LOG_PATH, index=False)


def save_selection(image_path, mask, stem):
    """Save the original image and the selected binary mask."""
    shutil.copy2(
        image_path,
        OUTPUT_IMAGE_DIR / image_path.name
    )

    output_mask = OUTPUT_MASK_DIR / f"{stem}.png"

    if not cv2.imwrite(str(output_mask), binary_mask(mask)):
        raise IOError(f"Failed to save: {output_mask}")


def remove_selection(stem, image_path=None):
    """Remove any output associated with a reverted decision."""
    mask_path = OUTPUT_MASK_DIR / f"{stem}.png"

    if mask_path.exists():
        mask_path.unlink()

    # Also remove a copied image if it exists.
    if image_path is not None:
        output_image = OUTPUT_IMAGE_DIR / image_path.name

        if output_image.exists():
            output_image.unlink()


# ============================================================
# IOU AND MULTI-TARGET ANALYSIS
# ============================================================

def calculate_iou(gt, pred):
    gt_binary = gt > 0
    pred_binary = pred > 0

    intersection = np.logical_and(
        gt_binary, pred_binary
    ).sum()

    union = np.logical_or(
        gt_binary, pred_binary
    ).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def get_components(mask):
    """
    Return connected components and their statistics.
    Connectivity = 8, so diagonal pixels are connected.
    """
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
    )

    # Remove background component from the returned component list.
    components = []

    for label_id in range(1, count):
        components.append({
            "label": label_id,
            "area": int(stats[label_id, cv2.CC_STAT_AREA]),
            "centroid": centroids[label_id].tolist()
        })

    return labels, components


def component_iou(component_mask, pred_mask):
    """IoU between one GT component and the complete prediction."""
    a = component_mask > 0
    b = pred_mask > 0

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    return float(intersection / union) if union else 1.0


def analyze_masks(gt, pred):
    """
    Automatic acceptance requires:
      1. Nonempty GT and prediction.
      2. Overall pixel IoU >= threshold.
      3. Same number of connected components.
      4. Every GT component has sufficient IoU with prediction.

    If any condition fails, route to manual review.
    """
    overall_iou = calculate_iou(gt, pred)

    gt_labels, gt_components = get_components(gt)
    _, pred_components = get_components(pred)

    per_target_ious = []

    for component in gt_components:
        component_mask = np.where(
            gt_labels == component["label"], 255, 0
        ).astype(np.uint8)

        # IoU of this target against the full prediction.
        # Extra predicted pixels and merged targets are penalized.
        target_iou = component_iou(component_mask, pred)

        per_target_ious.append(target_iou)

    gt_nonempty = len(gt_components) > 0
    pred_nonempty = len(pred_components) > 0

    count_match = len(gt_components) == len(pred_components)

    all_targets_match = (
        gt_nonempty
        and len(per_target_ious) == len(gt_components)
        and all(value >= IOU_THRESHOLD for value in per_target_ious)
    )

    auto_accept = (
        gt_nonempty
        and pred_nonempty
        and overall_iou >= IOU_THRESHOLD
        and count_match
        and all_targets_match
    )

    return {
        "iou": overall_iou,
        "gt_targets": len(gt_components),
        "pred_targets": len(pred_components),
        "target_ious": per_target_ious,
        "count_match": count_match,
        "auto_accept": auto_accept,
        "gt_foreground_pixels": int(np.count_nonzero(gt)),
        "pred_foreground_pixels": int(np.count_nonzero(pred))
    }


# ============================================================
# VISUALIZATION
# ============================================================

def make_panel(image, title, is_mask=False):
    if is_mask:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 2:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        panel = image.copy()

    panel = cv2.resize(
        panel, (400, 400),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.rectangle(panel, (0, 0), (400, 36), (0, 0, 0), -1)

    cv2.putText(
        panel, title, (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA
    )

    return panel


def create_comparison(image, gt, pred):
    panels = [
        make_panel(image, "Raw Image"),
        make_panel(gt, "Ground Truth", is_mask=True),
        make_panel(pred, "SCTransNet Prediction", is_mask=True)
    ]

    return np.hstack(panels)


def review_image(stem, image, gt, pred, metrics, comparison_files):
    """
    Keyboard:
      P = use SCTransNet prediction
      G = use original ground truth
      S = skip
      Q = quit
      LEFT ARROW = undo previous image's decision
    """
    comparison_path = comparison_files.get(stem)

    if comparison_path is not None:
        comparison = cv2.imread(str(comparison_path))

        if comparison is None:
            comparison = create_comparison(image, gt, pred)
    else:
        comparison = create_comparison(image, gt, pred)

    height, width = comparison.shape[:2]

    max_width = 1400

    if width > max_width:
        scale = max_width / width

        comparison = cv2.resize(
            comparison,
            (int(width * scale), int(height * scale))
        )

    window = "Ground Truth Refinement"
    cv2.imshow(window, comparison)

    print(
        f"\nREVIEW: {stem} | "
        f"IoU={metrics['iou']:.4f} | "
        f"GT targets={metrics['gt_targets']} | "
        f"Predicted targets={metrics['pred_targets']}"
    )

    print("P=Prediction | G=Ground Truth | S=Skip | Q=Quit | LEFT=Back")

    while True:
        key = cv2.waitKeyEx(0)

        # LEFT arrow key codes differ across OpenCV platforms.
        left_arrow_codes = {
            2424832, 2555904, 65361, 81
        }

        if key in left_arrow_codes:
            cv2.destroyWindow(window)
            return "back"

        key_char = key & 0xFF

        if key_char in (ord("p"), ord("P")):
            cv2.destroyWindow(window)
            return "prediction"

        if key_char in (ord("g"), ord("G")):
            cv2.destroyWindow(window)
            return "ground_truth"

        if key_char in (ord("s"), ord("S")):
            cv2.destroyWindow(window)
            return "skipped"

        if key_char in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window)
            return "quit"


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    images = index_files(IMAGE_DIR)
    ground_truths = index_files(GT_DIR)
    predictions = index_files(PRED_DIR)

    # Index existing visualizations by stem, regardless of extension.
    comparison_files = index_files(COMPARISON_DIR)

    stems = sorted(images)

    missing_gt = sorted(set(images) - set(ground_truths))
    missing_pred = sorted(set(images) - set(predictions))

    print(f"Images: {len(images)}")
    print(f"Ground-truth masks: {len(ground_truths)}")
    print(f"Predictions: {len(predictions)}")
    print(f"Missing GT: {len(missing_gt)}")
    print(f"Missing predictions: {len(missing_pred)}")

    decisions = load_decisions()

    # Error entries are retried. All other decisions, including
    # skipped images, count as completed on a normal restart.
    def is_completed(stem):
        record = decisions.get(stem)

        if record is None:
            return False

        return record.get("decision") not in (None, "error")

    # Resume at the first incomplete image.
    start_index = next(
        (i for i, stem in enumerate(stems)
         if not is_completed(stem)),
        len(stems)
    )

    index = start_index

    print(f"\nResume position: {index + 1} / {len(stems)}")
    print(f"Already completed: {sum(is_completed(s) for s in stems)}")
    print(f"Remaining: {sum(not is_completed(s) for s in stems)}")

    auto_count = 0
    reviewed_count = 0

    while index < len(stems):
        stem = stems[index]

        # Skip previously completed entries during normal progression.
        if is_completed(stem):
            index += 1
            continue

        # Progress counts all images in the ordered dataset.
        completed_count = sum(is_completed(s) for s in stems)
        remaining_count = len(stems) - completed_count

        print(
            f"\n[{index + 1}/{len(stems)}] "
            f"Completed: {completed_count} | "
            f"Remaining: {remaining_count} | "
            f"Image: {stem}"
        )

        # Missing files are logged and retried on the next run.
        if stem not in ground_truths or stem not in predictions:
            missing = []

            if stem not in ground_truths:
                missing.append("ground truth")

            if stem not in predictions:
                missing.append("prediction")

            decisions[stem] = {
                "filename": stem,
                "decision": "error",
                "error": f"Missing {' and '.join(missing)}",
                "iou": np.nan,
                "threshold": IOU_THRESHOLD
            }

            write_log(decisions)

            print(f"[ERROR] {stem}: missing {' and '.join(missing)}")
            index += 1
            continue

        try:
            image_path = images[stem]
            image = read_gray(image_path)
            gt = binary_mask(read_gray(ground_truths[stem]))
            pred = binary_mask(read_gray(predictions[stem]))

            if image.shape != gt.shape:
                raise ValueError(
                    f"Image shape {image.shape} != GT shape {gt.shape}"
                )

            pred_resized = pred.shape != gt.shape

            if pred_resized:
                pred = cv2.resize(
                    pred,
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                pred = binary_mask(pred)

                print(
                    f"[WARNING] {stem}: prediction resized "
                    "to ground-truth dimensions."
                )

            metrics = analyze_masks(gt, pred)

            common_record = {
                "filename": stem,
                "image_path": str(image_path),
                "gt_path": str(ground_truths[stem]),
                "prediction_path": str(predictions[stem]),
                "iou": round(metrics["iou"], 6),
                "threshold": IOU_THRESHOLD,
                "gt_targets": metrics["gt_targets"],
                "pred_targets": metrics["pred_targets"],
                "target_ious": str(metrics["target_ious"]),
                "count_match": metrics["count_match"],
                "gt_foreground_pixels": metrics["gt_foreground_pixels"],
                "pred_foreground_pixels": metrics["pred_foreground_pixels"],
                "prediction_resized": pred_resized,
                "both_masks_empty": (
                    metrics["gt_foreground_pixels"] == 0
                    and metrics["pred_foreground_pixels"] == 0
                )
            }

            # ----------------------------------------------------
            # AUTOMATIC ACCEPTANCE
            # ----------------------------------------------------
            if metrics["auto_accept"]:
                save_selection(image_path, pred, stem)

                decisions[stem] = {
                    **common_record,
                    "decision": "auto_prediction",
                    "error": ""
                }

                write_log(decisions)

                auto_count += 1

                print(
                    f"[AUTO ACCEPT] {stem} | "
                    f"IoU={metrics['iou']:.4f} | "
                    f"Targets={metrics['gt_targets']} | "
                    "SCTransNet mask saved."
                )

                index += 1
                continue

            # ----------------------------------------------------
            # MANUAL REVIEW
            # ----------------------------------------------------
            action = review_image(
                stem, image, gt, pred,
                metrics, comparison_files
            )

            if action == "back":
                # The currently displayed image has no new decision.
                # Undo the immediately preceding dataset image.
                if index == 0:
                    print("[BACK] Already at the first image.")
                    continue

                previous_stem = stems[index - 1]

                remove_selection(
                    previous_stem,
                    images.get(previous_stem)
                )

                if previous_stem in decisions:
                    del decisions[previous_stem]

                write_log(decisions)

                print(
                    f"[UNDO] Removed decision/output for "
                    f"{previous_stem}. Returning to that image."
                )

                index -= 1
                continue

            if action == "quit":
                print("[QUIT] Progress saved. Restart to resume.")
                break

            if action == "prediction":
                save_selection(image_path, pred, stem)
                decision = "prediction"

            elif action == "ground_truth":
                save_selection(image_path, gt, stem)
                decision = "ground_truth"

            else:
                # A skipped image has a CSV record but no output mask.
                remove_selection(stem, image_path)
                decision = "skipped"

            decisions[stem] = {
                **common_record,
                "decision": decision,
                "error": ""
            }

            write_log(decisions)

            reviewed_count += 1

            print(
                f"[SAVED DECISION] {stem} | "
                f"Action={decision} | IoU={metrics['iou']:.4f}"
            )

            index += 1

        except Exception as exc:
            decisions[stem] = {
                "filename": stem,
                "decision": "error",
                "error": str(exc),
                "iou": np.nan,
                "threshold": IOU_THRESHOLD
            }

            write_log(decisions)

            print(f"[ERROR] {stem}: {exc}")
            index += 1

    cv2.destroyAllWindows()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    completed = sum(is_completed(s) for s in stems)
    remaining = len(stems) - completed

    auto_total = sum(
        decisions.get(s, {}).get("decision") == "auto_prediction"
        for s in stems
    )

    manual_prediction_total = sum(
        decisions.get(s, {}).get("decision") == "prediction"
        for s in stems
    )

    gt_total = sum(
        decisions.get(s, {}).get("decision") == "ground_truth"
        for s in stems
    )

    skipped_total = sum(
        decisions.get(s, {}).get("decision") == "skipped"
        for s in stems
    )

    error_total = sum(
        decisions.get(s, {}).get("decision") == "error"
        for s in stems
    )

    print("\n========== REFINEMENT SUMMARY ==========")
    print(f"Total images: {len(stems)}")
    print(f"Completed: {completed}")
    print(f"Remaining: {remaining}")
    print(f"Automatically accepted SCTransNet masks: {auto_total}")
    print(f"Manually selected SCTransNet masks: {manual_prediction_total}")
    print(f"Original GT masks retained: {gt_total}")
    print(f"Skipped: {skipped_total}")
    print(f"Errors / missing inputs: {error_total}")
    print(f"Output masks: {OUTPUT_MASK_DIR}")
    print(f"Output images: {OUTPUT_IMAGE_DIR}")
    print(f"CSV log: {LOG_PATH}")


if __name__ == "__main__":
    main()
