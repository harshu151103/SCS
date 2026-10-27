
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset/custom_data")
IMAGE_DIR = DATASET_DIR / "images"
GT_DIR = DATASET_DIR / "masks"

PRED_DIR = Path("Result/custom_data")
COMPARISON_DIR = Path("Comparison/custom_data")

OUTPUT_DIR = Path("refined_dataset")
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_MASK_DIR = OUTPUT_DIR / "masks"
REVIEW_DIR = OUTPUT_DIR / "review"
AUDIT_FILE = OUTPUT_DIR / "mask_audit.csv"

IOU_THRESHOLD = 0.70

# Supported image formats
EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

# Set True to review even high-IoU cases.
REVIEW_AUTO_ACCEPTED = False


# ============================================================
# 2. DIRECTORY SETUP
# ============================================================

for directory in [
    OUTPUT_IMAGE_DIR,
    OUTPUT_MASK_DIR,
    REVIEW_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. FILE MATCHING
# ============================================================

def index_files(directory):
    """Index files by filename stem, ignoring extensions."""
    files = {}

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in files:
                raise ValueError(
                    f"Duplicate filename stem '{path.stem}' "
                    f"in {directory}. Resolve duplicates first."
                )

            files[path.stem] = path

    return files


images = index_files(IMAGE_DIR)
ground_truths = index_files(GT_DIR)
predictions = index_files(PRED_DIR)

if not images:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

common_names = sorted(
    set(images) & set(ground_truths) & set(predictions)
)

if not common_names:
    raise RuntimeError(
        "No matching image, ground-truth and prediction filenames found."
    )

print(f"Images found: {len(images)}")
print(f"Ground-truth masks found: {len(ground_truths)}")
print(f"Predictions found: {len(predictions)}")
print(f"Complete matching triplets: {len(common_names)}")


# ============================================================
# 4. MASK PROCESSING AND METRICS
# ============================================================

def read_binary_mask(path):
    """Read a mask and convert foreground pixels to 255."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")

    return (mask > 0).astype(np.uint8) * 255


def calculate_iou(gt, pred):
    """Calculate pixel-level intersection over union."""
    gt_binary = gt > 0
    pred_binary = pred > 0

    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()

    # Both masks empty: IoU is mathematically undefined.
    if union == 0:
        return None

    return float(intersection / union)


def count_targets(mask):
    """Count connected foreground components."""
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
    )

    # Component 0 is the background.
    return count - 1


def copy_image(source, destination):
    shutil.copy2(source, destination)


def save_mask(mask, destination):
    """Save a clean binary mask: background 0, target 255."""
    cv2.imwrite(str(destination), mask)


# ============================================================
# 5. MANUAL REVIEW
# ============================================================

def show_comparison(name, image_path, gt, pred, iou):
    """Show the raw image and both masks side by side."""

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Raw Image")

    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Original Ground Truth")

    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("SCTransNet Prediction")

    for ax in axes:
        ax.axis("off")

    score = "undefined" if iou is None else f"{iou:.4f}"

    fig.suptitle(
        f"{name} | IoU: {score}\n"
        "Close this window after making your selection in the terminal",
        fontsize=12
    )

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    return fig


def manual_review(name, image_path, gt, pred, iou):
    """
    Choices:
      P = select SCTransNet prediction
      G = select original ground truth
      S = skip this image
      Q = stop the review
    """

    fig = show_comparison(
        name, image_path, gt, pred, iou
    )

    try:
        while True:
            choice = input(
                f"\n[{name}] Choose "
                "[P]rediction / [G]round truth / "
                "[S]kip / [Q]uit: "
            ).strip().lower()

            if choice in {"p", "g", "s", "q"}:
                return choice

            print("Invalid choice. Enter P, G, S or Q.")

    finally:
        plt.close(fig)


# ============================================================
# 6. MAIN REFINEMENT PIPELINE
# ============================================================

audit_rows = []
processed = 0
auto_accepted = 0
manually_selected = 0
skipped = 0
errors = 0
quit_review = False


for name in sorted(images):

    image_path = images[name]
    gt_path = ground_truths.get(name)
    pred_path = predictions.get(name)

    row = {
        "filename": name,
        "image_path": str(image_path),
        "ground_truth_path": str(gt_path) if gt_path else "",
        "prediction_path": str(pred_path) if pred_path else "",
        "iou": "",
        "gt_target_count": "",
        "pred_target_count": "",
        "decision": "",
        "selected_source": "",
        "status": ""
    }

    if gt_path is None or pred_path is None:
        row["decision"] = "unmatched"
        row["status"] = "missing_mask"
        audit_rows.append(row)
        continue

    try:
        gt = read_binary_mask(gt_path)
        pred = read_binary_mask(pred_path)

        row["gt_target_count"] = count_targets(gt)
        row["pred_target_count"] = count_targets(pred)

        # Do not silently resize masks: alignment errors need review.
        if gt.shape != pred.shape:
            row["decision"] = "manual_review"
            row["status"] = (
                f"shape_mismatch_gt_{gt.shape}_pred_{pred.shape}"
            )
            audit_rows.append(row)
            print(f"[ERROR] {name}: GT/prediction dimensions differ.")
            errors += 1
            continue

        iou = calculate_iou(gt, pred)
        row["iou"] = "" if iou is None else round(iou, 6)

        # Empty masks are never auto-accepted.
        valid_nonempty = (
            np.any(gt > 0) and np.any(pred > 0)
        )

        high_agreement = (
            iou is not None
            and iou >= IOU_THRESHOLD
            and valid_nonempty
        )

        if high_agreement and not REVIEW_AUTO_ACCEPTED:
            # Automatically select SCTransNet's prediction.
            selected_mask = pred
            selected_source = "SCTransNet"
            decision = "auto_accept_prediction"

            auto_accepted += 1

        else:
            # Low-IoU or suspicious masks require manual inspection.
            choice = manual_review(
                name, image_path, gt, pred, iou
            )

            if choice == "q":
                row["decision"] = "quit"
                row["status"] = "not_processed"
                audit_rows.append(row)
                quit_review = True
                break

            if choice == "p":
                selected_mask = pred
                selected_source = "SCTransNet"
                decision = "manual_select_prediction"

            elif choice == "g":
                selected_mask = gt
                selected_source = "original_ground_truth"
                decision = "manual_select_ground_truth"

            else:
                row["decision"] = "skip"
                row["status"] = "not_selected"
                audit_rows.append(row)
                skipped += 1
                continue

            manually_selected += 1

        # Preserve the original image and save the chosen mask.
        copy_image(
            image_path,
            OUTPUT_IMAGE_DIR / image_path.name
        )

        save_mask(
            selected_mask,
            OUTPUT_MASK_DIR / f"{name}.png"
        )

        row["decision"] = decision
        row["selected_source"] = selected_source
        row["status"] = "saved"

        processed += 1

        print(
            f"[SAVED] {name} | "
            f"IoU={iou if iou is not None else 'undefined'} | "
            f"Source={selected_source}"
        )

    except Exception as exc:
        row["decision"] = "error"
        row["status"] = str(exc)
        errors += 1
        print(f"[ERROR] {name}: {exc}")

    finally:
        audit_rows.append(row)

    if quit_review:
        break


# ============================================================
# 7. SAVE AUDIT LOG
# ============================================================

fieldnames = [
    "filename",
    "image_path",
    "ground_truth_path",
    "prediction_path",
    "iou",
    "gt_target_count",
    "pred_target_count",
    "decision",
    "selected_source",
    "status"
]

with open(AUDIT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(audit_rows)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n========== REFINEMENT SUMMARY ==========")
print(f"Auto-accepted predictions: {auto_accepted}")
print(f"Manually selected masks:   {manually_selected}")
print(f"Saved image-mask pairs:    {processed}")
print(f"Skipped images:            {skipped}")
print(f"Errors:                    {errors}")
print(f"Audit log:                 {AUDIT_FILE}")
print(f"Refined dataset:           {OUTPUT_DIR}")
