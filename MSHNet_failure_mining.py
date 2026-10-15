"""
MSHNet failure-case mining for infrared small-target segmentation.

Run from the root of the MSHNet repository (next to main.py, model/ and utils/).
The script follows main.py's test-time preprocessing and threshold convention:
    prediction_mask = model_output > threshold
main.py's test mode (via utils/metric.py) thresholds the RAW LOGITS of MSHNet's fused output
at 0, so threshold defaults to 0.0 (<=> sigmoid probability 0.5).

MSHNet-specific notes (everything else is identical to the DNANet script):
  * Model: model.MSHNet.MSHNet(3). forward(x, warm_flag) returns (side_masks, output). main.py's
    test mode uses warm_flag=True, so the fused `final` head is used here as well.
  * Input: Image.open(...).convert('RGB'), resized to base_size x base_size (BILINEAR image,
    NEAREST mask), ToTensor + Normalize(ImageNet mean/std) -- exactly utils/data.py in 'val' mode.
  * base_size must be a multiple of 16 (the encoder max-pools 4 times).
  * Data layout (utils/data.py): <dataset_dir>/test.txt (ids without extension),
    images/<id>.png, masks/<id>.png.
  * Checkpoint: main.py loads torch.load(path)['state_dict']; the raw state dict (weight.pkl /
    the 'iou_67.87' weights) and the 'net' key of checkpoint.pkl are accepted as well.

Outputs:
  <dataset_dir>/failed_images/<checkpoint_stem>/
      comparison plots for failed cases only
      failure_metrics.csv (one row per evaluated image)
      summary.txt

Failure categories:
  - MISSED TARGET: one or more GT connected components were not matched
  - FALSE ALARM: one or more predicted connected components were unmatched
  - WRONG / POOR DETECTION: a matched target has poor component IoU or poor GT coverage
  - MIXED FAILURE: more than one of the above occurs

Important:
  Component matching is designed for tiny IR targets. A predicted component can match
  a GT component if their centroids are within the configured tolerance OR their
  masks overlap. This is more useful than relying on pixel IoU alone for tiny targets.

  Running script: python mshnet_failure_mining.py --dataset_dir dataset/IRSTD-1k --model_dir weight/MSHNet_weight.tar --base_size 256
  python mshnet_failure_mining.py --dataset_dir dataset/NUDT-SIRST --model_dir MSHNet/weight/NUDT-SIRST_weight.tar --base_size 256
"""

import argparse
import csv
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from tqdm import tqdm

from model.MSHNet import MSHNet

PAD_MULTIPLE = 16   # MSHNet's encoder max-pools 4 times


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def get_args():
    parser = argparse.ArgumentParser(
        description="Find and visualize MSHNet failure cases using GT/prediction component metrics."
    )

    # Repository-compatible arguments
    parser.add_argument("--root", type=str, default="dataset",
                        help="Dataset root directory; dataset folder is <root>/<dataset>.")
    parser.add_argument("--dataset", type=str, default="IRSTD-1k")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Full path of the dataset folder (like main.py's --dataset-dir); "
                             "overrides --root/--dataset.")
    parser.add_argument("--test_list", type=str, default="test.txt",
                        help="Image-id list inside the dataset folder (main.py's 'val' loader reads test.txt).")
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--base_size", type=int, default=256,
                        help="Resize target of the repo's val mode; must be a multiple of 16.")
    parser.add_argument("--model_dir", type=str, default="MSHNet/weight/IRSTD-1k_weight.tar",
                        help="Path to the MSHNet checkpoint (.tar / .pkl).")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', 'cuda:0', or 'cpu'.")

    # Failure mining controls
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Threshold applied directly to the raw model output (logits). "
                             "Repository convention is 0.0.")
    parser.add_argument("--centroid_tolerance", type=float, default=3.0,
                        help="Centroid distance (pixels at base_size resolution) allowed for a match.")
    parser.add_argument("--poor_component_iou", type=float, default=0.25,
                        help="Matched component IoU below this is considered a poor detection.")
    parser.add_argument("--poor_gt_coverage", type=float, default=0.50,
                        help="Fraction of GT component pixels recovered below this is considered poor.")
    parser.add_argument("--min_component_pixels", type=int, default=1,
                        help="Remove predicted/GT connected components smaller than this many pixels.")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8,
                        help="Connected-component connectivity; 8 is standard for small IR targets.")
    parser.add_argument("--save_probability_map", action="store_true",
                        help="Add the continuous model-output map as an extra panel for failed images.")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def load_checkpoint_weights(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:                                   # very old torch without weights_only
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # main.py (test mode) loads weight['state_dict']; training writes weight.pkl (raw state dict)
    # and checkpoint.pkl (key 'net').
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "net"):
            if isinstance(checkpoint.get(key), dict):
                state_dict = checkpoint[key]
                break

    # Support checkpoints saved from DataParallel (main.py --multi-gpus).
    while isinstance(state_dict, dict) and state_dict and all(
            isinstance(k, str) and k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def read_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def read_image_and_mask(dataset_dir, image_id, suffix, base_size, transform):
    image_path = dataset_dir / "images" / f"{image_id}{suffix}"
    mask_path = dataset_dir / "masks" / f"{image_id}{suffix}"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")

    # Match IRSTD_Dataset (val mode): RGB image resized bilinearly, mask resized with nearest-neighbour.
    raw = Image.open(image_path).convert("RGB")
    raw = raw.resize((base_size, base_size), Image.Resampling.BILINEAR)
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((base_size, base_size), Image.Resampling.NEAREST)

    raw_array = np.asarray(raw)
    mask_array = np.asarray(mask)
    gt = mask_array > (0 if mask_array.max() <= 1 else 127)     # masks are 0/255 in the repo, 0/1 also handled
    tensor = transform(raw)
    return raw_array, gt, tensor.unsqueeze(0), image_path


def connected_components(binary_mask, min_pixels=1, connectivity=8):
    structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    labels, count = ndimage.label(binary_mask.astype(bool), structure=structure)
    components = []
    for label_id in range(1, count + 1):
        ys, xs = np.where(labels == label_id)
        area = int(len(xs))
        if area < min_pixels:
            continue
        components.append({
            "label": label_id,
            "area": area,
            "ys": ys,
            "xs": xs,
            "centroid": (float(xs.mean()), float(ys.mean())),
            "mask": labels == label_id,
        })
    return components


def match_components(gt_components, pred_components, centroid_tolerance):
    """
    Hungarian one-to-one matching. A pair is eligible if centroid distance is within
    tolerance OR the two components overlap. Ineligible pairs receive a very high cost.
    """
    ng, npred = len(gt_components), len(pred_components)
    if ng == 0 or npred == 0:
        return [], list(range(ng)), list(range(npred))

    huge = 1e6
    cost = np.full((ng, npred), huge, dtype=np.float64)
    eligible = np.zeros((ng, npred), dtype=bool)
    for gi, gt_comp in enumerate(gt_components):
        gx, gy = gt_comp["centroid"]
        for pi, pred_comp in enumerate(pred_components):
            px, py = pred_comp["centroid"]
            distance = float(np.hypot(gx - px, gy - py))
            overlap = bool(np.any(gt_comp["mask"] & pred_comp["mask"]))
            if distance <= centroid_tolerance or overlap:
                eligible[gi, pi] = True
                # Prefer overlap, then close centroids; tiny epsilon makes assignments stable.
                cost[gi, pi] = distance - (1000.0 if overlap else 0.0) + (gi * 1e-6) + (pi * 1e-8)

    rows, cols = linear_sum_assignment(cost)
    matches = []
    matched_gt, matched_pred = set(), set()
    for gi, pi in zip(rows.tolist(), cols.tolist()):
        if eligible[gi, pi]:
            g = gt_components[gi]
            p = pred_components[pi]
            intersection = int(np.count_nonzero(g["mask"] & p["mask"]))
            union = int(np.count_nonzero(g["mask"] | p["mask"]))
            component_iou = intersection / union if union else 1.0
            gt_coverage = intersection / g["area"] if g["area"] else 1.0
            pred_purity = intersection / p["area"] if p["area"] else 1.0
            distance = float(np.hypot(
                g["centroid"][0] - p["centroid"][0],
                g["centroid"][1] - p["centroid"][1]
            ))
            matches.append({
                "gt_index": gi, "pred_index": pi, "distance": distance,
                "iou": component_iou, "gt_coverage": gt_coverage,
                "pred_purity": pred_purity, "intersection": intersection,
            })
            matched_gt.add(gi)
            matched_pred.add(pi)

    missed_gt = [i for i in range(ng) if i not in matched_gt]
    false_pred = [i for i in range(npred) if i not in matched_pred]
    return matches, missed_gt, false_pred


def evaluate_masks(gt, pred, args):
    gt_components = connected_components(gt, args.min_component_pixels, args.connectivity)
    pred_components = connected_components(pred, args.min_component_pixels, args.connectivity)
    matches, missed_gt, false_pred = match_components(
        gt_components, pred_components, args.centroid_tolerance
    )

    poor_matches = [
        m for m in matches
        if m["iou"] < args.poor_component_iou or m["gt_coverage"] < args.poor_gt_coverage
    ]

    intersection = int(np.count_nonzero(gt & pred))
    gt_pixels = int(np.count_nonzero(gt))
    pred_pixels = int(np.count_nonzero(pred))
    union = int(np.count_nonzero(gt | pred))
    pixel_iou = intersection / union if union else 1.0
    pixel_recall = intersection / gt_pixels if gt_pixels else (1.0 if pred_pixels == 0 else 1.0)
    pixel_precision = intersection / pred_pixels if pred_pixels else (1.0 if gt_pixels == 0 else 0.0)

    categories = []
    if missed_gt:
        categories.append("MISSED TARGET")
    if false_pred:
        categories.append("FALSE ALARM")
    if poor_matches:
        categories.append("WRONG / POOR DETECTION")
    if len(categories) > 1:
        failure_label = "MIXED FAILURE: " + " + ".join(categories)
    elif categories:
        failure_label = categories[0]
    else:
        failure_label = "PASS"

    failed = bool(categories)
    return {
        "failed": failed,
        "failure_label": failure_label,
        "gt_components": gt_components,
        "pred_components": pred_components,
        "matches": matches,
        "missed_gt": missed_gt,
        "false_pred": false_pred,
        "poor_matches": poor_matches,
        "gt_count": len(gt_components),
        "pred_count": len(pred_components),
        "matched_count": len(matches),
        "missed_count": len(missed_gt),
        "false_alarm_count": len(false_pred),
        "poor_match_count": len(poor_matches),
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "intersection_pixels": intersection,
        "pixel_iou": pixel_iou,
        "pixel_recall": pixel_recall,
        "pixel_precision": pixel_precision,
    }


def make_plot(raw, gt, pred, output_map, image_id, metrics, save_path, save_probability_map=False):
    panel_count = 4 if save_probability_map else 3
    fig, axes = plt.subplots(1, panel_count, figsize=(5.2 * panel_count, 5.7))
    if panel_count == 1:
        axes = [axes]

    axes[0].imshow(raw)
    axes[0].set_title("Raw image")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Ground truth\nTargets: {metrics['gt_count']}")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"Prediction — {metrics['failure_label']}\n"
        f"GT={metrics['gt_count']} | Pred={metrics['pred_count']} | "
        f"Missed={metrics['missed_count']} | FA={metrics['false_alarm_count']}\n"
        f"IoU={metrics['pixel_iou']:.3f} | Recall={metrics['pixel_recall']:.3f}"
    )
    if save_probability_map:
        im = axes[3].imshow(output_map, cmap="inferno")
        axes[3].set_title("Raw model output\n(thresholded for prediction)")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{image_id} | {metrics['failure_label']}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    repo_root = Path.cwd()
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(args.root) / args.dataset
    dataset_dir = (dataset_dir if dataset_dir.is_absolute() else repo_root / dataset_dir).resolve()
    checkpoint_path = Path(args.model_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.base_size <= 0 or args.base_size % PAD_MULTIPLE:
        raise ValueError(f"--base_size must be a positive multiple of {PAD_MULTIPLE} for MSHNet.")
    if args.min_component_pixels < 1:
        raise ValueError("--min_component_pixels must be >= 1.")

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    test_list_path = dataset_dir / args.test_list
    if not test_list_path.is_file():
        raise FileNotFoundError(f"Test list not found: {test_list_path}")
    image_ids = read_ids(test_list_path)
    if not image_ids:
        raise RuntimeError(f"No evaluation image IDs found in {test_list_path}.")

    # Match preprocessing used by utils/data.py.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = MSHNet(3).to(device)                       # main.py: model = MSHNet(3)
    load_checkpoint_weights(model, checkpoint_path, device)
    model.eval()

    # Folder is inside the dataset directory and grouped by checkpoint filename.
    checkpoint_stem = checkpoint_path.name
    for suffix in (".pth.tar", ".pt", ".pth", ".tar", ".pkl"):
        if checkpoint_stem.endswith(suffix):
            checkpoint_stem = checkpoint_stem[:-len(suffix)]
            break
    output_dir = dataset_dir / "failed_images" / checkpoint_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "failure_metrics.csv"
    rows = []
    failed_count = 0

    with torch.inference_mode():
        for image_id in tqdm(image_ids, desc="Mining failure cases"):
            raw, gt, input_tensor, _ = read_image_and_mask(
                dataset_dir, image_id, args.suffix, args.base_size, transform
            )
            input_tensor = input_tensor.to(device)
            _, model_result = model(input_tensor, True)     # main.py test: warm_flag = True -> fused logits

            output_map = model_result[0, 0].detach().float().cpu().numpy()
            pred = output_map > args.threshold
            metrics = evaluate_masks(gt, pred, args)

            match_ious = [m["iou"] for m in metrics["matches"]]
            match_coverages = [m["gt_coverage"] for m in metrics["matches"]]
            row = {
                "image_id": image_id,
                "failure": metrics["failed"],
                "failure_label": metrics["failure_label"],
                "gt_targets": metrics["gt_count"],
                "pred_components": metrics["pred_count"],
                "matched_targets": metrics["matched_count"],
                "missed_targets": metrics["missed_count"],
                "false_alarm_components": metrics["false_alarm_count"],
                "poor_matched_targets": metrics["poor_match_count"],
                "gt_pixels": metrics["gt_pixels"],
                "pred_pixels": metrics["pred_pixels"],
                "pixel_iou": round(metrics["pixel_iou"], 6),
                "pixel_recall": round(metrics["pixel_recall"], 6),
                "pixel_precision": round(metrics["pixel_precision"], 6),
                "mean_matched_component_iou": round(float(np.mean(match_ious)), 6) if match_ious else "",
                "mean_matched_gt_coverage": round(float(np.mean(match_coverages)), 6) if match_coverages else "",
                "threshold": args.threshold,
                "base_size": args.base_size,
            }
            rows.append(row)

            if metrics["failed"]:
                failed_count += 1
                safe_id = str(image_id).replace("/", "_").replace("\\", "_")
                plot_path = output_dir / f"{safe_id}_failure.png"
                make_plot(
                    raw, gt, pred, output_map, image_id, metrics, plot_path,
                    save_probability_map=args.save_probability_map
                )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("MSHNet Failure Mining Summary\n")
        f.write("=============================\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Evaluated images: {len(image_ids)}\n")
        f.write(f"Failed images saved: {failed_count}\n")
        f.write(f"Failure rate: {failed_count / len(image_ids):.4%}\n")
        f.write(f"Output threshold: {args.threshold}\n")
        f.write(f"Input resolution: {args.base_size} x {args.base_size}\n")
        f.write(f"Centroid tolerance: {args.centroid_tolerance} px\n")
        f.write(f"Poor component IoU threshold: {args.poor_component_iou}\n")
        f.write(f"Poor GT coverage threshold: {args.poor_gt_coverage}\n")
        f.write(f"Minimum component size: {args.min_component_pixels} px\n")
        f.write("\nFailure definition:\n")
        f.write("- Missed target: GT component has no eligible predicted component match.\n")
        f.write("- False alarm: predicted component has no eligible GT component match.\n")
        f.write("- Wrong/poor detection: matched component IoU or GT coverage is below threshold.\n")
        f.write("- Only failed cases receive comparison PNGs; CSV includes every evaluated image.\n")

    print("\nDone.")
    print(f"Evaluated images: {len(image_ids)}")
    print(f"Failed images: {failed_count}")
    print(f"Output folder: {output_dir}")
    print(f"Metrics CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()