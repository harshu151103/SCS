"""
SCTransNet failure-case mining for infrared small-target segmentation.

Run from the root of the SCTransNet repository (next to model/ and datasets/).
The script follows the repository's test-time preprocessing and threshold convention:
    prediction_mask = model_output_probability > threshold
SCTransNet (mode='test') returns sigmoid PROBABILITIES, and the repository's test.py uses
output > 0.5 (mIoU) and output > opt.threshold (Pd/Fa, default 0.5), so threshold defaults to 0.5.
(0.5 on the probability is the same decision as logit > 0, i.e. the DNANet convention.)

Outputs:
  datasets/<dataset>/failed_images/<checkpoint_stem>/
      comparison plots for failed cases only
      failure_metrics.csv (one row per evaluated image)
      summary.txt

Failure categories (identical to the DNANet script):
  - MISSED TARGET: one or more GT connected components were not matched
  - FALSE ALARM: one or more predicted connected components were unmatched
  - WRONG / POOR DETECTION: a matched target has poor component IoU or poor GT coverage
  - MIXED FAILURE: more than one of the above occurs

Important:
  Component matching is designed for tiny IR targets. A predicted component can match
  a GT component if their centroids are within the configured tolerance OR their
  masks overlap. This is more useful than relying on pixel IoU alone for tiny targets.

SCTransNet-SPECIFIC DIFFERENCES FROM THE DNANet SCRIPT
------------------------------------------------------
* Model: model.SCTransNet.SCTransNet(model.Config.get_SCTrans_config(), mode='test',
  deepsuper=...). No backbone / channel_size / num_blocks arguments exist for this network.
* Input: 1 channel, read with convert('I') (keeps 16-bit range) and normalised with the
  mean/std of the TRAINING dataset (--train_dataset), exactly like the repo's TestSetLoader.
  There is NO ImageNet RGB normalisation as in DNANet.
* Resolution: by default images are evaluated at ORIGINAL size, zero-padded (after
  normalisation) at the bottom/right to a multiple of 32 and the output is cropped back,
  exactly like the repo's test.py. '--native_res False' resizes to base_size x base_size.
* Split: <root>/<dataset>/img_idx/test_<dataset>.txt (no load_dataset helper / split_method).
* Checkpoint: the released weights store keys with a wrapper prefix ('model.' / 'module.');
  it is stripped automatically.

Running script:
  python sctransnet_failure_mining.py --dataset NUAA-SIRST --train_dataset SIRST3 --model_dir SIRST3/SCTransNet_NUAA_NUDT_IRSTD1K.pth.tar
"""

import argparse
import csv
import os
from collections import Counter
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
from tqdm import tqdm

import model.Config as config
from model.SCTransNet import SCTransNet


# Per-dataset intensity statistics, copied from get_img_norm_cfg() in the SCTransNet repo's utils.py
# (same table as evaluate_SCTransNet.py).
NORM_CFGS = {
    "NUAA-SIRST": dict(mean=101.06385040283203, std=34.619606018066406),
    "NUDT-SIRST": dict(mean=107.80905151367188, std=33.02274703979492),
    "IRSTD-1K": dict(mean=87.4661865234375, std=39.71953201293945),
    "SIRST2": dict(mean=101.06385040283203, std=34.619606018066406),
    "SIRST3": dict(mean=101.06385040283203, std=34.619606018066406),
    "NUDT-SIRST-Sea": dict(mean=43.62403869628906, std=18.91838264465332),
    "SIRST4": dict(mean=101.06385040283203, std=34.619606018066406),
    "SIRST5": dict(mean=101.06385040283203, std=34.619606018066406),
    "SIRST6": dict(mean=101.06385040283203, std=34.619606018066406),
    "SIRST7": dict(mean=101.06385040283203, std=34.619606018066406),
    "IRDST-real": dict(mean=101.54053497314453, std=56.49856185913086),
}

PAD_MULTIPLE = 32  # SCTransNet halves the resolution 4 times; the repo pads inputs to a multiple of 32


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def get_args():
    parser = argparse.ArgumentParser(
        description="Find and visualize SCTransNet failure cases using GT/prediction component metrics."
    )

    # Repository-compatible arguments
    parser.add_argument("--root", type=str, default="datasets",
                        help="Dataset root directory; dataset folder is <root>/<dataset>.")
    parser.add_argument("--dataset", type=str, default="NUAA-SIRST",
                        help="Test dataset; uses <root>/<dataset>/img_idx/test_<dataset>.txt")
    parser.add_argument("--train_dataset", type=str, default="SIRST3",
                        help="Dataset the weights were trained on: its mean/std normalise the input "
                             "(the repo's train_dataset_name; SIRST3 for the released weights).")
    parser.add_argument("--img_mean", type=float, default=None, help="Override the normalisation mean.")
    parser.add_argument("--img_std", type=float, default=None, help="Override the normalisation std.")
    parser.add_argument("--suffix", type=str, default=".png",
                        help="Image/mask extension; '.bmp' is tried as a fallback, like the repo.")
    parser.add_argument("--base_size", type=int, default=256,
                        help="Only used with --native_res False (must be a multiple of 32).")
    parser.add_argument("--native_res", type=parse_bool, nargs="?", const=True, default=True,
                        help="Evaluate at original resolution (zero-pad to a multiple of 32, as the repo does). "
                             "'--native_res False' resizes to base_size x base_size instead.")
    parser.add_argument("--deep_supervision", type=parse_bool, default=True,
                        help="Must match how the checkpoint was trained (the released weights use True).")
    parser.add_argument("--model", type=str, default="SCTransNet", choices=["SCTransNet"])
    parser.add_argument("--model_dir", type=str,
                        default="SIRST3/SCTransNet_NUAA_NUDT_IRSTD1K.pth.tar",
                        help="Path to the .pth.tar checkpoint, e.g. "
                             "SIRST3/SCTransNet_NUAA_NUDT_IRSTD1K.pth.tar")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', 'cuda:0', or 'cpu'.")

    # Failure mining controls
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold applied to the model's sigmoid PROBABILITY output (0 < t < 1). "
                             "Repository convention is 0.5 (== logit > 0).")
    parser.add_argument("--centroid_tolerance", type=float, default=3.0,
                        help="Centroid distance (pixels at the evaluated resolution) allowed for a match.")
    parser.add_argument("--poor_component_iou", type=float, default=0.25,
                        help="Matched component IoU below this is considered a poor detection.")
    parser.add_argument("--poor_gt_coverage", type=float, default=0.50,
                        help="Fraction of GT component pixels recovered below this is considered poor.")
    parser.add_argument("--min_component_pixels", type=int, default=1,
                        help="Remove predicted/GT connected components smaller than this many pixels.")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8,
                        help="Connected-component connectivity; 8 is standard for small IR targets.")
    parser.add_argument("--save_probability_map", action="store_true",
                        help="Add the continuous model-output (probability) map as an extra panel for failed images.")
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
    except TypeError:  # very old torch without weights_only
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    # SCTransNet checkpoints are saved through a wrapper (prefix 'model.') and/or DataParallel
    # (prefix 'module.'). Strip only prefixes shared by EVERY key; repeat for 'module.model.*'.
    if isinstance(state_dict, dict):
        while state_dict:
            changed = False
            for prefix in ("module.", "model."):
                if all(isinstance(k, str) and k.startswith(prefix) for k in state_dict):
                    state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
                    changed = True
                    break
            if not changed:
                break
        if not state_dict:
            raise RuntimeError("Checkpoint state_dict is empty.")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "Checkpoint does not match the model definition. Check --deep_supervision "
            "against how it was trained (the released weights use True).\n" + str(e)
        )
    return checkpoint


def read_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()  # also strips the '\r' of CRLF files
            if s:
                ids.append(s)
    return ids


def find_file(folder, image_id, suffix):
    """The repo tries .png first and falls back to .bmp; --suffix replaces the first choice."""
    for sfx in (suffix, ".bmp"):
        p = folder / f"{image_id}{sfx}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing file: {folder / (image_id + suffix)} (also tried .bmp)")


def binarize_mask(arr):
    """Masks are 0/255 in the repo's datasets but 0/1 is common elsewhere."""
    return arr > (0 if arr.max() <= 1 else 127)


def get_norm_cfg(train_dataset, root_dir, suffix, override_mean=None, override_std=None):
    """Mean/std used to normalise the input; same rules as get_img_norm_cfg() in the repo."""
    if override_mean is not None and override_std is not None:
        return dict(mean=float(override_mean), std=float(override_std))
    if train_dataset in NORM_CFGS:
        return dict(NORM_CFGS[train_dataset])
    # Unknown (e.g. custom) dataset: compute like the repo from its train + test lists.
    base = root_dir / train_dataset
    ids = read_ids(base / "img_idx" / f"train_{train_dataset}.txt") + \
        read_ids(base / "img_idx" / f"test_{train_dataset}.txt")
    means, stds = [], []
    for image_id in tqdm(ids, desc=f"Computing mean/std of '{train_dataset}'"):
        a = np.array(Image.open(find_file(base / "images", image_id, suffix)).convert("I"), dtype=np.float32)
        means.append(a.mean())
        stds.append(a.std())
    return dict(mean=float(np.array(means).mean()), std=float(np.array(stds).mean()))


def pad_to_multiple(arr, times=PAD_MULTIPLE):
    """Zero-pad the last two axes (bottom / right) up to a multiple of `times` (repo's PadImg)."""
    h, w = arr.shape[-2:]
    ph, pw = (-h) % times, (-w) % times
    if not (ph or pw):
        return arr
    return np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)], mode="constant")


def read_image_and_mask(dataset_dir, image_id, suffix, norm_cfg, native_res, base_size, mismatch_log):
    image_path = find_file(dataset_dir / "images", image_id, suffix)
    mask_path = find_file(dataset_dir / "masks", image_id, suffix)

    img = Image.open(image_path)
    msk = Image.open(mask_path)
    if img.size != msk.size:
        mismatch_log.append(image_id)
        msk = msk.resize(img.size, resample=Image.Resampling.NEAREST)

    # Match TestSetLoader: single-channel image read with convert('I'), no ImageNet normalisation.
    gray = np.array(img.convert("I"), dtype=np.float32)
    gt = binarize_mask(np.asarray(msk.convert("L")))

    if not native_res:
        gray = np.asarray(Image.fromarray(gray).resize((base_size, base_size), Image.BILINEAR),
                          dtype=np.float32)
        gt_img = Image.fromarray(gt.astype(np.uint8) * 255).resize((base_size, base_size), Image.NEAREST)
        gt = np.asarray(gt_img) > 127

    # Same as the repo's Normalized(): (img - mean) / std, then [1, H, W].
    x = ((gray - norm_cfg["mean"]) / norm_cfg["std"])[np.newaxis, :].astype(np.float32)
    return gray, gt, np.ascontiguousarray(x), image_path


def to_display_image(gray):
    """Scale the raw single-channel image to [0, 1] for plotting (8-bit images keep their true range)."""
    lo, hi = float(gray.min()), float(gray.max())
    if lo >= 0.0 and hi <= 255.0:
        lo, hi = 0.0, 255.0
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def predict_probability(model, x_np, device, native_res):
    """Returns SCTransNet's final-head sigmoid probability as a float32 [H, W] array."""
    h, w = x_np.shape[-2:]
    if native_res:  # zeros AFTER normalisation, like the repo's PadImg
        x_np = pad_to_multiple(x_np)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0).to(device)
    out = model(x)
    if isinstance(out, (list, tuple)):  # train-mode safety: last element is the main head
        out = out[-1]
    prob = out[0, 0, :h, :w].detach().float().cpu().numpy()  # crop the padding away
    if not np.isfinite(prob).all():
        raise RuntimeError("Model produced NaN/Inf outputs.")
    return prob


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


def make_plot(raw, gt, pred, output_map, image_id, metrics, save_path, threshold,
              save_probability_map=False):
    panel_count = 4 if save_probability_map else 3
    fig, axes = plt.subplots(1, panel_count, figsize=(5.2 * panel_count, 5.7))
    if panel_count == 1:
        axes = [axes]

    axes[0].imshow(raw, cmap="gray", vmin=0, vmax=1)
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
        im = axes[3].imshow(output_map, cmap="inferno", vmin=0, vmax=1)
        axes[3].set_title(f"Raw model output (sigmoid probability)\n(thresholded at p > {threshold:g})")
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
    root_dir = (repo_root / args.root).resolve()
    dataset_dir = (root_dir / args.dataset).resolve()
    checkpoint_path = Path(args.model_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be strictly between 0 and 1 (SCTransNet outputs probabilities).")
    if args.min_component_pixels < 1:
        raise ValueError("--min_component_pixels must be >= 1.")
    if args.centroid_tolerance <= 0:
        raise ValueError("--centroid_tolerance must be positive.")
    if not args.native_res:
        if args.base_size < PAD_MULTIPLE or args.base_size % PAD_MULTIPLE:
            raise ValueError(f"--base_size must be a positive multiple of {PAD_MULTIPLE} for SCTransNet.")
    if (args.img_mean is None) != (args.img_std is None):
        raise ValueError("--img_mean and --img_std must be given together.")

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    test_list_path = dataset_dir / "img_idx" / f"test_{args.dataset}.txt"
    if not test_list_path.is_file():
        raise FileNotFoundError(f"Test list not found: {test_list_path}")
    image_ids = read_ids(test_list_path)
    if not image_ids:
        raise RuntimeError(f"{test_list_path.name} is empty; no evaluation image IDs.")

    # Match preprocessing used by the supplied test.py / TestSetLoader.
    norm_cfg = get_norm_cfg(args.train_dataset, root_dir, args.suffix, args.img_mean, args.img_std)
    print(f"Input normalisation ({args.train_dataset}): mean={norm_cfg['mean']:.4f} std={norm_cfg['std']:.4f}")
    resolution_text = (f"native (zero-padded to a multiple of {PAD_MULTIPLE})" if args.native_res
                       else f"{args.base_size} x {args.base_size} (resized)")
    print(f"Resolution: {resolution_text}")

    model = SCTransNet(config.get_SCTrans_config(), mode="test",
                       deepsuper=args.deep_supervision).to(device)
    load_checkpoint_weights(model, checkpoint_path, device)
    model.eval()

    # Folder is inside the dataset directory and grouped by checkpoint filename.
    checkpoint_stem = checkpoint_path.name
    for suffix in (".pth.tar", ".pt", ".pth", ".tar"):
        if checkpoint_stem.endswith(suffix):
            checkpoint_stem = checkpoint_stem[:-len(suffix)]
            break
    output_dir = dataset_dir / "failed_images" / checkpoint_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "failure_metrics.csv"
    rows = []
    failed_count = 0
    mismatch_log = []

    with torch.inference_mode():
        for image_id in tqdm(image_ids, desc="Mining failure cases"):
            gray, gt, x_np, _ = read_image_and_mask(
                dataset_dir, image_id, args.suffix, norm_cfg, args.native_res, args.base_size, mismatch_log
            )
            output_map = predict_probability(model, x_np, device, args.native_res)
            if output_map.shape != gt.shape:
                raise RuntimeError(f"Shape mismatch for {image_id}: {output_map.shape} vs {gt.shape}")
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
                "image_height": int(gt.shape[0]),
                "image_width": int(gt.shape[1]),
            }
            rows.append(row)

            if metrics["failed"]:
                failed_count += 1
                safe_id = str(image_id).replace("/", "_").replace("\\", "_")
                plot_path = output_dir / f"{safe_id}_failure.png"
                make_plot(
                    to_display_image(gray), gt, pred, output_map, image_id, metrics, plot_path,
                    args.threshold, save_probability_map=args.save_probability_map
                )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(r["failure_label"] for r in rows if r["failure"])

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("SCTransNet Failure Mining Summary\n")
        f.write("=================================\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Evaluated images: {len(image_ids)}\n")
        f.write(f"Failed images saved: {failed_count}\n")
        f.write(f"Failure rate: {failed_count / len(image_ids):.4%}\n")
        f.write(f"Output threshold: p > {args.threshold} (sigmoid probability)\n")
        f.write(f"Input resolution: {resolution_text}\n")
        f.write(f"Input normalisation ({args.train_dataset}): mean={norm_cfg['mean']:.4f}, std={norm_cfg['std']:.4f}\n")
        f.write(f"Centroid tolerance: {args.centroid_tolerance} px\n")
        f.write(f"Poor component IoU threshold: {args.poor_component_iou}\n")
        f.write(f"Poor GT coverage threshold: {args.poor_gt_coverage}\n")
        f.write(f"Minimum component size: {args.min_component_pixels} px\n")
        f.write("\nFailed images by category:\n")
        if label_counts:
            for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write(f"- {label}: {count}\n")
        else:
            f.write("- none\n")
        if mismatch_log:
            f.write(f"\nNote: {len(mismatch_log)} image/mask pairs had different sizes; "
                    f"the mask was resized (nearest-neighbour) to the image size.\n")
        f.write("\nFailure definition:\n")
        f.write("- Missed target: GT component has no eligible predicted component match.\n")
        f.write("- False alarm: predicted component has no eligible GT component match.\n")
        f.write("- Wrong/poor detection: matched component IoU or GT coverage is below threshold.\n")
        f.write("- Only failed cases receive comparison PNGs; CSV includes every evaluated image.\n")

    print("\nDone.")
    print(f"Evaluated images: {len(image_ids)}")
    print(f"Failed images: {failed_count}")
    if mismatch_log:
        print(f"[WARNING] {len(mismatch_log)} image/mask pairs differed in size; masks were resized to the image.")
    print(f"Output folder: {output_dir}")
    print(f"Metrics CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()