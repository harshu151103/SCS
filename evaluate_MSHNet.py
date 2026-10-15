"""
MSHNet evaluation for small / dim infrared aerial targets (drones, aircraft, ...).

Run from the MSHNet repository root (next to model/, utils/ and main.py).

    python evaluate_MSHNet.py --dataset_dir dataset/NUDT-SIRST --model_dir "MSHNe/tweight/NUDT-SIRST_weight.tar"

    python evaluate_MSHNet.py --self_test        # verifies the metric code, no GPU/data needed

MSHNet-SPECIFIC NOTES (everything else below is identical to the SCTransNet script)
-----------------------------------------------------------------------------------
* Model: model.MSHNet.MSHNet(3). forward(x, warm_flag) returns (list_of_side_masks, output).
  main.py's test path calls it with warm_flag=True (warm_epoch = -1 and epoch = 1), i.e. the
  output is the fused `final` head over the four scales. That output is RAW LOGITS (no
  sigmoid); the repo's mIoU / PD_FA threshold it with (output > 0). The logits are used
  directly here, and a probability threshold p is converted with logit(p) (p > t <=> logit >
  log(t / (1 - t)); p = 0.5 <=> logit > 0).
* Input: 3 channels. The repo reads Image.open(...).convert('RGB'), applies ToTensor
  (uint8 -> [0, 1]) and Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]) (ImageNet
  statistics, fixed, not dataset-dependent).
* Resolution: the repo's val mode (IRSTD_Dataset._testval_sync_transform) resizes image
  (BILINEAR) and mask (NEAREST) to base_size x base_size (default 256), and PD_FA reshapes
  to base_size x base_size. This is therefore the DEFAULT here (--native_res False).
  --native_res True evaluates at original resolution instead: no resize, zero-padding AFTER
  normalisation to a multiple of 16 (4 max-pools in the encoder) and cropping back. The
  repo itself never does this, so its numbers will differ from the repo's.
* Data layout (as in utils/data.py): <dataset_dir>/test.txt (ids without extension; this is
  what main.py's 'val' loader reads), <dataset_dir>/trainval.txt, images/<id>.png,
  masks/<id>.png.
* Checkpoint: main.py's test mode loads torch.load(path)['state_dict'] (strict). This script
  also accepts the raw state dict (the 'iou_67.87' weights / weight.pkl written by training)
  and the 'net' key of the training checkpoint.pkl.

WHAT IS REPORTED (and why)
--------------------------
Headline, all at one fixed probability threshold (default 0.5 == logit > 0, the
repo's convention), with 95% bootstrap confidence intervals over images:

  mIoU    pooled pixel IoU = sum(TP) / sum(TP+FP+FN). Same definition the repo's
          utils/metric.py reports as "mIoU".
  nIoU    mean of per-image IoU over images that contain a target. Not dominated
          by large targets (a common companion metric in IRSTD papers).
  Pd      detected targets / all targets (target-level recall).
  Fa      area of false-alarm components / total pixels   (repo convention;
          shown in units of 1e-6 like the README tables).
  FA/img  number of false-alarm components per image (easier to interpret).
  Alert precision   detected targets / predicted components.

Diagnostics that matter for *small and dim* targets:
  * Pd split by target size and by local contrast (SCR), with counts.
  * Pd / Fa / mIoU across a sweep of thresholds (the operating-point trade-off).
  * Per-target CSV (area, contrast, peak score, detected?) to inspect misses.
  * Target-free images: how many produce false alarms.

Deliberately NOT reported: pixel accuracy, specificity, NPV, MCC, balanced
accuracy, FPR/FNR, Brier, MAE, ROC-AUC, Dice (== F1). With ~99.9% background
pixels these are all ~1.0 or redundant and say nothing about small-target
detection quality.

ALSO REPORTED (pixel-level, pooled over all images, with 95% bootstrap CIs):
  Precision  TP / (TP + FP)       Recall  TP / (TP + FN)       F1  2PR / (P + R)
  They are also reported across the threshold sweep. (This overrides the "NOT reported"
  note above for Precision, Recall and F1 only.)

MATCHING RULE (target-level)
----------------------------
A predicted component can detect a GT component if their centroid distance is
< --centroid_tolerance pixels (strict '<', as in the repo). Assignment is
one-to-one and optimal (Hungarian), so it does not depend on component order.
Unmatched predicted components are false alarms; Fa = their pixel area / pixels.
Use --match_rule overlap to instead count any pixel overlap as a hit (fairer for
larger aircraft whose centroid can shift by more than 3 px).
The repo's PD_FA matches greedily (first predicted component within 3 px, per GT component)
and removes matched areas from the false-alarm list BY AREA VALUE, so an unmatched
component whose area equals a matched one is dropped from Fa. Pd / Fa here can therefore
differ slightly from the repo's printed numbers; mIoU is computed the same way.

TRUST CAVEATS (printed in every report as well)
-----------------------------------------------
* main.py keeps weight.pkl at the epoch with the best mIoU on test.txt (its 'val' loader
  reads test.txt). If you evaluate that checkpoint on that list, the score is optimistic.
  For an honest number use a held-out split that was never used for checkpoint choice.
* By default evaluation runs at base_size x base_size (the repo's protocol). Resizing masks
  can erase tiny targets, and the script counts GT objects lost this way.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows OpenMP duplicate-lib workaround

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **_):
        return iterable


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Input normalisation of utils/data.py (transforms.Normalize), applied to the RGB image scaled to [0, 1].
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PAD_MULTIPLE = 16   # MSHNet's encoder max-pools 4 times (only used with --native_res True)

SWEEP_PROBS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
               0.70, 0.80, 0.90, 0.95, 0.99]

# Target-size bins in pixels of the *evaluated* resolution.
SIZE_BINS = [(1, 9, "tiny   (1-9 px)"),
             (10, 36, "small  (10-36 px)"),
             (37, 100, "medium (37-100 px)"),
             (101, math.inf, "large  (>100 px)")]

# Local-contrast estimation: background ring starts GAP px outside the target
# and is RING px wide (Chebyshev distance).
CONTRAST_GAP = 2
CONTRAST_RING = 8

STRUCT = {4: ndimage.generate_binary_structure(2, 1),
          8: ndimage.generate_binary_structure(2, 2)}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def prob_to_logit(p):
    return math.log(p / (1.0 - p))


def sigmoid(x):
    """Scalar sigmoid for the per-target 'peak_prob' column (MSHNet outputs raw logits)."""
    return float(1.0 / (1.0 + math.exp(-max(min(float(x), 700.0), -700.0))))


def finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def fmt(x, digits=4, scale=1.0):
    x = finite_or_none(x)
    return "n/a" if x is None else f"{x * scale:.{digits}f}"


def wilson_interval(k, n, z=1.96):
    """95% Wilson interval for a binomial proportion."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return finite_or_none(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# --------------------------------------------------------------------------- #
# Connected components + object matching (torch-free, unit-tested by --self_test)
# --------------------------------------------------------------------------- #
def label_components(mask, connectivity=8, min_pixels=1):
    """
    Returns (labels, areas, centroids_xy).
    labels: int array, 0 = background, 1..n = kept components (contiguous ids).
    areas: (n,) int. centroids_xy: (n, 2) float, columns are (x, y).
    Components smaller than min_pixels are removed.
    """
    labels, n = ndimage.label(mask, structure=STRUCT[connectivity])
    empty = (np.zeros(mask.shape, np.int64), np.zeros(0, np.int64), np.zeros((0, 2)))
    if n == 0:
        return empty

    areas_all = np.bincount(labels.ravel(), minlength=n + 1)
    keep = areas_all >= min_pixels
    keep[0] = False
    n_keep = int(keep.sum())
    if n_keep == 0:
        return empty
    if n_keep != n:
        lookup = np.zeros(n + 1, dtype=np.int64)
        lookup[keep] = np.arange(1, n_keep + 1)
        labels = lookup[labels]
        n = n_keep
    labels = labels.astype(np.int64, copy=False)

    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    cx = np.bincount(lab, weights=xs, minlength=n + 1)[1:] / areas
    cy = np.bincount(lab, weights=ys, minlength=n + 1)[1:] / areas
    return labels, areas.astype(np.int64), np.stack([cx, cy], axis=1)


def overlap_matrix(gt_labels, pred_labels, n_gt, n_pred):
    both = (gt_labels > 0) & (pred_labels > 0)
    idx = (gt_labels[both] - 1) * n_pred + (pred_labels[both] - 1)
    return np.bincount(idx, minlength=n_gt * n_pred).reshape(n_gt, n_pred)


def match_objects(gt_cent, pred_cent, tolerance, rule="centroid", overlap=None):
    """
    One-to-one optimal matching.
    Returns (gt_matched[bool n_gt], pred_matched[bool n_pred], gt_dist[float n_gt]).
    """
    n_gt, n_pred = len(gt_cent), len(pred_cent)
    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)
    gt_dist = np.full(n_gt, np.nan)
    if n_gt == 0 or n_pred == 0:
        return gt_matched, pred_matched, gt_dist

    dist = np.hypot(gt_cent[:, None, 0] - pred_cent[None, :, 0],
                    gt_cent[:, None, 1] - pred_cent[None, :, 1])
    eligible = (dist < tolerance) if rule == "centroid" else (overlap > 0)
    if not eligible.any():
        return gt_matched, pred_matched, gt_dist

    # Any eligible pair is far cheaper than any ineligible one, so the assignment
    # maximises the number of matches first, then minimises centroid distance.
    cost = np.where(eligible, dist, 1e9)
    rows, cols = linear_sum_assignment(cost)
    ok = eligible[rows, cols]
    rows, cols = rows[ok], cols[ok]
    gt_matched[rows] = True
    pred_matched[cols] = True
    gt_dist[rows] = dist[rows, cols]
    return gt_matched, pred_matched, gt_dist


def score_at_threshold(gt, gt_labels, gt_cent, logit, logit_thr, cfg):
    """All per-image counts at one threshold. `gt` is a bool array."""
    pred = logit > logit_thr                       # strict '>' like the repo (logit > 0)
    tp = int(np.count_nonzero(gt & pred))
    pred_px = int(np.count_nonzero(pred))
    gt_px = int(np.count_nonzero(gt))

    p_labels, p_areas, p_cent = label_components(pred, cfg.connectivity, cfg.min_component_pixels)
    n_gt, n_pred = len(gt_cent), len(p_cent)

    overlap = None
    if cfg.match_rule == "overlap" and n_gt and n_pred:
        overlap = overlap_matrix(gt_labels, p_labels, n_gt, n_pred)

    gt_m, pred_m, gt_dist = match_objects(gt_cent, p_cent, cfg.centroid_tolerance,
                                          cfg.match_rule, overlap)
    unmatched = ~pred_m
    counts = {
        "tp": tp,
        "fp": pred_px - tp,
        "fn": gt_px - tp,
        "n_pred": n_pred,
        "matched": int(gt_m.sum()),
        "fa_objs": int(unmatched.sum()),
        "fa_area": int(p_areas[unmatched].sum()) if n_pred else 0,
    }
    return counts, gt_m, gt_dist


# --------------------------------------------------------------------------- #
# Target descriptors (size / local contrast) for the "small and dim" analysis
# --------------------------------------------------------------------------- #
def target_descriptors(gray, gt_labels, n_gt):
    """Per GT component: local contrast (gray levels) and SCR vs. a background ring."""
    out = []
    slices = ndimage.find_objects(gt_labels)
    margin = CONTRAST_GAP + CONTRAST_RING + 1
    h, w = gt_labels.shape
    for k in range(1, n_gt + 1):
        sl = slices[k - 1]
        y0, y1 = max(sl[0].start - margin, 0), min(sl[0].stop + margin, h)
        x0, x1 = max(sl[1].start - margin, 0), min(sl[1].stop + margin, w)
        lab = gt_labels[y0:y1, x0:x1]
        g = gray[y0:y1, x0:x1]
        comp = lab == k
        others = (lab > 0) & ~comp
        inner = ndimage.binary_dilation(comp, STRUCT[8], iterations=CONTRAST_GAP)
        outer = ndimage.binary_dilation(comp, STRUCT[8], iterations=CONTRAST_GAP + CONTRAST_RING)
        ring = outer & ~inner & ~others
        if ring.sum() < 8:
            out.append((float("nan"), float("nan")))
            continue
        mu_t = float(g[comp].mean())
        mu_b = float(g[ring].mean())
        sd_b = max(float(g[ring].std()), 1.0)       # floor: avoids blow-up on flat backgrounds
        out.append((mu_t - mu_b, (mu_t - mu_b) / sd_b))
    return out


# --------------------------------------------------------------------------- #
# Aggregation, bootstrap
# --------------------------------------------------------------------------- #
def stats_from_sums(S):
    """Works for scalars and for arrays (bootstrap replicates)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "miou": S["tp"] / (S["tp"] + S["fp"] + S["fn"]),
            "pd": S["matched"] / S["n_gt"],
            "fa": S["fa_area"] / S["pixels"],
            "fa_per_image": S["fa_objs"] / S["n_img"],
            "alert_precision": S["matched"] / S["n_pred"],
            # ---- added: pixel-level Precision / Recall / F1 (pooled over all images) ----
            "precision": S["tp"] / (S["tp"] + S["fp"]),
            "recall": S["tp"] / (S["tp"] + S["fn"]),
            "f1": 2 * S["tp"] / (2 * S["tp"] + S["fp"] + S["fn"]),   # == 2PR/(P+R)
        }


def totals(A, t):
    S = {k: A[k][:, t].sum() for k in ("tp", "fp", "fn", "n_pred", "matched", "fa_objs", "fa_area")}
    S["n_gt"] = A["n_gt"].sum()
    S["pixels"] = A["pixels"].sum()
    S["n_img"] = len(A["n_gt"])
    return {k: np.float64(v) for k, v in S.items()}


def bootstrap_headline(A, t, n_boot, seed):
    """Image-level (cluster) bootstrap: resample whole images with replacement."""
    n = len(A["n_gt"])
    if n_boot <= 0 or n < 2:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    S = {k: A[k][:, t][idx].sum(axis=1).astype(np.float64)
         for k in ("tp", "fp", "fn", "n_pred", "matched", "fa_objs", "fa_area")}
    S["n_gt"] = A["n_gt"][idx].sum(axis=1).astype(np.float64)
    S["pixels"] = A["pixels"][idx].sum(axis=1).astype(np.float64)
    S["n_img"] = np.float64(n)
    boot = stats_from_sums(S)

    has_gt = (A["n_gt"] > 0).astype(np.float64)
    per_img_iou = A["niou_terms"][:, t]
    with np.errstate(divide="ignore", invalid="ignore"):
        boot["niou"] = (per_img_iou * has_gt)[idx].sum(axis=1) / has_gt[idx].sum(axis=1)

    return {k: (finite_or_none(np.nanpercentile(v, 2.5)), finite_or_none(np.nanpercentile(v, 97.5)))
            for k, v in boot.items() if np.isfinite(v).any()}


# --------------------------------------------------------------------------- #
# Data loading (torch-free)
# --------------------------------------------------------------------------- #
def read_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()           # also strips the '\r' of CRLF files
            if s:
                ids.append(s)
    return ids


def binarize_mask(arr):
    """Masks are 0/255 in the repo's datasets but 0/1 is common elsewhere."""
    return arr > (0 if arr.max() <= 1 else 127)


def find_file(folder, image_id, suffix):
    """The repo tries .png first and falls back to .bmp; --suffix replaces the first choice."""
    for sfx in (suffix, ".bmp"):
        p = folder / f"{image_id}{sfx}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing file: {folder / (image_id + suffix)} (also tried .bmp)")


def load_pair(dataset_dir, image_id, suffix, size, notes):
    """
    Mirrors IRSTD_Dataset.__getitem__ in 'val' mode: image -> convert('RGB'), mask as stored;
    with `size`, image is resized BILINEAR and mask NEAREST to size x size; then ToTensor + Normalize.
    Returns (x[3,H,W] float32 normalised, gray[H,W] float32 in 0..255 (luma of the RGB image, only
    used for the local-contrast descriptors), gt bool[H,W], n_native, n_resized).
    """
    img_path = find_file(dataset_dir / "images", image_id, suffix)
    msk_path = find_file(dataset_dir / "masks", image_id, suffix)

    img = Image.open(img_path)
    msk = Image.open(msk_path)
    if img.mode in ("I;16", "I", "F"):
        notes["non_8bit_images"] += 1       # convert('RGB') is applied exactly as in the repo
    notes["orig_sizes"][img.size] += 1

    # In resize mode the repo resizes image and mask independently to base_size, so a size mismatch is harmless.
    # Without resizing the two must be on the same grid.
    if size is None and img.size != msk.size:
        print(
            f"[WARNING] Dimension mismatch for {image_id}: "
            f"Image={img.size}, Mask={msk.size}. "
            f"Resizing mask to match image."
        )
        msk = msk.resize(img.size, resample=Image.Resampling.NEAREST)

    rgb = img.convert("RGB")                                  # same read as the repo's IRSTD_Dataset
    mask_arr = np.asarray(msk.convert("L"))
    vals = set(np.unique(mask_arr).tolist())
    if not (vals <= {0, 1} or vals <= {0, 255}):
        notes["odd_mask_values"] += 1
    gt = binarize_mask(mask_arr)

    n_native = n_resized = None
    if size is not None:
        n_native = label_components(gt, 8)[1].size
        rgb = rgb.resize((size, size), Image.BILINEAR)
        gt_img = Image.fromarray(gt.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)
        gt = np.asarray(gt_img) > 127
        n_resized = label_components(gt, 8)[1].size

    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    return to_input_array(rgb), gray, gt, n_native, n_resized


def pad_to_multiple(arr, times=PAD_MULTIPLE):
    """Zero-pad the last two axes (bottom / right) up to a multiple of `times`."""
    h, w = arr.shape[-2:]
    ph, pw = (-h) % times, (-w) % times
    if not (ph or pw):
        return arr
    return np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)], mode="constant")


def to_input_array(rgb):
    """ToTensor + Normalize(ImageNet mean/std) of utils/data.py on a PIL RGB image (or uint8 HxWx3 array)."""
    a = np.asarray(rgb, dtype=np.uint8).astype(np.float32) / 255.0     # ToTensor: uint8 -> [0, 1]
    a = a.transpose(2, 0, 1)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((a - mean) / std, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def parse_bool(v):
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def build_and_load_model(args, device):
    import torch
    from model.MSHNet import MSHNet

    model = MSHNet(3).to(device)                      # main.py: model = MSHNet(3)

    try:
        ckpt = torch.load(str(args.model_path), map_location=device, weights_only=False)
    except TypeError:                                 # very old torch without weights_only
        ckpt = torch.load(str(args.model_path), map_location=device)

    # main.py (test mode) loads weight['state_dict']; training writes weight.pkl (raw state dict)
    # and checkpoint.pkl (key 'net').
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("state_dict", "net"):
            if isinstance(ckpt.get(key), dict):
                state = ckpt[key]
                break

    # Handle checkpoints saved through nn.DataParallel (main.py --multi-gpus): strip a shared "module." prefix.
    if isinstance(state, dict):
        while state and all(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}
        if not state:
            raise RuntimeError("Checkpoint state_dict is empty.")

        print("Checkpoint parameter prefix handling completed.")
        print("Sample checkpoint keys:")
        for key in list(state.keys())[:5]:
            print(f"  {key}")

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError("Checkpoint does not match the MSHNet(3) definition in model/MSHNet.py.\n" + str(e))
    model.eval()

    meta = {}
    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            try:
                meta["epoch"] = int(ckpt["epoch"])
            except (TypeError, ValueError):
                pass
        if "iou" in ckpt:                             # checkpoint.pkl: best mIoU so far, NOT necessarily of these weights
            try:
                meta["stored_best_iou"] = float(ckpt["iou"])
            except (TypeError, ValueError):
                pass
    return model, meta


def predict_logits(model, x_np, device, native_res):
    """Returns the fused-head logits as float64 [H, W] (MSHNet outputs raw logits, no sigmoid)."""
    import torch

    h, w = x_np.shape[-2:]
    if native_res:                                    # MSHNet needs H, W divisible by 16
        x_np = pad_to_multiple(x_np)                  # zeros AFTER normalisation
    x = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0).to(device)
    _, out = model(x, True)                           # main.py test: tag = True (epoch 1 > warm_epoch -1)
    logit = out[0, 0, :h, :w].float().cpu().numpy().astype(np.float64)
    if not np.isfinite(logit).all():
        raise RuntimeError("Model produced NaN/Inf outputs.")
    return logit


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def stratify(targets, key, bins):
    rows = []
    for lo, hi, name in bins:
        sel = [t for t in targets if t[key] is not None and lo <= t[key] <= hi]
        k = sum(t["detected"] for t in sel)
        n = len(sel)
        lo_ci, hi_ci = wilson_interval(k, n)
        rows.append({"bin": name, "targets": n, "detected": k,
                     "pd": (k / n) if n else None, "pd_ci95": [lo_ci, hi_ci]})
    return rows


def contrast_bins(targets):
    """Tertiles of SCR over this dataset: dimmest / middle / brightest third."""
    vals = np.array([t["scr"] for t in targets if t["scr"] is not None])
    if len(vals) < 6:
        return []
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    return [(-math.inf, q1, f"dimmest third  (SCR <= {q1:.2f})"),
            (q1 + 1e-12, q2, f"middle third   (SCR {q1:.2f}..{q2:.2f})"),
            (q2 + 1e-12, math.inf, f"brightest third (SCR > {q2:.2f})")]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(json_safe(rows))


def ci_str(ci, scale=1.0, digits=4):
    if not ci or ci[0] is None or ci[1] is None:
        return ""
    return f"  [{ci[0] * scale:.{digits}f}, {ci[1] * scale:.{digits}f}]"


# --------------------------------------------------------------------------- #
# Self test (synthetic cases with hand-computed answers)
# --------------------------------------------------------------------------- #
def run_self_test():
    cfg = SimpleNamespace(connectivity=8, centroid_tolerance=3.0, min_component_pixels=1,
                          match_rule="centroid")
    H = W = 32
    failures = []

    def case(name, gt_pixels, pred_pixels, expect, **over):
        c = SimpleNamespace(**{**vars(cfg), **over})
        gt = np.zeros((H, W), bool)
        pr = np.zeros((H, W), bool)
        for (r0, r1, c0, c1) in gt_pixels:
            gt[r0:r1, c0:c1] = True
        for (r0, r1, c0, c1) in pred_pixels:
            pr[r0:r1, c0:c1] = True
        logit = np.where(pr, 5.0, -5.0).astype(np.float32)
        gl, _, gc = label_components(gt, c.connectivity, 1)
        got, _, _ = score_at_threshold(gt, gl, gc, logit, 0.0, c)
        bad = {k: (got[k], v) for k, v in expect.items() if got[k] != v}
        print(f"  [{'FAIL' if bad else ' ok '}] {name}")
        if bad:
            failures.append((name, bad))

    print("Running metric self-tests ...")
    G = [(10, 13, 10, 13)]                                   # one 3x3 target
    case("perfect prediction", G, G,
         dict(tp=9, fp=0, fn=0, matched=1, fa_objs=0, fa_area=0))
    case("empty prediction", G, [],
         dict(tp=0, fn=9, matched=0, fa_objs=0, fa_area=0, n_pred=0))
    case("detection + distant 2x2 false alarm", G, G + [(25, 27, 25, 27)],
         dict(matched=1, fa_objs=1, fa_area=4))
    case("equal-area FA is NOT swallowed (repo value-matching bug)", G,
         [(10, 12, 10, 12), (25, 27, 25, 27)],
         dict(matched=1, fa_objs=1, fa_area=4))
    case("two preds near one target: 1 hit + 1 false alarm",
         [(14, 19, 14, 19)], [(16, 17, 15, 16), (16, 17, 17, 18)],     # both 1 px from the centroid
         dict(matched=1, fa_objs=1, fa_area=1))
    case("distance exactly == tolerance is a miss (strict <)",
         [(16, 17, 16, 17)], [(16, 17, 19, 20)],
         dict(matched=0, fa_objs=1, fa_area=1))
    case("optimal one-to-one assignment (greedy would find only 1)",
         [(16, 17, 10, 11), (16, 17, 14, 15)], [(16, 17, 12, 13), (17, 18, 9, 10)],
         dict(matched=2, fa_objs=0))
    case("pixel counts of a half-overlapping box",
         [(10, 14, 10, 14)], [(10, 14, 12, 16)],
         dict(tp=8, fp=8, fn=8))
    case("centroid rule: 4px shift misses", [(10, 15, 10, 15)], [(10, 15, 14, 19)],
         dict(matched=0, fa_objs=1, fa_area=25))
    case("overlap rule: 4px shift still hits", [(10, 15, 10, 15)], [(10, 15, 14, 19)],
         dict(matched=1, fa_objs=0), match_rule="overlap")
    case("min_component_pixels drops 1-px speck", G, G + [(25, 26, 25, 26)],
         dict(fa_objs=0, fa_area=0), min_component_pixels=2)

    # contrast descriptor: flat bg 100, 3x3 target 150 -> contrast 50, SCR 50 (std floored to 1)
    gt = np.zeros((H, W), bool)
    gt[10:13, 10:13] = True
    gray = np.full((H, W), 100.0, np.float32)
    gray[gt] = 150.0
    gl, _, _ = label_components(gt, 8)
    (contrast, scr), = target_descriptors(gray, gl, 1)
    ok = abs(contrast - 50) < 1e-6 and abs(scr - 50) < 1e-6
    print(f"  [{' ok ' if ok else 'FAIL'}] local contrast / SCR")
    if not ok:
        failures.append(("contrast", (contrast, scr)))

    # bootstrap / aggregate plumbing
    A = {k: np.array([[3], [5]]) for k in ("tp", "fp", "fn", "n_pred", "matched", "fa_objs", "fa_area")}
    A.update(n_gt=np.array([1, 1]), pixels=np.array([100, 100]), niou_terms=np.array([[0.5], [0.5]]))
    s = stats_from_sums(totals(A, 0))
    ok = abs(s["miou"] - 8 / (8 + 8 + 8)) < 1e-12 and s["pd"] == 4.0 and s["fa"] == 8 / 200
    print(f"  [{' ok ' if ok else 'FAIL'}] aggregation arithmetic")
    if not ok:
        failures.append(("aggregation", s))

    # added: precision / recall / F1 with hand-computed values (tp=6, fp=2, fn=3)
    S = {"tp": 6.0, "fp": 2.0, "fn": 3.0, "matched": 1.0, "n_gt": 1.0, "fa_area": 0.0,
         "pixels": 100.0, "fa_objs": 0.0, "n_img": 1.0, "n_pred": 1.0}
    s = stats_from_sums(S)
    p_, r_ = 6 / 8, 6 / 9
    ok = (abs(s["precision"] - p_) < 1e-12 and abs(s["recall"] - r_) < 1e-12
          and abs(s["f1"] - 2 * p_ * r_ / (p_ + r_)) < 1e-12)
    print(f"  [{' ok ' if ok else 'FAIL'}] precision / recall / F1")
    if not ok:
        failures.append(("prf1", s))

    # added (MSHNet): probability <-> logit helpers keep the repo's "logit > 0" decision == "p > 0.5"
    ok = (prob_to_logit(0.5) == 0.0 and abs(sigmoid(prob_to_logit(0.9)) - 0.9) < 1e-12
          and sigmoid(0.0) == 0.5 and sigmoid(1e6) == 1.0 and 0.0 <= sigmoid(-1e6) < 1e-300)
    print(f"  [{' ok ' if ok else 'FAIL'}] MSHNet logit / probability helpers")
    if not ok:
        failures.append(("logit_helpers", None))

    # added (MSHNet): zero padding to a multiple of 16 and RGB / ImageNet input normalisation
    a = np.ones((3, 17, 34), np.float32)
    pa = pad_to_multiple(a)
    ok = (pa.shape == (3, 32, 48) and pa[:, :17, :34].min() == 1 and pa.sum() == a.sum()
          and pad_to_multiple(np.ones((3, 32, 16), np.float32)).shape == (3, 32, 16))
    px = np.array([[[0, 255, 128]]], np.uint8)                       # one RGB pixel
    xin = to_input_array(px)
    ref = (np.array([0, 255, 128], np.float32) / 255.0 - np.array(IMAGENET_MEAN, np.float32)) \
        / np.array(IMAGENET_STD, np.float32)
    ok = ok and xin.shape == (3, 1, 1) and xin.dtype == np.float32 and np.allclose(xin[:, 0, 0], ref, atol=1e-6)
    print(f"  [{' ok ' if ok else 'FAIL'}] MSHNet padding / ImageNet normalisation")
    if not ok:
        failures.append(("pad_norm", (pa.shape, xin)))

    if failures:
        print("\nSELF-TEST FAILED:")
        for f in failures:
            print("  ", f)
        sys.exit(1)
    print("All self-tests passed.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser(description="Evaluate a MSHNet checkpoint for small/dim IR targets.")
    p.add_argument("--self_test", action="store_true", help="Run synthetic metric tests and exit.")
    p.add_argument("--root", default="dataset")
    p.add_argument("--dataset", default="IRSTD-1k",
                   help="Test dataset folder name; uses <root>/<dataset>/<test_list>")
    p.add_argument("--dataset_dir", default=None,
                   help="Full path of the dataset folder (like main.py's --dataset-dir); overrides --root/--dataset.")
    p.add_argument("--test_list", default="test.txt",
                   help="Image-id list inside the dataset folder (main.py's 'val' loader reads test.txt).")
    p.add_argument("--train_list", default="trainval.txt",
                   help="Only used for the train/test leakage check (main.py's 'train' loader reads trainval.txt).")
    p.add_argument("--suffix", default=".png")
    p.add_argument("--model_dir", default=None, help="Path to the checkpoint (e.g. the .tar / .pkl weight file).")
    p.add_argument("--base_size", type=int, default=256,
                   help="Resize target of the repo's val mode (main.py --base-size). Used unless --native_res True "
                        "(must be a multiple of 16).")
    p.add_argument("--native_res", type=parse_bool, nargs="?", const=True, default=False,
                   help="Evaluate at original resolution (zero-pad to /16). Default False = the repo's protocol: "
                        "resize to base_size x base_size.")
    p.add_argument("--device", default="auto")

    p.add_argument("--prob_threshold", type=float, default=0.5,
                   help="Operating point as sigmoid(logit) > p; 0.5 == logit > 0 (repo default).")
    p.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    p.add_argument("--match_rule", choices=["centroid", "overlap"], default="centroid")
    p.add_argument("--centroid_tolerance", type=float, default=3.0)
    p.add_argument("--min_component_pixels", type=int, default=1,
                   help="Ignore PREDICTED components smaller than this (GT is never filtered).")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap replicates (0 = off).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.self_test and not args.model_dir:
        p.error("--model_dir is required (unless --self_test).")
    if not 0.0 < args.prob_threshold < 1.0:
        p.error("--prob_threshold must be strictly between 0 and 1.")
    if args.base_size < PAD_MULTIPLE or args.min_component_pixels < 1 or args.centroid_tolerance <= 0:
        p.error(f"base_size >= {PAD_MULTIPLE}, min_component_pixels >= 1, centroid_tolerance > 0 required.")
    if not args.native_res and args.base_size % PAD_MULTIPLE:
        p.error(f"--base_size must be a multiple of {PAD_MULTIPLE} for MSHNet.")
    return args


def main():
    args = get_args()
    if args.self_test:
        run_self_test()
        return

    import torch

    repo_root = Path.cwd()
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(args.root) / args.dataset
    dataset_dir = (dataset_dir if dataset_dir.is_absolute() else repo_root / dataset_dir).resolve()
    dataset_name = dataset_dir.name
    ckpt_path = Path(args.model_dir)
    args.model_path = ckpt_path if ckpt_path.is_absolute() else (repo_root / ckpt_path).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

    test_list_name = args.test_list
    train_list_name = args.train_list
    test_list_path = dataset_dir / test_list_name
    train_list_path = dataset_dir / train_list_name
    if not test_list_path.is_file():
        raise FileNotFoundError(f"Test list not found: {test_list_path}")
    eval_ids = read_ids(test_list_path)
    train_ids = read_ids(train_list_path) if train_list_path.is_file() else []
    if not eval_ids:
        raise RuntimeError(f"{test_list_name} is empty.")

    warnings = []
    if len(set(eval_ids)) != len(eval_ids):
        warnings.append(f"{test_list_name} contains {len(eval_ids) - len(set(eval_ids))} duplicate IDs (they are counted twice).")
    overlap_ids = set(eval_ids) & set(train_ids)
    if overlap_ids:
        warnings.append(f"DATA LEAKAGE: {len(overlap_ids)} IDs appear in BOTH {train_list_name} and {test_list_name} "
                        f"(same file names; if these are different datasets, check they are really the same images).")

    print(f"Device: {device} | Dataset: {dataset_dir} | Images: {len(eval_ids)}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Input: RGB, ImageNet normalisation mean={IMAGENET_MEAN} std={IMAGENET_STD}")
    model, ckpt_meta = build_and_load_model(args, device)

    cfg = SimpleNamespace(connectivity=args.connectivity, centroid_tolerance=args.centroid_tolerance,
                          min_component_pixels=args.min_component_pixels, match_rule=args.match_rule)
    thresholds = sorted(set(SWEEP_PROBS + [args.prob_threshold]))
    t_primary = thresholds.index(args.prob_threshold)
    logit_thrs = [prob_to_logit(p) for p in thresholds]
    n_img, n_thr = len(eval_ids), len(thresholds)

    keys = ("tp", "fp", "fn", "n_pred", "matched", "fa_objs", "fa_area")
    A = {k: np.zeros((n_img, n_thr), dtype=np.int64) for k in keys}
    A["niou_terms"] = np.zeros((n_img, n_thr))
    A["n_gt"] = np.zeros(n_img, dtype=np.int64)
    A["pixels"] = np.zeros(n_img, dtype=np.int64)

    notes = {"non_8bit_images": 0, "odd_mask_values": 0, "orig_sizes": Counter()}
    lost_by_resize = 0
    native_objects = 0
    target_rows, per_image_rows = [], []
    resize_to = None if args.native_res else args.base_size

    with torch.inference_mode():
        for i, image_id in enumerate(tqdm(eval_ids, desc="Evaluating")):
            x_in, gray, gt, n_native, n_resized = load_pair(
                dataset_dir, image_id, args.suffix, resize_to, notes)
            if n_native is not None:
                native_objects += n_native
                lost_by_resize += max(0, n_native - n_resized)

            logit = predict_logits(model, x_in, device, args.native_res)
            if logit.shape != gt.shape:
                raise RuntimeError(f"Shape mismatch for {image_id}: {logit.shape} vs {gt.shape}")

            gt_labels, gt_areas, gt_cent = label_components(gt, args.connectivity, 1)
            n_gt = len(gt_areas)
            A["n_gt"][i] = n_gt
            A["pixels"][i] = gt.size

            for t, lthr in enumerate(logit_thrs):
                counts, gt_m, gt_dist = score_at_threshold(gt, gt_labels, gt_cent, logit, lthr, cfg)
                for k in keys:
                    A[k][i, t] = counts[k]
                union = counts["tp"] + counts["fp"] + counts["fn"]
                A["niou_terms"][i, t] = counts["tp"] / union if union else 0.0

                if t == t_primary:
                    per_image_rows.append({
                        "image_id": image_id, "gt_targets": n_gt, "pred_components": counts["n_pred"],
                        "detected": counts["matched"], "missed": n_gt - counts["matched"],
                        "false_alarm_components": counts["fa_objs"],
                        "false_alarm_pixels": counts["fa_area"],
                        "iou": (counts["tp"] / (counts["tp"] + counts["fp"] + counts["fn"])
                                if (counts["tp"] + counts["fp"] + counts["fn"]) else None),
                    })
                    if n_gt:
                        desc = target_descriptors(gray, gt_labels, n_gt)
                        peaks = ndimage.maximum(logit, gt_labels, index=np.arange(1, n_gt + 1))
                        for k in range(n_gt):
                            contrast, scr = desc[k]
                            target_rows.append({
                                "image_id": image_id, "target_index": k,
                                "area_px": int(gt_areas[k]),
                                "centroid_x": float(gt_cent[k, 0]), "centroid_y": float(gt_cent[k, 1]),
                                "contrast_gray_levels": finite_or_none(contrast),
                                "scr": finite_or_none(scr),
                                "peak_prob": sigmoid(peaks[k]),            # sigmoid of the peak logit
                                "detected": bool(gt_m[k]),
                                "centroid_error_px": finite_or_none(gt_dist[k]),
                            })

    # ---------------- aggregate ----------------
    point = stats_from_sums(totals(A, t_primary))
    has_gt = A["n_gt"] > 0
    point["niou"] = float(A["niou_terms"][has_gt, t_primary].mean()) if has_gt.any() else float("nan")
    ci = bootstrap_headline(A, t_primary, args.bootstrap, args.seed)

    S = totals(A, t_primary)
    empty_imgs = ~has_gt
    empty_with_fa = int(((A["fa_objs"][:, t_primary] > 0) & empty_imgs).sum())

    sweep = []
    for t, p in enumerate(thresholds):
        s = stats_from_sums(totals(A, t))
        sweep.append({"prob_threshold": p, "mIoU": s["miou"], "Pd": s["pd"],
                      "Fa_x1e-6": s["fa"] * 1e6, "false_alarms_per_image": s["fa_per_image"],
                      "alert_precision": s["alert_precision"]})
        sweep[-1].update({"pixel_precision": s["precision"], "pixel_recall": s["recall"],
                          "pixel_f1": s["f1"]})                       # added

    size_rows = stratify([{**r, "v": r["area_px"]} for r in target_rows], "v", SIZE_BINS)
    cb = contrast_bins(target_rows)
    contrast_rows = stratify(target_rows, "scr", cb) if cb else []

    # ---------------- sanity checks / warnings ----------------
    if S["n_gt"] == 0:
        warnings.append("No ground-truth targets in the evaluated split: Pd / mIoU are undefined.")
    if resize_to is not None:
        sizes = notes["orig_sizes"]
        off = {k: v for k, v in sizes.items() if k != (resize_to, resize_to)}
        if off:
            top = ", ".join(f"{w}x{h} (x{c})" for (w, h), c in Counter(off).most_common(3))
            warnings.append(f"{sum(off.values())}/{n_img} images are not {resize_to}x{resize_to} "
                            f"(e.g. {top}) and were RESIZED (the repo's own val protocol); non-square images are "
                            f"aspect-distorted and tiny targets shrink. --native_res True evaluates at original "
                            f"resolution instead (not what the repo does).")
        if lost_by_resize:
            warnings.append(f"Resizing erased {lost_by_resize} of {native_objects} GT objects "
                            f"({100 * lost_by_resize / max(native_objects, 1):.2f}%) from the ground truth; "
                            f"they are not in Pd. Use --native_res for a lossless evaluation.")
    if notes["non_8bit_images"]:
        warnings.append(f"{notes['non_8bit_images']} images are 16-bit/float; they go through convert('RGB') exactly as in "
                        f"the repo's IRSTD_Dataset, so the model input depends on PIL's conversion. Check the "
                        f"resulting range matches what the model was trained on.")
    if notes["odd_mask_values"]:
        warnings.append(f"{notes['odd_mask_values']} masks contain values other than {{0,1}} or {{0,255}} "
                        f"(binarised at >0 or >127).")
    warnings.append(f"Selection bias: main.py saves weight.pkl at the epoch with the best mIoU on the test list (test.txt). If this split "
                    f"({test_list_name}) is that same list, scores are optimistic; use a never-seen split for the final number.")
    warnings.append("Pd / Fa use one-to-one Hungarian matching; the repo's PD_FA matches greedily and removes matched "
                    "areas from the false-alarm list by area value, so Pd / Fa can differ slightly from the repo's "
                    "printed numbers (mIoU is computed identically).")
    small_n = S["n_gt"] < 100
    if small_n:
        warnings.append(f"Only {int(S['n_gt'])} targets: Pd resolution is coarse; rely on the confidence intervals.")

    # ---------------- write outputs ----------------
    stem = args.model_path.name
    for ext in (".pth.tar", ".pt", ".pth", ".tar", ".pkl"):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = dataset_dir / "value_result" / stem / f"eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "dataset": dataset_name, "test_list": test_list_name, "images": n_img,
        "input_normalisation": {"channels": 3, "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "model_call": "MSHNet(3); _, logits = model(x, warm_flag=True)",
        "targets": int(S["n_gt"]), "checkpoint": str(args.model_path), "checkpoint_meta": ckpt_meta,
        "resolution": f"native (zero-padded to /{PAD_MULTIPLE})" if args.native_res else f"{args.base_size}x{args.base_size} resize",
        "prob_threshold": args.prob_threshold, "logit_threshold": prob_to_logit(args.prob_threshold),
        "match_rule": args.match_rule, "centroid_tolerance_px": args.centroid_tolerance,
        "connectivity": args.connectivity, "min_pred_component_pixels": args.min_component_pixels,
        "bootstrap_replicates": args.bootstrap, "seed": args.seed,
    }
    headline = {
        "mIoU": point["miou"], "nIoU": point["niou"], "Pd": point["pd"],
        "Fa": point["fa"], "Fa_x1e-6": point["fa"] * 1e6,
        "false_alarms_per_image": point["fa_per_image"],
        "alert_precision": point["alert_precision"],
        "ci95": {"mIoU": ci.get("miou"), "nIoU": ci.get("niou"), "Pd": ci.get("pd"),
                 "Fa": ci.get("fa"), "false_alarms_per_image": ci.get("fa_per_image")},
        "counts": {"targets": int(S["n_gt"]), "detected": int(S["matched"]),
                   "predicted_components": int(S["n_pred"]),
                   "false_alarm_components": int(S["fa_objs"]),
                   "false_alarm_pixels": int(S["fa_area"]), "total_pixels": int(S["pixels"])},
        "target_free_images": {"count": int(empty_imgs.sum()), "with_false_alarm": empty_with_fa},
    }
    # ---- added: Precision / Recall / F1 in the headline block of metrics.json ----
    headline.update({"precision": point["precision"], "recall": point["recall"], "f1": point["f1"]})
    headline["ci95"].update({"precision": ci.get("precision"), "recall": ci.get("recall"),
                             "f1": ci.get("f1")})
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({"config": config, "headline": headline, "by_target_size": size_rows,
                             "by_local_contrast": contrast_rows, "threshold_sweep": sweep,
                             "warnings": warnings}), f, indent=2)
    write_csv(out_dir / "per_image.csv", per_image_rows)
    write_csv(out_dir / "per_target.csv", target_rows)
    write_csv(out_dir / "threshold_sweep.csv", sweep)

    L = []
    L.append("MSHNet evaluation")
    L.append("=" * 60)
    L.append(f"Dataset: {dataset_name} ({test_list_name})   Images: {n_img}   Targets: {int(S['n_gt'])}")
    L.append(f"Checkpoint: {args.model_path.name}" + (f" (epoch {ckpt_meta['epoch']})" if "epoch" in ckpt_meta else ""))
    L.append(f"Resolution: {config['resolution']}   Threshold: p > {args.prob_threshold} (logit > {config['logit_threshold']:.3f})")
    L.append(f"Input: 3-channel RGB, ImageNet normalisation mean={IMAGENET_MEAN}, std={IMAGENET_STD} (as in the repo's IRSTD_Dataset)")
    L.append(f"Match rule: {args.match_rule}" + (f", centroid distance < {args.centroid_tolerance} px" if args.match_rule == "centroid" else ", any pixel overlap")
             + f", {args.connectivity}-connectivity, one-to-one")
    L.append("")
    L.append(f"HEADLINE  (point estimate  [95% bootstrap CI over images, n={args.bootstrap}])")
    L.append(f"  mIoU              {fmt(point['miou'])}{ci_str(ci.get('miou'))}   pooled pixel IoU (repo metric)")
    L.append(f"  nIoU              {fmt(point['niou'])}{ci_str(ci.get('niou'))}   mean per-image IoU, target images only")
    L.append(f"  Pd                {fmt(point['pd'])}{ci_str(ci.get('pd'))}   {int(S['matched'])}/{int(S['n_gt'])} targets detected")
    L.append(f"  Fa (x1e-6)        {fmt(point['fa'], 3, 1e6)}{ci_str(ci.get('fa'), 1e6, 3)}   false-alarm pixels / all pixels (repo metric)")
    L.append(f"  False alarms/img  {fmt(point['fa_per_image'], 4)}{ci_str(ci.get('fa_per_image'))}   {int(S['fa_objs'])} false-alarm components")
    L.append(f"  Alert precision   {fmt(point['alert_precision'])}   detected targets / predicted components")
    L.append(f"  Precision (pixel) {fmt(point['precision'])}{ci_str(ci.get('precision'))}   TP / (TP + FP), pooled pixels  [added]")
    L.append(f"  Recall (pixel)    {fmt(point['recall'])}{ci_str(ci.get('recall'))}   TP / (TP + FN), pooled pixels  [added]")
    L.append(f"  F1 (pixel)        {fmt(point['f1'])}{ci_str(ci.get('f1'))}   2PR / (P + R), pooled pixels  [added]")
    if empty_imgs.any():
        L.append(f"  Target-free images: {int(empty_imgs.sum())}, of which {empty_with_fa} "
                 f"({100 * empty_with_fa / int(empty_imgs.sum()):.1f}%) raise at least one false alarm")
    L.append("")
    L.append("Pd BY TARGET SIZE")
    for r in size_rows:
        L.append(f"  {r['bin']:<22} n={r['targets']:<5} Pd={fmt(r['pd'], 3)}{ci_str(r['pd_ci95'], 1, 3)}")
    if contrast_rows:
        L.append("")
        L.append("Pd BY LOCAL CONTRAST (SCR = (mean target - mean ring) / std ring; dim targets = low SCR)")
        for r in contrast_rows:
            L.append(f"  {r['bin']:<36} n={r['targets']:<5} Pd={fmt(r['pd'], 3)}{ci_str(r['pd_ci95'], 1, 3)}")
    L.append("")
    L.append("THRESHOLD SWEEP  (choose your operating point here; headline uses the marked row)")
    L.append(f"  {'p_thr':>6} {'mIoU':>7} {'Pd':>7} {'Fa x1e-6':>10} {'FA/img':>8} {'alert prec':>10}")
    for r in sweep:
        mark = "  <--" if r["prob_threshold"] == args.prob_threshold else ""
        L.append(f"  {r['prob_threshold']:>6.2f} {fmt(r['mIoU']):>7} {fmt(r['Pd']):>7} "
                 f"{fmt(r['Fa_x1e-6'], 3):>10} {fmt(r['false_alarms_per_image']):>8} "
                 f"{fmt(r['alert_precision']):>10}{mark}")
    L.append("")
    L.append("PIXEL-LEVEL PRECISION / RECALL / F1 ACROSS THRESHOLDS  [added]")
    L.append(f"  {'p_thr':>6} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for r in sweep:
        mark = "  <--" if r["prob_threshold"] == args.prob_threshold else ""
        L.append(f"  {r['prob_threshold']:>6.2f} {fmt(r['pixel_precision']):>10} "
                 f"{fmt(r['pixel_recall']):>8} {fmt(r['pixel_f1']):>8}{mark}")
    L.append("")
    L.append("WARNINGS / CAVEATS")
    for w in warnings:
        L.append(f"  - {w}")
    L.append("")
    L.append(f"Files: {out_dir}")
    L.append("  metrics.json  per_image.csv  per_target.csv (inspect misses here)  threshold_sweep.csv")
    text = "\n".join(L) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()