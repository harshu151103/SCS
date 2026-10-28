#!/usr/bin/env python3
"""
Ground-truth refinement for IR small/dim target datasets (NUDT-SIRST style).

Stage 1 (automatic): images where GT and SCTransNet prediction agree (global IoU,
         per-target IoU, target count) are saved to refined_dataset/correct_images
         using the PREDICTED mask as the refined ground truth.
Stage 2 (interactive, matplotlib window, no terminal input):
         P = use Predicted mask      -> refined_dataset/reviewed_images
         G = use original Ground truth -> refined_dataset/reviewed_images
         M = manual annotation needed -> refined_dataset/manual_annotation
         LEFT ARROW = undo previous decision and show that image again
         Q / Esc = quit (progress is saved, re-run to resume)

Expected layout:
    dataset/custom_data/images/*.png
    dataset/custom_data/masks/*.png
    Result/custom_data/*.png          (SCTransNet binary predictions)

Usage:
    python refine_ground_truth.py
    python refine_ground_truth.py --dataset dataset/custom_data --pred Result/custom_data \
        --out refined_dataset --iou-thr 0.6 --target-iou-thr 0.5
"""
import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PRED_SUFFIXES = ("", "_pred", "_prediction", "_result", "_out", "_mask", "_SCTransNet")

# action code -> (group directory, human readable label)
ACTIONS = {
    "auto": ("correct_images", "auto_accepted_prediction"),
    "P": ("reviewed_images", "predicted"),
    "G": ("reviewed_images", "ground_truth"),
    "M": ("manual_annotation", "manual_annotation"),
}
GROUPS = ("correct_images", "reviewed_images", "manual_annotation")
SUBDIRS = ("images", "mask", "visualization")

log = logging.getLogger("refine")


# ----------------------------------------------------------------------------
# Image / mask helpers
# ----------------------------------------------------------------------------
def build_index(directory):
    files = {}
    for p in sorted(Path(directory).iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if p.stem in files:
                raise ValueError(f"Duplicate filename stem '{p.stem}' in {directory}. Resolve first.")
            files[p.stem] = p
    return files


def find_by_stem(index, stem):
    for suf in PRED_SUFFIXES:
        if stem + suf in index:
            return index[stem + suf]
    return None


def read_raw(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image {path}")
    return img


def raw_for_display(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (255 * (img - img.min()) / max(img.max() - img.min(), 1e-6)).astype(np.uint8)
    return img


def read_binary_mask(path, shape):
    """Return (uint8 0/255 mask, shape_mismatch). Mismatched masks are resized (nearest) only
    so they can be displayed; the image is flagged and never auto-accepted. None path -> empty."""
    if path is None:
        return np.zeros(shape, np.uint8), False
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        log.warning("Cannot read mask %s -> treating as empty", path)
        return np.zeros(shape, np.uint8), False
    mismatch = m.shape != shape
    if mismatch:
        log.warning("Shape mismatch for %s (%s vs image %s) -> flagged for review", path.name, m.shape, shape)
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    thr = 0 if m.max() <= 1 else 127  # supports 0/1 and 0/255 masks
    return ((m > thr).astype(np.uint8)) * 255, mismatch


# ----------------------------------------------------------------------------
# Metrics (global IoU + per-target matching for multi-target images)
# ----------------------------------------------------------------------------
def compute_metrics(gt, pr):
    g, p = gt > 0, pr > 0
    inter, union = np.logical_and(g, p).sum(), np.logical_or(g, p).sum()
    iou = 1.0 if union == 0 else float(inter / union)

    ng, gl = cv2.connectedComponents(g.astype(np.uint8), connectivity=8)
    npd, pl = cv2.connectedComponents(p.astype(np.uint8), connectivity=8)
    ng, npd = ng - 1, npd - 1

    target_ious, hit = [], set()
    for i in range(1, ng + 1):
        gi = gl == i
        ids = np.unique(pl[gi])
        ids = ids[ids > 0]
        if len(ids) == 0:
            target_ious.append(0.0)
            continue
        hit.update(ids.tolist())
        pu = np.isin(pl, ids)  # all predicted blobs touching this GT target
        target_ious.append(float(np.logical_and(gi, pu).sum() / np.logical_or(gi, pu).sum()))

    if target_ious:
        min_t = min(target_ious)
    else:
        min_t = 1.0 if npd == 0 else 0.0
    return {
        "iou": iou, "n_gt": ng, "n_pred": npd,
        "min_target_iou": min_t, "extra_pred": npd - len(hit),
        "missed_gt": int(sum(t == 0.0 for t in target_ious)),
    }


def judge(m, args):
    """Return list of reasons the pair is NOT a match (empty list -> auto accept)."""
    if m["n_gt"] == 0 and m["n_pred"] == 0:
        return [] if args.auto_empty else ["both empty"]
    reasons = []
    if m["iou"] < args.iou_thr:
        reasons.append(f"low IoU<{args.iou_thr}")
    if m["min_target_iou"] < args.target_iou_thr:
        reasons.append(f"weak target<{args.target_iou_thr}")
    if m["missed_gt"]:
        reasons.append(f"{m['missed_gt']} GT target(s) missed")
    if m["extra_pred"]:
        reasons.append(f"{m['extra_pred']} extra pred blob(s)")
    if not args.allow_count_mismatch and m["n_gt"] != m["n_pred"]:
        reasons.append("target count differs")
    return reasons


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def draw_panels(axes, raw, gt, pr, m):
    for ax in axes:
        ax.clear()
        ax.axis("off")
    axes[0].imshow(raw_for_display(raw), cmap="gray")
    axes[0].set_title("Raw image")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Ground truth ({m['n_gt']} target(s))")
    axes[2].imshow(pr, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"SCTransNet prediction ({m['n_pred']} target(s))")


def summary_line(name, m, reasons=None):
    s = (f"{name}   IoU = {m['iou']:.3f}   GT/Pred targets = {m['n_gt']}/{m['n_pred']}   "
         f"min target IoU = {m['min_target_iou']:.3f}")
    if reasons:
        s += "\nFlagged: " + ", ".join(reasons)
    return s


def save_visualization(path, raw, gt, pr, m, title):
    fig = Figure(figsize=(15, 5.5))
    axes = fig.subplots(1, 3)
    draw_panels(axes, raw, gt, pr, m)
    fig.suptitle(title, fontsize=11)
    fig.savefig(str(path), dpi=100, bbox_inches="tight")


# ----------------------------------------------------------------------------
# Output manager (files, state, lists)
# ----------------------------------------------------------------------------
class Store:
    def __init__(self, out_dir, manual_init):
        self.out = Path(out_dir)
        self.manual_init = manual_init
        for g in GROUPS:
            for s in SUBDIRS:
                (self.out / g / s).mkdir(parents=True, exist_ok=True)
        self.state_file = self.out / "state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def _save_state(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_file)

    def save(self, item, action, raw, gt, pr, persist=True):
        group, label = ACTIONS[action]
        name, stem = item["img"].name, item["img"].stem
        m = item["metrics"]
        if action in ("auto", "P"):
            mask = pr
        elif action == "G":
            mask = gt
        else:  # manual
            mask = {"gt": gt, "pred": pr, "blank": np.zeros_like(gt)}[self.manual_init]
        base = self.out / group
        shutil.copy2(item["img"], base / "images" / name)
        cv2.imwrite(str(base / "mask" / f"{stem}.png"), mask)
        title = f"{summary_line(name, m)}   ->   {label.upper()}"
        save_visualization(base / "visualization" / f"{stem}.png", raw, gt, pr, m, title)
        self.state[name] = {"action": action, "time": datetime.now().isoformat(timespec="seconds")}
        if persist:
            self._save_state()
            self.write_lists(None)
        return base

    def undo(self, item):
        name, stem = item["img"].name, item["img"].stem
        for g in GROUPS:  # remove from every group to be safe
            for sub, fname in (("images", name), ("mask", f"{stem}.png"),
                               ("visualization", f"{stem}.png")):
                f = self.out / g / sub / fname
                if f.exists():
                    f.unlink()
        self.state.pop(name, None)
        self._save_state()

    def write_lists(self, items):
        if items is not None:
            self._items = items
        items = getattr(self, "_items", {})
        rows = {g: [] for g in GROUPS}
        for name, st in sorted(self.state.items()):
            g, label = ACTIONS[st["action"]]
            m = items.get(name, {}).get("metrics", {})
            rows[g].append([name, label, f"{m.get('iou', float('nan')):.4f}",
                            m.get("n_gt", ""), m.get("n_pred", ""), st["time"]])
        header = ["image", "action", "iou", "n_gt", "n_pred", "time"]
        (self.out / "correct_images" / "correct_images_list.txt").write_text(
            "\n".join(r[0] for r in rows["correct_images"]) + "\n")
        for g, fname in (("correct_images", "correct_images_list.csv"),
                         ("reviewed_images", "reviewed_images_list.csv"),
                         ("manual_annotation", "manual_annotation_list.csv")):
            with open(self.out / g / fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows[g])
        with open(self.out / "all_decisions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header + ["group"])
            for g in GROUPS:
                for r in rows[g]:
                    w.writerow(r + [g])


# ----------------------------------------------------------------------------
# Interactive reviewer
# ----------------------------------------------------------------------------
class Reviewer:
    def __init__(self, queue, items, store):
        self.queue, self.items, self.store = queue, items, store
        # resume at first undecided image
        self.idx = next((i for i, n in enumerate(queue) if n not in store.state), len(queue))
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []  # stop matplotlib shortcuts (p, g, s, q, left...) interfering
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 6.5))
        self.fig.canvas.manager.set_window_title("Ground truth refinement")
        self.title = self.fig.suptitle("", fontsize=11)
        self.fig.text(0.5, 0.02,
                      "P = use Prediction    G = use Ground truth    M = Manual annotation    "
                      "<- (left arrow) = Undo previous    Q = Quit",
                      ha="center", fontsize=12, weight="bold")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.show()

    def load(self, item):
        raw = read_raw(item["img"])
        shape = raw.shape[:2]
        gt, _ = read_binary_mask(item["gt"], shape)
        pr, _ = read_binary_mask(item["pred"], shape)
        return raw, gt, pr

    def show(self):
        n = len(self.queue)
        if self.idx >= n:
            for ax in self.axes:
                ax.clear()
                ax.axis("off")
            self.axes[1].text(0.5, 0.5, "All images reviewed!\n\nPress <- to undo the last one\nor Q to quit",
                              ha="center", va="center", fontsize=16, transform=self.axes[1].transAxes)
            self.title.set_text(f"Done: {n}/{n} reviewed")
            log.info("All %d images reviewed. Press LEFT to undo or Q to quit.", n)
        else:
            name = self.queue[self.idx]
            item = self.items[name]
            raw, gt, pr = self.load(item)
            draw_panels(self.axes, raw, gt, pr, item["metrics"])
            self.title.set_text(f"[{self.idx + 1}/{n}]  " + summary_line(name, item["metrics"], item["reasons"]))
            log.info("Showing [%d/%d] %s  IoU=%.3f  targets GT/Pred=%d/%d  flagged: %s",
                     self.idx + 1, n, name, item["metrics"]["iou"], item["metrics"]["n_gt"],
                     item["metrics"]["n_pred"], ", ".join(item["reasons"]) or "-")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            plt.close(self.fig)
            return
        if key == "left":
            if self.idx == 0:
                log.info("Already at the first image, nothing to undo.")
                return
            self.idx -= 1
            name = self.queue[self.idx]
            prev = self.store.state.get(name, {}).get("action", "none")
            self.store.undo(self.items[name])
            log.info("UNDO %s (was '%s'); files removed from refined dataset", name, prev)
            self.show()
            return
        if key in ("p", "g", "m") and self.idx < len(self.queue):
            action = key.upper()
            name = self.queue[self.idx]
            item = self.items[name]
            try:
                raw, gt, pr = self.load(item)
                base = self.store.save(item, action, raw, gt, pr)
            except Exception as e:  # keep UI alive, stay on the same image
                log.error("Failed to save %s: %s", name, e)
                return
            log.info("[%d/%d] %s -> %s saved to %s", self.idx + 1, len(self.queue), name,
                     ACTIONS[action][1].upper(), base)
            self.idx += 1
            self.show()

    def on_close(self, _):
        done = sum(1 for n in self.queue if n in self.store.state)
        log.info("Window closed. Reviewed %d/%d. Re-run the script to resume.", done, len(self.queue))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="dataset/custom_data")
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--masks-dir", default="masks")
    ap.add_argument("--pred", default="Result/custom_data")
    ap.add_argument("--out", default="refined_dataset")
    ap.add_argument("--iou-thr", type=float, default=0.7, help="global IoU needed for auto-accept")
    ap.add_argument("--target-iou-thr", type=float, default=0.5,
                    help="every GT target must reach this IoU against its predicted blob(s)")
    ap.add_argument("--allow-count-mismatch", action="store_true",
                    help="do not require #GT targets == #predicted targets for auto-accept")
    ap.add_argument("--auto-empty", action="store_true",
                    help="auto-accept images where both GT and prediction are empty (default: review them)")
    ap.add_argument("--manual-init", choices=["gt", "pred", "blank"], default="gt",
                    help="mask stored in manual_annotation/mask until you annotate it")
    ap.add_argument("--order", choices=["name", "iou"], default="name", help="review order")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "refine.log", encoding="utf-8")])

    ds = Path(args.dataset)
    img_idx = build_index(ds / args.images_dir)
    gt_idx = build_index(ds / args.masks_dir)
    pr_idx = build_index(args.pred)
    log.info("Found %d images, %d GT masks, %d predictions", len(img_idx), len(gt_idx), len(pr_idx))

    store = Store(out, args.manual_init)
    items = {}
    store._items = items
    n_auto = 0
    for k, (stem, img_path) in enumerate(img_idx.items(), 1):
        gt_path, pr_path = find_by_stem(gt_idx, stem), find_by_stem(pr_idx, stem)
        if gt_path is None:
            log.warning("%s: GT mask missing -> will be reviewed (GT treated as empty)", img_path.name)
        if pr_path is None:
            log.warning("%s: prediction missing -> will be reviewed (pred treated as empty)", img_path.name)
        try:
            raw = read_raw(img_path)
            gt, gt_bad = read_binary_mask(gt_path, raw.shape[:2])
            pr, pr_bad = read_binary_mask(pr_path, raw.shape[:2])
        except Exception as e:
            log.error("Skipping %s: %s", img_path.name, e)
            continue
        m = compute_metrics(gt, pr)
        reasons = judge(m, args)
        if gt_path is None or pr_path is None:
            reasons.append("missing GT" if gt_path is None else "missing prediction")
        if gt_bad or pr_bad:
            reasons.append("SHAPE MISMATCH (" + ("GT" if gt_bad else "pred") + ")")
        items[img_path.name] = {"img": img_path, "gt": gt_path, "pred": pr_path,
                                "metrics": m, "reasons": reasons}

        # Stage 1: automatic acceptance
        if not reasons and img_path.name not in store.state:
            store.save(items[img_path.name], "auto", raw, gt, pr, persist=False)
            n_auto += 1
            log.info("[auto %d/%d] %s accepted (IoU=%.3f, targets=%d)", k, len(img_idx),
                     img_path.name, m["iou"], m["n_gt"])

    store._save_state()
    store.write_lists(items)
    total_auto = sum(1 for s in store.state.values() if s["action"] == "auto")
    queue = [n for n, it in items.items()
             if store.state.get(n, {}).get("action") != "auto"]
    if args.order == "iou":
        queue.sort(key=lambda n: (items[n]["metrics"]["iou"], n))
    log.info("Auto-accepted this run: %d | total auto-accepted: %d | to review: %d",
             n_auto, total_auto, len(queue))

    if not queue:
        log.info("Nothing to review. Done.")
        return
    reviewer = Reviewer(queue, items, store)
    plt.show()
    log.info("Finished. Lists: %s", out / "all_decisions.csv")


if __name__ == "__main__":
    main()
