"""
MSHNet test script with saved prediction maps and 1x3 comparison visualizations.
Place this file in the MSHNet repository root, next to main.py.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data import IRSTD_Dataset
from utils.metric import PD_FA, mIoU, ROCMetric
from model.MSHNet import MSHNet


def parse_args():
    p = argparse.ArgumentParser(description="MSHNet evaluation with visual comparisons")
    p.add_argument("--dataset-dir", type=str, default="dataset/IRSTD-1k")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--warm-epoch", type=int, default=5)
    p.add_argument("--base-size", type=int, default=256)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--multi-gpus", action="store_true", default=False)
    p.add_argument("--if-checkpoint", action="store_true", default=False)
    p.add_argument("--mode", choices=["test"], default="test")
    p.add_argument("--weight-path", type=str, required=True)
    p.add_argument("--model-name", type=str, default=None,
                   help="Output folder name; defaults to checkpoint parent directory.")
    p.add_argument("--results-dir", type=str, default="./Results")
    p.add_argument("--comparisons-dir", type=str, default="./Comparisons")
    p.add_argument("--no-save-predictions", action="store_true")
    p.add_argument("--no-save-comparisons", action="store_true")
    return p.parse_args()


def safe_name(value):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value).strip())


def load_original_pair(dataset_dir, name):
    image_path = Path(dataset_dir) / "images" / f"{name}.png"
    mask_path = Path(dataset_dir) / "masks" / f"{name}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Original image not found: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")
    with Image.open(image_path) as im:
        raw_image = im.copy().convert("RGB")
    with Image.open(mask_path) as im:
        raw_mask = im.copy().convert("L")
    return raw_image, raw_mask


def save_comparison(raw_image, raw_mask, probability, save_path):
    width, height = raw_image.size
    prob_u8 = (np.clip(probability, 0, 1) * 255).round().astype(np.uint8)
    pred_img = Image.fromarray(prob_u8, mode="L").resize(
        (width, height), resample=Image.Resampling.BILINEAR
    )
    mask = np.asarray(raw_mask)
    mask = (mask > (127 if mask.max() > 1 else 0.5)).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_image)
    axes[0].set_title("Raw Image")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(np.asarray(pred_img), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("MSHNet Output")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


class VisualMSHNetTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.dataset = IRSTD_Dataset(args, mode="val")
        self.loader = Data.DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=False)

        self.model = MSHNet(3)
        if args.multi_gpus and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        self.model.to(self.device)

        checkpoint = torch.load(args.weight_path, map_location=self.device, weights_only = False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "net" in checkpoint:
            state = checkpoint["net"]
        else:
            state = checkpoint
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)
        self.model.eval()

        self.miou = mIoU(1)
        self.pd_fa = PD_FA(1, 10, args.base_size)
        self.roc = ROCMetric(1, 10)

        # checkpoint_parent = Path(args.weight_path).parent.name
        # model_name = args.model_name or checkpoint_parent or Path(args.weight_path).stem
        # self.model_name = safe_name(model_name)

        # Derive the output folder name from the checkpoint filename
        checkpoint_name = Path(args.weight_path).name
        
        if checkpoint_name == "IRSTD-1k_weight.tar":
            default_model_name = "IRSTD-1k"
        elif checkpoint_name == "NUDT-SIRST_weight.tar":
            default_model_name = "NUDT-SIRST"
        else:
            # Fallback for other checkpoint filenames
            default_model_name = Path(args.weight_path).stem.replace("_weight", "")
        
        model_name = args.model_name or default_model_name
        self.model_name = safe_name(model_name)
        
        self.results_dir = Path(args.results_dir) / self.model_name
        self.comparisons_dir = Path(args.comparisons_dir) / self.model_name

        self.save_predictions = not args.no_save_predictions
        self.save_comparisons = not args.no_save_comparisons
        if self.save_predictions:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.save_comparisons:
            self.comparisons_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results directory: {self.results_dir.resolve()}")
        print(f"Comparisons directory: {self.comparisons_dir.resolve()}")

    def test(self):
        self.model.eval()
        self.miou.reset()
        self.pd_fa.reset()
        self.roc.reset()
        # main.py uses warm_epoch=-1 for test mode, making warm_flag True.
        warm_flag = (1 > -1)

        with torch.no_grad():
            for i, (data, mask) in enumerate(tqdm(self.loader, desc="Testing MSHNet")):
                data = data.to(self.device)
                mask = mask.to(self.device)
                _, pred = self.model(data, warm_flag)

                # Keep the original repository's metric calls.
                self.miou.update(pred, mask)
                self.pd_fa.update(pred, mask)
                self.roc.update(pred, mask)

                image_name = self.dataset.names[i]
                filename = f"{safe_name(image_name)}.png"
                probability = torch.sigmoid(pred[0, 0]).detach().cpu().numpy()

                if self.save_predictions:
                    arr = (np.clip(probability, 0, 1) * 255).round().astype(np.uint8)
                    Image.fromarray(arr, mode="L").save(self.results_dir / filename)

                if self.save_comparisons:
                    raw_image, raw_mask = load_original_pair(self.args.dataset_dir, image_name)
                    save_comparison(raw_image, raw_mask, probability, self.comparisons_dir / filename)

        fa, pd = self.pd_fa.get(len(self.loader))
        _, mean_iou = self.miou.get()
        self.roc.get()
        print("\nMSHNet Evaluation Results")
        print(f"mIoU: {float(mean_iou):.6f}")
        print(f"Pd:   {float(pd[0]):.6f}")
        print(f"Fa:   {float(fa[0] * 1_000_000):.6f} (x10^-6)")


def main():
    args = parse_args()
    VisualMSHNetTester(args).test()


if __name__ == "__main__":
    main()
