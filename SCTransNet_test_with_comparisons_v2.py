import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import threading
from dataset import *
import time
from collections import OrderedDict
from model.SCTransNet import SCTransNet as SCTransNet
# from loss import *
import model.Config as config
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe plotting for servers
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
from skimage import measure


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    predict = (output > score_thresh).float()
    if len(target.shape) == 3:
        print('？？？？')  # 加一个维度 使得target与 output的size一致
        target = target.unsqueeze(dim=0)
        # target = np.expand_dims(target.float(), axis=1)
        target.to('cuda', torch.float)

    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    # 现在predict中高于阈值的部分为全1矩阵   target是GT

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()  # 对的预测为对的
    fp = (predict * ((predict != target).float())).sum()  # 错的预测为对的 虚警像素数
    tn = ((1 - predict) * ((predict == target).float())).sum()  # 错的预测为错的
    fn = (((predict != target).float()) * (1 - predict)).sum()  # 对的预测为错的
    pos = tp + fn  # 标签中 阳性的个数
    neg = fp + tn  # 标签中 阴性的个数
    class_pos = tp + fp  # 检测出的个数

    return tp, pos, fp, neg, class_pos


class SamplewiseSigmoidMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.lock = threading.Lock()
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.

        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """

        def evaluate_worker(self, label, pred):
            inter_arr, union_arr = batch_intersection_union_n(
                pred, label, self.nclass, self.score_thresh)
            with self.lock:
                self.total_inter = np.append(self.total_inter, inter_arr)
                self.total_union = np.append(self.total_union, union_arr)

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        nIoU = IoU.mean()
        return nIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])


def batch_intersection_union_n(output, target, nclass, score_thresh):
    """nIoU"""
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass
    outputnp = output.detach().cpu().numpy()
    # outputsig = F.sigmoid(output).detach().cpu().numpy()
    # outputsig = nd.sigmoid(output).asnumpy()
    predict = (outputnp > 0.5).astype('int64')
    # predict = predict.detach().cpu().numpy()
    # predict = (output.asnumpy() > 0).astype('int64') # P
    if len(target.shape) == 3:
        target = nd.expand_dims(target, axis=1).asnumpy().astype('int64')  # T
    elif len(target.shape) == 4:
        target = target.cpu().numpy().astype('int64')  # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target)  # TP  交集

    num_sample = intersection.shape[0]
    area_inter_arr = np.zeros(num_sample)
    area_pred_arr = np.zeros(num_sample)
    area_lab_arr = np.zeros(num_sample)
    area_union_arr = np.zeros(num_sample)
    for b in range(num_sample):
        # areas of intersection and union
        area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
        area_inter_arr[b] = area_inter

        area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
        area_pred_arr[b] = area_pred

        area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
        area_lab_arr[b] = area_lab

        area_union = area_pred + area_lab - area_inter
        area_union_arr[b] = area_union

        assert (area_inter <= area_union).all(), \
            "Intersection area should be smaller than Union area"

    return area_inter_arr, area_union_arr


class ROCMetric05():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins):
        # bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        # nclass :有几个类别 红外弱小目标检测只有一个类别
        super(ROCMetric05, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    # 网络输入的结果和标签 计算两者之前的东西
    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            # score_thresh = (iBin + 0.0) / self.bins
            score_thresh = (0.0 + iBin) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp  # 虚警像素数
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)  # tp_rates = recall = TP/(TP+FN)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)  # fp_rates =  FP/(FP+TN)
        FP = self.fp_arr / (self.neg_arr + self.pos_arr)
        recall = self.tp_arr / (self.pos_arr + 0.001)  # recall = TP/(TP+FN)
        precision = self.tp_arr / (self.class_pos + 0.001)  # precision = TP/(TP+FP)
        f1_score = (2.0 * recall[5] * precision[5]) / (recall[5] + precision[5] + 0.00001)

        return tp_rates, fp_rates, recall, precision, FP, f1_score

    def reset(self):
        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])


class mIoU():

    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)  # labeled: GT中目标的像素数目   correct:预测正确的像素数
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class PDFA():
    def __init__(self, ):
        super(PDFA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel += np.sum(self.dismatch)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel if self.all_pixel else 0.0
        Final_PD = self.PD / self.target if self.target else 0.0
        return float(Final_PD), float(Final_FA)

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class PD_FA():
    def __init__(self, ):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)  # 目标总数  直接就搞GT的连通域个数
        self.image_area_total = []  # 图像中预测的区域列表
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):  # image 与 label 之间 根据中心点 进行连通域的确定
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]  # 匹配上一个之后就 清除一个
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]  # 在image里面 但是不在label里面

        self.dismatch_pixel += np.sum(self.dismatch)  # Fa 虚警个数 像素的虚警
        # print(self.dismatch_pixel)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)  # 如果中心点之间距离在3一下 就算Pd  所以Pd 是匹配上了的目标的个数

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, float(Final_FA.cpu().detach().numpy())

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])




os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument('--ROC_thr', type=int, default=10, help='num')

# parser.add_argument("--model_names", default=['SCTrans'], type=list,
                    # help="model_name: 'ACM', 'Ours01', 'DNANet', 'ISNet', 'ACMNet', 'Ours01', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--model_names", default=['SCTrans'], nargs='+',
                    help="model_name: 'ACM', 'Ours01', 'DNANet', 'ISNet', 'ACMNet', 'Ours01', 'ISTDU-Net', 'U-Net', 'RISTDnet'")

parser.add_argument("--pth_dirs", default=["SIRST3/SCTransNet_NUAA_NUDT_IRSTD1K.pth.tar"], nargs="+")
parser.add_argument("--dataset_dir", default='datasets', type=str, help="train_dataset_dir")

# parser.add_argument("--dataset_names", default=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'], type=list,
#                     help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--dataset_names", default=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")

parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default="./Result/",
                    help="root directory for saved prediction images")
parser.add_argument("--save_comparison_dir", type=str, default="./Comparison/",
                    help="root directory for saved comparison plots")
parser.add_argument("--save_comparison", dest="save_comparison", action="store_true",
                    help="save Raw image / Ground truth / SCTransNet comparison plots (default: enabled)")
parser.add_argument("--no_save_comparison", dest="save_comparison", action="store_false",
                    help="disable comparison plot saving")
parser.set_defaults(save_comparison=True)
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()


def load_raw_image(dataset_dir, dataset_name, image_id):
    """
    Load the original, unnormalized grayscale image for visualization.
    TestSetLoader normalizes and pads its model input, so that tensor should
    not be used as the "Raw image" panel.
    """
    image_dir = Path(dataset_dir) / dataset_name / "images"
    for extension in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
        image_path = image_dir / f"{image_id}{extension}"
        if image_path.is_file():
            with Image.open(image_path) as image:
                return np.asarray(image.convert("I"), dtype=np.float32)

    raise FileNotFoundError(
        f"Could not find the original image for ID '{image_id}' in {image_dir}. "
        "Expected a .png, .bmp, .jpg, .jpeg, .tif, or .tiff file."
    )


# def save_comparison_plot(raw_image, gt_mask, prediction, save_path):
#     """
#     Save a 1x3 comparison: original grayscale image, ground-truth mask,
#     and the model's sigmoid probability map.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     axes[0].imshow(raw_image, cmap="gray")
#     axes[0].set_title("Raw Image")

#     axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
#     axes[1].set_title("Ground Truth Mask")

#     axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
#     axes[2].set_title("SCTransNet Output")

#     for ax in axes:
#         ax.axis("off")

#     fig.tight_layout()
#     fig.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)


def save_comparison_plot(raw_image, gt_mask, prediction, iou_score, save_path):
    """
    Save a 1x3 comparison and display per-image IoU
    in the SCTransNet Output panel title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_image, cmap="gray")
    axes[0].set_title("Raw Image")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"SCTransNet Output | IoU: {iou_score:.4f} "
        f"({iou_score * 100:.2f}%)"
    )

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def test():
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    # *************************固定阈值**********************

    BINARY_THRESHOLD = 0.2

    # 计算mIOU  完全OK
    IOU = mIoU()
    # 计算nIOU 完全OK
    nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)

    # 计算PD_FA   完全OK
    eval_05 = PD_FA()
    ROC_05 = ROCMetric05(nclass=1, bins=10)
    config_vit = config.get_SCTrans_config()


    # CPU
    # net = SCTransNet(config_vit, mode='test', deepsuper=True)
    # state_dict = torch.load(opt.pth_dir, map_location='cpu')
    # # CUDA
    net = SCTransNet(config_vit, mode='test', deepsuper=True).cuda()
    state_dict = torch.load(opt.pth_dir)

    new_state_dict = OrderedDict()
    #
    for k, v in state_dict['state_dict'].items():
        name = k[6:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    net.load_state_dict(new_state_dict)
    net.eval()
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(tbar):
            # img = Variable(img)

            # CPU
            # pred = net.forward(img)
            # pred = pred[:, :, :size[0], :size[1]]
            # gt_mask = gt_mask[:, :, :size[0], :size[1]]

            # CUDA:
            # pred = net.forward(img).cuda()
            # pred = pred[:, :, :size[0], :size[1]].cuda()
            # gt_mask = gt_mask[:, :, :size[0], :size[1]].cuda()

            # CUDA fixed: move inputs to the same device as the model BEFORE inference.
            img = img.cuda(non_blocking=True)
            gt_mask = gt_mask.cuda(non_blocking=True)

            pred = net(img)
            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]]

            # Convert prediction to binary mask
            pred_binary = (pred > BINARY_THRESHOLD).float()

            # Fix  threshold ##########################################################
            # IOU
            # IOU.update((pred > 0.5), gt_mask)  # 像素
            IOU.update(pred_binary, gt_mask) 

            # nIOU
            nIoU_metric.update(pred, gt_mask)  # 像素

            # PD_FA
            # eval_05.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)  # 目标
            eval_05.update((pred_binary[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)


            ROC_05.update(pred, gt_mask)
            # Save prediction output and/or 1x3 comparison visualization.
            # Directory layout: Result/<model_name>/ and Comparisons/<model_name>/.
            # Dataset prefixes prevent filename collisions across datasets.
            image_id = str(img_dir[0])
            output_name = f"{opt.test_dataset_name}_{image_id}.png"

            # prediction_map = pred[0, 0].detach().cpu().numpy()
            # ground_truth = gt_mask[0, 0].detach().cpu().numpy()

            prediction_map_raw = pred[0, 0].detach().cpu().numpy()
            prediction_map_binary = pred_binary[0, 0].detach().cpu().numpy().astype(np.uint8)
            ground_truth = gt_mask[0, 0].detach().cpu().numpy()

            if opt.save_img:
                output_dir = Path(opt.save_img_dir) / opt.model_name
                output_dir.mkdir(parents=True, exist_ok=True)

                # prediction_uint8 = (
                #     np.clip(prediction_map, 0.0, 1.0) * 255.0
                # ).round().astype(np.uint8)
                # Image.fromarray(prediction_uint8, mode="L").save(output_dir / output_name)

                binary_mask_uint8 = (prediction_map_binary * 255).astype(np.uint8)
                Image.fromarray(binary_mask_uint8, mode="L").save(output_dir/output_name)

            if opt.save_comparison:
                comparison_dir = Path(opt.save_comparison_dir) / opt.model_name
                comparison_dir.mkdir(parents=True, exist_ok=True)
                
                raw_image = load_raw_image(
                    opt.dataset_dir, opt.test_dataset_name, image_id
                )

                # gt_binary = ground_truth > 0.5

                pred_binary_np = prediction_map_binary > 0.0
                gt_binary = ground_truth > 0.5

                # Per-image IoU
                intersection = np.logical_and(
                    pred_binary_np, gt_binary
                ).sum()
            
                union = np.logical_or(
                    pred_binary_np, gt_binary
                ).sum()
            
                image_iou = intersection / union if union > 0 else 1.0


                # save_comparison_plot(
                #     raw_image=raw_image,
                #     gt_mask=(ground_truth > 0.5).astype(np.uint8),
                #     prediction=prediction_map,
                #     save_path=comparison_dir / output_name,
                # )

                # save_comparison_plot(
                #     raw_image=raw_image,
                #     gt_mask=(ground_truth > 0.5).astype(np.uint8),
                #     prediction=prediction_map_binary,
                #     save_path=comparison_dir / output_name,
                # )

                # Save comparison with IoU in the title
                save_comparison_plot(
                    raw_image=raw_image,
                    gt_mask=gt_binary.astype(np.uint8),
                    prediction=prediction_map_binary,
                    iou_score=image_iou,
                    save_path=comparison_dir / output_name,
                )

        # 0.5
        # IOU OK Good！
        pixAcc, mIOU = IOU.get()
        # # nIOU OK Good！
        nIoU = nIoU_metric.get()
        # # Pd Fa
        results2 = eval_05.get()
        #
        # # FP
        ture_positive_rate, false_positive_rate, recall, precision, FP, F1_score = ROC_05.get()

        print('pixAcc: %.4f| mIoU: %.4f | nIoU: %.4f | Pd: %.4f| Fa: %.4f |F1: %.4f'
              % (pixAcc * 100, mIOU * 100, nIoU * 100, results2[0] * 100, results2[1] * 1e+6, F1_score * 100))




if __name__ == '__main__':

    os.makedirs(opt.save_log, exist_ok = True)
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    # if dataset_name in pth_dir and model_name in pth_dir:
                    opt.test_dataset_name = dataset_name
                    opt.model_name = model_name
                    opt.train_dataset_name = pth_dir.split('/')[0]
                    print(pth_dir)
                    opt.f.write(pth_dir)
                    print(opt.test_dataset_name)
                    opt.f.write(opt.test_dataset_name + '\n')
                    # opt.pth_dir = opt.save_log + pth_dir
                    opt.pth_dir = pth_dir
                    test()
                    print('\n')
                    opt.f.write('\n')
        opt.f.close()
