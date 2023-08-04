import cv2
import numpy as np
from skimage import feature

DISTANCE_CUTOFF = 15
CLS_IDX_TO_NAME = {1: "BC", 2: "TC"}


def multiclass_dice_score(y_true, y_pred, eps=1e-7, ignore_bg=True):
    assert y_true.shape == y_pred.shape
    # y_pred = np.argmax(y_pred, axis=0) # H, W
    # y_pred = [(y_pred == v) for v in [0, 1, 2]]
    # y_pred = np.stack(y_pred, axis=0) # C, H, W

    # # extract certain classes from mask
    # y_true = [(y_true == v) for v in [0, 1, 2]]
    # y_true = np.stack(y_true, axis=0) # C, H, W

    # ignore BG
    if ignore_bg:
        y_true = y_true[1:] # C, H, W
        y_pred = y_pred[1:] # C, H, W

    intersection = np.sum(y_true * y_pred, axis=(1, 2))
    cardinality  = np.sum(y_true + y_pred, axis=(1, 2))
    dice_score   = (2. * intersection) / (cardinality + eps)

    return np.mean(dice_score)


def binary_dice_score(y_true, y_pred, eps=1e-7):
    assert y_true.shape == y_pred.shape

    intersection = np.sum(y_true * y_pred)
    cardinality  = np.sum(y_true + y_pred)
    dice_score   = (2. * intersection) / (cardinality + eps)

    return np.mean(dice_score)


def find_cells(heatmap):
    """This function detects the cells in the output heatmap

    Parameters
    ----------
    heatmap: torch.tensor
        output heatmap of the model,  shape: [1, 3, 1024, 1024] softmax

    Returns
    -------
        List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
    """

    bg, pred_wo_bg = np.split(heatmap, (1,), axis=0) # Background and non-background channels 0번이 BG, 1번이 CA, 2번이 TC
    bg  = np.squeeze(bg, axis=0)
    # obj = 1.0 - bg
    obj = np.sum(pred_wo_bg, axis=0)
    obj = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)
        
    peaks    = feature.peak_local_max(obj, min_distance=3, exclude_border=0, threshold_abs=0.0) # List[y, x]
    maxval   = np.max(pred_wo_bg, axis=0)
    maxcls_0 = np.argmax(pred_wo_bg, axis=0)

    # Filter out peaks if background score dominates
    peaks = np.array([peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]])
    if len(peaks) == 0:
        return []

    # Get score and class of the peaks
    scores     = maxval[peaks[:, 0], peaks[:, 1]]
    peak_class = maxcls_0[peaks[:, 0], peaks[:, 1]]

    predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]

    return predicted_cells


def _preprocess_distance_and_confidence(pred_all, gt_all):

    all_sample_result = []

    for pred, gt in zip(pred_all, gt_all):
        one_sample_result = {}

        for cls_idx in sorted(list(CLS_IDX_TO_NAME.keys())):
            pred_cls = np.array([p for p in pred if p[2] == cls_idx], np.float32)
            gt_cls   = np.array([g for g in gt if g[2] == cls_idx], np.float32)

            if len(gt_cls) == 0:
                gt_cls = np.zeros(shape=(0, 4))
            
            if len(pred_cls) == 0:
                distance   = np.zeros([0, len(gt_cls)])
                confidence = np.zeros([0, len(gt_cls)])

            else:
                pred_loc   = pred_cls[:, :2].reshape([-1, 1, 2])
                gt_loc     = gt_cls[:, :2].reshape([1, -1, 2])
                distance   = np.linalg.norm(pred_loc - gt_loc, axis=2)
                confidence = pred_cls[:, 2]

            one_sample_result[cls_idx] = (distance, confidence)

        all_sample_result.append(one_sample_result)

    return all_sample_result


def _calc_scores(all_sample_result, cls_idx, cutoff):    
    global_num_gt = 0
    global_num_tp = 0
    global_num_fp = 0

    for one_sample_result in all_sample_result:
        distance, confidence = one_sample_result[cls_idx]
        num_pred, num_gt = distance.shape
        assert len(confidence) == num_pred

        sorted_pred_indices = np.argsort(-confidence)
        bool_mask = (distance <= cutoff)

        num_tp = 0
        num_fp = 0
        for pred_idx in sorted_pred_indices:
            gt_neighbors = bool_mask[pred_idx].nonzero()[0]
            if len(gt_neighbors) == 0:  # No matching GT --> False Positive
                num_fp += 1
            else:  # Assign neares GT --> True Positive
                gt_idx = min(gt_neighbors, key=lambda gt_idx: distance[pred_idx, gt_idx])
                num_tp += 1
                bool_mask[:, gt_idx] = False

        assert num_tp + num_fp == num_pred
        global_num_gt += num_gt
        global_num_tp += num_tp
        global_num_fp += num_fp
        
    precision = global_num_tp / (global_num_tp + global_num_fp + 1e-7)
    recall = global_num_tp / (global_num_gt + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return round(f1, 4)


def mf1_metric(pred_all, gt_all):
    all_sample_result = _preprocess_distance_and_confidence(pred_all, gt_all)

    mf1 = (_calc_scores(all_sample_result, cls_idx=1, cutoff=DISTANCE_CUTOFF) + _calc_scores(all_sample_result, cls_idx=2, cutoff=DISTANCE_CUTOFF)) / 2.0
        
    return mf1