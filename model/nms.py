import numpy as np
from scipy.spatial.distance import cdist
import torchvision.ops as ops
import torch


def absolute_error_xy(coord1, coord2):
    """
    Calculates the absolute error between two coordinates containing
    an object score, x,y, additional coordinates
    :param coord1: array of shape (n, 1) where n is at least 3
    :type coord1: numpy array
    :param coord2: array of shape (n, 1) where n is at least 3
    :type coord2: numpy array
    """
    xy_1 = coord1[1:3]
    xy_2 = coord2[1:3]
    return np.mean(np.abs(xy_1 - xy_2))


def nms(
    bboxes, error_fn=absolute_error_xy, conf_threshold=0.5, threshold=5 / 448, M=1e6
):
    """
    Performs non max suppression on numpy array with positional predictions based on an error function
    :param bboxes: array with predictions of shape (N, D) where N is number of predictions and D
                   is the number of prediction coordinates where the first is the confidence score
                   of the prediction
    :type bboxes: numpy array
    :param error_fn: function to calculate the distance between two predictions
    :type error_fn: function
    :param conf_threshold: minimum threshold for confidence score under which all predictions are eliminated
    :type conf_threshold: float
    :param threshold: minimal threshold of distance between under which a prediction is eliminated
    :type threshold: float
    :param M: a large constant
    :type M: int
    :returns bboxes: array with remaining predictions after NMS
    :type bboxes: numpy array
    """
    bboxes = bboxes[bboxes[:, 0] > conf_threshold]
    indices = np.argsort((-bboxes[:, 0]))
    bboxes = bboxes[indices, :]
    cost_matrix = cdist(bboxes, bboxes, error_fn)
    cost_matrix[np.diag_indices(cost_matrix.shape[0])] = M
    duplicates = np.where(cost_matrix < threshold)
    duplicates = np.concatenate(
        (duplicates[0][:, np.newaxis], duplicates[1][:, np.newaxis]), axis=1
    )
    to_remove = np.unique(np.max(duplicates, axis=1))
    bboxes = np.delete(bboxes, to_remove, axis=0)

    return bboxes


def fast_nms(preds, conf_threshold, threshold, iou_thresh=1e-6):
    """
        Performs non max suppression on numpy array with positional predictions based on an error function
        :param preds: array with predictions of shape (N, D) where N is number of predictions and D
                       is the number of prediction coordinates where we have (object score, x, y, ...)
        :type bboxes: torch tensor
        :param conf_threshold: minimum threshold for confidence score under which all predictions are eliminated
        :type conf_threshold: float
        :param threshold: the extent of each positional prediction in width and height
        :type threshold: float
        :param iou_thresh: threshold for IOU calculation over which one prediction is eliminated
        :returns preds: array with remaining predictions after NMS
        :type preds: torch tensor
    """
    preds = preds[preds[:, 0] > conf_threshold]
    scores = preds[:, 0]
    bboxes = torch.cat(
        (preds[:, 1:3], threshold * torch.ones_like(preds[:, 1:3])), dim=1
    )
    indices = ops.nms(bboxes, scores, iou_threshold=iou_thresh)
    return preds[indices, :]
