import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import config
from tqdm import tqdm
from nms import nms
from detect_on_patches import run_on_patches
from utils import cells_to_bboxes

def rmse_xy(coord1, coord2):
    xy_1 = coord1[0:2]
    xy_2 = coord2[0:2]
    return np.sqrt(np.mean((xy_1 - xy_2) ** 2))


def rmse(coord1, coord2):
    return np.sum((coord1 - coord2) ** 2, axis=0)


def rel_error(pred, true):
    return np.sum(np.abs((pred - true) / true), axis=0)


def get_errors(pred_boxes, true_boxes, pixel_threshold, image_size):
    """

    This function calculates the matchings between two sets of coordinates and the number of true
    positivs

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        pixel_threshold (float): the mean number of pixels for where a prediction is considered a true positive

    Returns:
        TP (int): number of true positive predictions
        num_detections (int): number of detections in image
        num_ground_truths (int): number of ground truths in image
        coord_errors (np array): of mean absolute error for each coordinate in the image
    """

    threshold = pixel_threshold / image_size
    M = 1e8
    pred_boxes = np.array(pred_boxes)
    true_boxes = np.array(true_boxes)
    num_detections = pred_boxes.shape[0]
    num_ground_truths = true_boxes.shape[0]

    cost_matrix = cdist(pred_boxes, true_boxes, rmse_xy)
    cost_matrix[cost_matrix > threshold] = M
    pred_indices, true_indices = linear_sum_assignment(cost_matrix)

    true_positives = cost_matrix[pred_indices, true_indices] < M
    TP = np.sum(true_positives)
    if TP > 0:
        rmse_errors = rmse(
            pred_boxes[pred_indices[true_positives]],
            true_boxes[true_indices[true_positives]],
        )
        rel_errors = rel_error(
            pred_boxes[pred_indices[true_positives]],
            true_boxes[true_indices[true_positives]],
        )
    else:
        rmse_errors = np.zeros(true_boxes.shape[1])
        rel_errors = np.zeros(true_boxes.shape[1])

    return TP, num_detections, num_ground_truths, rmse_errors, rel_errors


def evaluate_experimental_data(
    loader,
    model,
    conf_threshold=0.5,
    pixel_threshold=5,
    image_size=2240,
    device=config.DEVICE,
    nms_threshold=7,
    batch_size=128,
    z_unit="micro",
    toggle_eval=False,
):
    """
    Evaluates the YOLOv3 model on the data in the loader inputted
    :param loader: PyTorch dataloader with images
    :param model: YOLOv3 model with loaded weights
    :param conf_threshold: confidence threshold over which to consider model predictions
    :param pixel_threshold: pixel_threshold under which to consider prediction true positive
    :param image_size: size of images in loader
    :param device: device to run model on
    :param nms_threshold: pixel threshold under which two predictions are considered to be duplicates
    :param batch_size: batch size for model inference with patches of size 448x448
    :param z_unit: if 'micro' the z predictions will be converted to micrometres according to simulation settings
                   used in our experiments. Do not use if your images differ.
    :param toggle_eval: boolean to indicate whether to set model to eval or train mode for inference i.e.
                        whether to use batch statistics from training or not in batch normalization
    :returns precision: (float) model's precision on loader
    :returns recall: (float) model's recall on loader
    :returns F1: (float) F1 score from precision and recall
    :returns rmse_error_rate: numpy array with rmse for x, y, z, radius, refractive index
    :returns rel_error_rate: numpy array with relative error for x, y, z, radius, refractive index
    """
    total_TP = 0
    num_detections = 0
    num_ground_truths = 0
    total_rmse_errors = 0
    total_rel_errors = 0
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        pred_bboxes = run_on_patches(
            x.squeeze(0).permute(1, 2, 0),
            model,
            conf_threshold,
            nms_threshold,
            batch_size=batch_size,
            z_unit=z_unit,
            toggle_eval=toggle_eval,
            device=device
        )
        # we just want one bbox for each label, not one for each scale
        # remove predictions below certain threshold
        pred_bboxes = pred_bboxes[pred_bboxes[:, 0] > conf_threshold, :][:, 1:]

        TP, detections, ground_truths, rmse_errors, rel_errors = get_errors(
            pred_bboxes, labels.squeeze(0), pixel_threshold, image_size
        )

        total_TP += TP
        num_detections += detections
        num_ground_truths += ground_truths
        total_rmse_errors += rmse_errors
        total_rel_errors += rel_errors

    precision = total_TP / (num_detections + 1e-6)
    recall = total_TP / (num_ground_truths + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    rmse_error_rate = np.sqrt(total_rmse_errors / (total_TP + 1e-6))
    rel_error_rate = total_rel_errors / (total_TP + 1e-6)
    return precision, recall, F1, rmse_error_rate, rel_error_rate


def evaluate_model(
    loader,
    model,
    conf_threshold=0.7,
    pixel_threshold=5,
    image_size=448,
    device=config.DEVICE,
    nms_threshold=2,
):
    """
    Evaluates the YOLOv3 model on the data in the loader inputted
    :param loader: PyTorch dataloader with images
    :param model: YOLOv3 model with loaded weights
    :param conf_threshold: confidence threshold over which to consider model predictions
    :param pixel_threshold: pixel_threshold under which to consider prediction true positive
    :param image_size: size of images in loader
    :param device: device to run model on
    :param nms_threshold: pixel threshold under which two predictions are considered to be duplicates
    :returns precision: (float) model's precision on loader
    :returns recall: (float) model's recall on loader
    :returns F1: (float) model's F1 score from precision and recall
    :returns rmse_error_rate: numpy array with rmse for x, y, z, radius, refractive index
    :returns rel_error_rate: numpy array with relative error for x, y, z, radius, refractive index
    """
    model.eval()
    total_TP = 0
    num_detections = 0
    num_ground_truths = 0
    total_rmse_errors = 0
    total_rel_errors = 0
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        if batch_idx > 50:
            break
        x = x.to(device).squeeze(0)

        with torch.no_grad():
            predictions = model(x)

        TP, detections, ground_truths, rmse_errors, rel_errors = get_batch_errors(
            predictions,
            labels,
            conf_threshold,
            pixel_threshold,
            image_size,
            nms_threshold,
        )
        total_TP += TP
        num_detections += detections
        num_ground_truths += ground_truths
        total_rmse_errors += rmse_errors
        total_rel_errors += rel_errors

    model.train()
    precision = total_TP / (num_detections + 1e-6)
    recall = total_TP / (num_ground_truths + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    rmse_error_rate = np.sqrt(total_rmse_errors / (total_TP + 1e-6))
    rel_error_rate = total_rel_errors / (total_TP + 1e-6)
    return precision, recall, F1, rmse_error_rate, rel_error_rate


def get_batch_errors(
    predictions,
    labels,
    conf_threshold,
    pixel_threshold,
    image_size,
    nms_threshold,
):
    """
    Returns number of true postives, detections and ground truths as well as total squared errors and relative errors
    for inputted predictions and labels
    :param predictions: list of tensors for predictions from model where each tensor has shape: (batch size, number of anchors on scale (3), grid size, grid size, 6)
    :param target: list of tensors for targets where each tensor has shape: (batch size, number of anchors on scale (3), grid size, grid size, 6)
    the 6 values signify (object score, x, y, z, radius, refractive index)
    :param conf_threshold: confidence threshold over which to consider model predictions
    :param pixel_threshold: pixel_threshold under which to consider a prediction true positive
    :param image_size: size of images in loader
    :param nms_threshold: pixel threshold under which two predictions are considered to be duplicates
    :returns total_TP: (int) number of true positive in the batch
    :returns num_detections: (int) number of detections in the batch
    :returns num_ground_truths: (int) number of targets in the batch
    :returns total_rmse_errors: (numpy array) total squared error for all true positive detections for each
                                x, y, z, radius, refractive index
    :returns total_rel_errors: (numpy array) sum of all relative errors for all true positive detections for each
                                x, y, z, radius, refractive index

    """
    total_TP = 0
    num_detections = 0
    num_ground_truths = 0
    total_rmse_errors = 0
    total_rel_errors = 0

    batch_size = predictions[0].shape[0]
    bboxes = [[] for _ in range(batch_size)]
    for i in range(3):
        S = predictions[i].shape[2]
        boxes_scale_i = cells_to_bboxes(predictions[i], S=S, is_preds=True)
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    # we just want one bbox for each label, not one for each scale
    true_bboxes = cells_to_bboxes(labels[2].squeeze(0), S=S, is_preds=False)

    for idx in range(batch_size):
        nms_boxes = nms(
            np.array(bboxes[idx]),
            conf_threshold=conf_threshold,
            threshold=nms_threshold / image_size,
        )
        cur_pred_bboxes = np.array(nms_boxes)
        cur_true_bboxes = np.array(true_bboxes[idx])

        # remove predictions below certain threshold
        cur_pred_bboxes = cur_pred_bboxes[cur_pred_bboxes[:, 0] > conf_threshold, :][
            :, 1:
        ]
        cur_true_bboxes = cur_true_bboxes[cur_true_bboxes[:, 0] > conf_threshold][:, 1:]
        TP, detections, ground_truths, rmse_errors, rel_errors = get_errors(
            cur_pred_bboxes, cur_true_bboxes, pixel_threshold, image_size
        )

        total_TP += TP
        num_detections += detections
        num_ground_truths += ground_truths
        total_rmse_errors += rmse_errors
        total_rel_errors += rel_errors

    return (
        total_TP,
        num_detections,
        num_ground_truths,
        total_rmse_errors,
        total_rel_errors,
    )


