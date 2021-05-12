import numpy as np
import argparse
import config
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.optim as optim
from utils import load_checkpoint, plot_particles
from nms import fast_nms
from math import ceil
import cv2
from model import YOLOv3


def cells_to_preds(predictions, S, x_splits, y_splits, x_pos, y_pos, z_unit="micro"):
    """
    Converts the model output to predictions of particles relative to the entire image
    :param predictions: torch tensor with model output of shape (batch_size, anchors, S, S, 6)
    :param S: grid size of model output
    :param x_splits: number of divisions in x-direction of image (no overlaps)
    :param y_splits: number of divisions in y_directions of image (no overlaps)
    :param x_pos: torch tensor with leftmost x-position of all patches relative to image
    :param y_pos: torch tensor with uppermost y_position of all pathes relative to image
    :param z_unit: if 'micro' the z predictions will be converted to micrometres according to simulation settings
                   used in our experiments. Do not use if your images differ.
    :returns converted_bboxes: torch tensor of shape as predictions with coordinates scales relative to entire image
    """
    box_predictions = predictions[..., 1:6]
    box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
    if z_unit == "micro":
        box_predictions[..., 2:3] = -box_predictions[..., 2:3] * 0.114
    box_predictions[..., 3:4] = (
        box_predictions[..., 3:4] * 1e-7
    )  # convert r predictions
    scores = torch.sigmoid(predictions[..., 0:1])
    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    # convert predictions relative to entire patch
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    # convert predictions relative to entire image
    x = 1 / x_splits * (x + x_pos.reshape(x.shape[0], 1, 1, 1, 1))
    y = 1 / y_splits * (y + y_pos.reshape(y.shape[0], 1, 1, 1, 1))

    converted_bboxes = torch.cat((scores, x, y, box_predictions[..., 2:5]), dim=-1)
    return converted_bboxes


def run_on_patches(
    image,
    model,
    conf_threshold,
    nms_threshold,
    batch_size=128,
    z_unit="micro",
    toggle_eval=False,
    device="cuda",
    plot_result=False,
):
    """
    Runs input image as 448x448 patches through model and returns resulting particle predictions
    :param image: torch tensor of image to detect on of shape (side, side, 3) where side is divisible by 448.
    :param model: torch YOLOv3 model instance with loaded weights
    :param conf_threshold: confidence threshold over which to consider model predictions
    :param nms_threshold: pixel threshold under which two predictions are considered to be duplicates
    :param batch_size: batch size for model inference with patches of size 448x448
    :param z_unit: if 'micro' the z predictions will be converted to micrometres according to simulation settings
                   used in our experiments. Do not use if your images differ.
    :param toggle_eval: boolean to indicate whether to set model to eval or train mode for inference i.e.
                        whether to use batch statistics from training or not in batch normalization
    :param device: device to run model on
    :returns nms_boxes: torch tensor of predictions of particle positions of shape (*, 6) where 6 indicates
                        (confidence, x, y, z, r, n), x and y coordinates are given relative to image size

    """

    height, width, channels = image.shape
    kernel_size = config.IMAGE_SIZE
    scale = 2
    stride = int(kernel_size / scale)
    S = [kernel_size // 32, kernel_size // 16, kernel_size // 8]
    stride_after = [
        kernel_size // (32 * scale),
        kernel_size // (16 * scale),
        kernel_size // (8 * scale),
    ]
    min_width = ceil(width / kernel_size) * kernel_size
    min_height = ceil(height / kernel_size) * kernel_size
    x_splits, y_splits = image.shape[0] // kernel_size, image.shape[1] // kernel_size
    patches = image.unfold(0, kernel_size, stride).unfold(1, kernel_size, stride)
    y_patches, x_patches = patches.shape[0], patches.shape[1]
    patches = patches.contiguous().view(-1, 3, kernel_size, kernel_size)

    model = model.to(device)

    outputs = [
        torch.zeros((y_patches * x_patches, 3, S, S, 6))
        for S in [kernel_size // 32, kernel_size // 16, kernel_size // 8]
    ]
    x_indices = (
        torch.arange(0, min_width // kernel_size - 1 + 1e-6, stride / kernel_size)
        .repeat(y_patches, 1)
        .reshape(y_patches, x_patches)
        .reshape(-1)
    )
    y_indices = (
        (
            torch.arange(0, min_height // kernel_size - 1 + 1e-6, stride / kernel_size)
            .unsqueeze(1)
            .repeat(1, x_patches)
        )
        .reshape(y_patches, x_patches)
        .reshape(-1)
    )

    output_size = [
        (x_splits * S[0], y_splits * S[0]),
        (x_splits * S[1], y_splits * S[1]),
        (x_splits * S[2], y_splits * S[2]),
    ]
    strides = [
        (stride_after[0], stride_after[0]),
        (stride_after[1], stride_after[1]),
        (stride_after[2], stride_after[2]),
    ]
    if toggle_eval:
        model.eval()
    else:
        model.train()  # if there is some difference in train/test data that means that batch statistics don't work

    for id in range(ceil(patches.shape[0] / batch_size)):
        from_idx = id * batch_size
        to_idx = min((id + 1) * batch_size, patches.shape[0])
        curr_patch = patches[from_idx:to_idx].to(config.DEVICE)
        x_pos = x_indices[from_idx:to_idx].to(config.DEVICE)
        y_pos = y_indices[from_idx:to_idx].to(config.DEVICE)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                out = model(curr_patch)
        out = [
            cells_to_preds(
                out[i], out[i].shape[2], x_splits, y_splits, x_pos, y_pos, z_unit=z_unit
            )
            for i in range(3)
        ]
        outputs[0][from_idx:to_idx, ...] = out[0]
        outputs[1][from_idx:to_idx, ...] = out[1]
        outputs[2][from_idx:to_idx, ...] = out[2]

    outputs = [
        outputs[i].permute(1, 4, 2, 3, 0).reshape(3 * 6, S[i] ** 2, -1)
        for i in range(3)
    ]
    new_outputs = [
        F.fold(
            outputs[i], output_size=output_size[i], kernel_size=S[i], stride=strides[i]
        )
        for i in range(3)
    ]
    recovery_mask = [
        F.fold(
            torch.ones_like(outputs[i]),
            output_size=output_size[i],
            kernel_size=S[i],
            stride=strides[i],
        )
        for i in range(3)
    ]
    outputs = [
        (new_outputs[i] / recovery_mask[i])
        .reshape(3, 6, output_size[i][0], output_size[i][1])
        .unsqueeze(0)
        .permute(0, 1, 3, 4, 2)
        for i in range(3)
    ]

    bboxes = torch.tensor([])
    for i in range(3):
        bboxes = torch.cat((bboxes, outputs[i].reshape(-1, 6)), dim=0)

    image_size = image.shape[0]
    nms_boxes = fast_nms(
        bboxes, conf_threshold=conf_threshold, threshold=nms_threshold / image_size
    )
    if plot_result:
        plot_particles(image[:, :, 2:3].to("cpu"), nms_boxes)
    return nms_boxes


def detect():
    """
    Pads input image and performs detection of particles on the image with YOLOv3.
    Note that results for x and y coordinates are given relative to the padded image.
    """
    parser = argparse.ArgumentParser(description="""
    Pads image  and performs detection of particles on the image with YOLOv3.
    Note that results for x and y coordinates are given relative to the padded image. 
    Returns particle positions and characteristics as tensor of shape (*, 6) where 6
    corresponds to (confidence, x, y, x, radius, refractive index)""")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="../weights.pth.tar",
        help="Path to weights or checkpoint file .pth.tar",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="../data/img.npy",
        help="Path to directory with images to inference (npy format expected)",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1, help="Size of each image batch"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.5, help="Object confidence threshold"
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=5,
        help="Pixel threshold for non-maximum suppression",
    )
    parser.add_argument(
        "--z_unit",
        type=str,
        default="",
        help="if 'micro' the z predictions will be converted to micrometres according to simulation settings\
                                                                used in our experiments. Do not use if your images differ.",
    )
    parser.add_argument(
        "--toggle_eval",
        type=bool,
        default=False,
        help="boolean to indicate whether to set model to eval or train mode for inference i.e \
                                                                        whether to use batch statistics from training or not in batch normalization",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run model on"
    )
    parser.add_argument(
        "--plot_result",
        type=bool,
        default=True,
        help="Whether the result should be plotted with Matplotlib",
    )
    args = parser.parse_args()

    image = np.load(args.image)
    height, width, channels = image.shape
    kernel_size = 448
    min_width = ceil(width / kernel_size) * kernel_size
    min_height = ceil(height / kernel_size) * kernel_size
    side = max(min_width, min_height)
    transform = A.Compose(
        [
            A.PadIfNeeded(
                min_width=side, min_height=side, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=1.0,),
            ToTensorV2(),
        ]
    )

    image = transform(image=image)["image"].permute(1, 2, 0)
    model = YOLOv3(in_channels=3, num_classes=config.NUM_CLASSES)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    load_checkpoint(args.weights, model, optimizer, config.LEARNING_RATE)

    model = model.to(args.device)
    results = run_on_patches(
        image,
        model,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        batch_size=args.batch_size,
        z_unit=args.z_unit,
        toggle_eval=args.toggle_eval,
        device=args.device,
        plot_result=args.plot_result,
    )
    return results


if __name__ == "__main__":
    detect()
