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


def cells_to_preds(predictions, S, z_unit="micro"):
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
    # convert predictions relative to image
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))

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
    plot_image = image
    image = torch.movedim(image, 2, 0)
    image = image[None, :, :, :]
    with torch.no_grad():
        out = model(image.to(config.DEVICE))

    predictions = [cells_to_preds(out[i], config.S[i]) for i in range(3)]
    bboxes = torch.tensor([])
    for i in range(3):
        bboxes = torch.cat((bboxes.to(config.DEVICE), predictions[i].reshape(-1, 6)), dim=0)

    nms_boxes = fast_nms(
        bboxes, conf_threshold=conf_threshold, threshold=nms_threshold / config.IMAGE_SIZE
    )
    if plot_result:
        plot_particles(plot_image[:, :, 2:3].to("cpu"), nms_boxes.to("cpu"))
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
        default="weights.pth.tar",
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
    pad_side = config.PAD_SIDE
    transform = A.Compose(
        [
            A.PadIfNeeded(
                min_width=pad_side, min_height=pad_side, border_mode=cv2.BORDER_CONSTANT
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
