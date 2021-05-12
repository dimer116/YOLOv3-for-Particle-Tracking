import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
from nms import nms
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

def plot_particles(image, boxes, scores=True, pixels=False):
    """Plots predicted  particles on the image"""
    im = np.array(image)
    height, width, _ = im.shape if not pixels else (1,1,1)

    # Create figure and axes
    bboxes = np.asarray(boxes)
    # bboxes = bboxes[bboxes[...,0]>0.5]
    if len(bboxes) > 0:
        if scores:
            plt.scatter(bboxes[..., 1] * width, bboxes[..., 2] * height)
        else:
            plt.scatter(bboxes[..., 0] * width, bboxes[..., 1] * height)
    # Display the image
    plt.imshow(im, cmap="gray")

    plt.show()


def cells_to_bboxes(predictions, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or compared to targets.
    :param predictions: tensor of size (N, 3, S, S, num_classes+5)
    :param S: the number of cells the image is divided in on the width (and height)
    :param is_preds: whether the input is predictions or the true bounding boxes
    :returns converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    box_predictions = predictions[..., 1:6]
    if is_preds:
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 3:4] = box_predictions[..., 3:4] * 1e-7  # r predictions
        scores = torch.sigmoid(predictions[..., 0:1])
    else:
        scores = predictions[..., 0:1]
    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    converted_bboxes = torch.cat(
        (scores, x, y, box_predictions[..., 2:5]), dim=-1
    ).reshape(BATCH_SIZE, 3 * S * S, 6)
    return converted_bboxes.tolist()




def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(train_dir, test_dir, batch_size, num_workers, pin_memory):
    from load_simulated_data import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        anchors = config.ANCHORS,
        datadir=train_dir,
        transform=config.transform,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
    )
    test_dataset = YOLODataset(
        anchors=config.ANCHORS,
        datadir=test_dir,
        transform=config.transform,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader


def plot_couple_examples(model, loader, threshold, nms_threshold):

    model.eval()
    x, y = next(iter(loader))
    x = x.to(config.DEVICE).squeeze(0)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            out = model(x)
            bboxes = [[] for _ in range(x.shape[0])]
            for i in range(3):
                batch_size, A, S, _, _ = out[i].shape
                boxes_scale_i = cells_to_bboxes(out[i], S=S, is_preds=True)
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

    model.train()

    for i in range(batch_size):
        nms_boxes = nms(np.array(bboxes[i]), conf_threshold=threshold, threshold=nms_threshold).tolist()
        print(len(nms_boxes))
        plot_particles(x[i, 2:3, :, :].permute(1, 2, 0).detach().cpu(), nms_boxes)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
