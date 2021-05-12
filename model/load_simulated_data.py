"""
Creates a Pytorch dataset to load simulated data by Deeptrack and add structural noise
"""

import config
import numpy as np
import os

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    plot_particles,
)
from StructuralNoise import struc_image


class YOLODataset(Dataset):
    def __init__(
        self,
        anchors,
        datadir,
        S=[14, 28, 56],
        transform=None,
    ):
        self.datadir = datadir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(
            anchors[0] + anchors[1] + anchors[2]
        )  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_thresh = 1e-8
        self.images = os.listdir(datadir + "/images/")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label_path = os.path.join(
            self.datadir + "/labels", self.images[index].replace("img", "boxes")
        )
        img_path = os.path.join(self.datadir + "/images", self.images[index])
        image = np.load(img_path)
        amplitude = (0.8 + 0.4 * np.random.random()) * np.max(image) / 70
        height, width, _ = image.shape
        image += struc_image(height, width, amplitude)[:, :, np.newaxis]
        image = np.concatenate((image.real, image.imag, np.abs(image)), axis=2)
        bboxes = np.load(label_path)

        if self.transform:
            xy = image.shape[1] * bboxes[:, 0:2]
            bboxes = np.concatenate((xy, bboxes[:, 2:]), axis=1)
            augmented = self.transform(image=image, keypoints=bboxes)
            image = augmented["image"]
            bboxes = np.array(augmented["keypoints"])
            if len(bboxes) > 0:
                bboxes[:, 0:2] /= image.shape[1]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            diff = torch.abs(torch.tensor([box[3:4]]) - self.anchors)
            mse_anchors = torch.mean(diff, dim=1)
            anchor_indices = mse_anchors.argsort(descending=False, dim=0)
            x, y, z, radius, n = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    box_coordinates = torch.tensor([x_cell, y_cell, z, radius, n])
                    targets[scale_idx][anchor_on_scale, i, j, 1:6] = box_coordinates
                    has_anchor[scale_idx] = True
                elif not anchor_taken and mse_anchors[anchor_idx] < self.ignore_thresh:
                    targets[scale_idx][
                        anchor_on_scale, i, j, 0
                    ] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    # transform = transforms.ToTensor()
    torch.manual_seed(0)

    dataset = YOLODataset(
        datadir="../training_data/",
        S=[14, 28, 56],
        anchors=anchors,
        transform=config.transform,
    )
    loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True
    )

    for x, y in tqdm(loader):
        for i in range(y[0].shape[1]):
           boxes = cells_to_bboxes(
               y[i],
               is_preds=False,
               S=y[i].shape[2],
           )[0]
        boxes = np.array(boxes)[np.array(boxes)[:,0]>0.5]
        print(len(boxes))
        plot_particles(x[0, 2:3].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()
