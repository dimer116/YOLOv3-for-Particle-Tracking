"""
Loads the experimental data, tests rmse, precision, recall, runs on patches and plots result
"""

import config
import numpy as np
import albumentations as A
import os
from detect_on_patches import run_on_patches
from torch.utils.data import Dataset, DataLoader
from utils import cells_to_bboxes, plot_particles, load_checkpoint
from metrics import evaluate_experimental_data
from model import YOLOv3
import torch.optim as optim
from tqdm import tqdm
from StructuralNoise import struc_image

class ExperimentalDataset(Dataset):
    """
    Dataset class to load experimental images or simulated images with larger size than training size
    """
    def __init__(self, datadir, transform=None, exp_images=True):
        self.datadir = datadir
        self.transform = transform
        self.exp_images = exp_images
        self.annotations = os.listdir(os.path.join(datadir, "images"))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(
            self.datadir + "/labels", self.annotations[index].replace("img", "boxes")
        )
        img_path = os.path.join(self.datadir + "/images", self.annotations[index])
        image = np.load(img_path, allow_pickle=True)
        bboxes = np.load(label_path)
        if self.exp_images:
            bboxes = np.concatenate(
                (bboxes[:, 1:2], bboxes[:, 0:1], bboxes[:, 2:]), axis=1
            )
            bboxes = np.concatenate(
                (
                    bboxes[:, 0:3],
                    2.2 * 1e-7 * np.ones((bboxes.shape[0], 1)),
                    1.58 * np.ones((bboxes.shape[0], 1)),
                ),
                axis=1,
            )
        else:
            amplitude = (0.8 + 0.4 * np.random.random()) * np.max(image) / 70
            height, width, _ = image.shape
            image += struc_image(height, width, amplitude)[:, :, np.newaxis]
            image = np.concatenate((image.real, image.imag, np.abs(image)), axis=2)

        # image = (image - np.mean(image, axis=(0,1)))/np.std(image, axis=(0,1))
        image_dims = image.shape[:2][::-1]
        if self.transform:
            bboxes[:, :2] *= image_dims
            augmented = self.transform(image=image, keypoints=bboxes)
            image = augmented["image"]
            bboxes = np.array(augmented["keypoints"])
            bboxes[:, 0:2] /= image.shape[1]
        return image, bboxes

def test_on_experimental_data():
    """
    Code for testing on dataset with experimental images
    """
    path_to_exp_data = "../../Experiment_images"
    check_metrics = False
    checkpoint = config.LOAD_CHECKPOINT_FILE
    dataset = ExperimentalDataset(
        datadir=path_to_exp_data, transform=config.exp_transforms, exp_images=True
    )
    convert_back = A.Compose([A.CenterCrop(height=config.EXP_SIZE[0], width=config.EXP_SIZE[1])], additional_targets={'keypoints1':'keypoints'},
              keypoint_params=A.KeypointParams(format="xy"))
    loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    model = YOLOv3(in_channels=3, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    load_checkpoint(checkpoint, model, optimizer, config.LEARNING_RATE)
    # haven't found best values for them, trades between precision and recall
    conf_threshold = 0.5
    nms_threshold = 7
    batch_size = 128
    # Evaluate on metrics
    if check_metrics:
        precision, recall, F1, rmse_errors, rel_errors = evaluate_experimental_data(
            loader,
            model,
            conf_threshold=conf_threshold,
            pixel_threshold=10,
            image_size=config.side,
            device=config.DEVICE,
            nms_threshold=nms_threshold,
            batch_size=batch_size,
            z_unit="micro",
            toggle_eval=True,
        )
        print(
            f"Precision is: {precision}, \n Recall is {recall}, \n F1: {F1}, \n"
            f"X rmse error: {rmse_errors[0]}, y rmse error: {rmse_errors[1]}, z rmse error: {rmse_errors[2]}, r rmse error: {rmse_errors[3]}, n rmse error: {rmse_errors[4]} \n"
            f"X relative error: {rel_errors[0]}, y relative error: {rel_errors[1]}, z relative error: {rel_errors[2]}, r relative error: {rel_errors[3]}, n relative error: {rel_errors[4]}"
        )
    # evaluate and plot images
    for x, y in tqdm(loader):
        x = x.to(config.DEVICE)
        # Run on experimental image
        nms_boxes = run_on_patches(
            x[0].permute(1, 2, 0),
            model,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            batch_size=batch_size,
            device=config.DEVICE,
            toggle_eval=False,
        )
        y[0,:,0:2], nms_boxes[:, 1:3] = config.side*y[0,:,0:2], config.side*nms_boxes[:,1:3]
        converted = convert_back(image = np.array(x[0].permute(1,2,0)), keypoints=y[0], keypoints1 = nms_boxes[:,1:])
        x, y, nms_boxes = converted['image'], np.array(converted['keypoints']), np.array(converted['keypoints1'])
        # Plot targets
        plot_particles(x[:, :, 2:3],y, scores=False, pixels=True)
        # PLot predictions
        plot_particles(
            x[:, :, 2:3], nms_boxes, scores=False, pixels=True
        )


if __name__ == "__main__":
    test_on_experimental_data()
