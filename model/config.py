"""
Configuration file to setup training of the model
"""
import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

DATASET = r"C:\Users\dimer\OneDrive\Skrivbord\kandidatarbete\Kandidatarbete\Edvin_local\data\training_2048x2048"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
BATCH_SIZE = 8
IMAGE_SIZE = 1952
NUM_CLASSES = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 2e-5
NUM_EPOCHS = 10
EVAL_INTERVAL = 5
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 2
PIXEL_THRESHOLD = 5
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
LOAD_CHECKPOINT_FILE = "weights.pth.tar"
SAVE_CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
PAD_SIDE = 1952
# EXP_SIZE = (1944, 1458)
ANCHORS = [
    [[2.91499325e-07], [2.74572732e-07], [2.57118761e-07]],
    [[2.39777548e-07], [2.22742292e-07], [2.06065526e-07]],
    [[1.89597521e-07], [1.73478734e-07], [1.57908925e-07]],
]


transform = A.Compose(
    [
        A.PadIfNeeded(min_width=PAD_SIDE, min_height=PAD_SIDE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=1.0,),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format="xy"),
)
# Transforms for experimental images used to test the model

exp_transforms = A.Compose(
    [
        A.PadIfNeeded(min_width=PAD_SIDE, min_height=PAD_SIDE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=1.0,),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format="xy"),
)

# Example of augmentations used during part of the training
augmentations = A.Compose(
    [
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=1.0,),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format="xy"),

)
