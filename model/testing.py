import albumentations as A
from albumentations.pytorch import ToTensorV2
t = A.Compose([A.CenterCrop(height=1944, width=1458)], additional_targets={'keypoints1':'keypoints'},
              keypoint_params=A.KeypointParams(format="xy"))