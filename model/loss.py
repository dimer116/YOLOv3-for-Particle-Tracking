"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
adapted to the task of particle tracking
"""
import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        # adapted during train so the loss terms have the same magnitude (approx)
        self.lambda_class = 1
        self.lambda_noobj = 100
        self.lambda_obj = 1
        self.lambda_xy = 50
        self.lambda_z = 0.005
        self.lambda_r = 10
        self.lambda_n = 50

    def forward(self, predictions, target):
        """
        :param predictions: output from model of shape: (batch size, number of anchors on scale (3), grid size, grid size, 6)
        :param target: targets on particular scale of shape: (batch size, number of anchors on scale (3), grid size, grid size, 6)
        the 6 values signify (object score, x, y, z, radius, refractive index)
        :return: returns the loss on the particular scale
        """

        # Check where obj and noobj (we ignore if target == -1)
        # Here we check where in the label matrix there is an object or not
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        # Compute the loss for the positions in the label matrix and the anchor boxes where there's no object.
        # The indexing noobj refers to the fact that we only apply the loss where there is no object
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # Here we compute the loss for the cells and anchor boxes that contain an object
        # Convert outputs from model to bounding boxes according to formulas in paper
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        err = torch.maximum(
            (
                1
                - torch.mean(
                    torch.abs(predictions[..., 1:3] - target[..., 1:3]),
                    dim=-1,
                    keepdim=True,
                )
            ),
            torch.tensor([0]).to(predictions.device),
        ).detach()
        # Only incur loss for the cells where there is an objects signified by indexing with obj
        object_loss = self.bce(
            predictions[..., 0:1][obj], (err * target[..., 0:1])[obj]
        )

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # compute mse loss for boxes
        xy_loss = self.mse(predictions[..., 1:3][obj], target[..., 1:3][obj])
        z_loss = self.mse((predictions[..., 3:4])[obj], target[..., 3:4][obj])
        # apply relu to predictions to avoid negative radii
        r_loss = self.mse(predictions[..., 4:5][obj], target[..., 4:5][obj] / (1e-7))
        n_loss = self.mse(predictions[..., 5:6][obj], target[..., 5:6][obj])

        return (
            self.lambda_xy * xy_loss
            + self.lambda_z * z_loss
            + self.lambda_r * r_loss
            + self.lambda_n * n_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
        )
