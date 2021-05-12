"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""
import argparse
import config
import torch
import torch.optim as optim
from metrics import evaluate_model
from model import YOLOv3
from tqdm import tqdm
from utils import (
    cells_to_bboxes,
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    plot_couple_examples,
)
from loss import YoloLoss

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE).squeeze(0)
        y0, y1, y2 = (
            y[0].to(config.DEVICE).squeeze(0),
            y[1].to(config.DEVICE).squeeze(0),
            y[2].to(config.DEVICE).squeeze(0),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0)
                + loss_fn(out[1], y1)
                + loss_fn(out[2], y2)
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def parse_arguments():
    """
    Parses the arguments for training the model
    """
    parser = argparse.ArgumentParser(description="""
            Training configuration for the YOLOv3 model on particle tracking""")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default=config.LOAD_CHECKPOINT_FILE,
        help="Path to weights or checkpoint file .pth.tar",
    )
    parser.add_argument(
        "-i",
        "--train_dir",
        type=str,
        default=config.DATASET,
        help="Path to directory with images and labels for training",
    )
    parser.add_argument(
        "-d",
        "--test_dir",
        type=str,
        default=config.DATASET,
        help="Path to directory with images and labels for testing",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=config.BATCH_SIZE, help="Size of each image batch"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=config.CONF_THRESHOLD, help="Object confidence threshold"
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=config.NMS_THRESHOLD,
        help="Pixel threshold for non-maximum suppression",
    )
    parser.add_argument(
        "--pixel_threshold",
        type=float,
        default=config.PIXEL_THRESHOLD,
        help="Pixel threshold for when to consider prediction true positive",
    )
    parser.add_argument(
        "--device", type=str, default=config.DEVICE, help="device to run model on"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate of the model",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config.WEIGHT_DECAY,
        help="Weight decay parameter for the optimizer",
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--load_model", type=bool, default=config.LOAD_MODEL, help="Whether to load model weights")
    parser.add_argument("--save_model", type=bool, default=config.SAVE_MODEL, help="Whether to save the model every epoch")
    parser.add_argument("--checkpoint_file", type=str, default=config.SAVE_CHECKPOINT_FILE, help="Path to where to save model checkpoint")
    parser.add_argument("--eval_interval", type=int, default=config.EVAL_INTERVAL, help="Number of epochs between model evaluations")
    parser.add_argument("--pin_memory", type=bool, default=config.PIN_MEMORY, help="Whether to pin memory in dataloader")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS, help="Number of workers in dataloaders")
    args = parser.parse_args()
    return args


def main():
    """
    Training loop for YOLOv3 model
    """
    args = parse_arguments()
    model = YOLOv3(in_channels=3, num_classes=config.NUM_CLASSES).to(args.device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader = get_loaders(args.train_dir, args.test_dir, args.batch_size, args.num_workers, args.pin_memory
    )
    if args.load_model:
        load_checkpoint(
            args.weights, model, optimizer, args.learning_rate
        )

    for epoch in range(args.epochs):
        # If evaluate model
        if epoch % args.eval_interval == 0 and epoch > 0:
            plot_couple_examples(model, test_loader, 0.5, args.nms_threshold)

            precision, recall, F1, rmse_errors, rel_errors = evaluate_model(
                test_loader, model, conf_threshold=args.conf_threshold, nms_threshold=args.nms_threshold,
                pixel_threshold=args.pixel_threshold
            )
            print(
                f"Precision is: {precision}, \n Recall is {recall}, \n F1: {F1}, \n"
                f"X rmse error: {rmse_errors[0]}, y rmse error: {rmse_errors[1]}, z rmse error: {rmse_errors[2]}, r rmse error: {rmse_errors[3]}, n rmse error: {rmse_errors[4]} \n"
                f"X relative error: {rel_errors[0]}, y relative error: {rel_errors[1]}, z relative error: {rel_errors[2]}, r relative error: {rel_errors[3]}, n relative error: {rel_errors[4]}"
            )

        train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler,
        )

        if args.save_model:
            save_checkpoint(model, optimizer, filename=args.checkpoint_file)


if __name__ == "__main__":
    main()
