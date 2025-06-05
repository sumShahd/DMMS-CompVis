import subprocess
import sys

# Install pycocotools at runtime
subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocotools"])


import torch
import argparse
from torch.utils.data import DataLoader, Subset
from engine import train_one_epoch, evaluate
from dataset import GoBData, get_transform, collate_fn
from model import get_model
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.ops as ops
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os # Added for os.makedirs


def compute_real_precision_recall(model, data_loader, device, iou_threshold=0.5, conf_threshold=0.5):
       
    model.eval()
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].detach().cpu()
                scores = output['scores'].detach().cpu()
                gt_boxes = target['boxes'].detach().cpu()

                # Filter predictions by confidence threshold
                keep = scores > conf_threshold
                pred_boxes = pred_boxes[keep]

                if len(pred_boxes) == 0:
                    fn += len(gt_boxes)
                    continue
                if len(gt_boxes) == 0:
                    fp += len(pred_boxes)
                    continue

                ious = ops.box_iou(pred_boxes, gt_boxes)
                matched_gt = set()
                for i in range(ious.size(0)):
                    max_iou, idx = torch.max(ious[i], dim=0)
                    if max_iou >= iou_threshold and idx.item() not in matched_gt:
                        tp += 1
                        matched_gt.add(idx.item())
                    else:
                        fp += 1

                fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1

def compute_validation_loss(model, data_loader_test, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in t.items()} for t in targets]

            loss_out = model(images, targets)

            def reduce_loss_dict(loss_dict):
                return sum(
                    v.float().mean() if isinstance(v, torch.Tensor) and v.ndim > 0 else float(v)
                    for v in loss_dict.values()
                    if isinstance(v, (torch.Tensor, float, int))
                )

            # Support both dict and list[dict]
            if isinstance(loss_out, dict):
                loss = reduce_loss_dict(loss_out)
            elif isinstance(loss_out, list):
                batch_losses = [reduce_loss_dict(ld) for ld in loss_out]
                loss = sum(batch_losses) / len(batch_losses)
            else:
                raise ValueError(f"Unexpected loss format: {type(loss_out)}")

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)

def train(args):
    # Setting up logging
    mlflow.start_run() # start logging
    # print("logging started")

    # mlflow.pytorch.autolog() # enable autologging
    # print("auto log enabled")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("device to cuda")

    dataset = GoBData(
        root=args.image_root,
        json_file=args.train_annotations, # Note: args.train_annotations is not used
        transforms=get_transform(train=True)
    )
    # print("dataset (train) created")

    dataset_test = GoBData(
        root=args.image_root,
        json_file=args.test_annotations, # Note: args.test_annotations is not used
        transforms=get_transform(train=False)
    )
    # print("dataset_test created")

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # print("both dataloaders done")

    # !!class weighting!!
    if ((args.model == "retinanet") and (args.num_classes == 6)):
        class_weights = torch.tensor([
            1.0,   # Background (index 0)
            1.0,   # Grassy
            1.0,   # Broadleaf
            1.0,  # Woody
            1.0,   # Obstacle
            1.0    # Generalised Green
        ]) #grassy gets underweighted since so many, obstacle and woody get overweighted since rare background class neutral
    else:
        class_weights = None

    model = get_model(model_name=args.model, num_classes=args.num_classes, class_weights=class_weights)
    # print("model created")
    model.to(device)
    # print("model moved to device")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    ##LEARNING RATE SCHEDULER##
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.3, verbose=True)

    # print("optimiser set up")

    # For torchmetrics mAP (if needed for other logging, keep, otherwise it's not used for the custom plot)
    # metric_map = MeanAveragePrecision(max_detection_thresholds=[1, 10, 300]).to(device)

    # Lists to store metrics for each epoch for plotting
    history_precision = []
    history_recall = []
    history_f1 = []

    epochs_range = list(range(args.epochs)) # Define once

    for epoch in range(args.epochs):
        metric_logger, loss, lr = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # lr_scheduler.step()
        # val_loss = compute_validation_loss(model, data_loader_test, device)

        # Custom precision, recall, F1 calculation
        precision, recall, f1 = compute_real_precision_recall(model, data_loader_test, device, iou_threshold=0.5, conf_threshold=0.5)
        print(f"Custom Evaluation -> Epoch {epoch}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        lr_scheduler.step(f1)
        mlflow.log_metric("f1", f1, step=epoch)

        
        # Standard evaluation (e.g., COCO eval, prints to console)
        # evaluate(model, data_loader_test, device=device) # This is the coco_evaluator



        # Append current epoch's metrics to history lists
        history_precision.append(precision)
        history_recall.append(recall)
        history_f1.append(f1)

        # Log metrics to MLflow per epoch
        mlflow.log_metric("precision", precision, step=epoch)
        mlflow.log_metric("recall", recall, step=epoch)
        mlflow.log_metric("f1_score", f1, step=epoch)
        mlflow.log_metric("loss", loss, step=epoch) #from engine.py
        mlflow.log_metric("lr", lr, step=epoch) #from engine.py

        # Logging parameters
        mlflow.log_param('batch_size', args.batch_size)
        mlflow.log_param('learning_rate', args.lr)
        mlflow.log_param('step_size', args.step_size)
        mlflow.log_param('gamma', args.gamma)
        mlflow.log_param('weight_decay', args.weight_decay)
        mlflow.log_param('epochs', args.epochs)
        mlflow.log_param('class_weights', class_weights)
        
    
    # Log model to MLflow (MLflow will version it)
    mlflow.pytorch.log_model(model, "RetinaNet_model_Jacob_S_modded")

    # print("Registering the model in workspace via MLFLow")
    # mlflow.pytorch.log_model(
    #     pytorch_model = model,
    #     registered_model_name = model_name,
    #     artifact_path = model_name,
    # )

    # After all epochs, create and log the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_precision, label='Precision')
    plt.plot(epochs_range, history_recall, label='Recall')
    plt.plot(epochs_range, history_f1, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Detection Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = f"{args.output_dir}/metrics_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path, "plots") # Optionally, log into a "plots" directory in MLflow artifacts
    plt.close() # Close the plot to free up memory

    # End the MLflow run only once at the very end
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='retinaNet', help="Model to choose")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the JSON annotation file for the dataset.")
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help="Directory to save checkpoints and plots.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=4, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--weight-decay', type=float, default=0.0005, help="Weight decay.")
    parser.add_argument('--step-size', type=int, default=3, help="Step size for LR scheduler.")
    parser.add_argument('--gamma', type=float, default=0.1, help="Gamma for LR scheduler.")
    parser.add_argument('--num-classes', type=int, default=2, help="Number of classes (1 class + background).")
    # Note: The following arguments were defined but not used in the original data loading part.
    # If they are intended to specify separate train/test annotation files,
    # the dataset loading logic in `train()` would need to be updated.
    parser.add_argument('--train_annotations', type=str, help="Path to train annotations JSON ")
    parser.add_argument('--test_annotations', type=str, help="Path to test annotations JSON ")
    parser.add_argument('--image_root', type=str, required=True, help="Root directory of the images.")
    args = parser.parse_args()

    train(args)

