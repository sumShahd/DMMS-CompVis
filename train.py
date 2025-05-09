import torch
import argparse
from torch.utils.data import DataLoader, Subset
from engine import train_one_epoch, evaluate
from dataset import GoBData, get_transform
from model import get_model
from utils import collate_fn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GoBData(args.data_path, get_transform(train=True))
    dataset_test = GoBData(args.data_path, get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = Subset(dataset, indices[:-50])
    dataset_test = Subset(dataset_test, indices[-50:])

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes=args.num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    metric = MeanAveragePrecision().to(device)

    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

        model.eval()
        metric.reset()
        with torch.no_grad():
            for images, targets in data_loader_test:
                images = [img.to(device) for img in images]
                outputs = model(images)
                outputs = [{k: v.to(device) for k, v in o.items()} for o in outputs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                metric.update(outputs, targets)
        result = metric.compute()
        precision = result['precision'][0].item()
        recall = result['recall'][0].item()
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        print(f"Epoch {epoch} mAP: {result['map']:.4f}, mAP@50: {result['map_50']:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1_score:.4f}")

    torch.save(model.state_dict(), f"{args.output_dir}/retinanet_final.pt")
    print(f"Model saved to {args.output_dir}/retinanet_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--step-size', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num-classes', type=int, default=2)  # 1 class + background
    args = parser.parse_args()

    train(args)
