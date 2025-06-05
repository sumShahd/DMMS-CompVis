import os
import argparse
import json
import torch
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from io import BytesIO
import requests
from PIL import Image
import torchvision.transforms.v2.functional as F
from torchvision import transforms
import torch
import numpy as np
from pycocotools import mask as coco_mask

class GoBData(torch.utils.data.Dataset):
    def __init__(self, root, json_file, transforms=None):
        self.root = root
        self.transforms = transforms

        # Load annotation JSON
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            self.data = json.load(f)

        self.imgs = {img['id']: img['file_name'] for img in self.data['images']}
        self.annotations = {ann['image_id']: [] for ann in self.data['annotations']}
        for ann in self.data['annotations']:
            self.annotations[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        # print(f"get item idx is {idx}")
        img_url = img_info.get('coco_url')
        # print(f"image url is {img_url}")
        image_id = img_info['id']
        # print(f"image id is {image_id}")
        
        if "cloud" in img_url:
            # print(f"'coco_url' for image ID {image_id} is invalid - moving to next idx")
            new_idx = idx+1
            img_info = self.data['images'][new_idx]
            # print(f"new get item idx is {new_idx}")
            img_url = img_info.get('coco_url')
            # print(f"new image url is {img_url}")
            image_id = img_info['id']
            # print(f"new image id is {image_id}")
            

        if not img_url:
            raise ValueError(f"No 'coco_url' for image ID {image_id}")

        # Load image from URL
        img = self.download_image(img_url)

        # Get image dimensions (height, width)
        height, width = img.shape[1], img.shape[2]

        # Load annotations
        annotations = self.annotations.get(image_id, [])
        masks = []
        boxes = []
        obj_ids = []

        for ann in annotations:
            segmentation = ann["segmentation"]

            rle = coco_mask.frPyObjects(segmentation, height, width)
            mask = coco_mask.decode(rle)

            if len(mask.shape) == 3:
                mask = mask.any(axis=2)

            masks.append(torch.tensor(mask, dtype=torch.uint8))
            boxes.append(ann["bbox"])  # [xmin, ymin, width, height]
            obj_ids.append(ann["category_id"])

        # Convert to tensors after appending
        masks = torch.stack(masks) if len(masks) > 0 else torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        obj_ids = torch.tensor(obj_ids, dtype=torch.int64) if len(obj_ids) > 0 else torch.zeros((0,), dtype=torch.int64)


        x, y, w, h = boxes.unbind(1)
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # Check if masks tensor is empty using .numel()
        if masks.numel() == 0:
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            obj_ids = torch.zeros((0,), dtype=torch.int64)

        boxes, masks, obj_ids = self.remove_invalid_boxes(boxes, masks, obj_ids)
        
        ## Class modification
        # labels = torch.ones((len(obj_ids),), dtype=torch.int64) # - current
        # labels = torch.tensor(obj_ids, dtype=torch.int64) # - new

        # # Binary class mapping: weed (1,2,3,5) → 1, obstacle(4), background(0) → 0
        weed_classes = {1, 2, 3, 5}
        labels = torch.tensor([1 if int(cls) in weed_classes else 0 for cls in obj_ids], dtype=torch.int64)

        img = tv_tensors.Image(img)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": idx,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(obj_ids),), dtype=torch.int64)
        }

        if self.transforms:
            img, target = self.transforms(img, target)
            
        
        return img, target

    def download_image(self, url):
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        transform = transforms.ToTensor()  # Old-style, widely supported
        return transform(img)

    def remove_invalid_boxes(self, boxes, masks, obj_ids):
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid], masks[valid], obj_ids[valid]


# --------------------------
# Transform Helpers
# --------------------------
def get_transform(train=True):
    transforms = []
    if train:
        transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    transforms.extend([T.ToDtype(torch.float, scale=True), T.ToPureTensor()])
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

# def cutmix(self, img, target):
#         # Simplified CutMix for object detection
#         rand_index = random.randint(0, len(self.image_ids) - 1)
#         img2_id = self.image_ids[rand_index]
#         ann_ids2 = self.coco.getAnnIds(imgIds=img2_id)
#         anns2 = self.coco.loadAnns(ann_ids2)
#         image_info2 = self.coco.loadImgs(img2_id)[0]
#         img2_path = os.path.join(self.images_dir, image_info2['file_name'])
#         img2 = read_image(img2_path).float() / 255.0

#         lam = np.random.beta(1.0, 1.0)
#         img = lam * img + (1 - lam) * img2

#         # merge boxes and labels
#         boxes2 = []
#         labels2 = []
#         for ann in anns2:
#             bbox = ann['bbox']
#             boxes2.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
#             labels2.append(ann['category_id'])

#         boxes2 = torch.tensor(boxes2, dtype=torch.float32)
#         labels2 = torch.tensor(labels2, dtype=torch.int64)

#         target['boxes'] = torch.cat([target['boxes'], boxes2], dim=0)
#         target['labels'] = torch.cat([target['labels'], labels2], dim=0)

#         return img, target

#     def mixup(self, img, target):
#         # Similar to CutMix but with full-image blending
#         rand_index = random.randint(0, len(self.image_ids) - 1)
#         img2_id = self.image_ids[rand_index]
#         ann_ids2 = self.coco.getAnnIds(imgIds=img2_id)
#         anns2 = self.coco.loadAnns(ann_ids2)
#         image_info2 = self.coco.loadImgs(img2_id)[0]
#         img2_path = os.path.join(self.images_dir, image_info2['file_name'])
#         img2 = read_image(img2_path).float() / 255.0

#         lam = np.random.beta(1.0, 1.0)
#         img = lam * img + (1 - lam) * img2

#         boxes2 = []
#         labels2 = []
#         for ann in anns2:
#             bbox = ann['bbox']
#             boxes2.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
#             labels2.append(ann['category_id'])

#         boxes2 = torch.tensor(boxes2, dtype=torch.float32)
#         labels2 = torch.tensor(labels2, dtype=torch.int64)

#         target['boxes'] = torch.cat([target['boxes'], boxes2], dim=0)
#         target['labels'] = torch.cat([target['labels'], labels2], dim=0)

#         return img, target

# --------------------------
# Main Training Entry Point
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, required=True)
    parser.add_argument("--test_annotations", type=str, required=True)
    args = parser.parse_args()

    # Root URL where images are hosted
    image_root_url = "FULL_PATH_/generalized_GOB/"

    # Dataset setup
    train_dataset = GoBData(
        root=image_root_url,
        json_file=args.train_annotations,
        transforms=get_transform(train=True)
    )
    test_dataset = GoBData(
        root=image_root_url,
        json_file=args.test_annotations,
        transforms=get_transform(train=False)
    )

    # DataLoader example
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    for images, targets in train_loader:
        print(f"Loaded batch of {len(images)} images")
        break  # For debug only

if __name__ == "__main__":
    main()

