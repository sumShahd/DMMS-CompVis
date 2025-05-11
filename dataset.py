import os
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

class GoBData(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        

    def remove_invalid_boxes(self, boxes, masks, obj_ids):
        valid = [(box[2] - box[0] > 0) and (box[3] - box[1] > 0) for box in boxes]
        valid_indices = torch.nonzero(torch.tensor(valid)).squeeze()
        return boxes[valid_indices], masks[valid_indices], obj_ids[valid_indices]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path).squeeze(0)

        obj_ids = torch.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        boxes, masks, obj_ids = self.remove_invalid_boxes(boxes, masks, obj_ids)
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)

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

    def __len__(self):
        return len(self.imgs)

def get_transform(train=True):
    transforms = []
    if train:
        transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    transforms.extend([T.ToDtype(torch.float, scale=True), T.ToPureTensor()])
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))
