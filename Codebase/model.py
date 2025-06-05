import torch 
import torchvision 
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead 
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from functools import partial
from utils import custom_sigmoid_focal_loss

def get_model(model_name, num_classes, class_weights=None):
    if model_name.lower() == "retinanet":
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        if class_weights is not None:
            model.head.classification_head.loss_func = lambda inputs, targets: custom_sigmoid_focal_loss(
                inputs, targets, class_weights=class_weights
            )
        return model

    elif model_name.lower() == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    elif model_name.lower() == "maskrcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")

