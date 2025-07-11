import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, resnet50
import cv2
import numpy as np
from typing import List, Tuple, Dict
import math

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomYOLOv8(nn.Module):
    def __init__(self, num_classes=80, input_size=640):
        super(CustomYOLOv8, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Backbone: EfficientNet-B0
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([112, 320, 1280], 256)
        
        # Detection heads
        self.detection_heads = nn.ModuleList([
            DetectionHead(256, num_classes) for _ in range(3)
        ])
        
        # Anchor generation
        self.anchor_generator = AnchorGenerator(
            sizes=[[32], [64], [128]], 
            aspect_ratios=[[0.5, 1.0, 2.0]] * 3
        )
        
    def forward(self, x):
        # Extract features
        features = self.extract_features(x)
        
        # FPN
        fpn_features = self.fpn(features)
        
        # Detection heads
        predictions = []
        for feat, head in zip(fpn_features, self.detection_heads):
            pred = head(feat)
            predictions.append(pred)
            
        return predictions
    
    def extract_features(self, x):
        features = []
        x = self.backbone.features[0](x)  # stem
        
        # Extract multi-scale features
        for i, layer in enumerate(self.backbone.features[1:]):
            x = layer(x)
            if i in [2, 5, 8]:  # Extract features at different scales
                features.append(x)
                
        return features

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
            
    def forward(self, inputs):
        # Build laterals
        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
        
        # Build top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )
            
        # Final conv
        outs = [conv(laterals[i]) for i, conv in enumerate(self.fpn_convs)]
        return outs

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, 1)
        )
        
        # Objectness head
        self.obj_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1)
        )
        
    def forward(self, x):
        cls_logits = self.cls_head(x)
        bbox_pred = self.reg_head(x)
        obj_logits = self.obj_head(x)
        
        return {
            'cls_logits': cls_logits,
            'bbox_pred': bbox_pred,
            'obj_logits': obj_logits
        }

class AnchorGenerator:
    def __init__(self, sizes, aspect_ratios):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        
    def generate_anchors(self, feature_map_size, device):
        anchors = []
        for size, aspect_ratio in zip(self.sizes, self.aspect_ratios):
            for s in size:
                for ar in aspect_ratio:
                    w = s * math.sqrt(ar)
                    h = s / math.sqrt(ar)
                    anchors.append([w, h])
        return torch.tensor(anchors, device=device)

class ObjectDetector:
    def __init__(self, device='cuda', model_path=None):
        self.device = device
        self.model = CustomYOLOv8(num_classes=80).to(device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if model_path:
            self.load_model(model_path)
            
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def detect(self, image: np.ndarray, conf_threshold=0.5, nms_threshold=0.5):
        """Detect objects in image"""
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
            
        # Post-process
        detections = self.post_process(predictions, conf_threshold, nms_threshold)
        
        return detections
    
    def post_process(self, predictions, conf_threshold, nms_threshold):
        """Post-process model predictions"""
        # Implementation of NMS and confidence filtering
        # This is a simplified version
        detections = []
        
        for pred in predictions:
            cls_logits = pred['cls_logits']
            bbox_pred = pred['bbox_pred']
            obj_logits = pred['obj_logits']
            
            # Apply sigmoid to get probabilities
            obj_probs = torch.sigmoid(obj_logits)
            cls_probs = torch.sigmoid(cls_logits)
            
            # Filter by confidence
            mask = obj_probs > conf_threshold
            
            # Extract valid detections
            valid_boxes = bbox_pred[mask]
            valid_scores = obj_probs[mask]
            valid_classes = cls_probs[mask]
            
            # Apply NMS (simplified)
            keep = self.nms(valid_boxes, valid_scores, nms_threshold)
            
            for i in keep:
                detections.append({
                    'bbox': valid_boxes[i].cpu().numpy(),
                    'confidence': valid_scores[i].item(),
                    'class_probs': valid_classes[i].cpu().numpy()
                })
                
        return detections
    
    def nms(self, boxes, scores, threshold):
        """Non-maximum suppression"""
        # Simple NMS implementation
        keep = []
        indices = torch.argsort(scores, descending=True)
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
                
            # Calculate IoU
            ious = self.calculate_iou(boxes[current], boxes[indices[1:]])
            
            # Remove boxes with high IoU
            indices = indices[1:][ious < threshold]
            
        return keep
    
    def calculate_iou(self, box1, boxes):
        """Calculate Intersection over Union"""
        # Simplified IoU calculation
        return torch.zeros(len(boxes))  # Placeholder

class ObjectDetectionTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.criterion = {
            'cls': FocalLoss(),
            'reg': nn.SmoothL1Loss(),
            'obj': nn.BCEWithLogitsLoss()
        }
        
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(images)
            
            # Calculate loss
            loss = self.calculate_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if scheduler:
                scheduler.step()
                
        return total_loss / len(dataloader)
    
    def calculate_loss(self, predictions, targets):
        """Calculate total loss"""
        # Simplified loss calculation
        return torch.tensor(0.0, requires_grad=True)  # Placeholder

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'weight_decay': 1e-4,
    'input_size': 640,
    'augmentation': {
        'horizontal_flip': 0.5,
        'vertical_flip': 0.1,
        'rotation': 15,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
}

def create_augmentation_pipeline():
    """Create data augmentation pipeline"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))