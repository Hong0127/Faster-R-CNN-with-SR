import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.sod4sb_dataset import SOD4SBDataset
from models.faster_rcnn import FasterRCNN
from config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, targets in progress_bar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        images = torch.stack(images)  # List of Tensors to a single Tensor

        proposals = [t["boxes"] for t in targets]  # Extract proposals (boxes) from targets
        proposals = [p.to(device) for p in proposals]  # Move proposals to device

        cls_score, bbox_pred, rpn_bbox_pred = model(images, proposals)  # 세 개의 값 반환

        targets_tensor = torch.tensor([t["labels"] for t in targets]).to(device)  # Convert targets to Tensor
        bbox_targets = torch.cat([t["boxes"] for t in targets]).view(-1, 4).to(device)  # Combine all bounding boxes into a single tensor with correct shape
        
        # Convert bbox_targets to the same size as bbox_pred
        bbox_targets = bbox_targets.repeat(1, cls_score.size(1)).view_as(bbox_pred)
        
        cls_loss, bbox_loss = criterion(cls_score, bbox_pred, targets_tensor, bbox_targets)
        loss = cls_loss + bbox_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_cls_loss += cls_loss.item()
        running_bbox_loss += bbox_loss.item()

        progress_bar.set_postfix(loss=loss.item(), cls_loss=cls_loss.item(), bbox_loss=bbox_loss.item())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_cls_loss = running_cls_loss / len(dataloader)
    epoch_bbox_loss = running_bbox_loss / len(dataloader)
    return epoch_loss, epoch_cls_loss, epoch_bbox_loss

def criterion(cls_score, bbox_pred, cls_targets, bbox_targets):
    cls_loss = nn.CrossEntropyLoss()(cls_score, cls_targets)
    bbox_loss = nn.SmoothL1Loss()(bbox_pred, bbox_targets)
    return cls_loss, bbox_loss

def evaluate(model, dataloader, device):
    model.eval()
    coco_gt = COCO(Config.VAL_ANN_FILE)
    coco_results = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            images = torch.stack(images)  # List of Tensors to a single Tensor

            proposals = [t["boxes"] for t in targets]  # Extract proposals (boxes) from targets
            proposals = [p.to(device) for p in proposals]  # Move proposals to device

            outputs = model(images, proposals)

            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": score
                    })

    # Save the results in a temporary json file
    result_file = "temp_coco_results.json"
    with open(result_file, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Clean up
    if os.path.exists(result_file):
        os.remove(result_file)

    return coco_eval.stats[0], coco_eval.stats[1]  # mAP 50, mAP 50-95

def train():
    device = Config.DEVICE
    model = FasterRCNN(num_classes=Config.NUM_CLASSES).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = SOD4SBDataset(root=Config.TRAIN_DATA_PATH, annFile=Config.TRAIN_ANN_FILE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    val_dataset = SOD4SBDataset(root=Config.VAL_DATA_PATH, annFile=Config.VAL_ANN_FILE, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)

    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        epoch_loss, epoch_cls_loss, epoch_bbox_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Cls Loss: {epoch_cls_loss:.4f}, BBox Loss: {epoch_bbox_loss:.4f}")

        mAP_50, mAP_50_95 = evaluate(model, val_dataloader, device)
        print(f"Epoch {epoch+1}, mAP@50: {mAP_50:.4f}, mAP@50-95: {mAP_50_95:.4f}")

if __name__ == '__main__':
    train()