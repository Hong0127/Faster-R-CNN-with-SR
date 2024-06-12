import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.sod4sb_dataset import SOD4SBDataset
from models.faster_rcnn import FasterRCNN
from config import Config

def test(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = [image.to(device) for image in images]
            cls_score, bbox_pred = model(images, proposals)  # Ensure `proposals` is available
            results.append((cls_score, bbox_pred))
    return results

def test():
    device = Config.DEVICE
    model = FasterRCNN(num_classes=Config.NUM_CLASSES).to(device)

    if Config.MODEL_WEIGHTS:
        model.load_state_dict(torch.load(Config.MODEL_WEIGHTS))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = SOD4SBDataset(root=Config.VAL_DATA_PATH, annFile=Config.VAL_ANN_FILE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    results = test(model, dataloader, device)
    print(results)

if __name__ == '__main__':
    test()