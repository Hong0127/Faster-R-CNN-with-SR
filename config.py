import torch

class Config:
    # 데이터 경로 설정
    TRAIN_DATA_PATH = '/Users/hongseongmin/source/faster_rcnn/datasets/bird-detection-1/train'
    VAL_DATA_PATH = '/Users/hongseongmin/source/faster_rcnn/datasets/bird-detection-1/val'
    TRAIN_ANN_FILE = '/Users/hongseongmin/source/faster_rcnn/datasets/bird-detection-1/train/_annotations.coco.json'
    VAL_ANN_FILE = '/Users/hongseongmin/source/faster_rcnn/datasets/bird-detection-1/val/_annotations.coco.json'
    
    # 학습 설정
    NUM_CLASSES = 2  # COCO has 80 classes + 1 background class
    BATCH_SIZE = 2
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    NUM_EPOCHS = 10
    
    # 기타 설정
    DEVICE = 'mps'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_WEIGHTS = 'path/to/trained_model_weights.pth'