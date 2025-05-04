import os
import torch

class Config:
    # 数据集配置
    # DATASET_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
    DATASET_PATH = r"C:\Users\CXY\Desktop\graduationDesign\dataset\AllVeinDataset\HFUT_split"

    # 预训练权重配置
    PRETRAINED_WEIGHTS = r"C:\Users\CXY\Desktop\graduationDesign\src\palmVeinRecognition\swin_tiny_patch4_window7_224.pth"  # 本地权重文件路径
    USE_PRE_WEIGHTS = True  # 是否使用本地权重

    # 模型配置
    MODEL_NAME = "swin_tiny"
    PRETRAINED = True
    EMBEDDING_SIZE = 512
    
    # 训练配置
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-2
    MARGIN = 0.3
    DISTANCE = "euclidean"  # "euclidean" or "cosine"
    
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 日志和保存
    LOG_INTERVAL = 1
    SAVE_DIR = "../checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

config = Config()