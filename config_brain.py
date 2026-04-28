import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ["resnet50", "efficientnet_b3"]
DATA_SPLITS = [0.25, 0.50, 1.0]
FREEZE_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]

NUM_EPOCHS = 15
BATCH_SIZE = 32  # smaller batch size for larger images
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224  # MRI images benefit from higher resolution than CIFAR
NUM_WORKERS = 2
RESULTS_DIR = "/content/drive/MyDrive/Project_Brain/results_brain"
NUM_CLASSES = 4  # glioma, meningioma, pituitary, no tumor