import os
import torch

# File Paths
BASE_PATH = r"D:\Telegram Downloads\Graduation Project\Latest from ALL\Hand_Gesture"
INPUT_FOLDER = os.path.join(BASE_PATH, "Hand images")
OUTPUT_CROPPED_FOLDER = os.path.join(BASE_PATH, "Hand_images_cropped")
OUTPUT_SPLIT_FOLDER = os.path.join(BASE_PATH, "Hand_images_cropped_split")
TRAIN_PATH = os.path.join(OUTPUT_SPLIT_FOLDER, "train")
VAL_PATH = os.path.join(OUTPUT_SPLIT_FOLDER, "val")
TEST_PATH = os.path.join(OUTPUT_SPLIT_FOLDER, "test")
MODEL_PATH = os.path.join(BASE_PATH, "models", "best_model.pth")

# Image Processing Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
DISPLAY_WIDTH = 700
DISPLAY_HEIGHT = 500

# MediaPipe Hands Parameters
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5

# Model Parameters
NUM_CLASSES = 4
CLASS_NAMES = {0: 'dislike', 1: 'like', 2: 'no_raised_hand', 3: 'raised_hand'}
CLASS_COUNTS = [89, 89, 470, 188]  # [dislike, like, no_raised_hand, raised_hand]

# Training Parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 40
PATIENCE_EARLY_STOPPING = 7
DROPOUT_RATE = 0.5

# Data Augmentation and Normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Dataset Splitting Parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")