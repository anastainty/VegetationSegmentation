import torch
import os

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

IMAGES_OUTPUT_DIR = os.path.join(DATA_PROCESSED_DIR, "images")
PLOTS_DIR = os.path.join(DATA_PROCESSED_DIR, "plots")

IMAGE_PATH = os.path.join(DATA_RAW_DIR, "DOM_zjsru_ms-5bands_8cm.tif")
DSM_PATH = os.path.join(DATA_RAW_DIR, "DSM_zjsru_op_8cm.tif")
MASK_PATH = os.path.join(DATA_RAW_DIR, "vegetation_classes.tif")

PATCH_SIZE = 256
BATCH_SIZE = 16
NUM_CHANNELS = 6
NUM_CLASSES = 5
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100