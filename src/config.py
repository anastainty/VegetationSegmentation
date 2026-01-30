import torch
import os

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MASKS_DIR = os.path.join(DATA_PROCESSED_DIR, "masks")
IMAGES_OUTPUT_DIR = os.path.join(DATA_PROCESSED_DIR, "images")
PLOTS_DIR = os.path.join(DATA_PROCESSED_DIR, "plots")

# --- ФАЙЛЫ ---
IMAGE_PATH = os.path.join(DATA_RAW_DIR, "DOM_zjsru_ms-5bands_8cm.tif")
DSM_PATH = os.path.join(DATA_RAW_DIR, "DSM_zjsru_op_8cm.tif")
# ВАЖНО: Добавили путь к NDVI файлу
NDVI_PATH = os.path.join(DATA_RAW_DIR, "DOM_NDVI_zjsru_ms-5bands_8cm.tif")
MASK_PATH = os.path.join(DATA_RAW_DIR, "vegetation_classes.tif")

# --- ГИПЕРПАРАМЕТРЫ ---
PATCH_SIZE = 256
BATCH_SIZE = 24  # Оставляем 24, раз он стабилен
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

# 7 каналов: 5 (Image) + 1 (DSM) + 1 (NDVI)
NUM_CHANNELS = 7
NUM_CLASSES = 5

DSM_MIN = -10.0
DSM_MAX = 50.0