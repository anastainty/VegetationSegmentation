import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

RAW_IMAGE_PATH = "data/raw/DOM_zjsru_ms-5bands_8cm.tif"
RAW_DSM_PATH = "data/raw/DSM_zjsru_op_8cm.tif"
RAW_MASK_PATH = "data/raw/vegetation_classes.tif" 

PATCH_SIZE = 256
BATCH_SIZE = 16
NUM_CHANNELS = 5
NUM_CLASSES = 4