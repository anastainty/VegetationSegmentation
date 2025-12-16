import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import random
import config


class VegetationDataset(Dataset):
    def __init__(self, image_path, dsm_path, mask_path, patch_size=256, transform=None, augment=False):
        self.patch_size = patch_size
        self.transform = transform
        self.augment = augment
        self.patches = []

        print(f"Loading data from:\n {image_path}\n {dsm_path}\n {mask_path}")
        print(f"Augmentation enabled: {self.augment}")

        # 1. Читаем изображение (5 каналов)
        with rasterio.open(image_path) as src_img:
            self.image = src_img.read().transpose(1, 2, 0).astype(np.float32)
            # Нормализация
            max_val = np.max(self.image)
            if max_val > 255:
                self.image /= 65535.0
            else:
                self.image /= 255.0

        # 2. Читаем высоту (1 канал) с ОЧИСТКОЙ
        with rasterio.open(dsm_path) as src_dsm:
            raw_dsm = src_dsm.read(1).astype(np.float32)
            valid_mask = raw_dsm > -100

            if valid_mask.any():
                dsm_min = np.min(raw_dsm[valid_mask])
                dsm_max = np.max(raw_dsm[valid_mask])
                dsm_clean = np.where(valid_mask, raw_dsm, dsm_min)
                dsm_clean = np.clip(dsm_clean, dsm_min, dsm_max)

                if dsm_max - dsm_min > 0:
                    self.dsm = (dsm_clean - dsm_min) / (dsm_max - dsm_min)
                else:
                    self.dsm = np.zeros_like(dsm_clean)
            else:
                self.dsm = np.zeros_like(raw_dsm)

            self.dsm = np.expand_dims(self.dsm, axis=2)

        # 3. Читаем маску
        with rasterio.open(mask_path) as src_mask:
            self.mask = src_mask.read(1).astype(np.int64)

        h, w, _ = self.image.shape

        # --- STRIDE STRATEGY ---
        # Если это обучение (augment=True), делаем шаг в половину патча (нахлест 50%)
        # Это дает в 4 раза больше патчей и учит сеть краям.
        stride = patch_size // 2 if augment else patch_size

        print("Generating patches...")
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                self.patches.append((y, x))

        print(f"Total patches created: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        y, x = self.patches[idx]

        # Вырезаем куски
        img_patch = self.image[y:y + self.patch_size, x:x + self.patch_size, :]
        dsm_patch = self.dsm[y:y + self.patch_size, x:x + self.patch_size, :]
        mask_patch = self.mask[y:y + self.patch_size, x:x + self.patch_size]

        # --- АУГМЕНТАЦИЯ ---
        if self.augment:
            # 1. Flip Left-Right
            if random.random() > 0.5:
                img_patch = np.fliplr(img_patch)
                dsm_patch = np.fliplr(dsm_patch)
                mask_patch = np.fliplr(mask_patch)

            # 2. Flip Up-Down
            if random.random() > 0.5:
                img_patch = np.flipud(img_patch)
                dsm_patch = np.flipud(dsm_patch)
                mask_patch = np.flipud(mask_patch)

            # 3. Rotate 90/180/270
            k = random.randint(0, 3)
            if k > 0:
                img_patch = np.rot90(img_patch, k)
                dsm_patch = np.rot90(dsm_patch, k)
                mask_patch = np.rot90(mask_patch, k)

        # Склеиваем: 5 цветов + 1 высота = 6 каналов
        combined_input = np.concatenate((img_patch, dsm_patch), axis=2)

        # .copy() обязателен после np.flip, иначе PyTorch ругается на negative strides
        combined_tensor = torch.from_numpy(combined_input.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy(mask_patch.copy()).long()

        return combined_tensor, mask_tensor