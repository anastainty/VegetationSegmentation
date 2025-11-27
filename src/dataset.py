import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import config


class VegetationDataset(Dataset):
    def __init__(self, image_path, dsm_path, mask_path, patch_size=256, transform=None):
        self.patch_size = patch_size
        self.transform = transform
        self.patches = []

        print(f"Loading data from:\n {image_path}\n {dsm_path}\n {mask_path}")

        # 1. Читаем изображение (5 каналов)
        with rasterio.open(image_path) as src_img:
            self.image = src_img.read().transpose(1, 2, 0).astype(np.float32)
            # Нормализация (16 бит -> 0..1 или 8 бит -> 0..1)
            max_val = np.max(self.image)
            if max_val > 255:
                self.image /= 65535.0
            else:
                self.image /= 255.0

        # 2. Читаем высоту (1 канал) с ОЧИСТКОЙ
        with rasterio.open(dsm_path) as src_dsm:
            raw_dsm = src_dsm.read(1).astype(np.float32)

            # Фильтруем nodata (все что меньше -100 считаем ошибкой сенсора)
            valid_mask = raw_dsm > -100

            if valid_mask.any():
                # Считаем мин/макс только по валидным данным
                dsm_min = np.min(raw_dsm[valid_mask])
                dsm_max = np.max(raw_dsm[valid_mask])

                # Заменяем мусор на минимум, чтобы не ломать график
                dsm_clean = np.where(valid_mask, raw_dsm, dsm_min)
                dsm_clean = np.clip(dsm_clean, dsm_min, dsm_max)

                # Нормализация 0..1
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
        print(f"Dataset shape: {h}x{w}. Channels: {config.NUM_CHANNELS}")

        # Генерация списка координат (Tiling)
        print("Generating patches...")
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                # Добавляем в список только если патч влезает целиком
                self.patches.append((y, x))

        print(f"Total patches created: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        y, x = self.patches[idx]

        # Вырезаем куски
        img_patch = self.image[y:y + self.patch_size, x:x + self.patch_size, :]
        dsm_patch = self.dsm[y:y + self.patch_size, x:x + self.patch_size, :]

        # Склеиваем: 5 цветов + 1 высота = 6 каналов
        combined_input = np.concatenate((img_patch, dsm_patch), axis=2)

        mask_patch = self.mask[y:y + self.patch_size, x:x + self.patch_size]

        # Конвертируем в тензоры PyTorch (Channel, H, W)
        combined_tensor = torch.from_numpy(combined_input.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_patch).long()

        return combined_tensor, mask_tensor