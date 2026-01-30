import rasterio
import torch
from torch.utils.data import Dataset
import numpy as np
import config
import os


class VegetationDataset(Dataset):
    # Добавили аргумент ndvi_path в инициализацию
    def __init__(self, image_path, dsm_path, ndvi_path, mask_path, roi_path=None, patch_size=256, augment=False):
        self.image_path = image_path
        self.dsm_path = dsm_path
        self.ndvi_path = ndvi_path  # Запоминаем путь
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.augment = augment

        # Загрузка ROI (оставляем как было)
        if roi_path and os.path.exists(roi_path):
            with rasterio.open(roi_path) as src:
                self.roi_mask = src.read(1)
                h, w = self.roi_mask.shape
                self.patches = []
                # Шаг 256 (без перекрытия для трейна)
                for y in range(0, h - patch_size + 1, patch_size):
                    for x in range(0, w - patch_size + 1, patch_size):
                        window_roi = self.roi_mask[y:y + patch_size, x:x + patch_size]
                        if np.any(window_roi):
                            self.patches.append((y, x))
        else:
            with rasterio.open(image_path) as src:
                h, w = src.height, src.width
            self.patches = []
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    self.patches.append((y, x))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        y, x = self.patches[idx]
        window = rasterio.windows.Window(x, y, self.patch_size, self.patch_size)

        # 1. Читаем Картинку (5 каналов)
        with rasterio.open(self.image_path) as src:
            image = src.read(window=window, boundless=True, fill_value=0)
            image = image.astype(np.float32)
            if np.max(image) > 255:
                image /= 65535.0
            else:
                image /= 255.0

        # 2. Читаем DSM
        with rasterio.open(self.dsm_path) as src:
            dsm = src.read(1, window=window, boundless=True, fill_value=config.DSM_MIN)
            dsm = dsm.astype(np.float32)
            dsm = np.clip(dsm, config.DSM_MIN, config.DSM_MAX)
            dsm = (dsm - config.DSM_MIN) / (config.DSM_MAX - config.DSM_MIN)
            dsm = np.expand_dims(dsm, axis=0)

        # 3. Читаем NDVI из ФАЙЛА (Вместо формулы)
        with rasterio.open(self.ndvi_path) as src:
            # Читаем 1-й канал (обычно NDVI там один)
            # fill_value=-1, так как пустота - это "не живое"
            ndvi = src.read(1, window=window, boundless=True, fill_value=-1.0)
            ndvi = ndvi.astype(np.float32)

            # NDVI обычно от -1 до 1.
            # Нейросети любят 0..1. Давайте сдвинем диапазон.
            # Если NDVI уже нормальный, это не повредит.
            ndvi = np.clip(ndvi, -1.0, 1.0)
            ndvi = (ndvi + 1.0) / 2.0  # Теперь диапазон 0..1
            ndvi = np.expand_dims(ndvi, axis=0)

        # 4. Читаем Маску
        with rasterio.open(self.mask_path) as src:
            mask = src.read(1, window=window, boundless=True, fill_value=0)
            mask = mask.astype(np.int64)

        # 5. Сборка (5 + 1 + 1 = 7 каналов)
        input_tensor = np.concatenate((image, dsm, ndvi), axis=0)

        # 6. Аугментация
        if self.augment:
            if np.random.random() > 0.5:
                input_tensor = np.flip(input_tensor, axis=2)
                mask = np.flip(mask, axis=1)
            if np.random.random() > 0.5:
                input_tensor = np.flip(input_tensor, axis=1)
                mask = np.flip(mask, axis=0)
            k = np.random.randint(0, 4)
            if k > 0:
                input_tensor = np.rot90(input_tensor, k, axes=(1, 2))
                mask = np.rot90(mask, k, axes=(0, 1))

        return torch.from_numpy(input_tensor.copy()).float(), torch.from_numpy(mask.copy()).long()