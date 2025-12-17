import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import os

import config
from model import UNet


def predict_full_map():
    # 1. Настройки путей
    # Создаем папку для результата, если нет
    os.makedirs(config.IMAGES_OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(config.IMAGES_OUTPUT_DIR, "predicted_vegetation_map.tif")

    print(f"--- Старт предсказания ---")
    print(f"Устройство: {config.DEVICE}")
    print(f"Сохранение в: {OUTPUT_PATH}")

    # 2. Загружаем модель
    print("Загрузка модели...")
    model = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES)

    # Загружаем веса на CPU для надежности, потом отправляем на устройство
    try:
        model.load_state_dict(torch.load("unet_vegetation_model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("❌ Ошибка: Файл 'unet_vegetation_model.pth' не найден!")
        print("Сначала запусти train.py, чтобы обучить модель.")
        return

    model.to(config.DEVICE)
    model.eval()

    # 3. Открываем исходные файлы
    with rasterio.open(config.IMAGE_PATH) as src_img, \
            rasterio.open(config.DSM_PATH) as src_dsm:

        # Копируем метаданные исходного снимка
        meta = src_img.meta.copy()
        height, width = src_img.height, src_img.width

        # ОБНОВЛЕНИЕ МЕТАДАННЫХ:
        # count=1 (один канал - карта классов)
        # dtype=uint8 (целые числа 0-255)
        # nodata=255 (значение для пустоты/прозрачности)
        meta.update(count=1, dtype=rasterio.uint8, nodata=255)

        print(f"Размер изображения: {width}x{height}")

        # Создаем выходной файл
        with rasterio.open(OUTPUT_PATH, 'w', **meta) as dst:
            patch_size = config.PATCH_SIZE
            # Считаем общее количество патчей для прогресс-бара
            total_patches = (height // patch_size + 1) * (width // patch_size + 1)
            pbar = tqdm(total=total_patches, desc="Генерация карты")

            # --- Глобальная статистика высот ---
            # Чтобы нормализовать высоту (DSM) корректно, нам нужны мин/макс значения всего файла.
            # Читаем уменьшенную копию (1/10 размера) для скорости.
            print("Вычисляем статистику высот...")
            dsm_overview = src_dsm.read(1, out_shape=(1, int(src_dsm.height // 10), int(src_dsm.width // 10)))

            # Фильтруем мусор (значения меньше -100)
            valid_mask = dsm_overview > -100
            if valid_mask.any():
                dsm_min_global = np.min(dsm_overview[valid_mask])
                dsm_max_global = np.max(dsm_overview[valid_mask])
            else:
                dsm_min_global, dsm_max_global = 0, 1

            print(f"DSM Min: {dsm_min_global:.2f}, DSM Max: {dsm_max_global:.2f}")

            # 4. Проход скользящим окном
            for y in range(0, height, patch_size):
                for x in range(0, width, patch_size):
                    # Определяем реальный размер окна (на краях может быть меньше 256)
                    window_h = min(patch_size, height - y)
                    window_w = min(patch_size, width - x)
                    window = Window(x, y, window_w, window_h)

                    # Читаем куски данных
                    img_chunk = src_img.read(window=window).transpose(1, 2, 0).astype(np.float32)
                    dsm_chunk = src_dsm.read(1, window=window).astype(np.float32)

                    # --- Нормализация Картинки ---
                    # Если данные 16-битные, делим на 65535, если 8-битные — на 255
                    if np.max(img_chunk) > 255:
                        img_chunk /= 65535.0
                    else:
                        img_chunk /= 255.0

                    # --- Нормализация Высоты ---
                    # Используем глобальные min/max, чтобы перепады высот сохранялись между патчами
                    dsm_clean = np.clip(dsm_chunk, dsm_min_global, dsm_max_global)
                    if dsm_max_global - dsm_min_global > 0:
                        dsm_chunk = (dsm_clean - dsm_min_global) / (dsm_max_global - dsm_min_global)
                    else:
                        dsm_chunk = np.zeros_like(dsm_clean)

                    # Добавляем измерение канала для высоты
                    dsm_chunk = np.expand_dims(dsm_chunk, axis=2)

                    # Склейка (Concat)
                    input_chunk = np.concatenate((img_chunk, dsm_chunk), axis=2)

                    # Паддинг (Padding)
                    # Если кусок меньше 256x256 (на краях карты), дополняем нулями
                    if window_h < patch_size or window_w < patch_size:
                        pad_h = patch_size - window_h
                        pad_w = patch_size - window_w
                        # pad только по высоте и ширине, каналы не трогаем
                        input_chunk = np.pad(input_chunk, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

                    # Перевод в тензор PyTorch
                    tensor = torch.from_numpy(input_chunk.transpose(2, 0, 1)).float()
                    tensor = tensor.unsqueeze(0).to(config.DEVICE)

                    # --- ПРЕДСКАЗАНИЕ ---
                    with torch.no_grad():
                        output = model(tensor)
                        # Берем индекс класса с максимальной вероятностью (Argmax)
                        pred = torch.argmax(output, dim=1).cpu().numpy()[0]

                    pred_cut = pred[:window_h, :window_w]

                    # Записываем результат в файл
                    dst.write(pred_cut.astype(rasterio.uint8), 1, window=window)

                    pbar.update(1)

            pbar.close()

    print(f"\n✅ Готово! Карта сохранена в: {OUTPUT_PATH}")


if __name__ == "__main__":
    predict_full_map()