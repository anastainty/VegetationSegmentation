import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import os

import config
from model import UNet


def predict_full_map():
    # 1. Настройки
    os.makedirs(config.IMAGES_OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(config.IMAGES_OUTPUT_DIR, "predicted_vegetation_map.tif")
    ROI_PATH = os.path.join("data", "processed", "masks", "roi_mask.tif")

    print(f"--- Старт предсказания ---")
    print(f"Устройство: {config.DEVICE}")

    # 2. Загрузка модели
    print("Загрузка модели...")
    model = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES)

    model_path = "unet_best_f1_model.pth"  # Берем лучшую модель по F1
    if not os.path.exists(model_path):
        print(f"⚠️ Файл {model_path} не найден, ищем старый...")
        model_path = "unet_vegetation_model.pth"

    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print(f"Загружены веса из: {model_path}")
    except FileNotFoundError:
        print("❌ Ошибка: Нет обученной модели.")
        return

    model.to(config.DEVICE)
    model.eval()

    # 3. Открываем файлы
    # Открываем также ROI маску, чтобы пропускать пустоту
    roi_exists = os.path.exists(ROI_PATH)
    if not roi_exists:
        print("⚠️ Маска ROI не найдена. Будет обработано все изображение (долго).")

    with rasterio.open(config.IMAGE_PATH) as src_img, \
            rasterio.open(config.DSM_PATH) as src_dsm:

        # Если есть ROI, открываем его
        src_roi = rasterio.open(ROI_PATH) if roi_exists else None

        meta = src_img.meta.copy()
        # Выходной файл: 1 канал, 255 = прозрачность/фон
        meta.update(count=1, dtype=rasterio.uint8, nodata=255)

        print(f"Размер изображения: {src_img.width}x{src_img.height}")

        # Глобальная статистика DSM (для нормализации)
        print("Вычисляем статистику высот...")
        dsm_ov = src_dsm.read(1, out_shape=(1, src_dsm.height // 10, src_dsm.width // 10))
        valid_dsm = dsm_ov > -100
        d_min = dsm_ov[valid_dsm].min() if valid_dsm.any() else 0
        d_max = dsm_ov[valid_dsm].max() if valid_dsm.any() else 1

        # Подготовка ROI маски (читаем в память, она легкая)
        if src_roi:
            roi_mask_small = src_roi.read(1)
            scale_h = src_img.height / src_roi.height
            scale_w = src_img.width / src_roi.width

        patch_size = config.PATCH_SIZE

        # --- СБОР СПИСКА ВАЛИДНЫХ ПАТЧЕЙ ---
        # Вместо тупого перебора всех координат, сначала проверим их по маске
        valid_coords = []
        for y in range(0, src_img.height, patch_size):
            for x in range(0, src_img.width, patch_size):
                if src_roi:
                    # Проверяем, попадает ли патч в белую зону маски
                    y_s = int(y / scale_h)
                    x_s = int(x / scale_w)
                    h_s = int(patch_size / scale_h)
                    w_s = int(patch_size / scale_w)
                    chunk = roi_mask_small[y_s:y_s + h_s, x_s:x_s + w_s]

                    # Если кусок полностью черный (0) - пропускаем
                    if not np.any(chunk):
                        continue
                valid_coords.append((y, x))

        print(
            f"Будет обработано патчей: {len(valid_coords)} (Сэкономлено: {(src_img.height // patch_size) * (src_img.width // patch_size) - len(valid_coords)})")

        with rasterio.open(OUTPUT_PATH, 'w', **meta) as dst:
            # Инициализируем файл значением 255 (прозрачность)
            # Чтобы пропущенные куски остались прозрачными
            # (Для больших файлов это может быть медленно, но надежно)
            # Если файл слишком большой, шаг инициализации можно пропустить, rasterio сам заполнит nodata

            for y, x in tqdm(valid_coords, desc="Предсказание"):
                window = Window(x, y, patch_size, patch_size)

                # Читаем с boundless=True (авто-паддинг нулями на краях)
                img = src_img.read(window=window, boundless=True, fill_value=0)
                img = img.transpose(1, 2, 0).astype(np.float32)

                dsm = src_dsm.read(1, window=window, boundless=True, fill_value=-9999).astype(np.float32)

                # Нормализация
                if np.max(img) > 255:
                    img /= 65535.0
                else:
                    img /= 255.0

                dsm = np.clip(dsm, d_min, d_max)
                if d_max - d_min > 0:
                    dsm = (dsm - d_min) / (d_max - d_min)
                else:
                    dsm = np.zeros_like(dsm)
                dsm = np.expand_dims(dsm, axis=2)

                # Инференс
                inp = np.concatenate((img, dsm), axis=2)
                tensor = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(config.DEVICE)

                with torch.no_grad():
                    out = model(tensor)
                    pred = torch.argmax(out, dim=1).cpu().numpy()[0]

                # Запись (обрезаем лишнее, если вышли за край при boundless чтении)
                real_h = min(patch_size, src_img.height - y)
                real_w = min(patch_size, src_img.width - x)

                dst.write(pred[:real_h, :real_w].astype(rasterio.uint8), 1,
                          window=Window(x, y, real_w, real_h))

        if src_roi: src_roi.close()

    print(f"\n✅ Готово! Карта сохранена в: {OUTPUT_PATH}")


if __name__ == "__main__":
    predict_full_map()