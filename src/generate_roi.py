import rasterio
import numpy as np
import config
import os
from scipy.ndimage import binary_fill_holes


def generate_roi():
    print("🤖 Генерация ROI маски (отсекаем черный фон)...")

    # Убедимся, что папка существует
    if not os.path.exists(config.MASKS_DIR):
        os.makedirs(config.MASKS_DIR)

    OUTPUT_PATH = os.path.join(config.MASKS_DIR, "roi_mask.tif")

    with rasterio.open(config.IMAGE_PATH) as src:
        # Читаем данные (уменьшаем в 10 раз для скорости, маска не нужна супер-точной)
        # Но для сохранения координат лучше читать 1:1, но блоками.
        # Для простоты прочитаем первый канал целиком (если RAM позволяет, 9ГБ занято, значит есть место)
        data = src.read(1)
        meta = src.meta.copy()

    # Создаем бинарную маску: всё, что не 0 (черный) — это полезные данные
    mask = (data > 0).astype(np.uint8)

    # Заполняем "дырки" внутри (например, черные крыши или тени), чтобы их не выкинуло
    # binary_fill_holes работает долго на больших картинках, можно пропустить если долго
    # mask = binary_fill_holes(mask).astype(np.uint8)

    meta.update(count=1, dtype=rasterio.uint8, nodata=0)

    with rasterio.open(OUTPUT_PATH, 'w', **meta) as dst:
        dst.write(mask, 1)

    print(f"✅ ROI маска сохранена: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_roi()