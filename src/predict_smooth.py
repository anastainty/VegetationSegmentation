import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import os
import scipy.signal.windows as windows
import config
from model import UNet


def spline_window(window_size, power=2):
    # Гауссово окно для мягкого сшивания
    wind = windows.gaussian(window_size, std=window_size / 3)
    wind = np.expand_dims(wind, 1)
    wind = np.dot(wind, wind.transpose())
    return wind


def predict_smooth():
    # Имя выходного файла
    OUTPUT_PATH = os.path.join(config.IMAGES_OUTPUT_DIR, "final_map_ndvi_focal_0936.tif")
    ROI_PATH = os.path.join(config.MASKS_DIR, "roi_mask.tif")

    print(f"🚀 Старт УМНОГО сглаживания (Gaussian)...")
    print(f"📡 Ожидается каналов: {config.NUM_CHANNELS} (RGB + DSM + NDVI)")

    # 1. ЗАГРУЗКА МОДЕЛИ
    model = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES).to(config.DEVICE)

    model_path = "unet_best_f1_model.pth"
    if not os.path.exists(model_path):
        print("❌ Ошибка: Нет модели unet_best_f1_model.pth!")
        return

    print(f"📂 Загружаем веса: {model_path}")
    state = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()

    # 2. ПОДГОТОВКА
    PATCH_SIZE = config.PATCH_SIZE
    STRIDE = 128  # Шаг сдвига (перекрытие 50%)

    # Готовим окно сглаживания
    window_weight = spline_window(PATCH_SIZE).astype(np.float32)

    # Открываем ВСЕ ТРИ ФАЙЛА (Image, DSM, NDVI)
    with rasterio.open(config.IMAGE_PATH) as src_img, \
            rasterio.open(config.DSM_PATH) as src_dsm, \
            rasterio.open(config.NDVI_PATH) as src_ndvi:  # <--- ВАЖНО: Добавлен NDVI

        h, w = src_img.height, src_img.width
        meta = src_img.meta.copy()
        meta.update(count=1, dtype=rasterio.uint8, nodata=255)

        print("Allocating memory (это может занять немного RAM)...")
        # Карты для накопления вероятностей и весов
        prob_map = np.zeros((config.NUM_CLASSES, h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        # Создаем сетку координат
        coords = []
        for y in range(0, h, STRIDE):
            for x in range(0, w, STRIDE):
                coords.append((y, x))

        print(f"🔄 Обработка {len(coords)} патчей...")

        for y, x in tqdm(coords):
            # Размеры текущего окна (на краях может быть меньше 256)
            window_h = min(PATCH_SIZE, h - y)
            window_w = min(PATCH_SIZE, w - x)
            window = Window(x, y, window_w, window_h)

            # --- 1. ЧИТАЕМ КАРТИНКУ (5 каналов) ---
            img = src_img.read(window=window, boundless=True, fill_value=0)
            img = img.transpose(1, 2, 0).astype(np.float32)
            if np.max(img) > 255:
                img /= 65535.0
            else:
                img /= 255.0

            # --- 2. ЧИТАЕМ DSM (1 канал) ---
            dsm = src_dsm.read(1, window=window, boundless=True, fill_value=config.DSM_MIN).astype(np.float32)
            dsm = np.clip(dsm, config.DSM_MIN, config.DSM_MAX)
            dsm = (dsm - config.DSM_MIN) / (config.DSM_MAX - config.DSM_MIN)
            dsm = np.expand_dims(dsm, axis=2)

            # --- 3. ЧИТАЕМ NDVI (1 канал) --- <--- НОВОЕ
            ndvi = src_ndvi.read(1, window=window, boundless=True, fill_value=-1.0).astype(np.float32)
            # Нормализация NDVI (было -1..1, станет 0..1)
            ndvi = np.clip(ndvi, -1.0, 1.0)
            ndvi = (ndvi + 1.0) / 2.0
            ndvi = np.expand_dims(ndvi, axis=2)

            # --- СБОРКА: 5 + 1 + 1 = 7 каналов ---
            inp = np.concatenate((img, dsm, ndvi), axis=2)

            # Паддинг (если окно меньше 256x256)
            pad_h = PATCH_SIZE - inp.shape[0]
            pad_w = PATCH_SIZE - inp.shape[1]
            if pad_h > 0 or pad_w > 0:
                inp = np.pad(inp, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

            # В тензор
            tensor = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(config.DEVICE)

            # Предсказание
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Обрезаем лишний паддинг (возвращаем размер окна)
            probs = probs[:, :window_h, :window_w]
            current_weight = window_weight[:window_h, :window_w]

            # Добавляем в общую карту с учетом весов Гаусса
            prob_map[:, y:y + window_h, x:x + window_w] += probs * current_weight
            weight_map[y:y + window_h, x:x + window_w] += current_weight

        print("Финализация карты...")
        # Делим на сумму весов, чтобы усреднить наложения
        valid_pixels = weight_map > 0
        final_probs = np.zeros_like(prob_map)
        final_probs[:, valid_pixels] = prob_map[:, valid_pixels] / weight_map[valid_pixels]

        # Выбираем класс с максимальной вероятностью
        result_map = np.argmax(final_probs, axis=0).astype(np.uint8)

        # Обрезаем лишнее по ROI маске (если есть)
        if os.path.exists(ROI_PATH):
            with rasterio.open(ROI_PATH) as roi:
                roi_mask = roi.read(1)
                if roi_mask.shape == result_map.shape:
                    result_map[roi_mask == 0] = 255

        # Сохраняем
        with rasterio.open(OUTPUT_PATH, 'w', **meta) as dst:
            dst.write(result_map, 1)

    print(f"✨ ГОТОВО! Карта сохранена: {OUTPUT_PATH}")


if __name__ == "__main__":
    predict_smooth()