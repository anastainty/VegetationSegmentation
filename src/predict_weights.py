import numpy as np
import rasterio
from sklearn.utils.class_weight import compute_class_weight
import config
import sys


def calculate_weights_smart():
    print("⚖️ --- УМНЫЙ РАСЧЕТ ВЕСОВ (Dr. Xu style) ---")

    with rasterio.open(config.IMAGE_PATH) as src_img:
        # Шаг 4 для ускорения (прореживание)
        step = 4
        h, w = src_img.height // step, src_img.width // step
        print("⏳ Чтение изображения...")
        # Читаем только 1 канал, чтобы понять где не пустота
        img_data = src_img.read(1, out_shape=(1, h, w))

    with rasterio.open(config.MASK_PATH) as src_mask:
        print("⏳ Чтение маски...")
        mask_data = src_mask.read(1, out_shape=(1, h, w))

    # 1. Валидная область (где есть фото)
    valid_pixels_mask = (img_data > 0)
    target_labels = mask_data[valid_pixels_mask]

    # Если в маске есть мусорные значения (например 255), уберем их, оставим 0..4
    target_labels = target_labels[target_labels < config.NUM_CLASSES]

    if target_labels.size == 0:
        print("❌ Ошибка: Нет данных!")
        return

    classes = np.unique(target_labels)
    classes.sort()
    print(f"🔎 Найденные классы: {classes}")

    # 2. Считаем математически сбалансированные веса
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=target_labels)

    # Создаем словарь {class_id: weight}
    weight_dict = {c: w for c, w in zip(classes, weights)}

    # 3. Формируем итоговый список
    final_weights = []
    labels_map = {0: "Фон", 1: "Асфальт", 2: "Трава", 3: "Деревья", 4: "Кусты"}

    for i in range(config.NUM_CLASSES):
        if i == 0:
            # --- ПРАВКА Dr. Xu ---
            # Не 0.0! Даем вес 0.1, чтобы модель училась, что на фоне (зданиях)
            # не должно быть растительности.
            w = 0.1
        elif i in weight_dict:
            w = weight_dict[i]
        else:
            w = 1.0  # Если класса нет в выборке (редко)

        final_weights.append(w)

    final_weights = np.array(final_weights, dtype=np.float32)

    # 4. Нормализация (чтобы средний вес был около 1.0)
    # Исключаем фон из расчета среднего, чтобы он не перекашивал
    mean_val = np.mean(final_weights[1:])
    final_weights = final_weights / mean_val

    # Фон снова фиксируем на 0.1 (или 0.2) от среднего уровня, если он улетел
    final_weights[0] = 0.1

    print("\n--- ГОТОВЫЙ РЕЗУЛЬТАТ ДЛЯ TRAIN.PY ---")
    tensor_str = ", ".join([f"{w:.4f}" for w in final_weights])
    print(f"class_weights = torch.tensor([{tensor_str}]).to(config.DEVICE)")
    print("-" * 50)

    for i, w in enumerate(final_weights):
        name = labels_map.get(i, "?")
        print(f"  {name:<10} (Id {i}): {w:.4f}")

    # Совет
    if final_weights[4] < final_weights[2]:
        print("\n⚠️ ВНИМАНИЕ: Вес 'Кустов' (4) получился меньше веса 'Травы' (2).")
        print("Это значит, кустов в разметке МНОГО. Если модель их путает,")
        print("можно вручную поднять вес кустов в train.py (например, до 1.0-1.2).")


if __name__ == "__main__":
    calculate_weights_smart()