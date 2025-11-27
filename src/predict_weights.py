import numpy as np
import rasterio
from sklearn.utils.class_weight import compute_class_weight
import config


def calculate_exact_weights():
    print(f"Читаем маску: {config.MASK_PATH}")

    with rasterio.open(config.MASK_PATH) as src:
        # Читаем маску
        mask = src.read(1).flatten()

    # Ищем уникальные классы (должны быть 0, 1, 2, 3, 4)
    classes = np.unique(mask)
    print(f"Найденные классы: {classes}")

    # Используем sklearn для расчета весов (методика 'balanced')
    # Формула: n_samples / (n_classes * np.bincount(y))
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=mask)

    print("\n--- РЕКОМЕНДУЕМЫЕ ВЕСА (SKLEARN) ---")
    print("-" * 40)

    # Выводим красиво
    for cls, w in zip(classes, weights):
        print(f"Класс {cls}: {w:.4f}")

    print("-" * 40)

    # Формируем строку для вставки в PyTorch
    # ВАЖНО: Класс 0 (фон) мы обычно обнуляем вручную, если хотим его игнорировать
    # Но sklearn посчитает вес и для него.

    final_weights = weights.copy()
    final_weights[0] = 0.0  # Принудительно зануляем фон, чтобы убрать черную рамку

    # Нормализуем так, чтобы Асфальт (класс 1) был равен 1.0 (для удобства восприятия)
    # Это не обязательно, но так понятнее
    if len(final_weights) > 1:
        factor = 1.0 / final_weights[1]
        final_weights = final_weights * factor

    print("\nКОПИРУЙ ЭТО В TRAIN.PY:")
    print(f"class_weights = torch.tensor({list(np.round(final_weights, 2))}).to(config.DEVICE)")


if __name__ == "__main__":
    calculate_exact_weights()
