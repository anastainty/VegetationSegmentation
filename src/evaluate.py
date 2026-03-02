import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

import config
from dataset import VegetationDataset
from model import UNet

# --- НАСТРОЙКИ ПОД НОВУЮ МОДЕЛЬ ---
MODEL_FILENAME = "unet_spatial_split.pth"
# Порядок классов должен СТРОГО совпадать с train.py!
CLASSES = ["Фон", "Асфальт", "Трава", "Деревья", "Кусты"]


def evaluate_model():
    print(f"📊 Начинаем оценку точности модели на устройстве: {config.DEVICE}")
    print(f"📡 Ожидается каналов: {config.NUM_CHANNELS} (RGB + DSM + NDVI)")

    # 1. Подготовка папки для результатов
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Поднимаемся на уровень выше src
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "data", "processed", "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"📂 Результаты будут сохранены в папку: {results_dir}")

    # Проверка путей
    if not os.path.exists(config.IMAGE_PATH):
        print(f"❌ ОШИБКА: Не найден файл изображения: {config.IMAGE_PATH}")
        return

    # 2. Загрузка данных (ВАЖНО: Добавили ndvi_path)
    # Мы используем тот же Dataset, что и при обучении, чтобы нарезать патчи
    roi_file = os.path.join(config.MASKS_DIR, "roi_mask.tif")

    val_dataset = VegetationDataset(
        image_path=config.IMAGE_PATH,
        dsm_path=config.DSM_PATH,
        ndvi_path=config.NDVI_PATH,  # <--- ВАЖНО: NDVI
        mask_path=config.MASK_PATH,
        roi_path=roi_file if os.path.exists(roi_file) else None,
        patch_size=config.PATCH_SIZE,
        augment=False  # Для валидации аугментация не нужна
    )

    # Batch size можно сделать побольше для скорости
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Загрузка модели
    # ВАЖНО: n_channels берем из конфига (там теперь 7)
    model = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Ищем модель
    possible_paths = [
        MODEL_FILENAME,
        os.path.join("src", MODEL_FILENAME),
        os.path.join(base_dir, MODEL_FILENAME)
    ]

    current_model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            current_model_path = p
            break

    if not current_model_path:
        print(f"❌ ОШИБКА: Не найден файл модели {MODEL_FILENAME}.")
        return

    print(f"📂 Загружаем веса из: {current_model_path}")
    state = torch.load(current_model_path, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_preds = []
    all_targets = []

    # 4. Прогон данных
    print(f"Всего патчей для оценки: {len(val_dataset)}")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Считаем метрики"):
            images = images.to(config.DEVICE)
            outputs = model(images)

            # Получаем предсказания
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            targets = masks.cpu().numpy().flatten()

            # --- ВАЖНО: Фильтрация ---
            # Исключаем пиксели со значением 255 (это паддинг или границы ROI)
            # Если их не убрать, они испортят матрицу ошибок
            valid_mask = targets != 255

            if np.any(valid_mask):
                all_preds.extend(preds[valid_mask])
                all_targets.extend(targets[valid_mask])

    if len(all_targets) == 0:
        print("❌ Ошибка: Нет валидных пикселей для оценки (возможно, вся маска = 255?)")
        return

    # 5. Расчет Матрицы ошибок
    print("\n🧮 Расчет матрицы ошибок и IoU...")
    # labels=range(len(CLASSES)) гарантирует, что матрица будет правильного размера
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(CLASSES)))

    # Считаем IoU для каждого класса
    iou_scores = []
    print("\n--- IoU по классам ---")
    for i in range(len(CLASSES)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denominator = tp + fp + fn

        if denominator > 0:
            iou = tp / denominator
        else:
            iou = 0.0  # Если класса вообще не было в выборке

        iou_scores.append(iou)
        print(f"{CLASSES[i]:<10}: {iou:.4f}")

    mean_iou = np.mean(iou_scores)
    print(f"{'-' * 20}\nMean IoU  : {mean_iou:.4f}")

    # 6. Формируем отчет
    print("📈 Генерация CSV отчета...")
    report_dict = classification_report(
        all_targets, all_preds,
        target_names=CLASSES,
        labels=range(len(CLASSES)),
        output_dict=True,
        zero_division=0
    )
    df_report = pd.DataFrame(report_dict).transpose()

    # Добавляем IoU в таблицу
    df_report['IoU'] = np.nan
    for i, class_name in enumerate(CLASSES):
        if class_name in df_report.index:
            df_report.loc[class_name, 'IoU'] = iou_scores[i]

    if 'macro avg' in df_report.index:
        df_report.loc['macro avg', 'IoU'] = mean_iou

    print("\n=== ПОЛНЫЙ ОТЧЕТ ===")
    print(df_report)

    # Сохраняем CSV
    csv_path = os.path.join(results_dir, "accuracy_report.csv")
    df_report.to_csv(csv_path)
    print(f"📄 Таблица сохранена: {csv_path}")

    # 7. Рисуем матрицу ошибок
    # Нормализация для красивых цветов (в процентах от истинного класса)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Убираем деление на 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix (Normalized)\nMean IoU: {mean_iou:.4f}")
    plt.ylabel("Истинный класс (Ground Truth)")
    plt.xlabel("Предсказанный класс (Prediction)")
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300)
    print(f"🖼 Матрица ошибок сохранена: {plot_path}")

    # 8. Текстовое саммари
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Model: {current_model_path}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Accuracy: {report_dict['accuracy']:.4f}\n")
        f.write("-" * 20 + "\n")
        for i, score in enumerate(iou_scores):
            f.write(f"{CLASSES[i]}: {score:.4f}\n")

    print("\n✅ Оценка завершена успешно!")


if __name__ == "__main__":
    evaluate_model()