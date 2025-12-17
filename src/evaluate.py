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

# --- НАСТРОЙКИ ---
MODEL_PATH = "unet_vegetation_model.pth"
CLASSES = ["Фон", "Асфальт", "Трава", "Кусты", "Лес"]


def evaluate_model():
    print(f"📊 Начинаем оценку точности модели на устройстве: {config.DEVICE}")
    print(f"Используем данные из: {config.IMAGE_PATH}")

    # 1. Подготовка папки для результатов
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "data", "processed", "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"📂 Результаты будут сохранены в папку: {results_dir}")

    if not os.path.exists(config.IMAGE_PATH) or not os.path.exists(config.MASK_PATH):
        print("❌ ОШИБКА: Не найдены файлы данных.")
        return

    # 2. Загрузка данных
    val_dataset = VegetationDataset(
        image_path=config.IMAGE_PATH,
        mask_path=config.MASK_PATH,
        dsm_path=config.DSM_PATH,
        transform=None
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 3. Загрузка модели
    model = UNet(n_channels=6, n_classes=5).to(config.DEVICE)

    if not os.path.exists(MODEL_PATH):
        alt_path = os.path.join("src", MODEL_PATH)
        current_model_path = alt_path if os.path.exists(alt_path) else None
    else:
        current_model_path = MODEL_PATH

    if not current_model_path:
        print(f"❌ ОШИБКА: Не найден файл модели {MODEL_PATH}.")
        return

    print(f"Загружаем веса из: {current_model_path}")
    model.load_state_dict(torch.load(current_model_path, map_location=config.DEVICE))
    model.eval()

    all_preds = []
    all_targets = []

    # 4. Прогон данных
    print(f"Всего тестовых примеров: {len(val_loader)}")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Считаем метрики"):
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            targets = masks.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)

    # --- ИЗМЕНЕНИЯ НАЧИНАЮТСЯ ЗДЕСЬ ---

    # 5. Сначала считаем Матрицу ошибок (нужна для IoU)
    print("\n🧮 Расчет матрицы ошибок и IoU...")
    cm = confusion_matrix(all_targets, all_preds)

    # Считаем IoU для каждого класса
    iou_scores = []
    for i in range(len(CLASSES)):
        if i < cm.shape[0] and i < cm.shape[1]:
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            denominator = tp + fp + fn
            iou = tp / denominator if denominator > 0 else 0.0
            iou_scores.append(iou)
        else:
            iou_scores.append(0.0)

    mean_iou = np.mean(iou_scores)

    # 6. Формируем отчет и добавляем в него IoU
    print("📈 Генерация CSV отчета...")
    report_dict = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()

    # Добавляем колонку IoU
    # Создаем пустую колонку
    df_report['IoU'] = np.nan

    # Заполняем IoU по классам
    for i, class_name in enumerate(CLASSES):
        if class_name in df_report.index:
            df_report.loc[class_name, 'IoU'] = iou_scores[i]

    # Заполняем Mean IoU в строку 'macro avg' (среднее)
    if 'macro avg' in df_report.index:
        df_report.loc['macro avg', 'IoU'] = mean_iou

    # Вывод красивой таблицы в консоль (заменяем NaN на прочерки для красоты вывода, но в CSV оставим числа)
    print("\n=== ПОЛНЫЙ ОТЧЕТ (с IoU) ===")
    print(df_report)

    # Сохраняем CSV
    csv_path = os.path.join(results_dir, "accuracy_report.csv")
    df_report.to_csv(csv_path)
    print(f"📄 Таблица метрик сохранена: {csv_path}")

    # 7. Рисуем матрицу ошибок (картинка)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix (Mean IoU: {mean_iou:.4f})")
    plt.ylabel("Истинный класс")
    plt.xlabel("Предсказанный класс")
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300)
    print(f"🖼 Матрица ошибок сохранена: {plot_path}")

    # 8. Краткий текстовый отчет (дублируем IoU туда тоже)
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Model: {current_model_path}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write("-" * 20 + "\n")
        for i, score in enumerate(iou_scores):
            f.write(f"IoU {CLASSES[i]}: {score:.4f}\n")


if __name__ == "__main__":
    evaluate_model()