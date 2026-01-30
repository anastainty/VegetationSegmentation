import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import csv  # Добавили для сохранения истории
import config
from dataset import VegetationDataset
from model import UNet


# FOCAL LOSS
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index, weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train():
    print("--- 🚀 START TRAINING (Dr. Xu: File-based NDVI + Focal Loss) ---")
    print(f"Device: {config.DEVICE} | Channels: {config.NUM_CHANNELS}")

    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    roi_file = os.path.join(config.MASKS_DIR, "roi_mask.tif")
    if not os.path.exists(roi_file):
        print("⚠️ Нет ROI маски! Рекомендуется создать.")

    # Передаем config.NDVI_PATH
    full_dataset = VegetationDataset(
        config.IMAGE_PATH, config.DSM_PATH, config.NDVI_PATH, config.MASK_PATH,
        roi_path=roi_file, patch_size=config.PATCH_SIZE, augment=True
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Dataset: {len(full_dataset)}. Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES).to(config.DEVICE)

    # --- ВЕСА КЛАССОВ (Dr. Xu Strategy) ---
    # Фон = 0.1 (чтобы видеть здания)
    # Кусты (Id 4) = 1.0 (Подняли вручную с 0.74, чтобы модель не путала их с травой)
    # weights = [0.1, 0.5933, 0.9752, 1.6905, 0.7410]
    weights = [0.1, 0.5933, 0.9752, 1.6905, 1.0]
    class_weights = torch.tensor(weights).float().to(config.DEVICE)
    print(f"⚖️ Веса классов: {weights}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_f1 = 0.0
    patience_limit = 35
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    try:
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            train_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

            for images, masks in loop:
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0
            all_preds, all_targets = [], []

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
                    targets = masks.cpu().numpy().flatten()
                    valid_indices = targets != 255
                    all_preds.extend(preds[valid_indices])
                    all_targets.extend(targets[valid_indices])

            avg_val_loss = val_loss / len(val_loader)
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro',
                                                                       zero_division=0)

            lr = optimizer.param_groups[0]['lr']
            print(
                f"\nSummary E{epoch + 1}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f} | F1={f1:.4f} | LR={lr:.6f}")

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_f1'].append(f1)
            scheduler.step(f1)

            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save(model.state_dict(), "unet_best_f1_model.pth")
                print(f"💾 NEW BEST! F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print("🛑 Stopping")
                    break

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by User (Ctrl+C)")

    # --- 📊 ГЕНЕРАЦИЯ ОТЧЕТА И ГРАФИКОВ ---
    print("📈 Генерация графиков...")

    # 1. Сохраняем историю в CSV (на всякий случай)
    csv_path = os.path.join(config.PLOTS_DIR, 'history_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_F1'])
        for i in range(len(history['train_loss'])):
            writer.writerow([i + 1, history['train_loss'][i], history['val_loss'][i], history['val_f1'][i]])
    print(f"💾 История сохранена в CSV: {csv_path}")

    # 2. Рисуем "Умные" графики
    plt.style.use('ggplot') # Красивый стиль
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # График Loss (с ограничением по Y)
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    ax1.set_title("Loss Dynamics (Zoomed)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Focal Loss")
    ax1.legend()
    ax1.grid(True)

    # ХИТРОСТЬ: Если график Loss "взорвался" до 3.0, мы это обрежем,
    # чтобы видеть динамику основного обучения (0.05 - 0.5)
    if len(history['val_loss']) > 0:
        median_loss = np.median(history['val_loss'])
        # Лимит = медиана * 4, но не меньше 1.0, чтобы случайно не обрезать слишком сильно
        limit = max(median_loss * 4.0, 0.5)
        # Если есть супер-пики, обрезаем их для красоты
        if max(history['val_loss']) > limit:
            ax1.set_ylim(0, limit)

    # График F1 Score
    ax2.plot(history['val_f1'], label='Validation F1', color='green', linewidth=2)
    ax2.set_title(f"F1 Score (Best: {best_f1:.4f})")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1.0) # F1 всегда от 0 до 1
    # Красная линия рекорда
    ax2.axhline(y=best_f1, color='red', linestyle='--', alpha=0.5, label='Best Result')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(config.PLOTS_DIR, 'training_xu_final.png')
    plt.savefig(plot_path, dpi=300)
    print(f"🖼 График сохранен: {plot_path}")


if __name__ == "__main__":
    train()