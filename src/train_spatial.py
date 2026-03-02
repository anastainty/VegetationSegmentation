import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import csv
import config
from dataset import VegetationDataset
from model import UNet


# --- FOCAL LOSS (Тот же, что и был) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.alpha = alpha

    def forward(self, inputs, targets):
        # inputs: [N, C, H, W], targets: [N, H, W]
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index, weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_spatial():
    print("--- ⚔️ HARDCORE MODE: SPATIAL SPLIT TRAINING ⚔️ ---")
    print("Мы проверяем гипотезу друга: учим на ВЕРХЕ карты, тестим на НИЗЕ.")
    print(f"Device: {config.DEVICE}")

    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # 1. Загружаем датасет
    # Используем ROI маску, чтобы не учиться на черных полях
    roi_file = os.path.join(config.MASKS_DIR, "roi_mask.tif")
    if not os.path.exists(roi_file):
        print("⚠️ ROI маска не найдена, используем всё изображение.")

    full_dataset = VegetationDataset(
        config.IMAGE_PATH, config.DSM_PATH, config.NDVI_PATH, config.MASK_PATH,
        roi_path=roi_file if os.path.exists(roi_file) else None,
        patch_size=config.PATCH_SIZE,
        augment=True  # Аугментация критически важна здесь!
    )

    # 2. ЖЕСТКОЕ РАЗДЕЛЕНИЕ (Spatial Split)
    # В твоем dataset.py патчи создаются циклами for y... for x...
    # Значит, в списке они идут: Верхний ряд -> Следующий ряд -> ... -> Нижний ряд.

    total_len = len(full_dataset)
    split_idx = int(0.8 * total_len)  # Режем на 80%

    # Индексы 0..split_idx = ВЕРХНЯЯ ЧАСТЬ КАРТЫ (Train)
    train_indices = list(range(0, split_idx))

    # Индексы split_idx..End = НИЖНЯЯ ЧАСТЬ КАРТЫ (Test)
    val_indices = list(range(split_idx, total_len))

    # Создаем подвыборки
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"\n🗺 ГЕОГРАФИЧЕСКОЕ РАЗДЕЛЕНИЕ:")
    print(f"   Train (Верх карты): {len(train_dataset)} патчей")
    print(f"   Test  (Низ карты):  {len(val_dataset)} патчей")
    print("-" * 50)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. МОДЕЛЬ
    model = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Веса классов (Dr. Xu style)
    # Фон=0.1, Кусты=1.0
    weights = [0.1, 0.5933, 0.9752, 1.6905, 1.0]
    class_weights = torch.tensor(weights).float().to(config.DEVICE)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_f1 = 0.0
    patience_limit = 25  # Чуть меньше, так как задача сложнее
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

            # Валидация на НИЖНЕЙ части карты
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
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='macro', zero_division=0
            )

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"\nSummary E{epoch + 1}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f} | F1={f1:.4f} | LR={current_lr:.6f}")

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_f1'].append(f1)

            scheduler.step(f1)

            # Сохраняем модель с пометкой SPATIAL
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save(model.state_dict(), "unet_spatial_split.pth")
                print(f"💾 SAVED SPATIAL MODEL! F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print("🛑 Early Stopping (Spatial)")
                    break

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")

    # Рисуем графики для этого эксперимента
    print("📈 Рисуем график Spatial Split...")
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(history['train_loss'], label='Train (Top Map)', color='blue')
    ax1.plot(history['val_loss'], label='Val (Bottom Map)', color='orange')
    ax1.set_title("Spatial Loss Dynamics")
    ax1.set_xlabel("Epochs")
    ax1.legend()

    # Обрезка пиков лосса
    if len(history['val_loss']) > 0:
        median_loss = np.median(history['val_loss'])
        limit = max(median_loss * 4.0, 0.5)
        ax1.set_ylim(0, limit)

    ax2.plot(history['val_f1'], label='Spatial F1', color='purple')
    ax2.set_title(f"Spatial F1 (Best: {best_f1:.4f})")
    ax2.set_xlabel("Epochs")
    ax2.set_ylim(0, 1.0)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'training_spatial_result.png'))
    print(f"🏁 Готово. Результат сохранен в plots/training_spatial_result.png")


if __name__ == "__main__":
    train_spatial()