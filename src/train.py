import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import config
from dataset import VegetationDataset
from model import UNet


def train():
    # 1. Инициализация данных
    print("--- Подготовка данных ---")

    # Включаем аугментацию (повороты, отражения)
    full_dataset = VegetationDataset(
        config.IMAGE_PATH,
        config.DSM_PATH,
        config.MASK_PATH,
        patch_size=config.PATCH_SIZE,
        augment=True
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Разбиваем данные
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Всего патчей: {len(full_dataset)}")
    print(f"Обучение: {len(train_dataset)} | Валидация: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Инициализация модели
    print(f"\n--- Инициализация нейросети (U-Net) на {config.DEVICE} ---")
    model = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    class_weights = torch.tensor([0.34, 1.36, 2.24, 3.88, 1.7]).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)

    # Оптимизатор с защитой от переобучения (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    # Планировщик (снижает скорость обучения, если вышли на плато)
    # verbose удален, так как устарел в новых версиях PyTorch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []

    # --- НАСТРОЙКИ EARLY STOPPING (Ранняя остановка) ---
    best_val_loss = float('inf')
    patience_limit = 15  # Сколько эпох ждать улучшения, прежде чем сдаться
    patience_counter = 0

    # 3. Цикл обучения
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

        for images, masks in loop:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Обновляем LR если нужно
        scheduler.step(avg_val_loss)

        # Выводим текущий LR вручную
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1} -> Train Loss: {epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        # --- ЛОГИКА СОХРАНЕНИЯ И ОСТАНОВКИ ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Сбрасываем счетчик, так как мы улучшились
            torch.save(model.state_dict(), "unet_vegetation_model.pth")
            print("💾 Модель сохранена (Новый лучший результат)!")
        else:
            patience_counter += 1
            print(f"⏳ Нет улучшений {patience_counter}/{patience_limit} эпох.")

            if patience_counter >= patience_limit:
                print("\n🛑 Ранняя остановка: Модель перестала обучаться.")
                break

    # 4. Сохранение графика
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(config.PLOTS_DIR, 'training_plot.png')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('График обучения')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"График сохранен в: {plot_path}")


if __name__ == "__main__":
    train()