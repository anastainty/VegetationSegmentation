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
    full_dataset = VegetationDataset(
        config.IMAGE_PATH,
        config.DSM_PATH,
        config.MASK_PATH,
        patch_size=config.PATCH_SIZE
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Всего патчей: {len(full_dataset)}")
    print(f"Обучение: {len(train_dataset)} | Валидация: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Инициализация модели
    print(f"\n--- Инициализация нейросети (U-Net) на {config.DEVICE} ---")
    model = UNet(n_channels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    # --- БАЛАНСИРОВКА (Версия 2) ---
    # 0 (Фон): 0.0 (Игнор)
    # 1 (Не раст.): ПОДНИМАЕМ до 2.0 (Асфальт теперь важен! Нейросеть будет бояться его закрасить)
    # 2 (Слабая): СНИЖАЕМ до 1.0 (Пусть сеть красит её, только если уверена)
    # 3 (Умеренная): 3.0 (Оставим высокой, так как кустов мало)
    # 4 (Плотная): 1.5 (Деревья определяются хорошо, оставляем как есть)

    class_weights = torch.tensor([0.0, 1.0, 1.64, 2.85, 1.25]).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_losses = []
    val_losses = []

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

        print(f"Epoch {epoch + 1} -> Train Loss: {epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 4. Сохранение модели
    print("\n--- Сохранение результатов ---")
    torch.save(model.state_dict(), "unet_vegetation_model.pth")
    print("Веса модели сохранены: unet_vegetation_model.pth")

    # 5. Сохранение графика
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