import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECA_Module(nn.Module):
    """Efficient Channel Attention (ECA) Module"""
    def __init__(self, channels, b=1, gamma=2):
        super(ECA_Module, self).__init__()
        # Динамический расчет размера ядра свертки в зависимости от числа каналов
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (Batch, Channels, Height, Width)
        y = self.avg_pool(x)
        # Подготавливаем тензор для 1D свертки
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        # Умножаем исходные признаки на веса внимания
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Спуск"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Подъем"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # --- ИНИЦИАЛИЗАЦИЯ ECA МОДУЛЕЙ ДЛЯ КАЖДОГО УРОВНЯ ---
        self.eca1 = ECA_Module(64)
        self.eca2 = ECA_Module(128)
        self.eca3 = ECA_Module(256)
        self.eca4 = ECA_Module(512)
        self.eca5 = ECA_Module(1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # --- DROPOUT для борьбы с переобучением ---
        self.dropout = nn.Dropout(p=0.5)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Проход кодировщика (Encoder)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # --- ПРИМЕНЕНИЕ ECA МОДУЛЯ ПЕРЕД ПРОПУСКНЫМИ СОЕДИНЕНИЯМИ ---
        # Фокусируем внимание на важных каналах (например, DSM и RedEdge)
        x1_eca = self.eca1(x1)
        x2_eca = self.eca2(x2)
        x3_eca = self.eca3(x3)
        x4_eca = self.eca4(x4)
        x5_eca = self.eca5(x5)

        # Проход декодировщика (Decoder)
        x = self.up1(x5_eca, x4_eca)
        x = self.up2(x, x3_eca)
        x = self.up3(x, x2_eca)
        x = self.up4(x, x1_eca)

        # Применяем Dropout перед выходом
        x = self.dropout(x)

        return self.outc(x)