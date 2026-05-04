import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from tqdm import tqdm
from logger import TrainingLogger
from device import (
    get_best_torch_device,
    get_dataloader_kwargs_for_device,
    get_device_display_info,
    get_mac_chip_info,
)

# 設為 False 可關閉訓練紀錄匯出
ENABLE_LOGGING = True


class ResidualBlock(nn.Module):
    """單一殘差塊：兩層 3×3 卷積 + shortcut 連接，避免深層網路梯度消失。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 當尺寸或通道數改變時，shortcut 需要 1×1 卷積對齊維度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 殘差相加
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """簡化版 ResNet，三組殘差層處理 CIFAR-10 的 32×32 彩色圖片（3 通道）。"""

    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # 初始卷積：3 通道輸入 → 64 特徵圖，不縮小尺寸
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)   # 32×32
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)  # 16×16
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2) # 8×8

        # AdaptiveAvgPool 讓輸出固定為 1×1，不受輸入尺寸影響
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 攤平成 (batch, 256)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """執行一個 epoch 的訓練，回傳平均 loss 與訓練準確率。"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="訓練", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 清除上一步殘留的梯度，再反向傳播更新權重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def test(model, test_loader, device):
    """在測試集上評估模型，回傳準確率。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # 推論不需要計算梯度，節省記憶體
        for images, labels in tqdm(test_loader, desc="測試", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    # Windows 多行程需要在 __main__ 保護下啟動，否則 DataLoader worker 會重複執行整個腳本
    device = get_best_torch_device(prefer_mps=True)
    device_type, device_detail = get_device_display_info(device)
    print(f"使用設備: {device} ({device_type})")
    if device_detail:
        print(f"裝置資訊: {device_detail}")
    chip = get_mac_chip_info()
    if chip.is_macos:
        print(f"Mac 晶片辨識: machine={chip.machine}, apple_silicon={chip.is_apple_silicon}")

    logger = TrainingLogger(enabled=ENABLE_LOGGING, device=device)

    batch_size = 256
    learning_rate = 0.001
    epochs = 100

    # 訓練集：RandomCrop + HFlip + ColorJitter；色彩抖動減少模型對特定光照的依賴
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 測試集只做標準化，不做增強
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    print("\n加載 CIFAR-10 數據...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)

    # pin_memory=True 讓 CPU→GPU 資料傳輸更快；num_workers 使用多行程預載資料
    dl_kwargs = get_dataloader_kwargs_for_device(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dl_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **dl_kwargs)

    model = ResNet(num_classes=10).to(device)
    # label_smoothing 讓模型不過度自信，緩解 train loss 壓到極低但 test acc 停滯的問題
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # weight_decay 為 L2 正則化，直接懲罰過大的權重，是對抗 overfitting 最有效的手段之一
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # CosineAnnealingLR 平滑衰減，避免 StepLR 在 epoch 10 造成 test acc 驟降（83%→76%）的情況
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n開始訓練...\n")
    start_time = time.time()
    logger.start()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        logger.log_epoch(epoch + 1, epochs, train_loss, train_acc, test_acc, elapsed)

    total_time = time.time() - start_time
    print(f"\n訓練完成！耗時: {total_time:.2f} 秒")

    logger.finish(total_time)
    logger.export(title="CIFAR-10 ResNet 訓練紀錄", output_dir="./logs/CIFAR10")

    torch.save(model.state_dict(), 'cifar10_resnet.pth')
    print("模型已保存為 cifar10_resnet.pth")
