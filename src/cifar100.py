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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """四組殘差層的 ResNet，處理 CIFAR-100 的 32×32 彩色圖片（100 分類）。
    比 CIFAR-10 版多一層（256→512），以容納更多類別所需的特徵容量。"""

    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)  # 32×32
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)  # 16×16
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)  # 8×8
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)  # 4×4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

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
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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

    with torch.no_grad():
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
    device = get_best_torch_device(prefer_mps=True)
    device_type, device_detail = get_device_display_info(device)
    print(f"使用設備: {device} ({device_type})")
    if device_detail:
        print(f"裝置資訊: {device_detail}")
    chip = get_mac_chip_info()
    if chip.is_macos:
        print(f"Mac 晶片辨識: machine={chip.machine}, apple_silicon={chip.is_apple_silicon}")

    logger = TrainingLogger(enabled=ENABLE_LOGGING, device=device)

    batch_size = 128
    learning_rate = 0.001
    epochs = 100

    # CIFAR-100 官方統計值（各通道 mean/std）
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std  = (0.2675, 0.2565, 0.2761)

    # 訓練集：隨機裁切、水平翻轉、色彩抖動，提升對 100 類的泛化能力
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    # 測試集只做標準化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    print("\n加載 CIFAR-100 數據...")
    train_dataset = datasets.CIFAR100(root='./data', train=True,  transform=train_transform, download=True)
    test_dataset  = datasets.CIFAR100(root='./data', train=False, transform=test_transform,  download=True)

    dl_kwargs = get_dataloader_kwargs_for_device(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **dl_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **dl_kwargs)

    model = ResNet(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # CosineAnnealingLR 讓學習率平滑衰減至接近 0，比 StepLR 在多類別任務上收斂更穩定
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
    logger.export(title="CIFAR-100 ResNet 訓練紀錄", output_dir="./logs/CIFAR100")

    torch.save(model.state_dict(), 'cifar100_resnet.pth')
    print("模型已保存為 cifar100_resnet.pth")
