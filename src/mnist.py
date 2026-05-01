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


class CNN(nn.Module):
    """兩層卷積 + 兩層全連接的 CNN，用於 MNIST 10 類分類。"""

    def __init__(self):
        super(CNN, self).__init__()
        # 卷積層：1 通道輸入 → 32 特徵圖，padding=2 保持 28×28 尺寸
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        # 卷積層：32 → 64 特徵圖
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # 每次 Pooling 將尺寸減半：28→14→7
        self.pool = nn.MaxPool2d(2, 2)
        # 攤平後 64×7×7 = 3136 個特徵
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        # 訓練時隨機關閉 50% 神經元，防止過擬合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32×28×28 → 32×14×14
        x = self.pool(self.relu(self.conv2(x)))  # 64×14×14 → 64×7×7
        x = x.view(x.size(0), -1)               # 攤平成 (batch, 3136)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                          # 輸出 10 個 logits
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """執行一個 epoch 的訓練，回傳平均 loss 與訓練準確率。"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="訓練"):
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
        for images, labels in tqdm(test_loader, desc="測試"):
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

    batch_size = 128
    learning_rate = 0.001
    epochs = 10

    # 標準化參數來自 MNIST 全資料集的均值與標準差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("\n加載 MNIST 數據...")
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # pin_memory=True 讓 CPU→GPU 資料傳輸更快；num_workers 使用多行程預載資料
    dl_kwargs = get_dataloader_kwargs_for_device(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dl_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **dl_kwargs)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n開始訓練...\n")
    start_time = time.time()
    logger.start()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        logger.log_epoch(epoch + 1, epochs, train_loss, train_acc, test_acc, elapsed)

    total_time = time.time() - start_time
    print(f"\n訓練完成！耗時: {total_time:.2f} 秒")

    logger.finish(total_time)
    logger.export(title="MNIST CNN 訓練紀錄", output_dir="./logs/MNIST")

    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("模型已保存為 mnist_cnn.pth")