import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
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

# 可選架構：resnet18 / resnet34 / resnet50
ARCH = "resnet18"


def build_resnet(arch: str, num_classes: int) -> nn.Module:
    """
    從 torchvision 載入標準 ResNet，針對 CIFAR-10（32×32）做兩處調整：
    1. 第一層換成 3×3 conv（stride=1），避免原本 7×7 stride=2 把 32×32 壓成 16×16
    2. 移除 MaxPool，防止特徵圖在殘差塊前就縮到 8×8
    """
    constructor = getattr(models, arch)
    model = constructor(weights=None)

    # 適配 CIFAR 小尺寸輸入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 替換最終分類層
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, gpu_transform=None):
    """執行一個 epoch 的訓練，回傳平均 loss 與訓練準確率。"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="訓練", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if gpu_transform is not None:
            images = gpu_transform(images)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    # Windows 多行程需要在 __main__ 保護下啟動，否則 DataLoader worker 會重複執行整個腳本
    device = get_best_torch_device(prefer_mps=True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    device_type, device_detail = get_device_display_info(device)
    print(f"使用設備: {device} ({device_type})")
    if device_detail:
        print(f"裝置資訊: {device_detail}")
    chip = get_mac_chip_info()
    if chip.is_macos:
        print(f"Mac 晶片辨識: machine={chip.machine}, apple_silicon={chip.is_apple_silicon}")

    logger = TrainingLogger(enabled=ENABLE_LOGGING, device=device)

    batch_size = 256
    learning_rate = 0.1
    epochs = 30

    # 訓練集 CPU 只做最基本的轉換，augmentation 搬到 GPU 做
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 測試集只做標準化，不做增強
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # GPU augmentation：在 GPU tensor 上做，省掉 CPU 的瓶頸
    gpu_train_transform = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("\n加載 CIFAR-10 數據...")
    train_dataset = datasets.CIFAR10(root='./data', train=True,  transform=train_transform, download=True)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=test_transform,  download=True)

    dl_kwargs = get_dataloader_kwargs_for_device(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **dl_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **dl_kwargs)

    print(f"\n建構模型: {ARCH}（CIFAR-10 適配版）...")
    model = build_resnet(ARCH, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # SGD + momentum 是訓練 ResNet 的標準組合，比 Adam 在大 batch 下收斂更穩
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # WarmupCosine：前 5 epoch 線性升溫，之後餘弦衰減，避免初期 lr 過大導致不穩定
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    print("\n開始訓練...\n")
    start_time = time.time()
    logger.start()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, gpu_transform=gpu_train_transform)
        test_acc = test(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {current_lr:.5f}")
        logger.log_epoch(epoch + 1, epochs, train_loss, train_acc, test_acc, elapsed)

    total_time = time.time() - start_time
    print(f"\n訓練完成！耗時: {total_time:.2f} 秒")

    logger.finish(total_time)
    logger.export(title=f"CIFAR-10 {ARCH.upper()} 訓練紀錄", output_dir="./logs/ResNet")

    torch.save(model.state_dict(), f'cifar10_{ARCH}.pth')
    print(f"模型已保存為 cifar10_{ARCH}.pth")
