# CIFAR-10 — 10 類彩色圖片分類

## 概述

CIFAR-10 是影像分類的標準基準之一，包含 10 個類別的 32×32 彩色圖片。本模組以三層殘差網路（ResNet）實作，引入 BatchNorm 與 shortcut 連接，在 20 個 epoch 內達到 85%+ 測試準確率。

- **腳本**：`src/cifar10.py`
- **執行腳本**：`cifat10.sh`
- **日誌目錄**：`logs/CIFAR10/`
- **模型輸出**：`cifar10_resnet.pth`

---

## 資料集

| 項目 | 內容 |
|------|------|
| 來源 | `torchvision.datasets.CIFAR10`（自動下載至 `./data/`） |
| 訓練集 | 50,000 張 |
| 測試集 | 10,000 張 |
| 圖片大小 | 32 × 32（RGB，3 通道） |
| 類別數 | 10 |

### 類別列表

| 編號 | 類別（中） | 類別（英） |
|------|-----------|-----------|
| 0 | 飛機 | airplane |
| 1 | 汽車 | automobile |
| 2 | 鳥 | bird |
| 3 | 貓 | cat |
| 4 | 鹿 | deer |
| 5 | 狗 | dog |
| 6 | 青蛙 | frog |
| 7 | 馬 | horse |
| 8 | 船 | ship |
| 9 | 卡車 | truck |

### 前處理與資料增強

**訓練集**
```python
transforms.RandomCrop(32, padding=4)      # 先填充 4 像素後隨機裁切
transforms.RandomHorizontalFlip()          # 50% 機率水平翻轉
transforms.ToTensor()
transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010)
)
```

**測試集**（無增強）
```python
transforms.ToTensor()
transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010)
)
```

---

## 模型架構

### ResidualBlock

```
輸入 (C_in × H × W)
├─ Conv2d(C_in→C_out, 3×3, stride) + BN + ReLU
├─ Conv2d(C_out→C_out, 3×3, stride=1) + BN
└─ shortcut: 1×1 Conv + BN（僅在 stride≠1 或通道改變時）
   輸出 = main_path + shortcut(x)  →  ReLU
```

### ResNet（整體）

```
輸入 (3 × 32 × 32)
│
├─ Conv2d(3→64, 3×3) + BN + ReLU          →  64 × 32 × 32
│
├─ Layer1: 2× ResidualBlock(64→64,  s=1)  →  64 × 32 × 32
├─ Layer2: 2× ResidualBlock(64→128, s=2)  →  128 × 16 × 16
├─ Layer3: 2× ResidualBlock(128→256,s=2)  →  256 × 8 × 8
│
├─ AdaptiveAvgPool2d(1,1)                  →  256 × 1 × 1
├─ Flatten                                 →  256
└─ Linear(256 → 10)
     輸出：10 個 logits
```

### 設計重點

- **shortcut 連接**：梯度可以跳層傳回，解決深層網路梯度消失問題。
- **BatchNorm**：穩定每層輸出分佈，加速收斂並允許較大 learning rate。
- **AdaptiveAvgPool**：輸出固定為 1×1，不受輸入尺寸限制，方便未來更換圖片大小。

---

## 訓練設定

| 超參數 | 值 |
|--------|----|
| Batch Size | 256 |
| Learning Rate | 0.001 |
| Epochs | 20 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| LR Scheduler | StepLR(step\_size=10, gamma=0.5) |

StepLR 讓第 11 epoch 起 lr 降至 0.0005，第 21 epoch 降至 0.00025，後期訓練更穩定收斂。

---

## 執行

```bash
# 直接執行
python src/cifar10.py

# 或透過腳本（需先建立 venv/）
bash cifat10.sh
```

---

## 預期結果

| Epoch | 大致 Test Acc |
|-------|--------------|
| 5     | ~70%         |
| 10    | ~80%         |
| 20    | ~85%+        |

---

## 與 MNIST 的主要差異

| 差異點 | MNIST | CIFAR-10 |
|--------|-------|----------|
| 圖片通道 | 1（灰階） | 3（RGB） |
| 解析度 | 28×28 | 32×32 |
| 模型 | 簡單 CNN | ResNet（殘差連接） |
| 資料增強 | 無 | RandomCrop + HFlip |
| LR Scheduler | 無 | StepLR |
| 難度 | 入門 | 中等 |
