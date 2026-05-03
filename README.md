# CNN 模型運行 GPU 測試程式 - PyTorch

用 CNN / ResNet 模型訓練影像分類，包含 MNIST 手寫數字、CIFAR-10 與 CIFAR-100 彩色圖片，並說明如何讓訓練跑在 GPU 上。

各模型詳細技術說明請見 [`docs/`](docs/) 資料夾。

---

## MNIST（手寫數字辨識）

**腳本**：`src/mnist.py`

### 資料集

- 訓練集：60,000 張 28×28 灰階圖
- 測試集：10,000 張
- 類別：數字 0 ~ 9，共 10 類

### 模型架構

兩層卷積 + 兩層全連接的簡單 CNN：

```
輸入 (1×28×28)
→ Conv2d(1→32) + ReLU + MaxPool  →  32×14×14
→ Conv2d(32→64) + ReLU + MaxPool →  64×7×7
→ Flatten → FC(3136→128) + ReLU + Dropout(0.5)
→ FC(128→10)  →  10 個類別
```

### 超參數

| 項目 | 值 |
|------|----|
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Epochs | 10 |
| Optimizer | Adam |

### 執行

```bash
python src/mnist.py
```

訓練紀錄輸出至 `logs/MNIST/`

---

## CIFAR-10（彩色圖片分類）

**腳本**：`src/cifar10.py`

### 資料集

- 訓練集：50,000 張 32×32 彩色圖（RGB 3 通道）
- 測試集：10,000 張
- 類別：飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車，共 10 類

### 模型架構

簡化版 ResNet，三組殘差層逐步縮小特徵圖：

```
輸入 (3×32×32)
→ Conv2d(3→64) + BN + ReLU
→ Layer1: 2× ResidualBlock(64→64)   →  64×32×32
→ Layer2: 2× ResidualBlock(64→128)  →  128×16×16
→ Layer3: 2× ResidualBlock(128→256) →  256×8×8
→ AdaptiveAvgPool → Flatten
→ FC(256→10)  →  10 個類別
```

ResidualBlock 的 shortcut 連接讓梯度可以跳層傳遞，解決深層網路難以訓練的問題。

### 超參數

| 項目 | 值 |
|------|----|
| Batch Size | 256 |
| Learning Rate | 0.001（每 10 epoch × 0.5） |
| Epochs | 20 |
| Optimizer | Adam |
| LR Scheduler | StepLR(step=10, gamma=0.5) |

訓練集使用隨機裁切（RandomCrop）與水平翻轉（RandomHorizontalFlip）做資料增強。

### 執行

```bash
python src/cifar10.py
```

訓練紀錄輸出至 `logs/CIFAR10/`

---

## CIFAR-100（百類彩色圖片分類）

**腳本**：`src/cifar100.py`

### 資料集

- 訓練集：50,000 張 32×32 彩色圖（RGB 3 通道）
- 測試集：10,000 張
- 類別：100 類（20 大類，每大類 5 小類），涵蓋動物、交通工具、日常物品等

### 模型架構

比 CIFAR-10 版多一組殘差層，以容納 100 個分類所需的特徵容量：

```
輸入 (3×32×32)
→ Conv2d(3→64) + BN + ReLU
→ Layer1: 2× ResidualBlock(64→64)    →  64×32×32
→ Layer2: 2× ResidualBlock(64→128)   →  128×16×16
→ Layer3: 2× ResidualBlock(128→256)  →  256×8×8
→ Layer4: 2× ResidualBlock(256→512)  →  512×4×4
→ AdaptiveAvgPool → Dropout(0.3) → Flatten
→ FC(512→100)  →  100 個類別
```

### 超參數

| 項目 | 值 |
|------|----|
| Batch Size | 128 |
| Learning Rate | 0.001（CosineAnnealing 衰減） |
| Epochs | 30 |
| Optimizer | Adam |
| LR Scheduler | CosineAnnealingLR(T_max=30) |
| Dropout | 0.3（FC 前） |

訓練集使用 RandomCrop、RandomHorizontalFlip 與 ColorJitter 資料增強。

### 執行

```bash
python src/cifar100.py
```

訓練紀錄輸出至 `logs/CIFAR100/`

---

## 如何讓模型跑在 GPU 上

本專案已在 `src/device.py` 內集中處理裝置選擇，會依序嘗試：

- NVIDIA GPU：`cuda`
- Apple Silicon（macOS）：`mps`（若 PyTorch 支援）
- 其他：`cpu`

### NVIDIA GPU（CUDA）

PyTorch 用一行就能偵測 NVIDIA GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

有 NVIDIA GPU 且安裝了 CUDA 驅動時，`device` 會是 `cuda`，否則自動退回 `cpu`。

### Mac（Apple Silicon）GPU：MPS

在 Apple Silicon（M1/M2/M3…）的 macOS 上，PyTorch 可透過 `mps` 使用 GPU（Metal）。

你可以用以下方式檢查：

```python
import torch

print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
```

若 `is_available()` 為 `True`，本專案會自動選用 `mps`。

接著只要把**模型**和**資料**都搬到同一個 device 上即可：

```python
model = ResNet().to(device)   # 模型搬到 GPU

images = images.to(device)   # 每個 batch 的資料也要搬
labels = labels.to(device)
```

這樣前向傳播、反向傳播、梯度更新全部都會在 GPU 上執行。

### DataLoader 的 pin_memory

```python
DataLoader(..., num_workers=4, pin_memory=True)
```

- `pin_memory=True`：將資料預先鎖定在記憶體，讓 CPU → GPU 的傳輸更快
- `num_workers=4`：用多個子行程預先載入資料，避免 GPU 等待

> **Mac（MPS）注意事項**：本專案在 `mps` 模式下預設使用 `num_workers=0`，避免部分環境出現多行程 DataLoader 的穩定性問題（設定在 `src/device.py` 的 `get_dataloader_kwargs_for_device`）。

> **Windows 注意事項**：`num_workers > 0` 在 Windows 上需要把訓練程式包在 `if __name__ == '__main__':` 裡，否則會因為 spawn 機制重複執行腳本而出錯。

---

## 訓練紀錄

每次訓練結束後自動將結果匯出為 txt，依資料集分資料夾存放：

```
logs/
├── MNIST/
│   └── training_log_20260502_001118.txt
├── CIFAR10/
│   └── training_log_20260502_012345.txt
└── CIFAR100/
    └── training_log_20260502_023456.txt
```

報告表頭會包含 **電腦名稱（hostname）**、**完整 Python 版本**（例如 `3.10.12`，與 `platform.python_version()` 一致）、**PyTorch 版本**（`torch.__version__`）、**訓練裝置與裝置名稱**。若實際使用 **NVIDIA GPU（CUDA）** 訓練，會額外寫入 **CUDA 版本**（對應 PyTorch 建置時綁定的 `torch.version.cuda`，與驅動程式顯示的 CUDA 可能略有差異）。

透過頂部的 `ENABLE_LOGGING` 旗標控制是否匯出：

```python
ENABLE_LOGGING = True   # 改為 False 可關閉
```

---

## 環境需求

- Python 3.10.12
- torch 2.0.1
- torchvision 0.15.2
- numpy 1.26.4
- matplotlib 3.10.9
- tqdm 4.67.3

```bash
pip install torch torchvision tqdm
```

若你需要 **CUDA** 或 **macOS MPS** 對應的 PyTorch 版本，請至 [pytorch.org](https://pytorch.org) 依照作業系統與硬體選擇安裝指令。


## 作者資訊

- 姓名 Name: 王建葦 Albert W.
- 電子郵件 Email: albert@mail.jw-albert.tw

## 貢獻

歡迎提交 Issue 和 Pull Request 來改善這個專案

## 授權

本專案採用 MIT 授權條款