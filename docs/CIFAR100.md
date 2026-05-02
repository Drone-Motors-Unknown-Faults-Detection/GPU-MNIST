# CIFAR-100 — 100 類彩色圖片分類

## 概述

CIFAR-100 與 CIFAR-10 使用相同的圖片來源（32×32 彩色圖），但分類數從 10 擴充至 100，難度顯著提升。本模組在 CIFAR-10 ResNet 的基礎上加深一層（256→512），並採用 CosineAnnealingLR 與更豐富的資料增強，目標在 30 個 epoch 內達到 60%+ 測試準確率。

- **腳本**：`src/cifar100.py`
- **執行腳本**：`cifar100.sh`
- **日誌目錄**：`logs/CIFAR100/`
- **模型輸出**：`cifar100_resnet.pth`

---

## 資料集

| 項目 | 內容 |
|------|------|
| 來源 | `torchvision.datasets.CIFAR100`（自動下載至 `./data/`） |
| 訓練集 | 50,000 張 |
| 測試集 | 10,000 張 |
| 圖片大小 | 32 × 32（RGB，3 通道） |
| 類別數 | 100（20 大類，每大類含 5 小類） |

### 20 大類（Superclass）

| 大類（英） | 代表小類 |
|-----------|---------|
| aquatic mammals | 海狸、海豚、水獺、海豹、鯨魚 |
| fish | 比目魚、鱸魚、鯊魚、鱒魚、鯉魚 |
| flowers | 蘭花、罌粟花、玫瑰、向日葵、鬱金香 |
| food containers | 瓶子、碗、罐頭、杯子、盤子 |
| fruit and vegetables | 蘋果、蘑菇、橘子、梨子、甜椒 |
| household electrical devices | 時鐘、電腦鍵盤、燈、電話、電視 |
| household furniture | 床、椅子、沙發、桌子、衣櫃 |
| insects | 蜜蜂、甲蟲、蝴蝶、毛毛蟲、蟑螂 |
| large carnivores | 熊、豹、獅子、老虎、狼 |
| large man-made outdoor things | 橋、城堡、房屋、路、摩天樓 |
| large natural outdoor scenes | 雲、森林、山、平原、海 |
| large omnivores and herbivores | 駱駝、牛、黑猩猩、大象、袋鼠 |
| medium-sized mammals | 狐狸、豪豬、負鼠、浣熊、臭鼬 |
| non-insect invertebrates | 蟹、龍蝦、蝸牛、蜘蛛、蠕蟲 |
| people | 嬰兒、男孩、女孩、男人、女人 |
| reptiles | 鱷魚、恐龍、蜥蜴、蛇、烏龜 |
| small mammals | 倉鼠、老鼠、兔子、松鼠、鼩鼱 |
| trees | 楓樹、橡樹、棕櫚樹、松樹、柳樹 |
| vehicles 1 | 自行車、公車、摩托車、轎車、卡車 |
| vehicles 2 | 割草機、火箭、電車、坦克、拖拉機 |

### 前處理與資料增強

**訓練集**
```python
transforms.RandomCrop(32, padding=4)
transforms.RandomHorizontalFlip()
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)  # CIFAR-10 版無此項
transforms.ToTensor()
transforms.Normalize(
    mean=(0.5071, 0.4867, 0.4408),   # CIFAR-100 專屬統計值
    std=(0.2675, 0.2565, 0.2761)
)
```

**測試集**（無增強）
```python
transforms.ToTensor()
transforms.Normalize(
    mean=(0.5071, 0.4867, 0.4408),
    std=(0.2675, 0.2565, 0.2761)
)
```

---

## 模型架構

### ResidualBlock

與 CIFAR-10 版完全相同：兩層 3×3 卷積 + BN + shortcut 連接。

### ResNet（整體）

比 CIFAR-10 版多一層 Layer4（256→512），以容納 100 個分類的特徵容量需求：

```
輸入 (3 × 32 × 32)
│
├─ Conv2d(3→64, 3×3) + BN + ReLU           →  64 × 32 × 32
│
├─ Layer1: 2× ResidualBlock(64→64,  s=1)   →  64 × 32 × 32
├─ Layer2: 2× ResidualBlock(64→128, s=2)   →  128 × 16 × 16
├─ Layer3: 2× ResidualBlock(128→256,s=2)   →  256 × 8 × 8
├─ Layer4: 2× ResidualBlock(256→512,s=2)   →  512 × 4 × 4   ← 新增層
│
├─ AdaptiveAvgPool2d(1,1)                   →  512 × 1 × 1
├─ Flatten                                  →  512
├─ Dropout(0.3)                             ← 新增，抑制過擬合
└─ Linear(512 → 100)
     輸出：100 個 logits
```

### 與 CIFAR-10 模型的差異

| 差異點 | CIFAR-10 ResNet | CIFAR-100 ResNet |
|--------|----------------|-----------------|
| 殘差層數 | 3 層（64→128→256） | 4 層（64→128→256→512） |
| 全連接前 Dropout | 無 | Dropout(0.3) |
| 最終 FC 輸出 | 256→10 | 512→100 |
| 參數量（約） | ~1.7M | ~6.6M |

---

## 訓練設定

| 超參數 | 值 |
|--------|----|
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Epochs | 30 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| LR Scheduler | CosineAnnealingLR(T\_max=30) |
| Dropout | 0.3（FC 前） |

### 為何改用 CosineAnnealingLR

StepLR 以固定步距階梯式降低 lr，在多分類任務中容易在降 lr 前後出現明顯的準確率震盪。CosineAnnealingLR 以餘弦曲線平滑地從 0.001 衰減至接近 0，整體訓練曲線更穩定，尤其在 epoch 15 ~ 30 的收斂階段效果更佳。

### 為何 Batch Size 從 256 降至 128

100 個類別代表每個 mini-batch 平均每類只有約 1.28 個樣本，梯度估計雜訊較大。縮小 batch size 讓每步更新涵蓋更多類別的代表性樣本，有助於收斂品質。

---

## 執行

```bash
# 直接執行
python src/cifar100.py

# 或透過腳本（需先建立 venv/）
bash cifar100.sh
```

---

## 預期結果

| Epoch | 大致 Test Acc |
|-------|--------------|
| 5     | ~35%         |
| 10    | ~48%         |
| 20    | ~58%         |
| 30    | ~62%+        |

CIFAR-100 的基線難度遠高於 CIFAR-10，本模組未使用預訓練權重，純從頭訓練。如需更高準確率，可考慮：

- 更深的 ResNet（ResNet-50 / WideResNet）
- 混合增強（Mixup / CutMix）
- 知識蒸餾（Knowledge Distillation）

---

## 三模型對比

| 項目 | MNIST | CIFAR-10 | CIFAR-100 |
|------|-------|----------|-----------|
| 類別數 | 10 | 10 | 100 |
| 圖片通道 | 1（灰階） | 3（RGB） | 3（RGB） |
| 解析度 | 28×28 | 32×32 | 32×32 |
| 模型 | 2 層 CNN | ResNet 3 層 | ResNet 4 層 |
| 參數量（約） | ~0.4M | ~1.7M | ~6.6M |
| Dropout | 0.5（FC） | 無 | 0.3（FC） |
| LR Scheduler | 無 | StepLR | CosineAnnealing |
| 資料增強 | 無 | Crop + HFlip | Crop + HFlip + ColorJitter |
| Batch Size | 128 | 256 | 128 |
| Epochs | 10 | 20 | 30 |
| 預期 Test Acc | ~99% | ~85% | ~62% |
