# GPU-MNIST

用 CNN 模型訓練 MNIST 手寫數字辨識，並說明如何讓訓練跑在 GPU 上。

## 這個程式在做什麼

1. 下載 MNIST 資料集（60,000 張訓練圖、10,000 張測試圖）
2. 建立一個兩層卷積的 CNN 模型
3. 用 Adam 優化器訓練 10 個 epoch
4. 每個 epoch 結束後輸出 loss 與準確率
5. 訓練完成後將模型存成 `mnist_cnn.pth`

## 如何讓模型跑在 GPU 上

PyTorch 用一行就能偵測 GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

有 NVIDIA GPU 且安裝了 CUDA 驅動時，`device` 會是 `cuda`，否則自動退回 `cpu`。

接著只要把**模型**和**資料**都搬到同一個 device 上即可：

```python
# 模型搬到 GPU
model = CNN().to(device)

# 每個 batch 的資料也要搬
images = images.to(device)
labels = labels.to(device)
```

這樣前向傳播、反向傳播、梯度更新全部都會在 GPU 上執行。

### DataLoader 的 pin_memory

```python
DataLoader(..., num_workers=4, pin_memory=True)
```

- `pin_memory=True`：將資料預先鎖定在記憶體，讓 CPU → GPU 的傳輸更快
- `num_workers=4`：用多個子行程預先載入資料，避免 GPU 等待

> **Windows 注意事項**：`num_workers > 0` 在 Windows 上需要把訓練程式包在 `if __name__ == '__main__':` 裡，否則會因為 spawn 機制重複執行腳本而出錯。

## 環境需求

- Python 3.8+
- PyTorch（建議安裝 CUDA 版本）
- torchvision
- tqdm

安裝指令：

```bash
pip install torch torchvision tqdm
```

CUDA 版 PyTorch 請至 [pytorch.org](https://pytorch.org) 依照作業系統與 CUDA 版本選擇對應的安裝指令。

## 執行

```bash
python src/app.py
```

有 GPU 時輸出會顯示：

```
使用設備: cuda
GPU名稱: NVIDIA GeForce ...
```

沒有 GPU 則顯示：

```
使用設備: cpu
```
