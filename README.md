# AI-CUP-2025-Team_7531
## 專案說明與環境安裝

### 1. 執行環境
- 作業系統：Windows 11
- 開發語言：Python 3.10.14
- 運算資源：NVIDIA RTX 4070 SUPER GPU（CUDA 11.8）

### 2. 主要套件安裝
請於專案資料夾下執行以下指令安裝所需套件：
```bash
pip install torch==2.6.0+cu118 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-optimizer==0.3.0
pip install tsai==0.4.0
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install scikit-learn==1.6.1
pip install tqdm==4.66.4
```

### 3. 重要模塊輸入／輸出說明
#### 資料格式
- 訓練資訊：train_info.csv 或 train_info_augmented.csv
- 測試資訊：test_info.csv
- 感測器數據：每筆對應一個 .txt 檔，存放於 train_data/、test_data/ 目錄下

#### 主程式參數
|參數名稱|說明|範例|
|:--:|:--:|:--:|
| `--train_info` | 訓練集 CSV 檔案 | `train_info_augmented.csv`|
| `--train_data` | 訓練資料資料夾路徑  | `train_data_augmented_level_year`|
| `--test_info`  | 測試集 CSV 檔案 | `test_info.csv`|
| `--test_data`  | 測試資料資料夾路徑  |`test_data`|
| `--mode`       | 執行模式       |`train_val`, `train_full`, `test` |
| `--epochs`     | 訓練回合數     | `100`|
| `--batch`      | 批次大小       | `64`|
| `--lr`         | 學習率         | `1e-3`|
| `--device`     | 運算裝置       | `cuda`, `cpu`|
| `--model`      | 欲儲存/測試的模型名稱    |`best_inception_model.pt`|

#### 重要輸出
- 模型權重：best_inception_model.pt
- 預測結果：submission.csv（依競賽要求格式）
- 訓練曲線圖：training_curves.png（顯示 Loss、Accuracy、AUC）

### 4. 專案架構
```bash
├── main.py                  # 主訓練／推論流程
├── model.py                 # InceptionTime 多分支模型定義
├── dataset.py               # 資料集處理
├── aug.py                   # 資料增強腳本（可選）
└── README.md                # 說明文件
```

### 5. 執行流程範例
#### 訓練（訓練＋驗證模式）
```bash
python main.py --mode train_val
```
#### 全資料集訓練（提交用）
```bash
python main.py --mode train_full
```
#### 測試集推論
```bash
python main.py --mode test
```
### 6. 重現結果說明
- 所有程式可於上述安裝環境下直接運行。
- 訓練、驗證、推論及結果儲存請參照主程式說明及預設參數，亦可依需求調整 batch size、learning rate、epochs 等超參數。

