# 手勢識別系統

這是一個基於 MediaPipe 和 OpenCV 的即時手勢識別系統，能夠識別多種手勢，並對不雅手勢自動進行馬賽克處理。

## 功能特色

-  **即時手勢識別**：支援多種手勢識別，包括數字 0-9、讚/倒讚、OK 手勢、搖滾手勢等
-  **自動馬賽克處理**：偵測到不雅手勢時自動進行馬賽克模糊處理
-  **靈活配置**：所有參數都可透過 `config.py` 輕鬆調整
-  **高精準度**：使用 MediaPipe 的手部追蹤技術，識別準確度高
-  **高效能**：即時處理，延遲低

## 支援的手勢

### 數字手勢
- **0-9**：拳頭、單指到五指張開等數字手勢

### 特殊手勢
- **good**：豎起大拇指（讚）
- **ROCK!**：搖滾手勢（食指與小指伸直）
- **ok**：OK 手勢

### 不雅手勢（會被馬賽克處理）
- **bad!!!**：倒讚（大拇指朝下）
- **no!!!**：比中指
- **thumb_mid_pinky**：大拇指+中指+小指同時伸直
- **ok**：部分設定可能將 OK 手勢列為敏感手勢

## 專案結構

```
hci_final/
├── demo.ipynb              # 原始 Jupyter Notebook（保留）
├── main.py                 # 主程式入口
├── utils.py                # 工具函數模組（角度計算、手勢判定）
├── config.py               # 配置檔案
├── requirements.txt        # Python 依賴套件
└── README.md              # 本說明文件
```

## 安裝步驟

### 1. 確認 Python 版本

請確保您的系統已安裝 Python 3.8 或更高版本：

```bash
python --version
```

### 2. 安裝依賴套件

在專案目錄下執行：

```bash
pip install -r requirements.txt
```

主要依賴套件包括：
- `mediapipe`：Google 的機器學習解決方案，用於手部追蹤
- `opencv-python`：影像處理函式庫
- `numpy`：數值運算函式庫

## 使用方法

### 基本使用

直接執行主程式：

```bash
python main.py
```

程式會自動開啟攝影機，並開始即時識別手勢。

### 退出程式

在視窗中按下 `q` 鍵即可退出程式。

## 配置說明

所有可調整的參數都在 `config.py` 中，您可以根據需求修改：

### 攝影機設定

```python
CAMERA_INDEX = 0        # 攝影機編號（0 為預設攝影機）
FRAME_WIDTH = 720       # 影像寬度
FRAME_HEIGHT = 540      # 影像高度
```

### MediaPipe 參數

```python
MODEL_COMPLEXITY = 0              # 模型複雜度（0: 快速, 1: 準確但較慢）
MIN_DETECTION_CONFIDENCE = 0.5    # 手部偵測信心閾值（0.0-1.0）
MIN_TRACKING_CONFIDENCE = 0.5     # 手部追蹤信心閾值（0.0-1.0）
```

### 手勢識別參數

```python
FINGER_BEND_THRESHOLD = 50        # 手指彎曲判定閾值（角度）
```

### 馬賽克效果設定

```python
BLACKLIST_GESTURES = ('no!!!', 'bad!!!', 'thumb_mid_pinky', 'ok')  # 需要馬賽克的手勢
DEBOUNCE_FRAMES = 3               # 連續出現多少幀才觸發馬賽克
BBOX_SMOOTH_ALPHA = 0.65          # 邊界框平滑係數（0.0-1.0）
```

### 顯示設定

```python
WINDOW_NAME = 'Hand Gesture Recognition'  # 視窗名稱
TEXT_FONT_SCALE = 5                       # 文字大小
WARNING_TEXT = 'BAD!!!'                   # 警告文字
EXIT_KEY = 'q'                            # 退出按鍵
```

## 技術說明

### 手勢識別原理

1. **手部偵測**：使用 MediaPipe 偵測手部的 21 個關鍵點
2. **角度計算**：根據關鍵點座標計算每根手指的彎曲角度
3. **手勢判定**：根據五根手指的角度組合判定手勢類型
4. **方向判定**：對於讚/倒讚手勢，額外判斷大拇指的朝向

### 馬賽克處理流程

1. **手勢確認**：連續數幀偵測到同一不雅手勢才觸發（避免誤判）
2. **區域計算**：計算手部的凸包（Convex Hull）並加上邊距
3. **邊界平滑**：使用指數移動平均（EMA）平滑邊界框，避免抖動
4. **馬賽克生成**：將區域縮小再放大，產生像素化效果
5. **警告顯示**：在馬賽克區域上方顯示警告文字

## 疑難排解

### 攝影機無法開啟

- 確認攝影機已正確連接
- 檢查其他程式是否正在使用攝影機
- 嘗試修改 `config.py` 中的 `CAMERA_INDEX`（改為 1、2 等）

### 識別不準確

- 增加 `MIN_DETECTION_CONFIDENCE` 和 `MIN_TRACKING_CONFIDENCE` 的值
- 調整 `FINGER_BEND_THRESHOLD` 參數
- 改善光線條件
- 確保手部完整出現在畫面中

### 馬賽克過於敏感

- 增加 `DEBOUNCE_FRAMES` 的值（需要更多連續幀才觸發）
- 調整 `BLACKLIST_GESTURES`，移除不需要馬賽克的手勢

### 馬賽克抖動嚴重

- 增加 `BBOX_SMOOTH_ALPHA` 的值（更平滑但反應稍慢）
- 減少 `DEBOUNCE_FRAMES` 避免延遲過大

## 系統需求

- **作業系統**：Windows 10/11、macOS 10.14+、Linux
- **Python**：3.8 或更高版本
- **記憶體**：建議 4GB 以上
- **攝影機**：任何 USB 或內建攝影機