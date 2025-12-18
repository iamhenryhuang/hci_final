# 手勢識別系統

這是一個用 MediaPipe 和 OpenCV 做的即時手勢識別系統，可以偵測手勢，如果比出不雅手勢會自動打馬賽克，累積太多次還會把臉也打馬賽克。

## 功能

- 即時識別手勢（讚、倒讚、OK、搖滾手勢、拳頭、比中指等）
- 偵測到不雅手勢會自動打馬賽克
- 會記錄不雅手勢次數，累積到 5 次就會把臉也打馬賽克
- 超過 10 次會進入 Shut Down 模式（畫面全黑）

## 支援的手勢

**一般手勢：**
- `good` - 豎大拇指（讚）
- `ROCK!` - 搖滾手勢
- `fist` - 拳頭

**不雅手勢（會被馬賽克）：**
- `bad!!!` - 倒讚
- `no!!!` - 比中指
- `thumb_mid_pinky` - 大拇指+中指+小指
- `ok` - OK 手勢

## 安裝

先確認有 Python 3.8 以上：

```bash
python --version
```

然後安裝需要的套件：

```bash
pip install -r requirements.txt
```

主要會用到：
- `mediapipe` - 手部追蹤
- `opencv-python` - 影像處理
- `numpy` - 數值運算

## 使用方式

進到 `finger_detection` 資料夾，執行主程式：

```bash
cd finger_detection
python main.py
```

程式會自動開啟攝影機開始偵測。比出不雅手勢會被馬賽克，左上角會顯示累積次數。按 `q` 可以退出程式。

## 專案結構

```
hci_final/
├── finger_detection/          # 手勢識別主程式
│   ├── main.py                # 主程式
│   ├── gesture_tracker.py     # 手勢計數追蹤
│   ├── gesture_recognizer.py  # 手勢識別邏輯
│   ├── visualizer.py          # 畫面顯示和馬賽克
│   ├── face_detector.py       # 臉部偵測
│   ├── geometry.py            # 手指角度計算
│   └── config.py              # 各種設定參數
├── face_detection/            # 臉部偵測模組
│   └── face_mosaic.py
└── requirements.txt           # 套件清單
```

## 設定

所有參數都在 `finger_detection/config.py` 可以調整：

- `CAMERA_INDEX` - 攝影機編號（預設 0）
- `BAD_GESTURE_THRESHOLD` - 觸發臉部馬賽克的次數（預設 5 次）
- `DEBOUNCE_FRAMES` - 連續幾幀才判定為手勢（預設 3）
- 其他 MediaPipe 和顯示相關的參數

## 注意事項

- 程式會自動記錄每日的不雅手勢次數到 `gesture_log.json`
- 每天會自動重置計數
- 按 `q` 退出時會重置計數器
