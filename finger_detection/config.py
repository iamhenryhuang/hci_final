"""
手勢識別系統配置檔案
可以根據需求調整各項參數
"""

# ==================== 攝影機設置 ====================
CAMERA_INDEX = 0  # 攝影機編號，0 為預設攝影機
FRAME_WIDTH = 720  # 影像寬度
FRAME_HEIGHT = 540  # 影像高度

# ==================== MediaPipe 設置 ====================
# 模型複雜度：0(最快), 1(較準確但較慢)
MODEL_COMPLEXITY = 0
# 手部偵測信心閾值 (0.0 ~ 1.0)
MIN_DETECTION_CONFIDENCE = 0.5
# 手部追蹤信心閾值 (0.0 ~ 1.0)
MIN_TRACKING_CONFIDENCE = 0.5

# ==================== 手勢識別參數 ====================
# 手指彎曲判定閾值（角度小於此值視為伸直，大於等於此值視為彎曲）
FINGER_BEND_THRESHOLD = 50

# 讚/倒讚判定時，使用的相對高度閾值比例
THUMB_DIRECTION_THRESHOLD_RATIO = 0.12

# ==================== 馬賽克效果設置 ====================
# 需要馬賽克處理的不雅手勢列表
BLACKLIST_GESTURES = ('no!!!', 'bad!!!', 'thumb_mid_pinky', 'ok')

# 多幀確認（debounce）參數：要求連續出現多少幀才觸發馬賽克
DEBOUNCE_FRAMES = 3

# Bounding box padding 比例
BBOX_PADDING_RATIO = 0.20
BBOX_EXTRA_PADDING = 8

# Bounding box 最小尺寸（像素）
BBOX_MIN_DIMENSION = 50

# Frame-to-frame 平滑係數（0.0 ~ 1.0，越接近 1 越穩定但延遲越大）
BBOX_SMOOTH_ALPHA = 0.65

# 馬賽克下採樣尺寸範圍
MOSAIC_DOWN_SAMPLE_MIN = 8
MOSAIC_DOWN_SAMPLE_MAX = 16
MOSAIC_DOWN_SAMPLE_DIVISOR = 4

# ==================== 顯示設置 ====================
# 視窗名稱
WINDOW_NAME = 'Hand Gesture Recognition'

# 文字顯示設置
TEXT_FONT_SCALE = 5
TEXT_THICKNESS = 10
TEXT_COLOR = (255, 255, 255)  # 白色

# 警告文字設置（馬賽克時顯示）
WARNING_TEXT = 'BAD!!!'
WARNING_FONT_SCALE = 1.2
WARNING_THICKNESS = 4
WARNING_COLOR = (0, 0, 255)  # 紅色
WARNING_BG_COLOR = (0, 0, 0)  # 黑色底
WARNING_BG_PADDING = 6

# 文字顯示位置
TEXT_POSITION = (30, 120)

# ==================== 後台追蹤設置 ====================
# 觸發臉部馬賽克的不雅手勢次數閾值
BAD_GESTURE_THRESHOLD = 5

# 手勢追蹤記錄檔案
GESTURE_LOG_FILE = 'gesture_log.json'

# ==================== 臉部偵測與馬賽克設置 ====================
# 臉部偵測 Haar Cascade 檔案名稱
FACE_CASCADE_NAME = 'haarcascade_frontalface_default.xml'

# 臉部偵測參數
FACE_SCALE_FACTOR = 1.1  # 偵測縮放比例
FACE_MIN_NEIGHBORS = 5    # 最小鄰居數（越大越嚴格）
FACE_MIN_SIZE = (30, 30)  # 最小臉部尺寸

# 臉部馬賽克效果等級（數字越大馬賽克效果越粗糙）
FACE_MOSAIC_LEVEL = 15

# 臉部馬賽克警告文字
FACE_MOSAIC_WARNING_TEXT = 'FACE BLOCKED!'
FACE_MOSAIC_WARNING_COLOR = (0, 0, 255)  # 紅色
FACE_MOSAIC_WARNING_FONT_SCALE = 0.8
FACE_MOSAIC_WARNING_THICKNESS = 2

# ==================== 其他設置 ====================
# 退出按鍵
EXIT_KEY = 'q'