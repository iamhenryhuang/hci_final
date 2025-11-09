"""
情緒檢測系統 - 基於 DeepFace
整合版本 - 高準確度情緒識別
"""
import os
import cv2
import numpy as np
from collections import deque
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont


# ============================================
# 配置參數 (可在此調整)
# ============================================

# 檢測器選擇
# 選項: 'opencv' (快速), 'ssd', 'mtcnn' (準確,需安裝), 'retinaface' (最準確,需安裝), 'mediapipe'
DETECTOR_BACKEND = 'opencv'

# 平滑窗口大小 (幀數) - 越大越穩定但反應越慢
SMOOTH_WINDOW = 15

# 置信度閾值 (0-100) - 只顯示超過此值的情緒
CONFIDENCE_THRESHOLD = 30

# 處理間隔 (每 N 幀分析一次) - 越大越快但更新越慢
PROCESS_EVERY_N_FRAMES = 2

# 攝像頭解析度
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 情緒映射 (英文 -> 繁體中文)
EMOTION_MAP = {
    'angry': '😠 生氣',
    'disgust': '🤢 噁心',
    'fear': '😨 害怕',
    'happy': '😊 開心',
    'sad': '😢 難過',
    'surprise': '😲 驚訝',
    'neutral': '😐 平靜'
}


# ============================================
# 工具函數
# ============================================

def find_chinese_font():
    """尋找系統中的中文字體"""
    # Windows 常見中文字體
    fonts = [
        r'C:\Windows\Fonts\msjh.ttc',      # 微軟正黑體
        r'C:\Windows\Fonts\msjhbd.ttc',
        r'C:\Windows\Fonts\mingliu.ttc',   # 細明體
        r'C:\Windows\Fonts\simsun.ttc',    # 宋體
        r'C:\Windows\Fonts\simhei.ttf',    # 黑體
    ]
    
    for font in fonts:
        if os.path.exists(font):
            return font
    
    # 嘗試掃描字體目錄
    fonts_dir = r'C:\Windows\Fonts'
    try:
        if os.path.isdir(fonts_dir):
            for f in os.listdir(fonts_dir):
                lf = f.lower()
                if any(k in lf for k in ('noto', 'msj', 'ming', 'kai', 'sim', 'hei')):
                    full = os.path.join(fonts_dir, f)
                    if os.path.exists(full):
                        return full
    except Exception:
        pass
    
    return None


def draw_chinese_text(img, text, position, font_path, font_size=30, color=(255, 255, 255)):
    """在圖像上繪製中文文字"""
    if not font_path:
        # 如果沒有中文字體,使用 OpenCV (僅支援英文)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_size/30, color, 2)
        return img
    
    # 轉換為 PIL 圖像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 載入字體
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        return img
    
    # 繪製文字
    draw.text(position, text, font=font, fill=color)
    
    # 轉回 OpenCV 格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================
# 情緒檢測器類
# ============================================

class EmotionDetector:
    """情緒檢測器類 - 封裝所有情緒檢測邏輯"""
    
    def __init__(self, detector_backend='opencv', smooth_window=15, confidence_threshold=30):
        """
        初始化情緒檢測器
        
        Args:
            detector_backend: 檢測器類型 ('opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe')
            smooth_window: 平滑窗口大小 (建議 10-20)
            confidence_threshold: 置信度閾值 (建議 25-40)
        """
        self.detector_backend = detector_backend
        self.smooth_window = smooth_window
        self.confidence_threshold = confidence_threshold
        self.emotion_buffer = deque(maxlen=smooth_window)
        self.font_path = find_chinese_font()
        self.last_face_region = None
        
        print(f"初始化情緒檢測器...")
        print(f"  檢測器: {detector_backend}")
        print(f"  平滑窗口: {smooth_window} 幀")
        print(f"  置信度閾值: {confidence_threshold}%")
        
        # 預載入模型
        self._warmup()
    
    def _warmup(self):
        """預熱模型"""
        try:
            print("  正在載入 DeepFace 模型...")
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(
                dummy,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                silent=True
            )
            print("  ✓ 模型載入完成!")
        except Exception as e:
            print(f"  ⚠ 模型預載入警告: {e}")
    
    def analyze_frame(self, frame):
        """
        分析單幀圖像的情緒
        
        Args:
            frame: OpenCV BGR 格式的圖像
            
        Returns:
            dict: 包含情緒分析結果,或 None
        """
        try:
            # 使用 DeepFace 分析
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=True,
                detector_backend=self.detector_backend,
                silent=True
            )
            
            # 處理返回結果
            if isinstance(result, list):
                result = result[0]
            
            return result
            
        except Exception as e:
            # 沒有檢測到臉部或其他錯誤
            return None
    
    def get_smoothed_emotion(self):
        """
        獲取平滑後的情緒結果
        
        Returns:
            tuple: (情緒標籤, 置信度, 所有情緒分數) 或 None
        """
        if len(self.emotion_buffer) == 0:
            return None
        
        # 使用指數加權移動平均 (最近的結果權重更高)
        weights = np.exp(np.linspace(0, 2, len(self.emotion_buffer)))
        weights = weights / weights.sum()
        
        # 累加情緒分數
        emotion_sum = {}
        for i, emotion_dict in enumerate(self.emotion_buffer):
            for emotion, score in emotion_dict.items():
                if emotion not in emotion_sum:
                    emotion_sum[emotion] = 0
                emotion_sum[emotion] += score * weights[i]
        
        # 找出主導情緒
        dominant_emotion = max(emotion_sum.items(), key=lambda x: x[1])
        emotion_label, confidence = dominant_emotion
        
        # 只有當置信度超過閾值才返回
        if confidence >= self.confidence_threshold:
            return emotion_label, confidence, emotion_sum
        
        return None
    
    def process_frame(self, frame):
        """
        處理視頻幀並返回帶標註的圖像
        
        Args:
            frame: 原始 OpenCV 圖像
            
        Returns:
            annotated_frame: 帶情緒標註的圖像
        """
        # 分析情緒
        result = self.analyze_frame(frame)
        
        if result and 'emotion' in result:
            self.emotion_buffer.append(result['emotion'])
            self.last_face_region = result.get('region', {})
        
        # 繪製結果
        annotated = frame.copy()
        
        # 繪製臉部框
        if self.last_face_region:
            x = self.last_face_region.get('x', 0)
            y = self.last_face_region.get('y', 0)
            w = self.last_face_region.get('w', 0)
            h = self.last_face_region.get('h', 0)
            
            # 綠色框
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 獲取平滑後的情緒
        emotion_result = self.get_smoothed_emotion()
        
        # 繪製情緒信息
        self._draw_emotion_info(annotated, emotion_result)
        
        return annotated
    
    def _draw_emotion_info(self, img, emotion_result):
        """在圖像上繪製情緒信息"""
        h, w = img.shape[:2]
        
        # 創建半透明背景
        overlay = img.copy()
        
        if emotion_result:
            emotion_label, confidence, all_emotions = emotion_result
            
            # 主要情緒顯示區域
            main_height = 80
            cv2.rectangle(overlay, (0, 0), (w, main_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            # 主要情緒文字
            emotion_cn = EMOTION_MAP.get(emotion_label, emotion_label)
            main_text = f"{emotion_cn}  {confidence:.1f}%"
            
            if self.font_path:
                img[:] = draw_chinese_text(img, main_text, (20, 15), 
                                          self.font_path, font_size=40, 
                                          color=(255, 255, 255))
            else:
                cv2.putText(img, f"{emotion_label.upper()} {confidence:.1f}%", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # 詳細情緒分數 (右側)
            sorted_emotions = sorted(all_emotions.items(), key=lambda x: -x[1])
            
            detail_x = w - 280
            detail_y_start = 100
            bar_width = 250
            bar_height = 25
            
            for i, (emo, score) in enumerate(sorted_emotions[:5]):
                y_pos = detail_y_start + i * 40
                
                # 繪製進度條背景
                cv2.rectangle(img, (detail_x, y_pos), 
                            (detail_x + bar_width, y_pos + bar_height), 
                            (50, 50, 50), -1)
                
                # 繪製進度條
                bar_length = int((score / 100) * bar_width)
                color = (0, 255, 0) if emo == emotion_label else (100, 100, 100)
                cv2.rectangle(img, (detail_x, y_pos), 
                            (detail_x + bar_length, y_pos + bar_height), 
                            color, -1)
                
                # 情緒標籤
                emo_text = EMOTION_MAP.get(emo, emo)
                if self.font_path:
                    img[:] = draw_chinese_text(img, f"{emo_text} {score:.1f}%", 
                                              (detail_x + 5, y_pos + 2), 
                                              self.font_path, font_size=18,
                                              color=(255, 255, 255))
                else:
                    cv2.putText(img, f"{emo} {score:.0f}%", 
                               (detail_x + 5, y_pos + 18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        else:
            # 沒有檢測到情緒
            status_height = 60
            cv2.rectangle(overlay, (0, 0), (w, status_height), (0, 0, 100), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            buffer_size = len(self.emotion_buffer)
            if buffer_size == 0:
                status_text = "正在偵測臉部..."
            elif buffer_size < self.smooth_window // 2:
                status_text = f"正在收集數據 {buffer_size}/{self.smooth_window}"
            else:
                status_text = "置信度不足,請保持明顯表情"
            
            if self.font_path:
                img[:] = draw_chinese_text(img, status_text, (20, 12), 
                                          self.font_path, font_size=30,
                                          color=(255, 255, 255))
            else:
                cv2.putText(img, "Detecting...", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 底部信息欄
        info_text = f"Buffer: {len(self.emotion_buffer)}/{self.smooth_window} | Detector: {self.detector_backend}"
        cv2.rectangle(img, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(img, info_text, (10, h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ============================================
# 主程序
# ============================================

def main():
    """主程序"""
    print("="*60)
    print("情緒檢測系統 - 基於 DeepFace")
    print("="*60)
    print()
    
    # 初始化檢測器
    try:
        detector = EmotionDetector(
            detector_backend=DETECTOR_BACKEND,
            smooth_window=SMOOTH_WINDOW,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
    except Exception as e:
        print(f"初始化失敗: {e}")
        print("嘗試使用默認 opencv 檢測器...")
        detector = EmotionDetector(
            detector_backend='opencv',
            smooth_window=SMOOTH_WINDOW,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
    
    # 開啟攝像頭
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("錯誤: 無法開啟攝像頭")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("\n" + "="*60)
    print("系統啟動成功!")
    print("提示:")
    print("  - 面向攝像頭,保持表情明顯")
    print("  - 確保光線充足")
    print("  - 按 'q' 退出程序")
    print("="*60 + "\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝像頭畫面")
                break
            
            frame_count += 1
            
            # 每 N 幀處理一次
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                annotated_frame = detector.process_frame(frame)
            else:
                # 不處理時仍然繪製上次的結果
                annotated_frame = frame.copy()
                emotion_result = detector.get_smoothed_emotion()
                detector._draw_emotion_info(annotated_frame, emotion_result)
            
            # 顯示結果
            cv2.imshow('emotion detection system (q: quit)', annotated_frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n用戶中斷程序")
    
    finally:
        # 清理資源
        cap.release()
        cv2.destroyAllWindows()
        print("\n程序已關閉")


if __name__ == '__main__':
    main()
