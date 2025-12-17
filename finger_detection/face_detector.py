"""
臉部偵測模組
負責載入 Haar Cascade 模型並進行臉部偵測
"""

import cv2
import mediapipe as mp
from config import (
    FACE_DETECTION_MIN_CONFIDENCE,
    FACE_DETECTION_MODEL_SELECTION
)

class FaceDetector:
    """處理臉部偵測的類 (使用 MediaPipe)"""
    
    def __init__(self):
        """初始化臉部偵測器"""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=FACE_DETECTION_MIN_CONFIDENCE,
            model_selection=FACE_DETECTION_MODEL_SELECTION
        )
        self.valid = True

    def detect(self, img):
        """
        偵測影格中的臉部
        
        Args:
            img: BGR 格式的影像陣列
            
        Returns:
            list: 偵測到的臉部矩形列表 [(x, y, w, h), ...]
        """
        if not self.valid:
            return []
            
        # MediaPipe 需要 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        
        faces = []
        if results.detections:
            h, w, _ = img.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                # 轉換為像素座標
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # 邊界保護
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)
                
                faces.append((x, y, width, height))
        
        return faces
