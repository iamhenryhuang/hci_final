"""
視覺化模組
負責畫面繪製、馬賽克效果與文字顯示
"""

import cv2
import numpy as np
import mediapipe as mp
from config import (
    BBOX_PADDING_RATIO, BBOX_EXTRA_PADDING, BBOX_MIN_DIMENSION, BBOX_SMOOTH_ALPHA,
    MOSAIC_DOWN_SAMPLE_MIN, MOSAIC_DOWN_SAMPLE_MAX, MOSAIC_DOWN_SAMPLE_DIVISOR,
    TEXT_FONT_SCALE, TEXT_THICKNESS, TEXT_COLOR, TEXT_POSITION,
    WARNING_TEXT, WARNING_FONT_SCALE, WARNING_THICKNESS, WARNING_COLOR,
    WARNING_BG_COLOR, WARNING_BG_PADDING,
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE,
    FACE_MOSAIC_LEVEL, FACE_MOSAIC_WARNING_TEXT, FACE_MOSAIC_WARNING_COLOR,
    FACE_MOSAIC_WARNING_FONT_SCALE, FACE_MOSAIC_WARNING_THICKNESS
)

class Visualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.prev_bbox = None
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.lineType = cv2.LINE_AA

    def draw_landmarks(self, img, hand_landmarks):
        """繪製手部骨架"""
        self.mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

    def draw_gesture_text(self, img, text):
        """顯示手勢名稱"""
        if text:
            cv2.putText(img, text, TEXT_POSITION, self.fontFace, 
                      TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, self.lineType)

    def draw_stats(self, img, stats, threshold):
        """顯示統計資訊"""
        info_text = f"Bad Gestures: {stats['bad_gesture_count']}/{threshold}"
        cv2.putText(img, info_text, (10, 30), self.fontFace, 0.7, (255, 255, 0), 2, self.lineType)
        
        if stats['face_mosaic_enabled']:
            status_text = "Status: FACE MOSAIC ON"
            status_color = (0, 0, 255)  # 紅色
        else:
            status_text = f"Status: Normal ({stats['remaining_warnings']} warnings left)"
            status_color = (0, 255, 0)  # 綠色
        cv2.putText(img, status_text, (10, 60), self.fontFace, 0.6, status_color, 2, self.lineType)

    def apply_hand_mosaic(self, img, landmarks, fx, fy, w, h):
        """對手部區域應用馬賽克"""
        # 計算馬賽克區域
        pts = np.array(landmarks, dtype=np.int32)
        try:
            hull = cv2.convexHull(pts)
            x, y, w_box, h_box = cv2.boundingRect(hull)
        except Exception:
            x_min, x_max = min(fx), max(fx)
            y_min, y_max = min(fy), max(fy)
            x, y, w_box, h_box = x_min, y_min, x_max - x_min, y_max - y_min

        # Padding
        pad = int(max(w_box, h_box) * BBOX_PADDING_RATIO) + BBOX_EXTRA_PADDING
        x_min = max(0, x - pad)
        y_min = max(0, y - pad)
        x_max = min(w, x + w_box + pad)
        y_max = min(h, y + h_box + pad)

        # 最小尺寸保護
        if (x_max - x_min) < BBOX_MIN_DIMENSION:
            cx = (x_min + x_max) // 2
            x_min = cx - BBOX_MIN_DIMENSION // 2
            x_max = cx + BBOX_MIN_DIMENSION // 2
        if (y_max - y_min) < BBOX_MIN_DIMENSION:
            cy = (y_min + y_max) // 2
            y_min = cy - BBOX_MIN_DIMENSION // 2
            y_max = cy + BBOX_MIN_DIMENSION // 2

        # 邊界檢查
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)

        # Frame-to-frame 平滑
        if self.prev_bbox is not None:
            px1, py1, px2, py2 = self.prev_bbox
            alpha = BBOX_SMOOTH_ALPHA
            x_min = int(px1 * alpha + x_min * (1 - alpha))
            y_min = int(py1 * alpha + y_min * (1 - alpha))
            x_max = int(px2 * alpha + x_max * (1 - alpha))
            y_max = int(py2 * alpha + y_max * (1 - alpha))

        self.prev_bbox = (x_min, y_min, x_max, y_max)

        # 應用馬賽克
        if x_max > x_min and y_max > y_min:
            mosaic_w = x_max - x_min
            mosaic_h = y_max - y_min
            mosaic = img[y_min:y_max, x_min:x_max]
            
            down_w = max(MOSAIC_DOWN_SAMPLE_MIN, min(MOSAIC_DOWN_SAMPLE_MAX, mosaic_w // MOSAIC_DOWN_SAMPLE_DIVISOR))
            down_h = max(MOSAIC_DOWN_SAMPLE_MIN, min(MOSAIC_DOWN_SAMPLE_MAX, mosaic_h // MOSAIC_DOWN_SAMPLE_DIVISOR))
            
            try:
                mosaic = cv2.resize(mosaic, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
                mosaic = cv2.resize(mosaic, (mosaic_w, mosaic_h), interpolation=cv2.INTER_NEAREST)
                img[y_min:y_max, x_min:x_max] = mosaic
            except Exception:
                pass

        # 顯示警告文字
        self._draw_warning_box(img, x_min, y_min, w, h, WARNING_TEXT, WARNING_COLOR)

    def apply_face_mosaic(self, img, face_cascade):
        """對臉部區域應用馬賽克"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE
        )
        
        for (x, y, face_w, face_h) in faces:
            # 馬賽克處理
            face_mosaic = img[y:y+face_h, x:x+face_w]
            level = FACE_MOSAIC_LEVEL
            mh = max(1, int(face_h / level))
            mw = max(1, int(face_w / level))
            
            try:
                face_mosaic = cv2.resize(face_mosaic, (mw, mh), interpolation=cv2.INTER_LINEAR)
                face_mosaic = cv2.resize(face_mosaic, (face_w, face_h), interpolation=cv2.INTER_NEAREST)
                img[y:y+face_h, x:x+face_w] = face_mosaic
            except Exception:
                pass
            
            # 顯示警告
            self._draw_warning_box(img, x, y, img.shape[1], img.shape[0], 
                                 FACE_MOSAIC_WARNING_TEXT, FACE_MOSAIC_WARNING_COLOR, is_face=True)

    def _draw_warning_box(self, img, x, y, w_img, h_img, text, color, is_face=False):
        """繪製警告文字與背景"""
        if is_face:
            scale = FACE_MOSAIC_WARNING_FONT_SCALE
            thickness = FACE_MOSAIC_WARNING_THICKNESS
            offset_y = 10
        else:
            scale = WARNING_FONT_SCALE
            thickness = WARNING_THICKNESS
            offset_y = 10

        (tw, th), _ = cv2.getTextSize(text, self.fontFace, scale, thickness)
        
        tx = x
        ty = y - offset_y
        
        box_x1 = max(0, tx - WARNING_BG_PADDING)
        box_y1 = max(0, ty - th - WARNING_BG_PADDING)
        box_x2 = min(w_img, tx + tw + WARNING_BG_PADDING)
        box_y2 = min(h_img, ty + WARNING_BG_PADDING)
        
        # 黑色背景
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), WARNING_BG_COLOR, -1)
        
        # 文字
        cv2.putText(img, text, (tx, ty), self.fontFace, scale, color, thickness, self.lineType)
