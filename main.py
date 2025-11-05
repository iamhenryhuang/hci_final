"""
手勢識別主程式
使用 MediaPipe 和 OpenCV 進行即時手勢識別，並對不雅手勢進行馬賽克處理
"""

import cv2
import mediapipe as mp
import numpy as np
from utils import hand_angle, hand_pos
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    MODEL_COMPLEXITY, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    BLACKLIST_GESTURES, DEBOUNCE_FRAMES,
    BBOX_PADDING_RATIO, BBOX_EXTRA_PADDING, BBOX_MIN_DIMENSION,
    BBOX_SMOOTH_ALPHA,
    MOSAIC_DOWN_SAMPLE_MIN, MOSAIC_DOWN_SAMPLE_MAX, MOSAIC_DOWN_SAMPLE_DIVISOR,
    WINDOW_NAME, TEXT_FONT_SCALE, TEXT_THICKNESS, TEXT_COLOR, TEXT_POSITION,
    WARNING_TEXT, WARNING_FONT_SCALE, WARNING_THICKNESS, WARNING_COLOR,
    WARNING_BG_COLOR, WARNING_BG_PADDING,
    EXIT_KEY
)


def main():
    """主程式入口"""
    # 初始化 MediaPipe 手部偵測工具
    mp_hands = mp.solutions.hands

    # 初始化攝影機
    cap = cv2.VideoCapture(CAMERA_INDEX)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 文字字型
    lineType = cv2.LINE_AA               # 文字邊框

    print("=" * 50)
    print("手勢識別系統啟動中...")
    print("=" * 50)
    print(f"攝影機: {CAMERA_INDEX}")
    print(f"解析度: {FRAME_WIDTH} x {FRAME_HEIGHT}")
    print(f"按 '{EXIT_KEY}' 鍵退出程式")
    print("=" * 50)

    # 啟用 MediaPipe 手部偵測
    with mp_hands.Hands(
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:

        if not cap.isOpened():
            print("錯誤：無法開啟攝影機")
            return

        w, h = FRAME_WIDTH, FRAME_HEIGHT  # 影像尺寸

        # 用於 frame-to-frame 平滑的先前 bounding box
        prev_bbox = None
        alpha = BBOX_SMOOTH_ALPHA

        # 多幀確認（debounce）狀態
        gesture_buffer_text = ''
        gesture_buffer_count = 0

        while True:
            ret, img = cap.read()
            
            if not ret:
                print("錯誤：無法讀取影像")
                break
                
            img = cv2.resize(img, (w, h))  # 縮小尺寸，加快處理效率

            # 轉換成 RGB 色彩供 MediaPipe 處理
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img2)

            # 如果偵測到手部
            if results.multi_hand_landmarks:
                detections = []  # 暫存每隻手的判斷結果，稍後依 debounce 決定是否馬賽克

                for hand_landmarks in results.multi_hand_landmarks:
                    finger_points = []  # 記錄手指節點座標
                    fx = []             # 記錄所有 x 座標
                    fy = []             # 記錄所有 y 座標

                    # 取得 21 個手部關鍵點
                    for i in hand_landmarks.landmark:
                        x = int(i.x * w)
                        y = int(i.y * h)
                        finger_points.append((x, y))
                        fx.append(x)
                        fy.append(y)

                    if finger_points:
                        # 計算手指角度
                        finger_angle = hand_angle(finger_points)

                        # 識別手勢（傳入座標以便判斷方向）
                        text = hand_pos(finger_angle, finger_points)

                        detections.append({
                            'text': text,
                            'finger_points': finger_points,
                            'fx': fx,
                            'fy': fy,
                        })

                # 決定本幀的 candidate（若有多隻手發現不雅手勢，取出現次數最多的那個）
                frame_candidates = [d['text'] for d in detections if d['text'] in BLACKLIST_GESTURES]
                if frame_candidates:
                    # 選擇出現最多的 label
                    candidate = max(set(frame_candidates), key=frame_candidates.count)
                    if candidate == gesture_buffer_text:
                        gesture_buffer_count += 1
                    else:
                        gesture_buffer_text = candidate
                        gesture_buffer_count = 1
                else:
                    gesture_buffer_text = ''
                    gesture_buffer_count = 0

                # 根據 debounce 結果決定對每個偵測進行馬賽克或僅顯示文字
                for d in detections:
                    text = d['text']
                    finger_points = d['finger_points']
                    fx = d['fx']
                    fy = d['fy']

                    # 是否應該馬賽克：必須為 blacklist 且等於 buffer 且連續幀數已達閾值
                    should_mosaic = (text in BLACKLIST_GESTURES and 
                                    text == gesture_buffer_text and 
                                    gesture_buffer_count >= DEBOUNCE_FRAMES)

                    if should_mosaic:
                        # 計算馬賽克區域
                        pts = np.array(finger_points, dtype=np.int32)
                        try:
                            hull = cv2.convexHull(pts)
                            x, y, w_box, h_box = cv2.boundingRect(hull)
                        except Exception:
                            # fallback to min/max if convexHull fails
                            x_min = min(fx)
                            x_max = max(fx)
                            y_min = min(fy)
                            y_max = max(fy)
                            x, y, w_box, h_box = x_min, y_min, x_max - x_min, y_max - y_min

                        # padding 與最小尺寸
                        pad = int(max(w_box, h_box) * BBOX_PADDING_RATIO) + BBOX_EXTRA_PADDING
                        x_min = x - pad
                        y_min = y - pad
                        x_max = x + w_box + pad
                        y_max = y + h_box + pad

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
                        if x_max > w: x_max = w
                        if y_max > h: y_max = h
                        if x_min < 0: x_min = 0
                        if y_min < 0: y_min = 0

                        # frame-to-frame 平滑
                        if prev_bbox is not None:
                            px1, py1, px2, py2 = prev_bbox
                            x_min = int(px1 * alpha + x_min * (1 - alpha))
                            y_min = int(py1 * alpha + y_min * (1 - alpha))
                            x_max = int(px2 * alpha + x_max * (1 - alpha))
                            y_max = int(py2 * alpha + y_max * (1 - alpha))

                        prev_bbox = (x_min, y_min, x_max, y_max)

                        # 製作馬賽克效果（確保區域有效）
                        if x_max > x_min and y_max > y_min:
                            mosaic_w = x_max - x_min
                            mosaic_h = y_max - y_min
                            mosaic = img[y_min:y_max, x_min:x_max]
                            # 若區域太小，略微放大取樣以免變形
                            down_w = max(MOSAIC_DOWN_SAMPLE_MIN, 
                                       min(MOSAIC_DOWN_SAMPLE_MAX, 
                                           mosaic_w // MOSAIC_DOWN_SAMPLE_DIVISOR))
                            down_h = max(MOSAIC_DOWN_SAMPLE_MIN, 
                                       min(MOSAIC_DOWN_SAMPLE_MAX, 
                                           mosaic_h // MOSAIC_DOWN_SAMPLE_DIVISOR))
                            try:
                                mosaic = cv2.resize(mosaic, (down_w, down_h), 
                                                  interpolation=cv2.INTER_LINEAR)
                                mosaic = cv2.resize(mosaic, (mosaic_w, mosaic_h), 
                                                  interpolation=cv2.INTER_NEAREST)
                                img[y_min:y_max, x_min:x_max] = mosaic
                            except Exception:
                                pass

                        # 顯示警告文字
                        txt = WARNING_TEXT

                        # 將文字置於馬賽克上方，並加黑色底色提高可讀性
                        tx = x_min
                        ty = y_min - 10
                        (tw, th), _ = cv2.getTextSize(txt, fontFace, WARNING_FONT_SCALE, 
                                                      WARNING_THICKNESS)
                        box_x1 = max(0, tx - WARNING_BG_PADDING)
                        box_y1 = max(0, ty - th - WARNING_BG_PADDING)
                        box_x2 = min(w, tx + tw + WARNING_BG_PADDING)
                        box_y2 = min(h, ty + WARNING_BG_PADDING)
                        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), 
                                    WARNING_BG_COLOR, -1)
                        cv2.putText(img, txt, (tx, ty), fontFace, WARNING_FONT_SCALE, 
                                  WARNING_COLOR, WARNING_THICKNESS, lineType)
                    else:
                        # 尚未達 debounce 閾值，或並非 blacklist，顯示手勢文字（或不顯示）
                        if text:
                            cv2.putText(img, text, TEXT_POSITION, fontFace, 
                                      TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, lineType)

            # 顯示影像
            cv2.imshow(WINDOW_NAME, img)

            # 按退出鍵退出
            if cv2.waitKey(5) == ord(EXIT_KEY):
                print("\n程式結束")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("攝影機已關閉")
    print("感謝使用！")


if __name__ == "__main__":
    main()

