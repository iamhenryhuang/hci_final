"""
手勢識別主程式
使用 MediaPipe 和 OpenCV 進行即時手勢識別，並對不雅手勢進行馬賽克處理
"""

import cv2
import mediapipe as mp
from backend import GestureTracker
from gesture_recognizer import GestureRecognizer
from visualizer import Visualizer
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    MODEL_COMPLEXITY, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    BLACKLIST_GESTURES, DEBOUNCE_FRAMES,
    BAD_GESTURE_THRESHOLD, GESTURE_LOG_FILE,
    FACE_CASCADE_NAME, EXIT_KEY
)

def main():
    """主程式入口"""
    # 1. 初始化各個模組
    tracker = GestureTracker(data_file=GESTURE_LOG_FILE)
    tracker.threshold = BAD_GESTURE_THRESHOLD
    
    recognizer = GestureRecognizer()
    visualizer = Visualizer()
    
    # 2. 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    
    # 3. 初始化攝影機
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # 4. 初始化臉部偵測器
    cascade_path = cv2.data.haarcascades + FACE_CASCADE_NAME
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"警告: 無法載入臉部偵測器: {cascade_path}")
        face_cascade = None

    # 顯示啟動資訊
    stats = tracker.get_statistics()
    print("=" * 50)
    print("手勢識別系統啟動中...")
    print(f"攝影機: {CAMERA_INDEX}")
    print(f"解析度: {FRAME_WIDTH} x {FRAME_HEIGHT}")
    print(f"今日不雅手勢次數: {stats['bad_gesture_count']}")
    print(f"按 '{EXIT_KEY}' 鍵退出程式")
    print("=" * 50)

    # 5. 啟動主迴圈
    with mp_hands.Hands(
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:

        if not cap.isOpened():
            print("錯誤：無法開啟攝影機")
            return

        w, h = FRAME_WIDTH, FRAME_HEIGHT
        
        # 狀態變數
        gesture_buffer_text = ''
        gesture_buffer_count = 0
        current_gesture_logged = False

        while True:
            ret, img = cap.read()
            if not ret:
                break
                
            img = cv2.resize(img, (w, h))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # -----------------------------------------------------
            # 手部偵測與識別
            # -----------------------------------------------------
            if results.multi_hand_landmarks:
                detections = []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # 繪製骨架
                    visualizer.draw_landmarks(img, hand_landmarks)
                    
                    # 轉換座標格式
                    landmarks = []
                    fx, fy = [], []
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        landmarks.append((x, y))
                        fx.append(x)
                        fy.append(y)
                    
                    # 識別手勢
                    gesture_name = recognizer.recognize(landmarks)
                    
                    detections.append({
                        'text': gesture_name,
                        'landmarks': landmarks,
                        'fx': fx,
                        'fy': fy
                    })

                # -----------------------------------------------------
                # 判斷是否觸發不雅手勢計數 (Debounce Logic)
                # -----------------------------------------------------
                frame_candidates = [d['text'] for d in detections if d['text'] in BLACKLIST_GESTURES]
                
                if frame_candidates:
                    candidate = max(set(frame_candidates), key=frame_candidates.count)
                    if candidate == gesture_buffer_text:
                        gesture_buffer_count += 1
                        if gesture_buffer_count >= DEBOUNCE_FRAMES and not current_gesture_logged:
                            tracker.add_bad_gesture(candidate)
                            current_gesture_logged = True
                    else:
                        gesture_buffer_text = candidate
                        gesture_buffer_count = 1
                        current_gesture_logged = False
                else:
                    gesture_buffer_text = ''
                    gesture_buffer_count = 0
                    current_gesture_logged = False

                # -----------------------------------------------------
                # 繪製結果 (文字或馬賽克)
                # -----------------------------------------------------
                for d in detections:
                    text = d['text']
                    
                    # 判斷是否需要馬賽克
                    should_mosaic = (text in BLACKLIST_GESTURES and 
                                   text == gesture_buffer_text and 
                                   gesture_buffer_count >= DEBOUNCE_FRAMES)
                    
                    if should_mosaic:
                        visualizer.apply_hand_mosaic(img, d['landmarks'], d['fx'], d['fy'], w, h)
                    else:
                        visualizer.draw_gesture_text(img, text)

            # -----------------------------------------------------
            # 臉部馬賽克處理 (若已達閾值)
            # -----------------------------------------------------
            if tracker.is_face_mosaic_enabled() and face_cascade is not None:
                visualizer.apply_face_mosaic(img, face_cascade)

            # -----------------------------------------------------
            # 顯示統計資訊與畫面
            # -----------------------------------------------------
            visualizer.draw_stats(img, tracker.get_statistics(), BAD_GESTURE_THRESHOLD)
            cv2.imshow('Hand Gesture Recognition', img)

            if cv2.waitKey(5) == ord(EXIT_KEY):
                print("\n程式結束")
                tracker.reset()
                break

    cap.release()
    cv2.destroyAllWindows()
    print("攝影機已關閉")

if __name__ == "__main__":
    main()