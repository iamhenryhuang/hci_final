"""
手勢識別主程式
使用 MediaPipe 和 OpenCV 進行即時手勢識別，
並對不雅手勢進行多段懲罰（警告音、高風險提示、Shut Down 全黑畫面）與馬賽克處理。
"""

import cv2
import numpy as np
import mediapipe as mp

from gesture_tracker import GestureTracker
from gesture_recognizer import GestureRecognizer
from visualizer import Visualizer
from face_detector import FaceDetector
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    MODEL_COMPLEXITY, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    BLACKLIST_GESTURES, DEBOUNCE_FRAMES,
    BAD_GESTURE_THRESHOLD, GESTURE_LOG_FILE,
    EXIT_KEY, WINDOW_NAME,
)


class EnhancedGestureTracker(GestureTracker):
    """
    只在本檔案內使用的「加強版追蹤器」：
    - 不修改原本 gesture_tracker.GestureTracker 檔案
    - 內部依照 bad_gesture_count 推出 penalty_level（normal / high_warning / shutdown）
    - add_bad_gesture() 回傳 dict：{"level_changed": bool, "penalty_level": str, "face_mosaic_enabled": bool}
    - get_statistics() 會在原本 stats 的基礎上加上 "penalty_level"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始懲罰等級
        self.penalty_level = "normal"

    def _update_penalty_level(self):
        """
        根據當前 bad_gesture_count 計算懲罰等級。
        - normal: 未達 BAD_GESTURE_THRESHOLD
        - high_warning: >= BAD_GESTURE_THRESHOLD
        - shutdown: >= BAD_GESTURE_THRESHOLD * 2  （可依需要再調整）
        """
        count = self.bad_gesture_count

        if count >= BAD_GESTURE_THRESHOLD * 2:
            new_level = "shutdown"
        elif count >= BAD_GESTURE_THRESHOLD:
            new_level = "high_warning"
        else:
            new_level = "normal"

        level_changed = new_level != self.penalty_level
        self.penalty_level = new_level
        return level_changed, new_level

    def add_bad_gesture(self, gesture_name):
        """
        包一層在原本 GestureTracker.add_bad_gesture 外面：
        - 先呼叫原本的 add_bad_gesture（維持原本計數 & 臉部馬賽克邏輯）
        - 再根據最新的 bad_gesture_count 算出 penalty_level
        - 回傳 dict，讓主程式可以使用 result['penalty_level']
        """
        # 呼叫原本的邏輯（會更新 bad_gesture_count / face_mosaic_enabled）
        face_mosaic_now = super().add_bad_gesture(gesture_name)

        # 重新計算懲罰等級
        level_changed, new_level = self._update_penalty_level()

        return {
            "level_changed": level_changed,
            "penalty_level": new_level,
            "face_mosaic_enabled": face_mosaic_now,
        }

    def get_statistics(self):
        """
        在原本統計資訊上補上一個 "penalty_level"，
        讓程式可以直接用 stats["penalty_level"] 判斷是否進入 shutdown。
        """
        stats = super().get_statistics()
        stats["penalty_level"] = self.penalty_level
        return stats


class GestureRecognitionApp:
    def __init__(self):
        """初始化應用程式"""
        # 1. 初始化各個模組（使用加強版追蹤器，原檔案不變）
        self.tracker = EnhancedGestureTracker(data_file=GESTURE_LOG_FILE)
        # 仍沿用原本閾值設定，確保臉部馬賽克門檻一致
        self.tracker.threshold = BAD_GESTURE_THRESHOLD

        self.recognizer = GestureRecognizer()
        self.visualizer = Visualizer()
        self.face_detector = FaceDetector()

        # 2. 初始化 MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        # 3. 初始化攝影機
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        # 4. 狀態變數（debounce 用）
        self.gesture_buffer_text = ""
        self.gesture_buffer_count = 0
        self.current_gesture_logged = False

        # 5. 是否進入 Shut Down 模式（全黑畫面）
        self.shutdown_mode = False

        self.print_startup_info()

    # ---------------------------------------------------------
    # 啟動資訊與嗶聲
    # ---------------------------------------------------------
    def print_startup_info(self):
        """顯示啟動資訊"""
        stats = self.tracker.get_statistics()
        print("=" * 50)
        print("手勢識別系統啟動中...")
        print(f"攝影機: {CAMERA_INDEX}")
        print(f"解析度: {FRAME_WIDTH} x {FRAME_HEIGHT}")
        print(f"今日不雅手勢次數: {stats['bad_gesture_count']}")
        print(f"按 '{EXIT_KEY}' 鍵退出程式")
        print("=" * 50)

    def _play_warning_beep(self):
        """高風險階段時嗶一聲（Windows 有效，其他平台就略過）"""
        try:
            import winsound
            winsound.Beep(1000, 200)
        except Exception:
            pass

    # ---------------------------------------------------------
    # 處理單一影格
    # ---------------------------------------------------------
    def process_frame(self, img):
        """
        回傳處理後的影像：
        - 若 shutdown_mode=True：直接回傳全黑「STREAM PAUSED」畫面
        - 否則：做手勢偵測、馬賽克與狀態顯示
        """
        # ========= Shut Down 模式：完全黑畫面 & 停止偵測 =========
        if self.shutdown_mode:
            black = np.zeros_like(img)
            text = "STREAM PAUSED"

            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4
            )
            cx, cy = img.shape[1] // 2, img.shape[0] // 2

            cv2.putText(
                black,
                text,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )
            return black
        # =====================================================

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        detections = []

        # ---------------- 手部偵測與手勢識別 ----------------
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 繪製骨架
                self.visualizer.draw_landmarks(img, hand_landmarks)

                # 轉成像素座標
                landmarks, fx, fy = [], [], []
                for lm in hand_landmarks.landmark:
                    x_px, y_px = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x_px, y_px))
                    fx.append(x_px)
                    fy.append(y_px)

                # 識別手勢
                gesture_name = self.recognizer.recognize(landmarks)

                detections.append(
                    {
                        "text": gesture_name,
                        "landmarks": landmarks,
                        "fx": fx,
                        "fy": fy,
                    }
                )

            # 更新不雅手勢狀態 & 計數
            self.update_gesture_status(detections)

            # 決定是否對手部做馬賽克
            # 需求：不要再顯示白色的 bad!!! / fist / good 等文字，只保留紅色的 bad / blocked（由馬賽克警告框顯示）
            for d in detections:
                text = d["text"]
                should_mosaic = (
                    text in BLACKLIST_GESTURES
                    and text == self.gesture_buffer_text
                    and self.gesture_buffer_count >= DEBOUNCE_FRAMES
                )

                if should_mosaic:
                    self.visualizer.apply_hand_mosaic(
                        img, d["landmarks"], d["fx"], d["fy"], w, h
                    )
                # 否則不畫任何手勢文字（避免出現白色 bad!!! / fist / good 等字）

        # ---------------- 臉部馬賽克（達到閾值後） ----------------
        if self.tracker.face_mosaic_enabled:
            faces = self.face_detector.detect(img)
            self.visualizer.draw_face_mosaic(img, faces)

        # ---------------- 狀態顯示 & 檢查是否進入 Shut Down ----------------
        stats = self.tracker.get_statistics()

        # 如果懲罰等級已經到 shutdown，就從下一幀開始全黑
        if stats["penalty_level"] == "shutdown":
            self.shutdown_mode = True

        self.visualizer.draw_stats(img, stats, BAD_GESTURE_THRESHOLD)

        return img

    # ---------------------------------------------------------
    # 更新不雅手勢計數（無 Shut Down 時才會動）
    # ---------------------------------------------------------
    def update_gesture_status(self, detections):
        """更新手勢狀態與計數（含多段懲罰邏輯）"""
        if self.shutdown_mode:
            return  # 已經 shut down 就不再計數

        frame_candidates = [
            d["text"] for d in detections if d["text"] in BLACKLIST_GESTURES
        ]

        if frame_candidates:
            # 同一幀可能兩隻手，比較常出現的那一種
            candidate = max(set(frame_candidates), key=frame_candidates.count)

            if candidate == self.gesture_buffer_text:
                self.gesture_buffer_count += 1

                if (
                    self.gesture_buffer_count >= DEBOUNCE_FRAMES
                    and not self.current_gesture_logged
                ):
                    # 真的算一次不雅手勢
                    result = self.tracker.add_bad_gesture(candidate)
                    self.current_gesture_logged = True

                    # 剛從 high_warning 這一階升級時嗶一聲
                    if result["level_changed"] and result["penalty_level"] == "high_warning":
                        self._play_warning_beep()
            else:
                # 換另一種手勢 → 重置 buffer
                self.gesture_buffer_text = candidate
                self.gesture_buffer_count = 1
                self.current_gesture_logged = False
        else:
            # 這一幀沒有任何黑名單手勢
            self.gesture_buffer_text = ""
            self.gesture_buffer_count = 0
            self.current_gesture_logged = False

    def run(self):
        """啟動主迴圈"""
        if not self.cap.isOpened():
            print("錯誤：無法開啟攝影機")
            return

        print("系統運行中...")

        try:
            while True:
                ret, img = self.cap.read()
                if not ret:
                    break

                img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

                # 處理畫面（含多段懲罰與 Shut Down 邏輯）
                img = self.process_frame(img)

                # 顯示畫面
                cv2.imshow(WINDOW_NAME, img)

                if cv2.waitKey(5) == ord(EXIT_KEY):
                    print("\n程式結束，重置計數")
                    self.tracker.reset()
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        """清理資源"""
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
        print("攝影機已關閉")


def main():
    app = GestureRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()