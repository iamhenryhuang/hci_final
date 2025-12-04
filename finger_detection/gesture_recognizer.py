"""
手勢識別模組
負責根據手指角度與關鍵點座標判斷手勢類型
"""

from config import FINGER_BEND_THRESHOLD, THUMB_DIRECTION_THRESHOLD_RATIO
from geometry import calculate_hand_angles

class GestureRecognizer:
    def __init__(self):
        self.threshold = FINGER_BEND_THRESHOLD

    def recognize(self, landmarks):
        """
        識別手勢
        
        Args:
            landmarks: 21 個手部關鍵點座標列表 [(x, y), ...]
            
        Returns:
            str: 手勢名稱
        """
        if not landmarks:
            return ''
            
        # 計算手指角度
        angles = calculate_hand_angles(landmarks)
        
        f1, f2, f3, f4, f5 = angles
        threshold = self.threshold

        # 計算手部 bounding box 高度（用於正規化方向判斷）
        ys = [p[1] for p in landmarks]
        box_h = max(ys) - min(ys) if ys else 0

        # ---------------------------------------------------------
        # 1. 優先檢測特殊手勢 (Gang Sign)
        # ---------------------------------------------------------
        # Westside / Gang Sign 識別邏輯
        # 特徵：食指小指伸直 + (中指無名指交叉 OR 中指無名指緊貼)
        if f2 < threshold and f5 < threshold:
            # 取得關鍵點座標
            idx_tip = landmarks[8]   # 食指尖
            mid_tip = landmarks[12]  # 中指尖
            rng_tip = landmarks[16]  # 無名指尖
            mid_mcp = landmarks[9]   # 中指根
            rng_mcp = landmarks[13]  # 無名指根

            # 寬鬆檢查：中指與無名指只要「指尖高於指根」即可
            if mid_tip[1] < mid_mcp[1] and rng_tip[1] < rng_mcp[1]:
                # 判斷 A: 嚴格交叉檢測
                is_crossed = (mid_tip[0] - rng_tip[0]) * (mid_mcp[0] - rng_mcp[0]) < 0

                # 判斷 B: 視覺聚攏檢測
                dist_mid_rng = abs(mid_tip[0] - rng_tip[0])
                dist_idx_mid = abs(idx_tip[0] - mid_tip[0])
                is_touching = dist_mid_rng < (dist_idx_mid * 0.35)

                if is_crossed or is_touching:
                    return 'GangSign'

        # ---------------------------------------------------------
        # 2. 檢測其他手勢
        # ---------------------------------------------------------
        
        # 特殊不雅手勢：大拇指+中指+小指伸直，食指+無名指捲縮
        if f1<threshold and f2>=threshold and f3<threshold and f4>=threshold and f5<threshold:
            return 'thumb_mid_pinky'

        # 讚 / 倒讚：大拇指伸直且其他手指捲縮
        if f1<threshold and f2>=threshold and f3>=threshold and f4>=threshold and f5>=threshold:
            if len(landmarks) > 4:
                thumb_mcp = landmarks[2]
                thumb_tip = landmarks[4]
                dy = thumb_tip[1] - thumb_mcp[1]
                threshold_dy = (box_h * THUMB_DIRECTION_THRESHOLD_RATIO) if box_h > 0 else 10
                
                if dy > threshold_dy:
                    return 'bad!!!'   # 倒讚
                elif dy < -threshold_dy:
                    return 'good'     # 讚
                else:
                    return 'good'     # 保守判定
            return 'good'

        # 中指 (no!!!)
        if f1>=threshold and f2>=threshold and f3<threshold and f4>=threshold and f5>=threshold:
            return 'no!!!'

        # 搖滾 (ROCK!)
        if f1<threshold and f2<threshold and f3>=threshold and f4>=threshold and f5<threshold:
            return 'ROCK!'

        # 拳頭 (fist)
        if f1>=threshold and f2>=threshold and f3>=threshold and f4>=threshold and f5>=threshold:
            return 'fist'

        # OK
        if f1>=threshold and f2>=threshold and f3<threshold and f4<threshold and f5<threshold:
            return 'ok'
        
        # OK (另一種變體)
        if f1<threshold and f2>=threshold and f3<threshold and f4<threshold and f5<threshold:
            return 'ok'

        return ''
