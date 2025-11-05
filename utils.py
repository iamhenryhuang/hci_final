"""
手勢識別工具函數模組
包含角度計算、手勢判定等核心功能
"""

import math
from config import FINGER_BEND_THRESHOLD, THUMB_DIRECTION_THRESHOLD_RATIO


def vector_2d_angle(v1, v2):
    """
    根據兩點的座標，計算角度
    
    參數:
        v1: 第一個向量 (x, y)
        v2: 第二個向量 (x, y)

    返回:
        angle: 兩向量之間的角度（度數）
    """
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]

    try:
        # 使用餘弦定理計算角度
        angle_ = math.degrees(
            math.acos(
                (v1_x*v2_x + v1_y*v2_y) / 
                (((v1_x**2 + v1_y**2)**0.5) * ((v2_x**2 + v2_y**2)**0.5))
            )
        )
    except:
        angle_ = 180

    return angle_


def hand_angle(hand_):
    """
    根據傳入的 21 個手部節點座標，計算五根手指的角度
    
    參數:
        hand_: 包含 21 個手部關鍵點的座標列表
    
    返回:
        angle_list: 五根手指的角度列表 [大拇指, 食指, 中指, 無名指, 小拇指]
    """
    angle_list = []
    
    # 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)
    
    # 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    
    # 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    
    # 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    
    # 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    
    return angle_list


def hand_pos(finger_angle, finger_points=None):
    """
    根據手指角度的列表內容，返回對應的手勢名稱

    參數:
        finger_angle: 五根手指的角度列表
        finger_points: 可選，21 個手部關鍵點座標，用於判斷朝向（例如倒讚判斷大拇指朝上或朝下）

    返回:
        str: 手勢名稱

    角度規則: 小於 FINGER_BEND_THRESHOLD 表示手指伸直，大於等於表示手指捲縮
    """
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度

    threshold = FINGER_BEND_THRESHOLD

    # 計算手部 bounding box 高度（用於正規化方向判斷）
    box_h = None
    if finger_points is not None and len(finger_points) > 0:
        ys = [p[1] for p in finger_points]
        box_h = max(ys) - min(ys)

    # 新增：同時比大拇指 + 中指 + 小指（三指同時伸直），視為特殊不雅手勢
    # 條件：大拇指、中指、小指 伸直 (角度 < threshold)，且食指、無名指 捲縮 (角度 >= threshold)
    if f1<threshold and f2>=threshold and f3<threshold and f4>=threshold and f5<threshold:
        return 'thumb_mid_pinky'

    # 判斷讚 / 倒讚：大拇指伸直且其他手指捲縮
    if f1<threshold and f2>=threshold and f3>=threshold and f4>=threshold and f5>=threshold:
        # 若有座標資料，使用大拇指指尖與大拇指 MCP 的 y 差值（相對於手部高度）判斷朝向
        if finger_points is not None and len(finger_points) > 4:
            # 使用 thumb_mcp (index 2) 與 thumb_tip (index 4)
            thumb_mcp = finger_points[2]
            thumb_tip = finger_points[4]
            dy = thumb_tip[1] - thumb_mcp[1]
            # 使用相對手部高度作為閾值，避免不同距離造成差異
            threshold_dy = (box_h * THUMB_DIRECTION_THRESHOLD_RATIO) if box_h and box_h>0 else 10
            if dy > threshold_dy:
                return 'bad!!!'   # 指尖比 MCP 更往下 -> 倒讚
            elif dy < -threshold_dy:
                return 'good'      # 指尖比 MCP 更往上 -> 讚
            else:
                # 不明顯的情況，回傳 'good'（較保守）
                return 'good'
        else:
            # 沒有座標資訊時，保守回傳 'good'
            return 'good'

    # 中指 (no) 判定
    if f1>=threshold and f2>=threshold and f3<threshold and f4>=threshold and f5>=threshold:
        return 'no!!!'       # 比中指（會加馬賽克）
    elif f1<threshold and f2<threshold and f3>=threshold and f4>=threshold and f5<threshold:
        return 'ROCK!'       # 搖滾
    elif f1>=threshold and f2>=threshold and f3>=threshold and f4>=threshold and f5>=threshold:
        return '0'           # 拳頭
    elif f1>=threshold and f2<threshold and f3>=threshold and f4>=threshold and f5>=threshold:
        return '1'           # 1
    elif f1>=threshold and f2<threshold and f3<threshold and f4>=threshold and f5>=threshold:
        return '2'           # 2
    elif f1>=threshold and f2>=threshold and f3<threshold and f4<threshold and f5<threshold:
        return 'ok'          # OK
    elif f1<threshold and f2>=threshold and f3<threshold and f4<threshold and f5<threshold:
        return 'ok'          # OK (另一種)
    elif f1>=threshold and f2<threshold and f3<threshold and f4<threshold and f5>threshold:
        return '3'           # 3
    elif f1>=threshold and f2<threshold and f3<threshold and f4<threshold and f5<threshold:
        return '4'           # 4
    elif f1<threshold and f2<threshold and f3<threshold and f4<threshold and f5<threshold:
        return '5'           # 5
    elif f1<threshold and f2>=threshold and f3>=threshold and f4>=threshold and f5<threshold:
        return '6'           # 6
    elif f1<threshold and f2<threshold and f3>=threshold and f4>=threshold and f5>=threshold:
        return '7'           # 7
    elif f1<threshold and f2<threshold and f3<threshold and f4>=threshold and f5>=threshold:
        return '8'           # 8
    elif f1<threshold and f2<threshold and f3<threshold and f4<threshold and f5>=threshold:
        return '9'           # 9

    else:
        return ''            # 無法識別

