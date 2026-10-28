"""行为分类器 - 基于几何规则的智能行为识别

通过分析动物关键点的空间关系和运动模式来判断行为，
不依赖随机数，所有判断都基于可解释的几何特征。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import math

from utils.constants import BEHAVIORS


class RuleBasedBehaviorClassifier:
    """基于规则的行为分类器

    通过关键点几何特征（身体比例、关节角度、运动幅度）判断行为，
    使用滑动窗口平滑和稳定性机制减少误判。
    """

    def __init__(self):
        self.sequence_buffer: List[Dict] = []
        self.max_sequence_length = 8
        self.current_behavior = 'idle'
        self.behavior_stability = 0
        self.min_stability_frames = 2
        self.frame_count = 0

    def reset_buffer(self):
        self.sequence_buffer = []
        self.current_behavior = 'idle'
        self.behavior_stability = 0
        self.frame_count = 0

    def add_pose_features(self, keypoints_or_features):
        if isinstance(keypoints_or_features, dict):
            self.sequence_buffer.append(keypoints_or_features.copy())
        else:
            self.sequence_buffer.append({})

        if len(self.sequence_buffer) > self.max_sequence_length * 2:
            self.sequence_buffer = self.sequence_buffer[-self.max_sequence_length:]

        self.frame_count += 1

    def classify(self) -> Tuple[str, float]:
        if len(self.sequence_buffer) < 2:
            return 'idle', 0.0

        keypoints = self.sequence_buffer[-1] if self.sequence_buffer else {}

        if not keypoints:
            return 'idle', 0.3

        behavior, confidence = self._classify_from_keypoints(keypoints)

        if behavior == self.current_behavior:
            self.behavior_stability += 1
        else:
            self.behavior_stability = 0
            self.current_behavior = behavior

        if self.behavior_stability >= self.min_stability_frames:
            return behavior, confidence
        else:
            return behavior, confidence * 0.8

    def _classify_from_keypoints(self, keypoints: Dict) -> Tuple[str, float]:
        """基于关键点几何特征进行行为分类"""
        if not keypoints:
            return 'idle', 0.3

        try:
            # 提取关键点坐标
            left_eye = keypoints.get('left_eye', (0, 0, 0))
            right_eye = keypoints.get('right_eye', (0, 0, 0))
            nose = keypoints.get('nose', (0, 0, 0))
            left_ear = keypoints.get('left_ear', (0, 0, 0))
            right_ear = keypoints.get('right_ear', (0, 0, 0))

            left_front_paw = keypoints.get('left_front_paw', (0, 0, 0))
            right_front_paw = keypoints.get('right_front_paw', (0, 0, 0))
            left_back_paw = keypoints.get('left_back_paw', (0, 0, 0))
            right_back_paw = keypoints.get('right_back_paw', (0, 0, 0))

            left_front_shoulder = keypoints.get('left_front_shoulder', (0, 0, 0))
            right_front_shoulder = keypoints.get('right_front_shoulder', (0, 0, 0))
            left_back_shoulder = keypoints.get('left_back_shoulder', (0, 0, 0))
            right_back_shoulder = keypoints.get('right_back_shoulder', (0, 0, 0))

            left_front_elbow = keypoints.get('left_front_elbow', (0, 0, 0))
            right_front_elbow = keypoints.get('right_front_elbow', (0, 0, 0))
            left_back_elbow = keypoints.get('left_back_elbow', (0, 0, 0))
            right_back_elbow = keypoints.get('right_back_elbow', (0, 0, 0))

            # 计算几何特征
            body_height = self._body_height(keypoints)
            body_width = self._body_width(keypoints)
            paws_spread = self._paws_spread(keypoints)
            motion_level = self._detect_motion()
            head_angle = self._head_angle(nose, left_eye, right_eye)
            ear_spread = self._distance(left_ear, right_ear)
            has_ears = (len(left_ear) >= 2 and len(right_ear) >= 2
                        and left_ear[1] > 0 and right_ear[1] > 0)
            body_aspect = body_height / max(body_width, 0.01)

            # 计算前肢伸展程度（用于伸懒腰检测）
            front_leg_extension = self._front_leg_extension(keypoints)
            # 计算后肢伸展程度
            back_leg_extension = self._back_leg_extension(keypoints)

            # 检查关键点置信度
            avg_conf = self._average_confidence(keypoints)
            if avg_conf < 0.3:
                return 'idle', 0.3

            # --- 行为判断逻辑（按优先级排序）---

            # 1. 睡觉：身体扁平，几乎无运动
            if body_height < 0.15 and motion_level < 0.1:
                return 'sleeping', 0.90

            # 2. 伸懒腰：前肢大幅伸展，身体拉长
            if front_leg_extension > 0.6 and body_height > 0.4:
                return 'stretching', 0.85

            # 3. 坐着：身体高度中等，宽度较窄，低运动
            if 0.25 < body_height < 0.55 and body_aspect > 0.8 and motion_level < 0.15:
                return 'sitting', 0.80

            # 4. 行走：中等运动，四肢有规律位移
            if 0.15 < motion_level < 0.5:
                paw_motion = self._paw_motion_symmetry()
                if paw_motion > 0.3:
                    return 'walking', 0.75

            # 5. 奔跑：高运动幅度
            if motion_level > 0.5:
                return 'running', 0.80

            # 6. 玩耍：运动幅度大且不规则
            if motion_level > 0.35 and self._motion_irregularity() > 0.5:
                return 'playing', 0.75

            # 7. 踩奶：前爪有节律性开合运动
            if self._detect_kneading():
                return 'kneading', 0.80

            # 8. 理毛：头部靠近身体某部位，有小幅运动
            if self._detect_grooming(nose, left_front_paw, right_front_paw):
                return 'grooming', 0.75

            # 9. 捕猎：身体压低，高度集中，微小运动
            if body_height < 0.3 and 0.05 < motion_level < 0.2:
                return 'hunting', 0.70

            # 10. 飞机耳：耳朵间距异常小（耳朵向后压平）
            # AP-10K 等无耳部关键点的模型下跳过，避免 ear_spread=0 误判
            if has_ears and ear_spread < 0.08 and body_width > 0.2:
                return 'airplane_ears', 0.75

            # 11. 缓慢眨眼：眼睛区域有节律变化
            if self._detect_slow_blinking(left_eye, right_eye):
                return 'slow_blinking', 0.70

            # 12. 蹭物体：头部有明显的侧向运动
            if self._detect_head_bunting():
                return 'head_bunting', 0.70

            # 13. 露肚皮：身体高度极小，宽度大，仰卧姿态
            if body_height < 0.2 and body_width > 0.4:
                return 'belly_exposure', 0.80

            # 14. 尾巴竖起：尾部关键点位置高
            if self._detect_tail_raised(keypoints):
                return 'tail_raising', 0.70

            # 15. 尾巴甩动：尾部有快速运动
            if self._detect_tail_lashing():
                return 'tail_lashing', 0.70

            return 'idle', 0.50

        except Exception:
            return 'idle', 0.3

    def _distance(self, p1: Tuple, p2: Tuple) -> float:
        if len(p1) < 2 or len(p2) < 2:
            return 0.0
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _body_height(self, keypoints: Dict) -> float:
        """计算身体高度占比（身体顶点到后爪的距离 / 画面高度）

        优先使用耳朵；AP-10K 等动物姿态模型没有耳部关键点，
        此时回退到背线(withers)、鼻尖或眼睛估算身体顶点。
        """
        top_points = [keypoints.get('left_ear', (0, 0, 0)),
                      keypoints.get('right_ear', (0, 0, 0))]
        valid_top = [p for p in top_points if len(p) >= 2 and p[1] > 0]
        if not valid_top:
            fallback_points = [keypoints.get('withers', (0, 0, 0)),
                               keypoints.get('nose', (0, 0, 0)),
                               keypoints.get('left_eye', (0, 0, 0)),
                               keypoints.get('right_eye', (0, 0, 0))]
            valid_top = [p for p in fallback_points if len(p) >= 2 and p[1] > 0]

        bottom_points = [keypoints.get('left_back_paw', (0, 0, 0)),
                         keypoints.get('right_back_paw', (0, 0, 0))]

        valid_bottom = [p for p in bottom_points if len(p) >= 2 and p[1] > 0]

        if not valid_top or not valid_bottom:
            return 0.5

        top_y = min(p[1] for p in valid_top)
        bottom_y = max(p[1] for p in valid_bottom)

        return abs(bottom_y - top_y) / 480.0

    def _body_width(self, keypoints: Dict) -> float:
        """计算身体宽度占比"""
        left_points = [keypoints.get('left_front_shoulder', (0, 0, 0)),
                       keypoints.get('left_back_shoulder', (0, 0, 0))]
        right_points = [keypoints.get('right_front_shoulder', (0, 0, 0)),
                        keypoints.get('right_back_shoulder', (0, 0, 0))]

        valid_left = [p for p in left_points if len(p) >= 2 and p[0] > 0]
        valid_right = [p for p in right_points if len(p) >= 2 and p[0] > 0]

        if not valid_left or not valid_right:
            return 0.3

        left_x = min(p[0] for p in valid_left)
        right_x = max(p[0] for p in valid_right)

        return abs(right_x - left_x) / 640.0

    def _paws_spread(self, keypoints: Dict) -> float:
        """计算四肢展开程度"""
        left_front = keypoints.get('left_front_paw', (0, 0, 0))
        right_front = keypoints.get('right_front_paw', (0, 0, 0))
        left_back = keypoints.get('left_back_paw', (0, 0, 0))
        right_back = keypoints.get('right_back_paw', (0, 0, 0))

        front_spread = self._distance(left_front, right_front)
        back_spread = self._distance(left_back, right_back)

        return (front_spread + back_spread) / 2 / 640.0

    def _detect_motion(self) -> float:
        """检测帧间运动幅度（归一化到 0-1）"""
        if len(self.sequence_buffer) < 3:
            return 0.0

        motion = 0.0
        count = 0
        for i in range(1, min(4, len(self.sequence_buffer))):
            prev = self.sequence_buffer[-i - 1]
            curr = self.sequence_buffer[-i]

            for key in prev:
                if key in curr and len(prev[key]) >= 2 and len(curr[key]) >= 2:
                    motion += self._distance(prev[key], curr[key])
                    count += 1

        if count == 0:
            return 0.0

        return min(1.0, motion / (count * 50.0))

    def _head_angle(self, nose, left_eye, right_eye) -> float:
        """计算头部倾斜角度"""
        if len(nose) < 2 or len(left_eye) < 2 or len(right_eye) < 2:
            return 0.0
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        dx = nose[0] - eye_center_x
        dy = nose[1] - eye_center_y
        return math.atan2(dy, dx)

    def _front_leg_extension(self, keypoints: Dict) -> float:
        """计算前肢伸展程度（肩-肘-爪的角度）"""
        shoulder = keypoints.get('left_front_shoulder', (0, 0, 0))
        elbow = keypoints.get('left_front_elbow', (0, 0, 0))
        paw = keypoints.get('left_front_paw', (0, 0, 0))

        if len(shoulder) < 2 or len(elbow) < 2 or len(paw) < 2:
            return 0.0

        # 计算肘部角度：角度越大说明伸展越充分
        v1 = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        v2 = (paw[0] - elbow[0], paw[1] - elbow[1])

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 * mag2 == 0:
            return 0.0

        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        angle = math.acos(cos_angle)

        # 角度接近 π 说明完全伸展
        return angle / math.pi

    def _back_leg_extension(self, keypoints: Dict) -> float:
        """计算后肢伸展程度"""
        shoulder = keypoints.get('left_back_shoulder', (0, 0, 0))
        elbow = keypoints.get('left_back_elbow', (0, 0, 0))
        paw = keypoints.get('left_back_paw', (0, 0, 0))

        if len(shoulder) < 2 or len(elbow) < 2 or len(paw) < 2:
            return 0.0

        v1 = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        v2 = (paw[0] - elbow[0], paw[1] - elbow[1])

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 * mag2 == 0:
            return 0.0

        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        angle = math.acos(cos_angle)
        return angle / math.pi

    def _paw_motion_symmetry(self) -> float:
        """检测四肢运动的对称性（行走特征：对角肢同步）"""
        if len(self.sequence_buffer) < 3:
            return 0.0

        prev = self.sequence_buffer[-2]
        curr = self.sequence_buffer[-1]

        paw_keys = ['left_front_paw', 'right_front_paw', 'left_back_paw', 'right_back_paw']
        motions = {}

        for key in paw_keys:
            if key in prev and key in curr and len(prev[key]) >= 2 and len(curr[key]) >= 2:
                motions[key] = self._distance(prev[key], curr[key])

        if len(motions) < 4:
            return 0.0

        # 对角肢运动差（行走时对角肢应同步）
        diag1_diff = abs(motions.get('left_front_paw', 0) - motions.get('right_back_paw', 0))
        diag2_diff = abs(motions.get('right_front_paw', 0) - motions.get('left_back_paw', 0))

        avg_motion = sum(motions.values()) / len(motions)
        if avg_motion < 1:
            return 0.0

        symmetry = 1.0 - (diag1_diff + diag2_diff) / (2 * avg_motion * 2)
        return max(0, symmetry)

    def _motion_irregularity(self) -> float:
        """检测运动不规则性（玩耍特征：随机方向变化）"""
        if len(self.sequence_buffer) < 4:
            return 0.0

        motions = []
        for i in range(1, min(4, len(self.sequence_buffer))):
            prev = self.sequence_buffer[-i - 1]
            curr = self.sequence_buffer[-i]
            total = 0
            count = 0
            for key in prev:
                if key in curr and len(prev[key]) >= 2 and len(curr[key]) >= 2:
                    total += self._distance(prev[key], curr[key])
                    count += 1
            if count > 0:
                motions.append(total / count)

        if len(motions) < 2:
            return 0.0

        # 计算运动变化的方差
        mean_motion = sum(motions) / len(motions)
        if mean_motion < 1:
            return 0.0

        variance = sum((m - mean_motion) ** 2 for m in motions) / len(motions)
        return min(1.0, variance / (mean_motion ** 2))

    def _detect_kneading(self) -> bool:
        """检测踩奶行为：前爪有节律性的开合运动"""
        if len(self.sequence_buffer) < 4:
            return False

        # 检测前爪距离的周期性变化
        distances = []
        for i in range(min(4, len(self.sequence_buffer))):
            kp = self.sequence_buffer[-(i + 1)]
            lf = kp.get('left_front_paw', (0, 0, 0))
            rf = kp.get('right_front_paw', (0, 0, 0))
            if len(lf) >= 2 and len(rf) >= 2:
                distances.append(self._distance(lf, rf))

        if len(distances) < 3:
            return False

        # 检测距离是否有先增后减或先减后增的模式
        diffs = [distances[i] - distances[i + 1] for i in range(len(distances) - 1)]
        sign_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)

        # 有节律性变化且运动幅度不大
        avg_dist = sum(distances) / len(distances)
        return sign_changes >= 1 and avg_dist < 200

    def _detect_grooming(self, nose, left_front_paw, right_front_paw) -> bool:
        """检测理毛行为：头部靠近前爪"""
        if len(nose) < 2:
            return False

        for paw in [left_front_paw, right_front_paw]:
            if len(paw) >= 2:
                dist = self._distance(nose, paw)
                if dist < 80:  # 头部非常靠近前爪
                    return True

        return False

    def _detect_slow_blinking(self, left_eye, right_eye) -> bool:
        """检测缓慢眨眼：眼睛置信度有节律变化"""
        if len(self.sequence_buffer) < 3:
            return False

        eye_confs = []
        for i in range(min(3, len(self.sequence_buffer))):
            kp = self.sequence_buffer[-(i + 1)]
            le = kp.get('left_eye', (0, 0, 0))
            re = kp.get('right_eye', (0, 0, 0))
            avg = ((le[2] if len(le) > 2 else 0) + (re[2] if len(re) > 2 else 0)) / 2
            eye_confs.append(avg)

        if len(eye_confs) < 2:
            return False

        # 置信度有变化说明眼睛在开合
        conf_range = max(eye_confs) - min(eye_confs)
        return 0.1 < conf_range < 0.5

    def _detect_head_bunting(self) -> bool:
        """检测蹭物体行为：头部有明显的侧向运动"""
        if len(self.sequence_buffer) < 3:
            return False

        head_positions = []
        for i in range(min(3, len(self.sequence_buffer))):
            kp = self.sequence_buffer[-(i + 1)]
            nose = kp.get('nose', (0, 0, 0))
            if len(nose) >= 2 and nose[0] > 0:
                head_positions.append((nose[0], nose[1]))

        if len(head_positions) < 2:
            return False

        # 检测水平方向的运动
        dx = abs(head_positions[0][0] - head_positions[-1][0])
        dy = abs(head_positions[0][1] - head_positions[-1][1])

        # 水平运动大于垂直运动
        return dx > 20 and dx > dy * 1.5

    def _detect_tail_raised(self, keypoints: Dict) -> bool:
        """检测尾巴竖起：尾部关键点的 y 坐标高于肩部"""
        tail_base = keypoints.get('tail_base', (0, 0, 0))
        withers = keypoints.get('withers', (0, 0, 0))

        if len(tail_base) < 2 or len(withers) < 2:
            return False

        # 尾巴基部高于肩部（y 值更小）
        return tail_base[1] < withers[1] and tail_base[1] > 0

    def _detect_tail_lashing(self) -> bool:
        """检测尾巴甩动：尾部有快速运动"""
        if len(self.sequence_buffer) < 3:
            return False

        tail_positions = []
        for i in range(min(3, len(self.sequence_buffer))):
            kp = self.sequence_buffer[-(i + 1)]
            # 尝试用尾部关键点，如果没有则用后肩区域估算
            tail = kp.get('tail_base', kp.get('withers', (0, 0, 0)))
            if len(tail) >= 2 and tail[0] > 0:
                tail_positions.append(tail)

        if len(tail_positions) < 2:
            return False

        # 检测尾部位置变化
        total_motion = 0
        for i in range(len(tail_positions) - 1):
            total_motion += self._distance(tail_positions[i], tail_positions[i + 1])

        return total_motion > 30

    def _average_confidence(self, keypoints: Dict) -> float:
        """计算关键点平均置信度"""
        confs = []
        for v in keypoints.values():
            if len(v) >= 3:
                confs.append(v[2])
        return sum(confs) / len(confs) if confs else 0.0

    def classify_with_probs(self) -> Dict:
        behavior, confidence = self.classify()
        probs = np.zeros(len(BEHAVIORS))
        if behavior in BEHAVIORS:
            idx = BEHAVIORS.index(behavior)
            probs[idx] = confidence
        return {
            'behavior': behavior,
            'confidence': confidence,
            'probabilities': probs
        }

    def get_top_k_behaviors(self, k: int = 3) -> List[Tuple[str, float]]:
        """返回最可能的 k 个行为（基于当前分类结果）"""
        behavior, confidence = self.classify()
        results = [(behavior, confidence)]

        # 基于几何特征给出相近的备选行为
        related_behaviors = {
            'stretching': [('sitting', 0.5), ('grooming', 0.4)],
            'walking': [('running', 0.6), ('playing', 0.4)],
            'running': [('walking', 0.5), ('playing', 0.4)],
            'sitting': [('idle', 0.5), ('sleeping', 0.3)],
            'sleeping': [('sitting', 0.4), ('idle', 0.3)],
            'playing': [('running', 0.5), ('walking', 0.3)],
            'grooming': [('sitting', 0.4), ('idle', 0.3)],
            'hunting': [('sitting', 0.4), ('playing', 0.3)],
            'kneading': [('sitting', 0.5), ('grooming', 0.3)],
        }

        if behavior in related_behaviors:
            for alt_behavior, alt_conf in related_behaviors[behavior]:
                if len(results) >= k:
                    break
                results.append((alt_behavior, alt_conf * confidence))

        return results[:k]


class BehaviorClassifier:
    """行为分类器包装类，提供统一接口"""

    def __init__(self, model: Optional[object] = None, device: str = 'cpu', sequence_length: int = 8):
        self.device = device
        self.sequence_length = sequence_length
        self.feature_size = 81
        self.classifier = RuleBasedBehaviorClassifier()
        self.sequence_buffer: List[np.ndarray] = []

    def reset_buffer(self):
        self.sequence_buffer = []
        self.classifier.reset_buffer()

    def add_pose_features(self, features: np.ndarray):
        actual_size = len(features)
        if actual_size != self.feature_size:
            self.feature_size = actual_size
        self.sequence_buffer.append(features)
        if len(self.sequence_buffer) > self.sequence_length * 2:
            self.sequence_buffer = self.sequence_buffer[-self.sequence_length:]

    def classify(self) -> Tuple[str, float]:
        if len(self.sequence_buffer) < 2:
            return 'idle', 0.0
        return self.classifier.classify()

    def classify_with_probs(self) -> Dict:
        return self.classifier.classify_with_probs()

    def get_top_k_behaviors(self, k: int = 3) -> List[Tuple[str, float]]:
        return self.classifier.get_top_k_behaviors(k)


def create_behavior_classifier(device: str = 'cpu', sequence_length: int = 8):
    classifier = BehaviorClassifier(device=device, sequence_length=sequence_length)
    return None, classifier
