"""行为分类器 - 归一化几何特征 + 时序规则

几何量一律按「身体尺度」(背线 withers → 尾根 tail_base 的距离) 归一化，
运动量按「身体长度/秒」计算，因此结果与画面分辨率、动物远近、采样帧率无关。

静态姿态采用打分而非固定优先级，避免规则顺序造成的系统性偏差
（例如所有趴着的猫都被判成睡觉、站着的猫被判成坐着）。

AP-10K 没有耳部/尾尖关键点，因此缓慢眨眼、飞机耳、尾巴竖起、尾巴甩动
无法从当前关键点可靠推断，已移出规则集，避免凭置信度噪声产生假标签。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.constants import BEHAVIORS

# 低于该置信度的关键点视为缺失，避免把模型的随机猜测当作几何证据
MIN_KEYPOINT_CONFIDENCE = 0.30
DEFAULT_FRAME_INTERVAL = 1.0 / 15.0

# 运动阈值，单位为「身体长度/秒」
STILL_SPEED = 0.10
CRAWL_SPEED = 0.35
RUN_SPEED = 2.5

# 持续静止多久才判定为「睡觉」：刚趴下/短暂发呆不应直接报睡觉
SLEEP_MIN_SECONDS = 2.0

# 偶发漏检的容忍时长：单帧丢失不应把标签瞬间打回「发呆」
MISS_TOLERANCE_SECONDS = 1.0

_TRACKED_KEYPOINTS = (
    'nose', 'withers', 'tail_base',
    'left_front_paw', 'right_front_paw', 'left_back_paw', 'right_back_paw',
)


def _keypoint(keypoints: Dict, name: str, min_conf: float = MIN_KEYPOINT_CONFIDENCE):
    """取有效关键点；缺坐标或置信度不足一律返回 None"""
    value = keypoints.get(name)
    if value is None or len(value) < 2:
        return None
    x, y = float(value[0]), float(value[1])
    if x <= 0 or y <= 0:
        return None
    if len(value) >= 3 and float(value[2]) < min_conf:
        return None
    return (x, y)


def _distance(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mean_point(points) -> Optional[Tuple[float, float]]:
    valid = [p for p in points if p is not None]
    if not valid:
        return None
    return (
        sum(p[0] for p in valid) / len(valid),
        sum(p[1] for p in valid) / len(valid),
    )


def _ramp(value: float, low: float, high: float) -> float:
    """把 value 线性映射到 0~1，用于把连续特征转成证据强度"""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


class _FrameGeometry:
    """单帧归一化几何特征

    scale 取「背线长度」与「检测框对角线一半」的较大值：
    - 侧视时背线长度最稳定，不受动物远近影响；
    - 正对镜头时背线被压缩，用检测框兜底。
    """

    __slots__ = (
        'scale', 'height_ratio', 'rear_drop', 'axis_angle', 'paw_drop_diff',
        'front_leg_extension', 'front_reach', 'nose_paw_distance',
        'nose_shoulder_distance', 'paw_y_spread', 'ground_y', 'bbox_aspect',
    )

    def __init__(self, keypoints: Dict, bbox=None):
        front_paws = [p for p in (
            _keypoint(keypoints, 'left_front_paw'),
            _keypoint(keypoints, 'right_front_paw'),
        ) if p]
        back_paws = [p for p in (
            _keypoint(keypoints, 'left_back_paw'),
            _keypoint(keypoints, 'right_back_paw'),
        ) if p]
        shoulders = [p for p in (
            _keypoint(keypoints, 'left_front_shoulder'),
            _keypoint(keypoints, 'right_front_shoulder'),
        ) if p]
        back_shoulders = [p for p in (
            _keypoint(keypoints, 'left_back_shoulder'),
            _keypoint(keypoints, 'right_back_shoulder'),
        ) if p]

        withers = _keypoint(keypoints, 'withers')
        tail_base = _keypoint(keypoints, 'tail_base')
        nose = _keypoint(keypoints, 'nose')
        eyes = _mean_point([_keypoint(keypoints, 'left_eye'), _keypoint(keypoints, 'right_eye')])

        top = withers or _mean_point(shoulders) or eyes or nose
        rear = tail_base or _mean_point(back_shoulders)
        paws = front_paws + back_paws

        bbox_diag = 0.0
        self.bbox_aspect = 1.0
        if bbox is not None and len(bbox) == 4:
            bw = abs(float(bbox[2]) - float(bbox[0]))
            bh = abs(float(bbox[3]) - float(bbox[1]))
            bbox_diag = math.hypot(bw, bh)
            if bh > 1e-3:
                self.bbox_aspect = bw / bh

        scale = _distance(top, rear)
        if scale < 1e-3:
            scale = bbox_diag * 0.5
        if scale < 1e-3:
            all_points = [p for p in (top, rear, nose, eyes) + tuple(paws) if p]
            if len(all_points) >= 2:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                scale = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.5
        self.scale = max(scale, 1.0)

        if top is not None and paws:
            self.ground_y = max(p[1] for p in paws)
            self.height_ratio = max(0.0, (self.ground_y - top[1]) / self.scale)
        else:
            self.ground_y = None
            self.height_ratio = 0.0

        if rear is not None and self.ground_y is not None:
            self.rear_drop = max(0.0, (self.ground_y - rear[1]) / self.scale)
        else:
            # 尾根与后肩都缺失时取中性值，避免把未知姿态推向站立
            self.rear_drop = 0.3

        if rear is not None and top is not None:
            dx = abs(rear[0] - top[0])
            dy = abs(rear[1] - top[1])
            self.axis_angle = math.degrees(math.atan2(dy, dx)) if dx + dy > 0 else 0.0
        else:
            self.axis_angle = 0.0

        if front_paws and back_paws:
            front_y = sum(p[1] for p in front_paws) / len(front_paws)
            back_y = sum(p[1] for p in back_paws) / len(back_paws)
            self.paw_drop_diff = abs(front_y - back_y) / self.scale
        else:
            self.paw_drop_diff = 0.0

        extensions = []
        for side in ('left', 'right'):
            shoulder = _keypoint(keypoints, f'{side}_front_shoulder')
            elbow = _keypoint(keypoints, f'{side}_front_elbow')
            paw = _keypoint(keypoints, f'{side}_front_paw')
            if shoulder and elbow and paw:
                extensions.append(_joint_extension(shoulder, elbow, paw))
        self.front_leg_extension = max(extensions) if extensions else 0.0

        if top is not None and front_paws:
            self.front_reach = max(_distance(top, p) for p in front_paws) / self.scale
        else:
            self.front_reach = 0.0

        if nose is not None and front_paws:
            self.nose_paw_distance = min(_distance(nose, p) for p in front_paws) / self.scale
        else:
            self.nose_paw_distance = 1.0

        if nose is not None and shoulders:
            self.nose_shoulder_distance = min(_distance(nose, p) for p in shoulders) / self.scale
        else:
            self.nose_shoulder_distance = 1.0

        if paws:
            ys = [p[1] for p in paws]
            self.paw_y_spread = (max(ys) - min(ys)) / self.scale
        else:
            self.paw_y_spread = 0.0


def _joint_extension(a, b, c) -> float:
    """三点的伸直程度：0 = 完全折叠，1 = 完全伸直"""
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 * mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)))
    return math.acos(cos_angle) / math.pi


class RuleBasedBehaviorClassifier:
    """基于归一化几何特征与时序规则的行为分类器"""

    def __init__(self):
        self.sequence_buffer: List[Dict] = []
        self.max_sequence_length = 8
        self.current_behavior = 'idle'
        self.behavior_stability = 0
        self.min_stability_frames = 2
        self.frame_count = 0

        self._pending_behavior: Optional[str] = None
        self._pending_count = 0
        self._last_confidence = 0.0
        self._last_scores: Dict[str, float] = {}
        self._still_since: Optional[float] = None

    def reset_buffer(self):
        self.sequence_buffer = []
        self.current_behavior = 'idle'
        self.behavior_stability = 0
        self.frame_count = 0
        self._pending_behavior = None
        self._pending_count = 0
        self._last_confidence = 0.0
        self._last_scores = {}
        self._still_since = None

    # ==================== 输入 ====================

    def add_pose_features(self, keypoints_or_features, bbox=None, timestamp=None):
        """追加一帧关键点

        Args:
            keypoints_or_features: 关键点字典；传空字典表示本帧没有检测到动物
            bbox: 目标检测框 (x1, y1, x2, y2)，用于归一化兜底
            timestamp: 帧时间戳（秒）；缺省时按 15 FPS 递增
        """
        if not isinstance(keypoints_or_features, dict):
            keypoints_or_features = {}

        keypoints = dict(keypoints_or_features)
        geometry = _FrameGeometry(keypoints, bbox) if keypoints else None

        if timestamp is None:
            timestamp = (self.sequence_buffer[-1]['timestamp'] + DEFAULT_FRAME_INTERVAL
                         if self.sequence_buffer else 0.0)

        self.sequence_buffer.append({
            'keypoints': keypoints,
            'geometry': geometry,
            'timestamp': float(timestamp),
        })
        if len(self.sequence_buffer) > self.max_sequence_length * 2:
            self.sequence_buffer = self.sequence_buffer[-self.max_sequence_length:]

        self.frame_count += 1

    # ==================== 分类入口 ====================

    def classify(self) -> Tuple[str, float]:
        """时序分类：需要至少两帧才能判断运动类行为"""
        if len(self.sequence_buffer) < 2:
            return 'idle', 0.0

        if self.sequence_buffer[-1]['geometry'] is None:
            last_ts = self.sequence_buffer[-1]['timestamp']
            last_valid_ts = None
            for frame in reversed(self.sequence_buffer[:-1]):
                if frame['geometry'] is not None:
                    last_valid_ts = frame['timestamp']
                    break
            # 短暂漏检（如转身、遮挡一帧）沿用上一个标签，避免画面闪烁
            if (last_valid_ts is not None
                    and last_ts - last_valid_ts <= MISS_TOLERANCE_SECONDS
                    and self.current_behavior != 'idle'):
                self._last_confidence *= 0.9
                return self.current_behavior, self._last_confidence

            self._reset_pending()
            self._still_since = None
            self.current_behavior = 'idle'
            self.behavior_stability = 0
            self._last_confidence = 0.3
            self._last_scores = {'idle': 0.3}
            return 'idle', 0.3

        candidate, score = self._best_behavior()
        confidence = max(0.0, min(0.95, score))

        if candidate == self.current_behavior:
            self.behavior_stability += 1
            self._reset_pending()
            self._last_confidence = confidence
            return candidate, confidence

        if candidate == self._pending_behavior:
            self._pending_count += 1
        else:
            self._pending_behavior = candidate
            self._pending_count = 1

        # idle 不作为先验保留，首个候选立即生效；其余行为需连续确认
        if self.current_behavior == 'idle' or self._pending_count >= self.min_stability_frames:
            self.current_behavior = candidate
            self.behavior_stability = 1
            self._reset_pending()
            self._last_confidence = confidence * 0.95
            return candidate, self._last_confidence

        return self.current_behavior, self._last_confidence

    def classify_static(self) -> Tuple[str, float]:
        """静态分类：单张图片没有帧间运动，只判断姿态/理毛/伸懒腰"""
        if not self.sequence_buffer or self.sequence_buffer[-1]['geometry'] is None:
            return 'idle', 0.0

        geometry = self.sequence_buffer[-1]['geometry']
        scores = self._posture_scores(geometry)

        grooming = self._grooming_score(geometry)
        if grooming > 0.8:
            # 口鼻贴住前爪/肩部是高精度证据，直接压过姿态判断
            scores = {'grooming': grooming}
        elif grooming > 0.55:
            scores['grooming'] = grooming

        stretching = self._stretching_score(geometry)
        if stretching > 0.55:
            scores['stretching'] = stretching

        self._last_scores = dict(scores)
        behavior = max(scores, key=scores.get)
        score = scores[behavior]
        if score < 0.45:
            return 'idle', 0.3

        self.current_behavior = behavior
        self._last_confidence = min(0.95, 0.55 + 0.4 * score)
        return behavior, self._last_confidence

    def _classify_from_keypoints(self, keypoints: Dict) -> Tuple[str, float]:
        """仅依据单帧关键点做静态分类（供兼容调用）"""
        if not keypoints:
            return 'idle', 0.3
        geometry = _FrameGeometry(keypoints)
        scores = self._posture_scores(geometry)
        grooming = self._grooming_score(geometry)
        if grooming > 0.55:
            scores['grooming'] = grooming
        behavior = max(scores, key=scores.get)
        score = scores[behavior]
        if score < 0.45:
            return 'idle', 0.3
        return behavior, min(0.95, 0.55 + 0.4 * score)

    # ==================== 静态姿态打分 ====================

    def _posture_scores(self, g: _FrameGeometry) -> Dict[str, float]:
        """站立/坐着/趴卧三类的证据打分，0~1

        - 站立：身体离地高，尾根也高（臀部离地）
        - 坐着：背线陡（身体直立），尾根贴地，身体离地高
        - 趴卧：身体离地低，尾根贴地
        """
        standing = (
            0.45 * _ramp(g.height_ratio, 0.40, 0.65)
            + 0.35 * _ramp(g.rear_drop, 0.30, 0.55)
            + 0.20 * (1.0 - _ramp(g.axis_angle, 30.0, 60.0))
        )
        sitting = (
            0.45 * _ramp(g.axis_angle, 35.0, 60.0)
            + 0.35 * (1.0 - _ramp(g.rear_drop, 0.15, 0.40))
            + 0.20 * _ramp(g.height_ratio, 0.45, 0.70)
        )
        low_body = (
            0.60 * (1.0 - _ramp(g.height_ratio, 0.25, 0.55))
            + 0.40 * (1.0 - _ramp(g.rear_drop, 0.20, 0.50))
        )
        # 狮身人面趴姿：后躯贴地但胸口抬起，背线向后下方倾斜。
        # 用臀部高度带 + 背线角度带与站立（背线平）、坐姿（背线陡）区分。
        semi_lying = (
            _ramp(g.rear_drop, 0.22, 0.32)
            * (1.0 - _ramp(g.rear_drop, 0.50, 0.65))
            * _ramp(g.axis_angle, 8.0, 20.0)
            * (1.0 - _ramp(g.axis_angle, 45.0, 65.0))
        )
        lying = max(low_body, 0.9 * semi_lying)
        return {'standing': standing, 'sitting': sitting, 'lying': lying}

    def _grooming_score(self, g: _FrameGeometry) -> float:
        """理毛：口鼻贴近前爪或肩部"""
        if g.nose_paw_distance <= g.nose_shoulder_distance:
            return 1.0 - _ramp(g.nose_paw_distance, 0.18, 0.45)
        return 0.9 * (1.0 - _ramp(g.nose_shoulder_distance, 0.15, 0.40))

    def _stretching_score(self, g: _FrameGeometry) -> float:
        """伸懒腰：前肢伸直且向前远伸，身体压低"""
        extension = _ramp(g.front_leg_extension, 0.85, 0.98)
        reach = _ramp(g.front_reach, 0.75, 1.05)
        body_low = 1.0 - abs(g.height_ratio - 0.45) / 0.55
        return max(0.0, 0.5 * extension + 0.3 * reach + 0.2 * max(0.0, body_low))

    # ==================== 时序特征 ====================

    def _speed_series(self, max_pairs: int = 4) -> List[float]:
        """相邻帧的移动速度（身体长度/秒）

        位移取所有可见关键点的中位数，减少单个抖动关键点的影响；
        dt 来自真实时间戳，因此 3 FPS 与 15 FPS 的采样结果一致。
        """
        frames = self.sequence_buffer
        speeds: List[float] = []
        recent = frames[-(max_pairs + 1):]
        for prev, curr in zip(recent[:-1], recent[1:]):
            if prev['geometry'] is None or curr['geometry'] is None:
                continue
            dt = curr['timestamp'] - prev['timestamp']
            if dt <= 1e-3 or dt > 3.0:
                continue
            scale = max(curr['geometry'].scale, 1.0)
            displacements = []
            for name in _TRACKED_KEYPOINTS:
                a = _keypoint(prev['keypoints'], name)
                b = _keypoint(curr['keypoints'], name)
                if a is not None and b is not None:
                    displacements.append(_distance(a, b))
            if len(displacements) < 3:
                continue
            speeds.append(_median(displacements) / dt / scale)
        return speeds

    def _motion_irregularity(self, speeds: List[float]) -> float:
        if len(speeds) < 3:
            return 0.0
        mean = sum(speeds) / len(speeds)
        if mean < 0.3:
            return 0.0
        variance = sum((s - mean) ** 2 for s in speeds) / len(speeds)
        return min(1.0, math.sqrt(variance) / mean)

    def _detect_kneading(self) -> bool:
        """踩奶：前爪上下起伏且方向反复变化，要求采样足够密"""
        frames = self.sequence_buffer[-6:]
        if len(frames) < 4:
            return False
        if any(f['geometry'] is None for f in frames):
            return False
        if frames[-1]['timestamp'] - frames[0]['timestamp'] < 0.4:
            return False

        heights = []
        for frame in frames:
            left = _keypoint(frame['keypoints'], 'left_front_paw')
            right = _keypoint(frame['keypoints'], 'right_front_paw')
            if left is None or right is None:
                return False
            heights.append((left[1] + right[1]) / 2.0)

        scale = max(frames[-1]['geometry'].scale, 1.0)
        amplitude = (max(heights) - min(heights)) / scale
        deltas = [heights[i + 1] - heights[i] for i in range(len(heights) - 1)]
        flips = sum(1 for i in range(len(deltas) - 1) if deltas[i] * deltas[i + 1] < 0)
        return amplitude >= 0.05 and flips >= 2

    def _detect_head_bunting(self, speed: float) -> bool:
        """蹭物体：身体基本静止，口鼻做横向往复运动"""
        if speed >= CRAWL_SPEED:
            return False
        frames = self.sequence_buffer[-6:]
        if len(frames) < 3 or any(f['geometry'] is None for f in frames):
            return False

        nose_points = [_keypoint(f['keypoints'], 'nose') for f in frames]
        if any(p is None for p in nose_points):
            return False

        xs = [p[0] for p in nose_points]
        ys = [p[1] for p in nose_points]
        scale = max(frames[-1]['geometry'].scale, 1.0)
        horizontal_range = (max(xs) - min(xs)) / scale
        vertical_range = (max(ys) - min(ys)) / scale
        if horizontal_range < 0.12 or vertical_range >= horizontal_range:
            return False

        deltas = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        return any(deltas[i] * deltas[i + 1] < 0 for i in range(len(deltas) - 1))

    # ==================== 行为决策 ====================

    def _best_behavior(self) -> Tuple[str, float]:
        geometry = self.sequence_buffer[-1]['geometry']
        speeds = self._speed_series()
        speed = _median(speeds[:-1]) if len(speeds) >= 2 else (speeds[-1] if speeds else 0.0)
        irregularity = self._motion_irregularity(speeds)

        last_ts = self.sequence_buffer[-1]['timestamp']
        if speed < STILL_SPEED:
            if self._still_since is None:
                self._still_since = last_ts
        else:
            self._still_since = None
        still_duration = 0.0 if self._still_since is None else last_ts - self._still_since

        scores: Dict[str, float] = {}

        if speed >= RUN_SPEED:
            if irregularity > 0.6:
                scores['playing'] = 0.92
                scores['running'] = 0.80
            else:
                scores['running'] = 0.90
        elif speed >= CRAWL_SPEED:
            scores['walking'] = 0.70 + 0.25 * _ramp(speed, CRAWL_SPEED, 1.6)

        if speed < CRAWL_SPEED:
            grooming = self._grooming_score(geometry)
            if grooming > 0.55:
                scores['grooming'] = grooming
            if self._detect_kneading():
                scores['kneading'] = 0.78
            if self._detect_head_bunting(speed):
                scores['head_bunting'] = 0.70

            stretching = self._stretching_score(geometry)
            if stretching > 0.55:
                scores['stretching'] = stretching

            # 静止姿态只在低速时参与竞争，否则移动中的猫会被误判成站/坐/趴
            postures = self._posture_scores(geometry)
            posture = max(postures, key=postures.get)
            posture_score = postures[posture]

            if posture == 'lying' and posture_score >= 0.5:
                if speed < STILL_SPEED:
                    # 趴着不动先报「趴卧」，持续静止足够久才升级为「睡觉」
                    if still_duration >= SLEEP_MIN_SECONDS:
                        scores['sleeping'] = 0.72 + 0.2 * posture_score
                    else:
                        scores['lying'] = 0.55 + 0.4 * posture_score
                elif speed < CRAWL_SPEED:
                    scores['hunting'] = 0.62 + 0.2 * posture_score
            elif posture_score >= 0.5:
                scores[posture] = 0.55 + 0.4 * posture_score

        if not scores:
            scores['idle'] = 0.5

        self._last_scores = dict(scores)
        behavior = max(scores, key=scores.get)
        return behavior, scores[behavior]

    # ==================== 兼容接口 ====================

    def classify_with_probs(self) -> Dict:
        behavior, confidence = self.classify()
        probs = np.zeros(len(BEHAVIORS))
        total = sum(self._last_scores.values()) or 1.0
        for name, score in self._last_scores.items():
            if name in BEHAVIORS:
                probs[BEHAVIORS.index(name)] = score / total
        if behavior in BEHAVIORS and probs[BEHAVIORS.index(behavior)] == 0.0:
            probs[BEHAVIORS.index(behavior)] = confidence
        return {
            'behavior': behavior,
            'confidence': confidence,
            'probabilities': probs,
        }

    def get_top_k_behaviors(self, k: int = 3) -> List[Tuple[str, float]]:
        if not self._last_scores:
            self.classify()
        ranked = sorted(self._last_scores.items(), key=lambda item: item[1], reverse=True)
        return [(name, min(0.95, score)) for name, score in ranked[:max(1, k)]]

    def _reset_pending(self):
        self._pending_behavior = None
        self._pending_count = 0

    def _body_height(self, keypoints: Dict, bbox=None) -> float:
        """身体离地高度 / 身体尺度（归一化，与分辨率无关）"""
        return _FrameGeometry(keypoints, bbox).height_ratio

    def _body_width(self, keypoints: Dict, bbox=None) -> float:
        return _FrameGeometry(keypoints, bbox).paw_y_spread

    def _detect_motion(self) -> float:
        speeds = self._speed_series()
        return _median(speeds) if speeds else 0.0


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

    def add_pose_features(self, features=None, keypoints=None, bbox=None, timestamp=None):
        """追加一帧数据

        features 为固定维度特征向量（可选，供外部特征管线使用），
        keypoints 为关键点字典（规则分类必需）。
        """
        if features is not None:
            array = np.asarray(features, dtype=np.float32)
            self.feature_size = len(array)
            self.sequence_buffer.append(array)
            if len(self.sequence_buffer) > self.sequence_length * 2:
                self.sequence_buffer = self.sequence_buffer[-self.sequence_length:]

        if keypoints is not None:
            self.classifier.add_pose_features(keypoints, bbox=bbox, timestamp=timestamp)

    def classify(self) -> Tuple[str, float]:
        if not self.classifier.sequence_buffer:
            return 'idle', 0.0
        return self.classifier.classify()

    def classify_static(self) -> Tuple[str, float]:
        return self.classifier.classify_static()

    def classify_with_probs(self) -> Dict:
        return self.classifier.classify_with_probs()

    def get_top_k_behaviors(self, k: int = 3) -> List[Tuple[str, float]]:
        return self.classifier.get_top_k_behaviors(k)


def create_behavior_classifier(device: str = 'cpu', sequence_length: int = 8):
    classifier = BehaviorClassifier(device=device, sequence_length=sequence_length)
    return None, classifier
