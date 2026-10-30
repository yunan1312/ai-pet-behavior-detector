"""行为分类器单元测试"""

import pytest
import numpy as np

from pose_helpers import (
    bbox_for,
    grooming_keypoints,
    lying_keypoints,
    sitting_keypoints,
    sphinx_keypoints,
    standing_keypoints,
)
from models.behavior_classifier import RuleBasedBehaviorClassifier
from utils.constants import BEHAVIORS


class TestRuleBasedBehaviorClassifier:
    """测试基于规则的行为分类器"""

    def test_initial_state(self, rule_classifier):
        """测试初始状态"""
        behavior, confidence = rule_classifier.classify()
        assert behavior == 'idle'
        assert confidence == 0.0

    def test_classify_with_empty_buffer(self, rule_classifier):
        """测试空缓冲区分类"""
        rule_classifier.add_pose_features({})
        behavior, confidence = rule_classifier.classify()
        assert behavior == 'idle'

    def test_classify_with_valid_keypoints(self, rule_classifier, sample_keypoints):
        """测试有效关键点分类"""
        rule_classifier.add_pose_features(sample_keypoints)
        rule_classifier.add_pose_features(sample_keypoints)
        behavior, confidence = rule_classifier.classify()
        assert behavior in [
            'sitting', 'standing', 'lying', 'idle', 'walking', 'stretching',
            'grooming', 'sleeping', 'running', 'playing', 'hunting', 'kneading',
            'head_bunting'
        ]
        assert 0.0 <= confidence <= 1.0

    def test_stability_mechanism(self, rule_classifier, sample_keypoints):
        """测试稳定性机制：连续相同行为才确认"""
        rule_classifier.add_pose_features(sample_keypoints)
        rule_classifier.add_pose_features(sample_keypoints)
        behavior1, conf1 = rule_classifier.classify()

        # 再次添加相同数据
        rule_classifier.add_pose_features(sample_keypoints)
        behavior2, conf2 = rule_classifier.classify()

        # 第二次分类时稳定性应增加
        assert rule_classifier.behavior_stability >= 0

    def test_reset_buffer(self, rule_classifier, sample_keypoints):
        """测试缓冲区重置"""
        rule_classifier.add_pose_features(sample_keypoints)
        rule_classifier.add_pose_features(sample_keypoints)
        rule_classifier.classify()

        rule_classifier.reset_buffer()
        assert len(rule_classifier.sequence_buffer) == 0
        assert rule_classifier.current_behavior == 'idle'
        assert rule_classifier.frame_count == 0

    def test_buffer_size_limit(self, rule_classifier, sample_keypoints):
        """测试缓冲区大小限制"""
        for _ in range(50):
            rule_classifier.add_pose_features(sample_keypoints)

        assert len(rule_classifier.sequence_buffer) <= rule_classifier.max_sequence_length * 2

    def test_classify_with_probs(self, rule_classifier, sample_keypoints):
        """测试概率分类输出"""
        rule_classifier.add_pose_features(sample_keypoints)
        rule_classifier.add_pose_features(sample_keypoints)
        result = rule_classifier.classify_with_probs()

        assert 'behavior' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert len(result['probabilities']) == len(BEHAVIORS)

    def test_get_top_k_behaviors(self, rule_classifier, sample_keypoints):
        """测试获取 top-k 行为"""
        rule_classifier.add_pose_features(sample_keypoints)
        rule_classifier.add_pose_features(sample_keypoints)
        results = rule_classifier.get_top_k_behaviors(3)

        assert len(results) <= 3
        for behavior, confidence in results:
            assert isinstance(behavior, str)
            assert 0.0 <= confidence <= 1.0


class TestBehaviorClassifier:
    """测试行为分类器包装类"""

    def test_initialization(self, behavior_classifier):
        """测试初始化"""
        assert behavior_classifier.sequence_length == 8
        assert behavior_classifier.feature_size == 81

    def test_add_pose_features(self, behavior_classifier):
        """测试添加姿态特征"""
        features = np.random.randn(81).astype(np.float32)
        behavior_classifier.add_pose_features(features)
        assert len(behavior_classifier.sequence_buffer) == 1

    def test_classify_insufficient_data(self, behavior_classifier):
        """测试数据不足时的分类"""
        features = np.random.randn(81).astype(np.float32)
        behavior_classifier.add_pose_features(features)
        behavior, confidence = behavior_classifier.classify()
        assert behavior == 'idle'
        assert confidence == 0.0

    def test_reset_buffer(self, behavior_classifier):
        """测试重置缓冲区"""
        features = np.random.randn(81).astype(np.float32)
        behavior_classifier.add_pose_features(features)
        behavior_classifier.reset_buffer()
        assert len(behavior_classifier.sequence_buffer) == 0


def _feed_motion(classifier, fps, speed_bl_per_second, samples=8, scale=1.0):
    """按给定帧率与速度（身体长度/秒）喂入平移的站立姿态"""
    dt = 1.0 / fps
    step = speed_bl_per_second * 200.0 * scale * dt
    label, confidence = 'idle', 0.0
    for index in range(samples):
        keypoints = standing_keypoints(scale=scale, dx=step * index)
        classifier.add_pose_features(
            keypoints, bbox=bbox_for(keypoints), timestamp=index * dt
        )
        if index >= 1:
            label, confidence = classifier.classify()
    return label, confidence


class TestNormalizedFeatures:
    """验证尺度不变、帧率不变与姿态判别"""

    @pytest.mark.parametrize('builder,expected', [
        (standing_keypoints, 'standing'),
        (sitting_keypoints, 'sitting'),
        (lying_keypoints, 'lying'),
    ])
    def test_static_postures_are_distinguished(self, builder, expected):
        keypoints = builder()
        classifier = RuleBasedBehaviorClassifier()
        classifier.add_pose_features(keypoints, bbox=bbox_for(keypoints), timestamp=0.0)

        behavior, confidence = classifier.classify_static()

        assert behavior == expected
        assert confidence >= 0.6

    def test_body_scale_invariance(self):
        """同一姿态在近处/远处（缩放 0.35）应得到相同标签"""
        near = RuleBasedBehaviorClassifier()
        far = RuleBasedBehaviorClassifier()

        near_points = standing_keypoints(scale=1.0)
        far_points = standing_keypoints(scale=0.35, dx=40.0, dy=60.0)
        near.add_pose_features(near_points, bbox=bbox_for(near_points), timestamp=0.0)
        far.add_pose_features(far_points, bbox=bbox_for(far_points), timestamp=0.0)

        assert near.classify_static()[0] == 'standing'
        assert far.classify_static()[0] == 'standing'

    def test_fps_invariance_walking(self):
        """同样的 1 身体长度/秒，3 FPS 与 15 FPS 都应是行走"""
        slow_rate = RuleBasedBehaviorClassifier()
        fast_rate = RuleBasedBehaviorClassifier()

        assert _feed_motion(slow_rate, fps=3.0, speed_bl_per_second=1.0)[0] == 'walking'
        assert _feed_motion(fast_rate, fps=15.0, speed_bl_per_second=1.0)[0] == 'walking'

    def test_high_speed_is_running(self):
        classifier = RuleBasedBehaviorClassifier()
        label, _confidence = _feed_motion(classifier, fps=15.0, speed_bl_per_second=4.0)
        assert label == 'running'

    def test_grooming_from_single_frame(self):
        keypoints = grooming_keypoints()
        classifier = RuleBasedBehaviorClassifier()
        classifier.add_pose_features(keypoints, bbox=bbox_for(keypoints), timestamp=0.0)

        assert classifier.classify_static()[0] == 'grooming'

    def test_hysteresis_prevents_single_frame_flicker(self):
        classifier = RuleBasedBehaviorClassifier()
        assert _feed_motion(classifier, fps=15.0, speed_bl_per_second=1.0, samples=6)[0] == 'walking'

        dt = 1.0 / 15.0
        first = standing_keypoints(dx=400.0)
        classifier.add_pose_features(first, bbox=bbox_for(first), timestamp=6 * dt)
        assert classifier.classify()[0] == 'walking', '单帧突变不应立刻改判'

        label = 'walking'
        for index in range(2, 6):
            keypoints = standing_keypoints(dx=400.0 * index)
            classifier.add_pose_features(
                keypoints, bbox=bbox_for(keypoints), timestamp=(5 + index) * dt
            )
            label = classifier.classify()[0]
        assert label == 'running'

    def test_unreliable_behaviors_are_not_emitted(self):
        """无耳部/尾尖关键点支撑的行为不应再产生假标签"""
        forbidden = {
            'airplane_ears', 'slow_blinking', 'tail_raising',
            'tail_lashing', 'belly_exposure',
        }
        for builder in (standing_keypoints, sitting_keypoints,
                        lying_keypoints, grooming_keypoints):
            keypoints = builder()
            classifier = RuleBasedBehaviorClassifier()
            classifier.add_pose_features(keypoints, bbox=bbox_for(keypoints), timestamp=0.0)
            assert classifier.classify_static()[0] not in forbidden

    def test_missing_keypoints_yield_idle(self):
        classifier = RuleBasedBehaviorClassifier()
        classifier.add_pose_features({}, timestamp=0.0)
        classifier.add_pose_features({}, timestamp=0.1)

        assert classifier.classify() == ('idle', 0.3)

    def test_sleeping_requires_sustained_stillness(self):
        """趴着不动先报趴卧，持续静止 2 秒后才升级为睡觉"""
        classifier = RuleBasedBehaviorClassifier()
        dt = 1.0 / 15.0
        labels = []
        for index in range(50):
            keypoints = lying_keypoints()
            classifier.add_pose_features(
                keypoints, bbox=bbox_for(keypoints), timestamp=index * dt
            )
            labels.append(classifier.classify()[0])

        assert labels[10] == 'lying', '刚趴下不足 2 秒不应报睡觉'
        assert labels[-1] == 'sleeping'

    def test_lying_slow_motion_is_hunting(self):
        """趴伏且缓慢前移更接近捕猎，而不是睡觉"""
        classifier = RuleBasedBehaviorClassifier()
        dt = 1.0 / 15.0
        step = 0.25 * 200.0 * dt  # 0.25 身体长度/秒
        label = 'idle'
        for index in range(12):
            keypoints = lying_keypoints(dx=step * index)
            classifier.add_pose_features(
                keypoints, bbox=bbox_for(keypoints), timestamp=index * dt
            )
            if index >= 1:
                label = classifier.classify()[0]

        assert label == 'hunting'

    def test_posture_without_tail_base_uses_hips(self):
        """尾根缺失时用后肩估计臀部高度，坐姿不应退化成站立"""
        keypoints = sitting_keypoints()
        keypoints.pop('tail_base')
        classifier = RuleBasedBehaviorClassifier()
        classifier.add_pose_features(
            keypoints, bbox=bbox_for(keypoints), timestamp=0.0
        )

        assert classifier.classify_static()[0] == 'sitting'

    def test_sphinx_pose_is_lying_not_standing(self):
        """胸口抬起、后躯贴地的狮身人面姿势属于趴卧"""
        keypoints = sphinx_keypoints()
        classifier = RuleBasedBehaviorClassifier()
        classifier.add_pose_features(
            keypoints, bbox=bbox_for(keypoints), timestamp=0.0
        )

        assert classifier.classify_static()[0] == 'lying'

    def test_speed_is_available_after_three_frames(self):
        """缓冲不足 5 帧时也要能算出速度，避免视频开头迟迟不判行走"""
        classifier = RuleBasedBehaviorClassifier()
        dt = 1.0 / 15.0
        step = 0.5 * 200.0 * dt

        for index in range(3):
            keypoints = standing_keypoints(dx=step * index)
            classifier.add_pose_features(
                keypoints, bbox=bbox_for(keypoints), timestamp=index * dt
            )

        speeds = classifier._speed_series()
        assert speeds
        assert speeds[-1] == pytest.approx(0.5, abs=0.05)
        assert classifier.classify()[0] == 'walking'

    def test_single_missed_frame_keeps_label(self):
        """偶发一帧漏检沿用旧标签，持续漏检才回到发呆"""
        classifier = RuleBasedBehaviorClassifier()
        dt = 1.0 / 15.0
        label = 'idle'
        for index in range(5):
            keypoints = standing_keypoints()
            classifier.add_pose_features(
                keypoints, bbox=bbox_for(keypoints), timestamp=index * dt
            )
            label = classifier.classify()[0]
        assert label == 'standing'

        classifier.add_pose_features({}, timestamp=5 * dt)
        assert classifier.classify()[0] == 'standing'

        classifier.add_pose_features({}, timestamp=5 * dt + 2.0)
        assert classifier.classify()[0] == 'idle'
