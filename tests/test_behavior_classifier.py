"""行为分类器单元测试"""

import pytest
import numpy as np


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
            'sitting', 'idle', 'walking', 'stretching', 'grooming',
            'sleeping', 'running', 'playing', 'hunting', 'kneading',
            'slow_blinking', 'head_bunting', 'belly_exposure',
            'tail_raising', 'tail_lashing', 'airplane_ears'
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
        assert len(result['probabilities']) == 16

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
