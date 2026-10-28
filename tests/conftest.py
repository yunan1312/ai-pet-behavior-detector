"""pytest 配置文件 - 共享测试夹具"""

import os
import sys
import pytest

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture
def config_loader():
    """提供 ConfigLoader 实例"""
    from utils.config_loader import ConfigLoader
    return ConfigLoader()


@pytest.fixture
def logger():
    """提供 Logger 实例"""
    from utils.logger import get_logger
    return get_logger()


@pytest.fixture
def behavior_classifier():
    """提供 BehaviorClassifier 实例"""
    from models.behavior_classifier import BehaviorClassifier
    return BehaviorClassifier(sequence_length=8)


@pytest.fixture
def rule_classifier():
    """提供 RuleBasedBehaviorClassifier 实例"""
    from models.behavior_classifier import RuleBasedBehaviorClassifier
    return RuleBasedBehaviorClassifier()


@pytest.fixture
def intent_mapper(config_loader):
    """提供 IntentMapper 实例"""
    from core.intent_mapper import IntentMapper
    return IntentMapper(config_loader)


@pytest.fixture
def emotion_analyzer():
    """提供 EmotionAnalyzer 实例"""
    from core.emotion_analyzer import EmotionAnalyzer
    return EmotionAnalyzer()


@pytest.fixture
def pose_processor():
    """提供 AnimalPoseProcessor 实例"""
    from models.pose_estimator import AnimalPoseProcessor
    return AnimalPoseProcessor()


@pytest.fixture
def sample_keypoints():
    """提供示例关键点数据（模拟一只坐着的猫）"""
    return {
        'left_eye': (300.0, 200.0, 0.9),
        'right_eye': (340.0, 200.0, 0.9),
        'nose': (320.0, 220.0, 0.95),
        'left_ear': (280.0, 170.0, 0.85),
        'right_ear': (360.0, 170.0, 0.85),
        'left_front_shoulder': (290.0, 280.0, 0.8),
        'right_front_shoulder': (350.0, 280.0, 0.8),
        'left_front_elbow': (275.0, 330.0, 0.75),
        'right_front_elbow': (365.0, 330.0, 0.75),
        'left_front_paw': (280.0, 380.0, 0.7),
        'right_front_paw': (360.0, 380.0, 0.7),
        'left_back_shoulder': (295.0, 320.0, 0.8),
        'right_back_shoulder': (345.0, 320.0, 0.8),
        'left_back_elbow': (285.0, 380.0, 0.75),
        'right_back_elbow': (355.0, 380.0, 0.75),
        'left_back_paw': (290.0, 420.0, 0.7),
        'right_back_paw': (350.0, 420.0, 0.7),
    }


@pytest.fixture
def sample_keypoints_stretching():
    """提供示例关键点数据（模拟一只伸懒腰的猫）"""
    return {
        'left_eye': (300.0, 180.0, 0.9),
        'right_eye': (340.0, 180.0, 0.9),
        'nose': (320.0, 200.0, 0.95),
        'left_ear': (280.0, 150.0, 0.85),
        'right_ear': (360.0, 150.0, 0.85),
        'left_front_shoulder': (290.0, 260.0, 0.8),
        'right_front_shoulder': (350.0, 260.0, 0.8),
        'left_front_elbow': (250.0, 340.0, 0.75),
        'right_front_elbow': (390.0, 340.0, 0.75),
        'left_front_paw': (230.0, 400.0, 0.7),
        'right_front_paw': (410.0, 400.0, 0.7),
        'left_back_shoulder': (295.0, 300.0, 0.8),
        'right_back_shoulder': (345.0, 300.0, 0.8),
        'left_back_elbow': (285.0, 370.0, 0.75),
        'right_back_elbow': (355.0, 370.0, 0.75),
        'left_back_paw': (290.0, 430.0, 0.7),
        'right_back_paw': (350.0, 430.0, 0.7),
    }
