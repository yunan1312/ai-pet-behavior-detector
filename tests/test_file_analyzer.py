"""文件分析器单元测试"""

import pytest
import numpy as np

from pose_helpers import bbox_for, sitting_keypoints, standing_keypoints


class TestFrameAnalyzer:
    """测试帧分析器"""

    def test_initialization(self, pose_processor, behavior_classifier, intent_mapper):
        """测试初始化"""
        from core.file_analyzer import FrameAnalyzer
        analyzer = FrameAnalyzer(
            pose_processor,
            behavior_classifier,
            intent_mapper,
            confidence_threshold=0.75
        )
        assert analyzer.confidence_threshold == 0.75
        assert analyzer.last_behavior == 'idle'

    def test_analyze_empty_frame(self, pose_processor, behavior_classifier, intent_mapper):
        """测试分析空帧"""
        from core.file_analyzer import FrameAnalyzer
        analyzer = FrameAnalyzer(
            pose_processor,
            behavior_classifier,
            intent_mapper
        )
        # 创建一个空帧
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyzer.analyze_frame(empty_frame)
        assert 'behavior' in result
        assert 'confidence' in result
        assert 'keypoints' in result

    def test_reset(self, pose_processor, behavior_classifier, intent_mapper):
        """测试重置功能"""
        from core.file_analyzer import FrameAnalyzer
        analyzer = FrameAnalyzer(
            pose_processor,
            behavior_classifier,
            intent_mapper
        )
        # 分析一些帧
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        analyzer.analyze_frame(frame)
        analyzer.analyze_frame(frame)

        # 重置
        analyzer.reset()
        assert analyzer.last_behavior == 'idle'
        assert analyzer.last_confidence == 0.0
        assert analyzer.last_intent == ''


class TestFileAnalyzer:
    """测试文件分析器"""

    def test_initialization(self, pose_processor, behavior_classifier, intent_mapper):
        """测试初始化"""
        from core.file_analyzer import FileAnalyzer
        analyzer = FileAnalyzer(
            pose_processor,
            behavior_classifier,
            intent_mapper,
            confidence_threshold=0.75
        )
        assert analyzer.confidence_threshold == 0.75

    def test_analyze_single_image(self, pose_processor, behavior_classifier, intent_mapper):
        """测试分析单张图片"""
        from core.file_analyzer import FileAnalyzer
        analyzer = FileAnalyzer(
            pose_processor,
            behavior_classifier,
            intent_mapper
        )
        # 创建测试图片
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyzer.analyze_single_image(test_image)

        assert 'behavior' in result
        assert 'confidence' in result
        assert 'intent' in result
        assert 'keypoints' in result

    def test_analyze_single_image_with_none(self, pose_processor, behavior_classifier, intent_mapper):
        """测试分析None图片"""
        from core.file_analyzer import FileAnalyzer
        analyzer = FileAnalyzer(
            pose_processor,
            behavior_classifier,
            intent_mapper
        )
        result = analyzer.analyze_single_image(None)
        assert result['behavior'] == 'idle'
        assert result['confidence'] == 0.0

    def test_analyze_video_frames(self, pose_processor, behavior_classifier, intent_mapper):
        """测试分析视频帧"""
        from core.file_analyzer import FileAnalyzer
        analyzer = FileAnalyzer(
            pose_processor,
            behavior_classifier,
            intent_mapper
        )
        # 创建测试帧
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]
        results = analyzer.analyze_video_frames(frames)
        assert len(results) == len(frames)
        for result in results:
            assert 'behavior' in result
            assert 'confidence' in result
            assert 'keypoints' in result

    def test_analyze_video_frames_with_progress_callback(self, pose_processor, behavior_classifier, intent_mapper):
        """测试带进度回调的视频帧分析"""
        from core.file_analyzer import FileAnalyzer
        analyzer = FileAnalyzer(
            pose_processor,
            behavior_classifier,
            intent_mapper
        )
        # 创建测试帧
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)
        ]

        progress_values = []
        def progress_callback(progress):
            progress_values.append(progress)

        results = analyzer.analyze_video_frames(frames, progress_callback)
        assert len(results) == len(frames)
        # 进度回调应该被调用
        assert len(progress_values) > 0

class _StubPoseProcessor:
    """返回固定关键点的姿态处理器替身，避免测试加载真实模型"""

    def __init__(self, keypoints, box=None):
        self.keypoints = keypoints
        self.last_detection = {'box': box} if box is not None else None

    def estimate(self, frame):
        return dict(self.keypoints)

    def extract_pose_features(self, keypoints):
        return None


class _MovingPoseProcessor(_StubPoseProcessor):
    """每次调用把姿态向右平移固定距离，模拟匀速行走"""

    def __init__(self, step, box=None):
        super().__init__({}, box)
        self.step = step
        self.index = -1

    def estimate(self, frame):
        self.index += 1
        keypoints = standing_keypoints(dx=self.step * self.index)
        self.keypoints = keypoints
        self.last_detection = {'box': bbox_for(keypoints)}
        return keypoints


class TestNormalizedAnalysis:
    """验证图片/视频路径能真正输出行为（修复恒为 idle 的问题）"""

    def test_single_image_returns_static_posture(self, intent_mapper, behavior_classifier):
        from core.file_analyzer import FileAnalyzer
        from pose_helpers import sitting_keypoints

        keypoints = sitting_keypoints()
        processor = _StubPoseProcessor(keypoints, bbox_for(keypoints))
        analyzer = FileAnalyzer(processor, behavior_classifier, intent_mapper, 0.6)

        result = analyzer.analyze_single_image(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result['behavior'] == 'sitting'
        assert result['confidence'] > 0.6
        assert result['intent'], '静态分类结果也应生成意图文案'

    def test_video_samples_detect_walking(self, intent_mapper, behavior_classifier):
        from core.file_analyzer import FileAnalyzer

        fps = 3.0
        step = 1.0 * 200.0 / fps  # 1 身体长度/秒
        processor = _MovingPoseProcessor(step)
        analyzer = FileAnalyzer(processor, behavior_classifier, intent_mapper, 0.6)

        behaviors = []
        for index in range(8):
            result = analyzer.analyze_video_sample(
                np.zeros((480, 640, 3), dtype=np.uint8), timestamp=index / fps
            )
            behaviors.append(result['behavior'])

        assert 'walking' in behaviors

    def test_video_sampling_rate_does_not_change_speed(self, intent_mapper):
        """3 FPS 与 15 FPS 采样同一速度的运动应得到同样的行为"""
        from models.behavior_classifier import BehaviorClassifier

        labels = []
        for fps in (3.0, 15.0):
            classifier = BehaviorClassifier(sequence_length=8)
            processor = _MovingPoseProcessor(1.0 * 200.0 / fps)
            from core.file_analyzer import FileAnalyzer
            analyzer = FileAnalyzer(processor, classifier, intent_mapper, 0.6)
            label = 'idle'
            for index in range(8):
                result = analyzer.analyze_video_sample(
                    np.zeros((480, 640, 3), dtype=np.uint8), timestamp=index / fps
                )
                label = result['behavior']
            labels.append(label)

        assert labels == ['walking', 'walking']
