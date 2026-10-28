"""文件分析器单元测试"""

import pytest
import numpy as np


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