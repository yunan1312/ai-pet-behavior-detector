"""视频处理模块单元测试"""

import pytest
import numpy as np
import tempfile
import os


class TestVideoStreamHandler:
    """测试视频流处理器"""

    def test_initialization(self):
        """测试初始化"""
        from core.video_handler import VideoStreamHandler
        handler = VideoStreamHandler(source=0, target_fps=15)
        assert handler.source == 0
        assert handler.target_fps == 15
        assert not handler.is_running

    def test_is_opened_when_not_running(self):
        """测试未运行时is_opened返回False"""
        from core.video_handler import VideoStreamHandler
        handler = VideoStreamHandler(source=0)
        assert not handler.is_opened()

    def test_get_fps_when_not_running(self):
        """测试未运行时get_fps返回0"""
        from core.video_handler import VideoStreamHandler
        handler = VideoStreamHandler(source=0)
        assert handler.get_fps() == 0.0

    def test_read_frame_when_not_running(self):
        """测试未运行时read_frame返回False"""
        from core.video_handler import VideoStreamHandler
        handler = VideoStreamHandler(source=0)
        ret, frame = handler.read_frame()
        assert not ret
        assert frame is None

    def test_stop_when_not_running(self):
        """测试未运行时stop不报错"""
        from core.video_handler import VideoStreamHandler
        handler = VideoStreamHandler(source=0)
        handler.stop()
        assert not handler.is_running


class TestFileVideoHandler:
    """测试文件视频处理器"""

    def test_initialization(self):
        """测试初始化"""
        from core.video_handler import FileVideoHandler
        handler = FileVideoHandler("test_video.mp4")
        assert handler.file_path == "test_video.mp4"
        assert handler.total_frames == 0

    def test_open_nonexistent_file(self):
        """测试打开不存在的文件"""
        from core.video_handler import FileVideoHandler
        handler = FileVideoHandler("nonexistent_video.mp4")
        result = handler.open()
        assert not result

    def test_get_info_when_not_opened(self):
        """测试未打开时get_info返回空信息"""
        from core.video_handler import FileVideoHandler
        handler = FileVideoHandler("test_video.mp4")
        info = handler.get_info()
        assert info['fps'] == 0.0
        assert info['total_frames'] == 0

    def test_read_frame_when_not_opened(self):
        """测试未打开时read_frame返回False"""
        from core.video_handler import FileVideoHandler
        handler = FileVideoHandler("test_video.mp4")
        ret, frame = handler.read_frame()
        assert not ret
        assert frame is None

    def test_seek_when_not_opened(self):
        """测试未打开时seek返回False"""
        from core.video_handler import FileVideoHandler
        handler = FileVideoHandler("test_video.mp4")
        result = handler.seek(0)
        assert not result

    def test_release_when_not_opened(self):
        """测试未打开时release不报错"""
        from core.video_handler import FileVideoHandler
        handler = FileVideoHandler("test_video.mp4")
        handler.release()
        assert handler.cap is None


class TestLoadImage:
    """测试图片加载函数"""

    def test_load_nonexistent_image(self):
        """测试加载不存在的图片"""
        from core.video_handler import load_image
        result = load_image("nonexistent_image.jpg")
        assert result is None

    def test_load_image_with_invalid_path(self):
        """测试加载无效路径的图片"""
        from core.video_handler import load_image
        result = load_image("")
        assert result is None

    def test_load_image_with_none_path(self):
        """测试加载None路径的图片"""
        from core.video_handler import load_image
        result = load_image(None)
        assert result is None


class TestListAvailableCameras:
    """测试摄像头列表函数"""

    def test_list_cameras_returns_list(self):
        """测试返回类型为列表"""
        from core.video_handler import list_available_cameras
        cameras = list_available_cameras()
        assert isinstance(cameras, list)

    def test_list_cameras_with_limit(self):
        """测试带限制的摄像头列表"""
        from core.video_handler import list_available_cameras
        cameras = list_available_cameras(max_devices=3)
        assert isinstance(cameras, list)
        assert len(cameras) <= 3