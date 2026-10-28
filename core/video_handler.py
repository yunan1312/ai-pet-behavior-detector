import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class VideoStreamHandler:
    def __init__(self, source: int = 0, target_fps: int = 15):
        self.source = source
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()

        self.logger = get_logger()
        self.capture_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if self.is_running:
            self.logger.warning("Video stream already running")
            return True

        try:
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.source}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            self.logger.info(f"Video stream started from source: {self.source}")
            return True

        except Exception as e:
            self.logger.error(f"Error starting video stream: {str(e)}")
            return False

    def _capture_loop(self):
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame.copy()

                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed
            else:
                self.logger.warning("Failed to read frame from video source")
                self.is_running = False
                break

            time.sleep(self.frame_interval)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            return False, None

    def stop(self):
        self.is_running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.logger.info("Video stream stopped")

    def is_opened(self) -> bool:
        return self.is_running and self.cap is not None and self.cap.isOpened()

    def get_fps(self) -> float:
        return self.fps

    def get_frame_size(self) -> Tuple[int, int]:
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    def restart(self) -> bool:
        self.stop()
        time.sleep(0.5)
        return self.start()


class FileVideoHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.duration = 0.0
        self.current_frame_idx = 0
        self.is_playing = False
        self.logger = get_logger()

    def open(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video file: {self.file_path}")
                return False

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0

            self.logger.info(f"Opened video file: {self.file_path}, FPS: {self.fps}, Duration: {self.duration:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"Error opening video file: {str(e)}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx += 1
        return ret, frame

    def seek(self, frame_idx: int) -> bool:
        if self.cap is None:
            return False

        if 0 <= frame_idx < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.current_frame_idx = frame_idx
            return True
        return False

    def get_current_timestamp(self) -> float:
        return self.current_frame_idx / self.fps if self.fps > 0 else 0

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_info(self) -> dict:
        return {
            'file_path': self.file_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'duration': self.duration
        }


def load_image(file_path: str) -> Optional[np.ndarray]:
    """加载图片文件
    
    Args:
        file_path: 图片文件路径
        
    Returns:
        numpy数组格式的图片，加载失败返回None
    """
    try:
        if not os.path.exists(file_path):
            print(f"Image file not found: {file_path}")
            return None
            
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to decode image: {file_path}")
            return None
            
        return img
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None


def list_available_cameras(max_devices: int = 10) -> list:
    available_cameras = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras
