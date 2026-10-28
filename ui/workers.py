"""后台任务：模型加载、摄像头扫描、实时/视频/图片分析。

所有耗时工作都放在守护线程中执行，通过 Qt 信号回到主线程更新界面，
避免推理或磁盘读取阻塞窗口事件循环。
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from core.video_handler import FileVideoHandler, VideoStreamHandler, list_available_cameras
from core.video_handler import load_image
from utils.logger import get_logger

if TYPE_CHECKING:  # 避免在启动阶段导入模型依赖，拖慢首屏
    from core.file_analyzer import FileAnalyzer


class ModelLoader(QObject):
    """分阶段加载姿态与行为模型，并回报阶段性状态。"""

    status = pyqtSignal(str)
    loaded = pyqtSignal(object, object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, sequence_length: int = 8, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.sequence_length = sequence_length
        self._thread: Optional[threading.Thread] = None
        self._cancelled = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="ModelLoader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            from models.pose_estimator import create_pose_estimator
            from models.behavior_classifier import create_behavior_classifier

            self.status.emit("正在加载姿态估计模型…")
            pose_model, pose_processor = create_pose_estimator()
            if self._cancelled.is_set():
                return

            self.status.emit("正在初始化行为分类器…")
            behavior_model, behavior_classifier = create_behavior_classifier(
                sequence_length=self.sequence_length
            )
            if self._cancelled.is_set():
                return

            self.loaded.emit(pose_model, pose_processor, behavior_model, behavior_classifier)
        except Exception as exc:  # pragma: no cover - 依赖真实模型环境
            get_logger().error(
                "Model loading failed:\n" + traceback.format_exc()
            )
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self._cancelled.set()


class CameraScanWorker(QObject):
    """异步检测可用摄像头，避免启动时界面等待。"""

    found = pyqtSignal(list)

    def __init__(self, max_devices: int = 6, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.max_devices = max_devices

    def start(self) -> None:
        threading.Thread(target=self._run, name="CameraScan", daemon=True).start()

    def _run(self) -> None:
        try:
            cameras = list_available_cameras(self.max_devices)
        except Exception:
            get_logger().error("Camera scan failed:\n" + traceback.format_exc())
            cameras = []
        self.found.emit(cameras)


class RealtimeAnalysisWorker(QObject):
    """实时分析：持续读取最新帧并推理，结果排队回到 UI 线程。"""

    result_ready = pyqtSignal(object, object)
    stats = pyqtSignal(float)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        video_handler: VideoStreamHandler,
        analyzer,
        target_fps: int = 15,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.video_handler = video_handler
        self.analyzer = analyzer
        self.target_fps = max(1, int(target_fps))
        self.results: List[dict] = []
        self.processed_frames = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._logger = get_logger()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="RealtimeAnalysis", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = 1.0 / self.target_fps
        last_frame_count = -1
        idle_rounds = 0
        last_stats = time.time()
        stats_frames = 0
        try:
            while not self._stop.is_set():
                frame_count = getattr(self.video_handler, "frame_count", 0)
                if frame_count == last_frame_count:
                    idle_rounds += 1
                    if idle_rounds > 100 and not self.video_handler.is_opened():
                        self.failed.emit("摄像头读取中断，请检查设备连接")
                        return
                    time.sleep(0.01)
                    continue
                last_frame_count = frame_count
                idle_rounds = 0

                started = time.perf_counter()
                ret, frame = self.video_handler.read_frame()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                result = self.analyzer.analyze_frame(frame)
                self.processed_frames += 1
                stats_frames += 1
                if result.get("triggered"):
                    item = {
                        "timestamp": round(
                            self.processed_frames / max(1, self.target_fps), 2
                        ),
                        "behavior": result.get("behavior", ""),
                        "confidence": result.get("confidence", 0.0),
                        "intent": result.get("intent", ""),
                    }
                    self.results.append(item)
                self.result_ready.emit(frame, result)

                now = time.time()
                if now - last_stats >= 1.0:
                    self.stats.emit(stats_frames / max(0.001, now - last_stats))
                    stats_frames = 0
                    last_stats = now

                elapsed = time.perf_counter() - started
                remaining = interval - elapsed
                if remaining > 0:
                    self._stop.wait(remaining)
        except Exception as exc:  # pragma: no cover - 依赖摄像头硬件
            self._logger.error("Realtime analysis error:\n" + traceback.format_exc())
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def stop(self, timeout: float = 2.5) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)


class VideoAnalysisWorker(QObject):
    """视频文件分析：按时间采样跳帧读取，支持真实进度与取消。"""

    progress = pyqtSignal(int, str)
    preview = pyqtSignal(object, object)
    finished = pyqtSignal(object, object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        file_path: str,
        analyzer: "FileAnalyzer",
        confidence_threshold: float = 0.75,
        analysis_fps: float = 3.0,
        max_frames: int = 1500,
        preview_interval: float = 0.25,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.file_path = file_path
        self.analyzer = analyzer
        self.confidence_threshold = confidence_threshold
        self.analysis_fps = max(0.5, float(analysis_fps))
        self.max_frames = max(30, int(max_frames))
        self.preview_interval = preview_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._logger = get_logger()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="VideoAnalysis", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        handler = FileVideoHandler(self.file_path)
        if not handler.open():
            self.failed.emit("无法打开视频文件，请确认格式是否受支持（MP4 / AVI / MOV）")
            return

        info = handler.get_info()
        total_frames = int(info.get("total_frames") or 0)
        fps = float(info.get("fps") or 0.0)
        if total_frames <= 0:
            self.failed.emit("视频内容为空或无法读取帧信息")
            handler.release()
            return

        # 按目标分析帧率跳帧，并限制最大分析帧数，兼顾速度与覆盖度
        step_by_fps = max(1, int(round(fps / self.analysis_fps))) if fps > 0 else 1
        step_by_cap = max(1, math.ceil(total_frames / self.max_frames))
        step = max(1, step_by_fps, step_by_cap)
        indices = list(range(0, total_frames, step))
        total_to_analyze = len(indices)

        results: List[dict] = []
        last_preview = 0.0
        analyzed = 0
        try:
            # 避免与实时分析的分类器状态交叉
            if hasattr(self.analyzer, "behavior_classifier"):
                self.analyzer.behavior_classifier.reset_buffer()

            for position, frame_index in enumerate(indices):
                if self._stop.is_set():
                    self.cancelled.emit()
                    return
                if not handler.seek(frame_index):
                    break
                ret, frame = handler.read_frame()
                if not ret or frame is None:
                    continue

                analyzed += 1
                result = self.analyzer.analyze_single_image(frame)
                behavior = result.get("behavior", "")
                confidence = float(result.get("confidence", 0.0) or 0.0)
                if (
                    behavior not in {"", "idle", "error"}
                    and confidence >= max(0.3, self.confidence_threshold * 0.6)
                ):
                    results.append(
                        {
                            "timestamp": round(frame_index / fps, 2) if fps > 0 else position,
                            "behavior": behavior,
                            "confidence": confidence,
                            "intent": result.get("intent", ""),
                        }
                    )

                now = time.time()
                if now - last_preview >= self.preview_interval or position == total_to_analyze - 1:
                    self.preview.emit(frame, result)
                    last_preview = now

                percent = int((position + 1) / max(1, total_to_analyze) * 100)
                self.progress.emit(
                    percent,
                    f"已分析 {position + 1}/{total_to_analyze} 个采样帧 · 识别 {len(results)} 次",
                )

            if hasattr(self.analyzer, "behavior_classifier"):
                self.analyzer.behavior_classifier.reset_buffer()
            summary = ""
            if results:
                summary = max(results, key=lambda item: item.get("confidence", 0)).get(
                    "intent", ""
                )
            payload = {
                "source_path": self.file_path,
                "duration": info.get("duration", 0.0),
                "total_frames": total_frames,
                "analyzed_frames": analyzed,
                "fps": fps,
                "summary": summary,
            }
            self.finished.emit(results, payload)
        except Exception as exc:  # pragma: no cover - 依赖视频编解码器
            self._logger.error("Video analysis error:\n" + traceback.format_exc())
            self.failed.emit(str(exc))
        finally:
            handler.release()


class ImageAnalysisWorker(QObject):
    """单张图片分析，避免大图推理阻塞界面。"""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image_path: str, analyzer: "FileAnalyzer", parent=None) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.analyzer = analyzer

    def start(self) -> None:
        threading.Thread(target=self._run, name="ImageAnalysis", daemon=True).start()

    def _run(self) -> None:
        image = load_image(self.image_path)
        if image is None:
            self.failed.emit("无法加载图片，请确认格式是否受支持（JPG / PNG / BMP）")
            return
        try:
            result = self.analyzer.analyze_single_image(image)
            self.finished.emit({"image": image, "result": result})
        except Exception as exc:  # pragma: no cover
            get_logger().error("Image analysis error:\n" + traceback.format_exc())
            self.failed.emit(str(exc))
