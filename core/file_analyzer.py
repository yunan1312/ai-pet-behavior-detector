import cv2
import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Callable
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pose_estimator import AnimalPoseProcessor as PoseProcessor
from models.behavior_classifier import BehaviorClassifier
from core.intent_mapper import IntentMapper, IntentSmoothing
from utils.logger import get_logger


class FrameAnalyzer:
    def __init__(
        self,
        pose_processor: PoseProcessor,
        behavior_classifier: BehaviorClassifier,
        intent_mapper: IntentMapper,
        confidence_threshold: float = 0.75
    ):
        self.pose_processor = pose_processor
        self.behavior_classifier = behavior_classifier
        self.intent_mapper = intent_mapper
        self.confidence_threshold = confidence_threshold
        self.smoother = IntentSmoothing(window_size=3)
        self.logger = get_logger()

        self.last_behavior = 'idle'
        self.last_intent = ''
        self.last_confidence = 0.0
        self.consecutive_same_behavior = 0
        self.min_consecutive_for_speech = 2

        self.callback: Optional[Callable] = None

    def set_callback(self, callback: Callable):
        self.callback = callback

    def _detection_box(self) -> Optional[Tuple[int, int, int, int]]:
        detection = getattr(self.pose_processor, 'last_detection', None)
        if detection:
            return detection.get('box')
        return None

    def analyze_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Dict:
        try:
            keypoints = self.pose_processor.estimate(frame)
            bbox = self._detection_box()
            features = (self.pose_processor.extract_pose_features(keypoints)
                        if keypoints else None)

            if timestamp is None:
                timestamp = time.monotonic()

            self.behavior_classifier.add_pose_features(
                features, keypoints=keypoints, bbox=bbox, timestamp=timestamp
            )

            if len(self.behavior_classifier.classifier.sequence_buffer) >= 2:
                behavior, confidence = self.behavior_classifier.classify()

                if behavior != 'idle':
                    self.smoother.add_behavior(behavior, confidence)
                    smoothed_behavior, smoothed_confidence = self.smoother.get_smoothed_behavior()
                else:
                    smoothed_behavior, smoothed_confidence = 'idle', confidence

                if smoothed_confidence >= self.confidence_threshold:
                    self.last_behavior = smoothed_behavior
                    self.last_confidence = smoothed_confidence
                    self.last_intent = self.intent_mapper.map_behavior_to_intent(
                        smoothed_behavior, smoothed_confidence
                    )

                    if self.callback:
                        self.callback({
                            'behavior': self.last_behavior,
                            'confidence': self.last_confidence,
                            'intent': self.last_intent,
                            'keypoints': keypoints
                        })

                    return {
                        'behavior': self.last_behavior,
                        'confidence': self.last_confidence,
                        'intent': self.last_intent,
                        'keypoints': keypoints,
                        'triggered': True
                    }

            return {
                'behavior': 'idle',
                'confidence': 0.0,
                'intent': '',
                'keypoints': keypoints,
                'triggered': False
            }

        except Exception as e:
            self.logger.error(f"Error analyzing frame: {str(e)}")
            return {
                'behavior': 'error',
                'confidence': 0.0,
                'intent': '',
                'keypoints': {},
                'triggered': False
            }

    def analyze_video_frames(self, frames: List[np.ndarray], fps: float = 15.0) -> List[Dict]:
        results = []
        for index, frame in enumerate(frames):
            result = self.analyze_frame(frame, timestamp=index / max(fps, 1e-3))
            results.append(result)
        return results

    def should_speak(self) -> bool:
        return self.last_behavior != 'idle'

    def reset(self):
        self.behavior_classifier.reset_buffer()
        self.smoother.reset()
        self.last_behavior = 'idle'
        self.last_intent = ''
        self.last_confidence = 0.0
        self.consecutive_same_behavior = 0


class FileAnalyzer:
    def __init__(
        self,
        pose_processor: PoseProcessor,
        behavior_classifier: BehaviorClassifier,
        intent_mapper: IntentMapper,
        confidence_threshold: float = 0.75
    ):
        self.pose_processor = pose_processor
        self.behavior_classifier = behavior_classifier
        self.intent_mapper = intent_mapper
        self.confidence_threshold = confidence_threshold
        self.logger = get_logger()

    def _detection_box(self) -> Optional[Tuple[int, int, int, int]]:
        detection = getattr(self.pose_processor, 'last_detection', None)
        if detection:
            return detection.get('box')
        return None

    def analyze_single_image(self, image: np.ndarray) -> Dict:
        """分析单张图片

        单张图片没有帧间运动，只能判断静态姿态（站立/坐着/趴卧）
        以及理毛、伸懒腰这类姿态性行为。
        """
        try:
            if image is None or image.size == 0:
                return {
                    'behavior': 'idle',
                    'confidence': 0.0,
                    'intent': '',
                    'keypoints': {}
                }

            keypoints = self.pose_processor.estimate(image)
            bbox = self._detection_box()

            self.behavior_classifier.reset_buffer()
            self.behavior_classifier.add_pose_features(
                None, keypoints=keypoints, bbox=bbox, timestamp=0.0
            )

            behavior, confidence = self.behavior_classifier.classify_static()
            intent = ''
            if behavior != 'idle' and confidence >= 0.3:
                intent = self.intent_mapper.map_behavior_to_intent(behavior, confidence)

            return {
                'behavior': behavior,
                'confidence': confidence,
                'intent': intent,
                'keypoints': keypoints
            }
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return {
                'behavior': 'error',
                'confidence': 0.0,
                'intent': '',
                'keypoints': {}
            }

    def analyze_video_sample(self, frame: np.ndarray, timestamp: float) -> Dict:
        """分析视频中的一个采样帧

        采样帧按真实时间戳喂入滑动窗口，因此 3 FPS 与 15 FPS 下
        计算出的运动速度一致，走路/奔跑判据不会随采样率漂移。
        """
        try:
            keypoints = self.pose_processor.estimate(frame)
            bbox = self._detection_box()
            features = (self.pose_processor.extract_pose_features(keypoints)
                        if keypoints else None)

            self.behavior_classifier.add_pose_features(
                features, keypoints=keypoints, bbox=bbox, timestamp=timestamp
            )

            behavior, confidence = 'idle', 0.0
            if len(self.behavior_classifier.classifier.sequence_buffer) >= 2:
                behavior, confidence = self.behavior_classifier.classify()

            intent = ''
            if behavior not in {'idle', 'error'} and confidence >= self.confidence_threshold:
                intent = self.intent_mapper.map_behavior_to_intent(behavior, confidence)

            return {
                'behavior': behavior,
                'confidence': confidence,
                'intent': intent,
                'keypoints': keypoints,
                'triggered': bool(intent)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing video sample: {str(e)}")
            return {
                'behavior': 'error',
                'confidence': 0.0,
                'intent': '',
                'keypoints': {},
                'triggered': False
            }

    def analyze_video_frames(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable] = None,
        fps: float = 15.0,
    ) -> List[Dict]:
        results = []
        total = len(frames)

        self.behavior_classifier.reset_buffer()

        for i, frame in enumerate(frames):
            result = self.analyze_video_sample(frame, timestamp=i / max(fps, 1e-3))
            results.append({
                'frame_idx': i,
                'behavior': result['behavior'],
                'confidence': result['confidence'],
                'intent': result['intent'],
                'keypoints': result['keypoints'],
            })

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback((i + 1) / total * 100)

        return results
