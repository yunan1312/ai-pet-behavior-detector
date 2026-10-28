import cv2
import numpy as np
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

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        try:
            keypoints = self.pose_processor.estimate(frame)

            features = self.pose_processor.extract_pose_features(keypoints)

            self.behavior_classifier.add_pose_features(features)
            self.behavior_classifier.classifier.add_pose_features(keypoints)

            if len(self.behavior_classifier.sequence_buffer) >= self.behavior_classifier.sequence_length:
                smoothed_behavior, smoothed_confidence = self.smoother.get_smoothed_behavior()

                behavior_result = self.behavior_classifier.classify()

                if behavior_result[0] != 'idle':
                    self.smoother.add_behavior(behavior_result[0], behavior_result[1])
                    smoothed_behavior, smoothed_confidence = self.smoother.get_smoothed_behavior()

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

    def analyze_video_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        results = []
        for frame in frames:
            result = self.analyze_frame(frame)
            results.append(result)
        return results

    def should_speak(self) -> bool:
        return (
            self.last_behavior != 'idle' and
            self.last_behavior != self.behavior_classifier.sequence_buffer[-1] if self.behavior_classifier.sequence_buffer else False
        )

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

    def analyze_single_image(self, image: np.ndarray) -> Dict:
        """分析单张图片
        
        Args:
            image: OpenCV格式的图片数组
            
        Returns:
            包含行为、置信度、意图和关键点的字典
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
            features = self.pose_processor.extract_pose_features(keypoints)

            self.behavior_classifier.reset_buffer()
            self.behavior_classifier.add_pose_features(features)

            behavior, confidence = self.behavior_classifier.classify()
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

    def analyze_video_frames(self, frames: List[np.ndarray], progress_callback: Optional[Callable] = None) -> List[Dict]:
        results = []
        total = len(frames)

        self.behavior_classifier.reset_buffer()

        for i, frame in enumerate(frames):
            keypoints = self.pose_processor.estimate(frame)
            features = self.pose_processor.extract_pose_features(keypoints)
            self.behavior_classifier.add_pose_features(features)

            if len(self.behavior_classifier.sequence_buffer) >= self.behavior_classifier.sequence_length:
                behavior, confidence = self.behavior_classifier.classify()
                if confidence >= self.confidence_threshold:
                    intent = self.intent_mapper.map_behavior_to_intent(behavior, confidence)
                else:
                    intent = ''
            else:
                behavior = 'idle'
                confidence = 0.0
                intent = ''

            results.append({
                'frame_idx': i,
                'behavior': behavior,
                'confidence': confidence,
                'intent': intent,
                'keypoints': keypoints
            })

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback((i + 1) / total * 100)

        return results
