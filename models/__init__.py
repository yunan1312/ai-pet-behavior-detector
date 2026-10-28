from .pose_estimator import AnimalPoseProcessor, create_pose_estimator, KEYPOINTS, KEYPOINT_CONNECTIONS
from .behavior_classifier import BehaviorClassifier, RuleBasedBehaviorClassifier, create_behavior_classifier, BEHAVIORS

PoseProcessor = AnimalPoseProcessor

__all__ = [
    'PoseProcessor', 'AnimalPoseProcessor', 'create_pose_estimator',
    'KEYPOINTS', 'KEYPOINT_CONNECTIONS',
    'BehaviorClassifier', 'RuleBasedBehaviorClassifier',
    'create_behavior_classifier', 'BEHAVIORS'
]
