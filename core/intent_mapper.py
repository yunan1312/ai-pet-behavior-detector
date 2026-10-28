import numpy as np
from typing import Dict, List, Tuple, Optional

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.constants import BEHAVIOR_TO_EMOTION


class IntentMapper:
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.config_loader = config_loader or ConfigLoader()
        self.intent_mapping = self.config_loader.get_intent_mapping()
        self.current_language = self.config_loader.get('speech.language', 'zh_CN')
        self.logger = get_logger()

        self.behavior_to_emotion = BEHAVIOR_TO_EMOTION

    def map_behavior_to_intent(self, behavior: str, confidence: float = 1.0) -> str:
        language_key = self.current_language if self.current_language in ['zh_CN', 'en_US'] else 'zh_CN'

        intents = self.intent_mapping.get('intents', {})

        if behavior in intents:
            intent_text = intents[behavior].get(language_key, intents[behavior].get('zh_CN', ''))
            if intent_text:
                self.logger.log_behavior(behavior, confidence, intent_text)
                return intent_text

        default_intents = {
            'stretching': {'zh_CN': '我正在舒展身体~', 'en_US': 'I am stretching my body~'},
            'head_bunting': {'zh_CN': '这是我的地盘~', 'en_US': 'This is my territory~'},
            'tail_raising': {'zh_CN': '我感到开心~', 'en_US': 'I am happy~'},
            'belly_exposure': {'zh_CN': '我完全信任你~', 'en_US': 'I completely trust you~'},
            'slow_blinking': {'zh_CN': '我感到很放松~', 'en_US': 'I feel relaxed~'},
            'airplane_ears': {'zh_CN': '我感到有点紧张...', 'en_US': 'I feel a bit nervous...'},
            'walking': {'zh_CN': '我在悠闲地散步~', 'en_US': 'I am taking a walk~'},
            'running': {'zh_CN': '我在奔跑~', 'en_US': 'I am running~'},
            'sitting': {'zh_CN': '我在安静地休息~', 'en_US': 'I am resting quietly~'},
            'sleeping': {'zh_CN': '我正在睡觉~', 'en_US': 'I am sleeping~'},
            'grooming': {'zh_CN': '我在梳理毛发~', 'en_US': 'I am grooming~'},
            'hunting': {'zh_CN': '我在准备捕猎！', 'en_US': 'I am hunting!~'},
            'playing': {'zh_CN': '我想玩耍~', 'en_US': 'I want to play~'},
            'tail_lashing': {'zh_CN': '我有点烦躁...', 'en_US': 'I am a bit agitated...'},
            'kneading': {'zh_CN': '我感到很放松~', 'en_US': 'I feel relaxed~'},
            'idle': {'zh_CN': '我在发呆~', 'en_US': 'I am just being idle~'}
        }

        intent_text = default_intents.get(behavior, {}).get(language_key, '')
        if not intent_text:
            intent_text = default_intents.get(behavior, {}).get('zh_CN', '我有点好奇~')

        self.logger.log_behavior(behavior, confidence, intent_text)
        return intent_text

    def set_language(self, language: str):
        if language in ['zh_CN', 'en_US']:
            self.current_language = language
            self.logger.info(f"Intent mapper language changed to: {language}")

    def get_language(self) -> str:
        return self.current_language

    def reload_mapping(self):
        self.config_loader.reload_intent_mapping()
        self.intent_mapping = self.config_loader.get_intent_mapping()
        self.logger.info("Intent mapping reloaded from config file")

    def get_available_behaviors(self) -> List[str]:
        intents = self.intent_mapping.get('intents', {})
        return list(intents.keys())

    def get_behavior_keywords(self, behavior: str) -> List[str]:
        intents = self.intent_mapping.get('intents', {})
        if behavior in intents:
            return intents[behavior].get('keywords', [])
        return []

    def get_all_intents_for_behavior(self, behavior: str) -> Dict[str, str]:
        intents = self.intent_mapping.get('intents', {})
        if behavior in intents:
            result = {}
            for lang in ['zh_CN', 'en_US']:
                if lang in intents[behavior]:
                    result[lang] = intents[behavior][lang]
            return result
        return {}

    def get_emotion_for_behavior(self, behavior: str) -> str:
        return self.behavior_to_emotion.get(behavior, 'neutral')


class IntentSmoothing:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.behavior_history: List[Tuple[str, float]] = []

    def add_behavior(self, behavior: str, confidence: float):
        self.behavior_history.append((behavior, confidence))
        if len(self.behavior_history) > self.window_size * 2:
            self.behavior_history = self.behavior_history[-self.window_size:]

    def get_smoothed_behavior(self) -> Tuple[str, float]:
        if not self.behavior_history:
            return 'idle', 0.0

        recent = self.behavior_history[-self.window_size:]

        behavior_counts: Dict[str, float] = {}
        for behavior, confidence in recent:
            if behavior not in behavior_counts:
                behavior_counts[behavior] = []
            behavior_counts[behavior].append(confidence)

        best_behavior = 'idle'
        best_score = 0.0

        for behavior, confidences in behavior_counts.items():
            avg_confidence = np.mean(confidences)
            count_bonus = len(confidences) / self.window_size * 0.1
            score = avg_confidence + count_bonus

            if score > best_score:
                best_score = score
                best_behavior = behavior

        return best_behavior, best_score

    def reset(self):
        self.behavior_history = []
