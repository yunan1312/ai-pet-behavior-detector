"""意图映射器单元测试"""

import pytest


class TestIntentMapper:
    """测试意图映射器"""

    def test_map_behavior_to_intent_zh(self, intent_mapper):
        """测试中文意图映射"""
        intent_mapper.set_language('zh_CN')
        intent = intent_mapper.map_behavior_to_intent('stretching', 0.8)
        assert isinstance(intent, str)
        assert len(intent) > 0

    def test_map_behavior_to_intent_en(self, intent_mapper):
        """测试英文意图映射"""
        intent_mapper.set_language('en_US')
        intent = intent_mapper.map_behavior_to_intent('stretching', 0.8)
        assert isinstance(intent, str)
        assert len(intent) > 0

    def test_map_all_behaviors(self, intent_mapper):
        """测试所有行为都能映射到意图"""
        behaviors = [
            'stretching', 'head_bunting', 'tail_raising', 'belly_exposure',
            'slow_blinking', 'airplane_ears', 'walking', 'running',
            'sitting', 'sleeping', 'grooming', 'hunting', 'playing',
            'tail_lashing', 'kneading', 'idle'
        ]
        for behavior in behaviors:
            intent = intent_mapper.map_behavior_to_intent(behavior, 0.8)
            assert isinstance(intent, str)
            assert len(intent) > 0

    def test_get_emotion_for_behavior(self, intent_mapper):
        """测试行为到情绪的映射"""
        assert intent_mapper.get_emotion_for_behavior('stretching') == 'relaxed'
        assert intent_mapper.get_emotion_for_behavior('head_bunting') == 'affectionate'
        assert intent_mapper.get_emotion_for_behavior('tail_raising') == 'happy'
        assert intent_mapper.get_emotion_for_behavior('sleeping') == 'sleepy'
        assert intent_mapper.get_emotion_for_behavior('unknown') == 'neutral'

    def test_set_language(self, intent_mapper):
        """测试语言切换"""
        intent_mapper.set_language('en_US')
        assert intent_mapper.get_language() == 'en_US'

        intent_mapper.set_language('zh_CN')
        assert intent_mapper.get_language() == 'zh_CN'

    def test_invalid_language(self, intent_mapper):
        """测试无效语言回退"""
        intent_mapper.set_language('invalid')
        assert intent_mapper.get_language() == 'zh_CN'


class TestIntentSmoothing:
    """测试意图平滑器"""

    def test_empty_smoothing(self):
        """测试空平滑器"""
        from core.intent_mapper import IntentSmoothing
        smoother = IntentSmoothing(window_size=3)
        behavior, confidence = smoother.get_smoothed_behavior()
        assert behavior == 'idle'
        assert confidence == 0.0

    def test_single_behavior(self):
        """测试单个行为平滑"""
        from core.intent_mapper import IntentSmoothing
        smoother = IntentSmoothing(window_size=3)
        smoother.add_behavior('sitting', 0.8)
        behavior, confidence = smoother.get_smoothed_behavior()
        assert behavior == 'sitting'

    def test_consistent_behavior(self):
        """测试一致行为的平滑"""
        from core.intent_mapper import IntentSmoothing
        smoother = IntentSmoothing(window_size=3)
        smoother.add_behavior('sitting', 0.8)
        smoother.add_behavior('sitting', 0.9)
        smoother.add_behavior('sitting', 0.85)
        behavior, confidence = smoother.get_smoothed_behavior()
        assert behavior == 'sitting'
        assert confidence > 0.8

    def test_mixed_behaviors(self):
        """测试混合行为的平滑"""
        from core.intent_mapper import IntentSmoothing
        smoother = IntentSmoothing(window_size=3)
        smoother.add_behavior('sitting', 0.8)
        smoother.add_behavior('walking', 0.7)
        smoother.add_behavior('sitting', 0.9)
        behavior, confidence = smoother.get_smoothed_behavior()
        # sitting 出现 2 次，应被选中
        assert behavior == 'sitting'

    def test_reset(self):
        """测试重置"""
        from core.intent_mapper import IntentSmoothing
        smoother = IntentSmoothing(window_size=3)
        smoother.add_behavior('sitting', 0.8)
        smoother.reset()
        assert len(smoother.behavior_history) == 0
