"""情感分析器单元测试"""

import pytest


class TestEmotionAnalyzer:
    """测试情感分析器"""

    def test_analyze_empty_text(self, emotion_analyzer):
        """测试空文本"""
        emotion, confidence = emotion_analyzer.analyze_emotion('')
        assert emotion == 'neutral'
        assert confidence == 0.0

    def test_analyze_happy_text(self, emotion_analyzer):
        """测试开心文本"""
        emotion, confidence = emotion_analyzer.analyze_emotion('我好开心啊！太棒了！')
        assert emotion == 'happy'
        assert confidence > 0

    def test_analyze_sleepy_text(self, emotion_analyzer):
        """测试困倦文本"""
        emotion, confidence = emotion_analyzer.analyze_emotion('好困啊，想睡觉了')
        assert emotion == 'sleepy'
        assert confidence > 0

    def test_analyze_neutral_text(self, emotion_analyzer):
        """测试中性文本"""
        emotion, confidence = emotion_analyzer.analyze_emotion('今天天气不错')
        assert emotion == 'neutral'

    def test_get_voice_params(self, emotion_analyzer):
        """测试获取语音参数"""
        params = emotion_analyzer.get_voice_params('happy')
        assert 'rate' in params
        assert 'pitch' in params
        assert 'description' in params

    def test_get_voice_params_unknown(self, emotion_analyzer):
        """测试未知情感的语音参数"""
        params = emotion_analyzer.get_voice_params('unknown_emotion')
        assert params == emotion_analyzer.get_voice_params('neutral')

    def test_get_available_emotions(self, emotion_analyzer):
        """测试获取可用情感列表"""
        emotions = emotion_analyzer.get_available_emotions()
        assert 'happy' in emotions
        assert 'neutral' in emotions
        assert len(emotions) > 5

    def test_get_emotion_description(self, emotion_analyzer):
        """测试获取情感描述"""
        desc = emotion_analyzer.get_emotion_description('happy')
        assert isinstance(desc, str)
        assert len(desc) > 0
