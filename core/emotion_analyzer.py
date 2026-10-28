import re
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class EmotionAnalyzer:
    def __init__(self):
        self.logger = get_logger()
        self._init_emotion_patterns()

    def _init_emotion_patterns(self):
        self.emotion_keywords = {
            'happy': {
                'keywords': ['开心', '快乐', '高兴', '幸福', '满足', '愉悦', '愉快', '兴奋', '激动', '太棒了', '好开心', '哈哈', '嘻嘻', '😊', '😄', '🥰', '开心', 'happy', 'joy', 'delighted', 'pleased'],
                'pattern': r'开心|快乐|高兴|幸福|满足|愉悦|愉快|兴奋|激动|太棒了|好开心|哈哈|嘻嘻|😊|😄|🥰|happy|joy|delighted|pleased'
            },
            'excited': {
                'keywords': ['哇', '哇塞', '哇哦', '天啊', '天哪', '太厉害了', '太神奇了', '好棒', '好厉害', '惊喜', '惊讶', '震惊', '😲', '🤩', '✨', 'wow', 'amazing', 'incredible', 'fantastic'],
                'pattern': r'哇|哇塞|哇哦|天啊|天哪|太厉害了|太神奇了|好棒|好厉害|惊喜|惊讶|震惊|😲|🤩|✨|wow|amazing|incredible|fantastic'
            },
            'curious': {
                'keywords': ['什么', '怎么', '为什么', '呢', '吗', '?', '？', '好奇', '想知道', '想了解', '这是', '那是', '🤔', '❓', 'curious', 'wonder', 'what', 'why', 'how'],
                'pattern': r'什么|怎么|为什么|呢|吗|\?|？|好奇|想知道|想了解|这是|那是|🤔|❓|curious|wonder|what|why|how'
            },
            'affectionate': {
                'keywords': ['爱你', '喜欢你', '爱', '喜欢', '宝贝', '亲爱的', '心爱', '❤️', '💕', '💖', '💗', 'love', 'adore', 'cherish', 'dear'],
                'pattern': r'爱你|喜欢你|爱|喜欢|宝贝|亲爱的|心爱|❤️|💕|💖|💗|love|adore|cherish|dear'
            },
            'relaxed': {
                'keywords': ['放松', '休息', '舒服', '惬意', '悠闲', '安详', '宁静', '平静', '😌', '😴', '🌙', 'relaxed', 'calm', 'peaceful', 'comfortable'],
                'pattern': r'放松|休息|舒服|惬意|悠闲|安详|宁静|平静|😌|😴|🌙|relaxed|calm|peaceful|comfortable'
            },
            'playful': {
                'keywords': ['玩', '玩耍', '游戏', '有趣', '好玩', '来玩', '一起玩', '🎾', '🎮', '⚽', 'play', 'fun', 'game', 'playful'],
                'pattern': r'玩|玩耍|游戏|有趣|好玩|来玩|一起玩|🎾|🎮|⚽|play|fun|game|playful'
            },
            'hungry': {
                'keywords': ['饿', '想吃', '好饿', '肚子饿', '觅食', '捕猎', '🍖', '🐟', '🥩', 'hungry', 'starving', 'food', 'eat'],
                'pattern': r'饿|想吃|好饿|肚子饿|觅食|捕猎|🍖|🐟|🥩|hungry|starving|food|eat'
            },
            'sleepy': {
                'keywords': ['困', '想睡', '好困', '睡觉', '打盹', '瞌睡', '😴', '💤', '🌙', 'sleepy', 'tired', 'sleep', 'nap'],
                'pattern': r'困|想睡|好困|睡觉|打盹|瞌睡|😴|💤|🌙|sleepy|tired|sleep|nap'
            },
            'nervous': {
                'keywords': ['紧张', '不安', '担心', '害怕', '有点', '😰', '😟', '😨', 'nervous', 'anxious', 'worried', 'afraid'],
                'pattern': r'紧张|不安|担心|害怕|有点|😰|😟|😨|nervous|anxious|worried|afraid'
            },
            'agitated': {
                'keywords': ['烦躁', '生气', '恼火', '烦', '😾', '😠', '😡', 'agitated', 'angry', 'annoyed', 'irritated'],
                'pattern': r'烦躁|生气|恼火|烦|😾|😠|😡|agitated|angry|annoyed|irritated'
            }
        }

        self.emotion_voice_params = {
            'happy': {
                'rate': 1.1,
                'pitch': '+10%',
                'description': '开心愉快'
            },
            'excited': {
                'rate': 1.2,
                'pitch': '+15%',
                'description': '兴奋激动'
            },
            'curious': {
                'rate': 1.0,
                'pitch': '+5%',
                'description': '好奇疑问'
            },
            'affectionate': {
                'rate': 0.9,
                'pitch': '+8%',
                'description': '温柔亲切'
            },
            'relaxed': {
                'rate': 0.85,
                'pitch': '-5%',
                'description': '放松安详'
            },
            'playful': {
                'rate': 1.15,
                'pitch': '+12%',
                'description': '活泼 playful'
            },
            'hungry': {
                'rate': 1.05,
                'pitch': '+3%',
                'description': '饥饿渴求'
            },
            'sleepy': {
                'rate': 0.8,
                'pitch': '-10%',
                'description': '困倦慵懒'
            },
            'nervous': {
                'rate': 1.1,
                'pitch': '+5%',
                'description': '紧张不安'
            },
            'agitated': {
                'rate': 1.15,
                'pitch': '+8%',
                'description': '烦躁激动'
            },
            'neutral': {
                'rate': 1.0,
                'pitch': '+0%',
                'description': '中性自然'
            }
        }

    def analyze_emotion(self, text: str) -> Tuple[str, float]:
        if not text or not text.strip():
            return 'neutral', 0.0

        text_lower = text.lower()
        emotion_scores = {}

        for emotion, data in self.emotion_keywords.items():
            matches = re.findall(data['pattern'], text_lower)
            if matches:
                emotion_scores[emotion] = len(matches)

        if not emotion_scores:
            return 'neutral', 0.5

        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        primary_emotion = sorted_emotions[0][0]
        confidence = min(sorted_emotions[0][1] / 3.0, 1.0)

        return primary_emotion, confidence

    def get_voice_params(self, emotion: str) -> Dict:
        if emotion in self.emotion_voice_params:
            return self.emotion_voice_params[emotion].copy()
        return self.emotion_voice_params['neutral'].copy()

    def get_available_emotions(self) -> List[str]:
        return list(self.emotion_keywords.keys()) + ['neutral']

    def get_emotion_description(self, emotion: str) -> str:
        if emotion in self.emotion_voice_params:
            return self.emotion_voice_params[emotion]['description']
        return self.emotion_voice_params['neutral']['description']
