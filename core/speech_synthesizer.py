"""语音合成模块 - 基于 edge-tts 的 TTS 引擎

支持多语言、多音色、情感驱动的语音参数调整。
使用线程锁保证线程安全，使用临时文件自动清理避免泄漏。
"""

import asyncio
import edge_tts
import numpy as np
from typing import Optional, Callable, Dict
import sys
import os
import subprocess
import tempfile
import threading

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from core.emotion_analyzer import EmotionAnalyzer


VOICE_MAP = {
    'zh_CN': {
        'female_warm': 'zh-CN-XiaoxiaoNeural',
        'male_gentle': 'zh-CN-YunxiNeural',
        'child_lively': 'zh-CN-XiaoyiNeural'
    },
    'en_US': {
        'female_warm': 'en-US-JennyNeural',
        'male_gentle': 'en-US-BrandonNeural',
        'child_lively': 'en-US-SaraNeural'
    }
}

VOICE_DISPLAY_NAMES = {
    'zh_CN': {
        'female_warm': '亲切女声',
        'male_gentle': '温和男声',
        'child_lively': '活泼童声'
    },
    'en_US': {
        'female_warm': 'Warm Female',
        'male_gentle': 'Gentle Male',
        'child_lively': 'Lively Child'
    }
}


class SpeechSynthesizer:
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.config_loader = config_loader or ConfigLoader()
        self.language = self.config_loader.get('speech.language', 'zh_CN')
        self.voice_name = self.config_loader.get('speech.voice', 'female_warm')
        self.speed = self.config_loader.get('speech.speed', 1.0)
        self.volume = self.config_loader.get('speech.volume', 0.8)
        self.logger = get_logger()
        self.emotion_analyzer = EmotionAnalyzer()

        self.communicate: Optional[edge_tts.Communicate] = None
        self._is_speaking = False
        self._speaking_lock = threading.Lock()
        self.current_task: Optional[asyncio.Task] = None
        self._pending_text = ""
        self._pygame_initialized = False
        self._pygame_lock = threading.Lock()
        self.current_emotion = 'neutral'
        self.current_emotion_confidence = 0.0

    @property
    def is_speaking(self) -> bool:
        with self._speaking_lock:
            return self._is_speaking

    @is_speaking.setter
    def is_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value

    def _get_voice_key(self) -> str:
        if self.voice_name in VOICE_MAP.get(self.language, VOICE_MAP['zh_CN']):
            return self.voice_name
        return 'female_warm'

    def _get_voice_id(self) -> str:
        voice_map = VOICE_MAP.get(self.language, VOICE_MAP['zh_CN'])
        return voice_map.get(self._get_voice_key(), voice_map['female_warm'])

    def _init_pygame(self):
        with self._pygame_lock:
            if not self._pygame_initialized:
                try:
                    import pygame
                    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                    self._pygame_initialized = True
                except Exception as e:
                    self.logger.error(f"Failed to init pygame: {e}")

    def _play_audio_file(self, file_path: str) -> bool:
        try:
            import pygame
            self._init_pygame()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.set_volume(max(0.0, min(1.0, self.volume)))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except Exception as e:
            self.logger.error(f"Pygame playback failed: {e}")

        try:
            if sys.platform == 'win32':
                import winsound
                winsound.PlaySound(file_path, winsound.SND_FILENAME)
                return True
        except Exception as e:
            self.logger.error(f"Winsound playback failed: {e}")

        try:
            if sys.platform == 'win32':
                os.startfile(file_path)
                import time
                time.sleep(2)
                return True
            elif sys.platform == 'darwin':
                subprocess.run(['afplay', file_path], check=True)
                return True
            else:
                subprocess.run(['aplay', file_path], check=True)
                return True
        except Exception as e:
            self.logger.error(f"System playback failed: {e}")

        return False

    async def _speak_async(self, text: str, emotion: Optional[str] = None):
        """异步语音合成
        
        Args:
            text: 要合成的文本
            emotion: 情感类型（可选）
        """
        temp_path = None
        try:
            self.is_speaking = True
            self._pending_text = text

            voice_id = self._get_voice_id()

            if emotion is None:
                emotion, confidence = self.emotion_analyzer.analyze_emotion(text)
                self.current_emotion = emotion
                self.current_emotion_confidence = confidence
            else:
                self.current_emotion = emotion
                self.current_emotion_confidence = 1.0

            emotion_params = self.emotion_analyzer.get_voice_params(emotion)

            base_rate = self.speed
            emotion_rate = emotion_params['rate']
            final_rate = base_rate * emotion_rate

            rate_value = int((final_rate - 1) * 100)
            if rate_value > 0:
                speed_str = f"+{rate_value}%"
            elif rate_value < 0:
                speed_str = f"{rate_value}%"
            else:
                speed_str = "+0%"

            volume_value = int((max(0.0, min(1.0, self.volume)) - 1) * 100)
            volume_str = f"{volume_value}%" if volume_value else "+0%"

            emotion_desc = self.emotion_analyzer.get_emotion_description(emotion)
            self.logger.log_speech(text, self.language, self.voice_name, emotion_desc)

            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            self.communicate = edge_tts.Communicate(
                text,
                voice=voice_id,
                rate=speed_str,
                volume=volume_str
            )

            await self.communicate.save(temp_path)

            self._play_audio_file(temp_path)

        except Exception as e:
            self.logger.error(f"TTS synthesis error: {str(e)}")
        finally:
            self.is_speaking = False
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def speak(self, text: str, blocking: bool = False, emotion: Optional[str] = None):
        if not text:
            return

        if self.is_speaking:
            return

        def run_speak():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._speak_async(text, emotion))
                loop.close()
            except Exception as e:
                self.logger.error(f"Speech thread error: {str(e)}")
                self.is_speaking = False

        if blocking:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._speak_async(text, emotion))
                loop.close()
            except Exception as e:
                self.logger.error(f"Blocking speech error: {str(e)}")
                self.is_speaking = False
        else:
            thread = threading.Thread(target=run_speak, daemon=True)
            thread.start()

    def speak_in_thread(self, text: str, emotion: Optional[str] = None):
        self.speak(text, blocking=False, emotion=emotion)

    def set_emotion(self, emotion: str):
        if emotion in self.emotion_analyzer.get_available_emotions():
            self.current_emotion = emotion
            self.logger.info(f"Speech emotion set to: {emotion}")

    def get_current_emotion(self) -> str:
        return self.current_emotion

    def get_emotion_confidence(self) -> float:
        return self.current_emotion_confidence

    def get_available_emotions(self) -> list:
        return self.emotion_analyzer.get_available_emotions()

    def set_language(self, language: str):
        if language in VOICE_MAP:
            self.language = language
            self.config_loader.update_config('speech.language', language)
            self.logger.info(f"TTS language changed to: {language}")

    def set_voice(self, voice_name: str):
        voice_map = VOICE_MAP.get(self.language, VOICE_MAP['zh_CN'])
        if voice_name in voice_map:
            self.voice_name = voice_name
            self.config_loader.update_config('speech.voice', voice_name)
            self.logger.info(f"TTS voice changed to: {voice_name}")

    def set_speed(self, speed: float):
        self.speed = max(0.5, min(2.0, speed))
        self.config_loader.update_config('speech.speed', self.speed)

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))
        self.config_loader.update_config('speech.volume', self.volume)

    def get_available_languages(self) -> list:
        return list(VOICE_MAP.keys())

    def get_available_voices(self, language: Optional[str] = None) -> dict:
        lang = language or self.language
        voice_map = VOICE_MAP.get(lang, VOICE_MAP['zh_CN'])
        display_names = VOICE_DISPLAY_NAMES.get(lang, VOICE_DISPLAY_NAMES['zh_CN'])
        return {key: display_names.get(key, key) for key in voice_map.keys()}

    def stop(self):
        if self.communicate:
            try:
                self.communicate = None
            except Exception:
                pass
        self.is_speaking = False
        if self.current_task:
            self.current_task.cancel()
            self.current_task = None

        try:
            import pygame
            if self._pygame_initialized:
                pygame.mixer.music.stop()
        except Exception:
            pass

    def get_current_settings(self) -> dict:
        return {
            'language': self.language,
            'voice': self.voice_name,
            'speed': self.speed,
            'volume': self.volume,
            'is_speaking': self.is_speaking
        }
