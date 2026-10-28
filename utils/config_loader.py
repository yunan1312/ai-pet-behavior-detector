import yaml
import os
from typing import Any, Dict, Optional


class ConfigLoader:
    _instance: Optional['ConfigLoader'] = None
    _config: Dict[str, Any] = {}
    _intent_mapping: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_configs()
        return cls._instance

    def _load_configs(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        app_config_path = os.path.join(base_dir, 'config', 'app_config.yaml')
        intent_mapping_path = os.path.join(base_dir, 'config', 'intent_mapping.yaml')

        if os.path.exists(app_config_path):
            with open(app_config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._get_default_config()

        if os.path.exists(intent_mapping_path):
            with open(intent_mapping_path, 'r', encoding='utf-8') as f:
                self._intent_mapping = yaml.safe_load(f) or {}
        else:
            self._intent_mapping = {}

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'recognition': {
                'confidence_threshold': 0.75,
                'target_fps': 15,
                'analysis_fps': 3.0,
                'max_analysis_frames': 1500,
                'sequence_length': 8,
                'model_path': './models/pose_model.pth'
            },
            'speech': {
                'enabled': True,
                'language': 'zh_CN',
                'voice': 'female_warm',
                'speed': 1.0,
                'volume': 0.8
            },
            'file': {
                'import_path': './imports',
                'export_path': './exports',
                'supported_video_formats': ['mp4', 'avi', 'mov'],
                'supported_image_formats': ['jpg', 'jpeg', 'png', 'bmp']
            },
            'database': {
                'path': './data/records.db'
            },
            'logging': {
                'level': 'INFO',
                'file': './data/app.log'
            },
            'app': {
                'show_welcome': True,
                'is_first_run': True,
                'auto_save_records': True
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_intent_mapping(self) -> Dict[str, Any]:
        return self._intent_mapping

    def reload_intent_mapping(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        intent_mapping_path = os.path.join(base_dir, 'config', 'intent_mapping.yaml')
        if os.path.exists(intent_mapping_path):
            with open(intent_mapping_path, 'r', encoding='utf-8') as f:
                self._intent_mapping = yaml.safe_load(f) or {}

    def get_all_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def update_config(self, key: str, value: Any):
        """更新配置项并立即持久化。

        Args:
            key: 配置键，支持点分隔符（如 'speech.language'）
            value: 配置值
        """
        self.update_many({key: value})

    def update_many(self, updates: Dict[str, Any]):
        """批量更新配置，只写一次文件，减少磁盘 IO。"""
        if not updates:
            return
        try:
            for key, value in updates.items():
                keys = key.split('.')
                config = self._config
                for k in keys[:-1]:
                    if k not in config or not isinstance(config[k], dict):
                        config[k] = {}
                    config = config[k]
                config[keys[-1]] = value
            self._save()
        except Exception as e:
            print(f"Error updating config: {str(e)}")

    def _save(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        app_config_path = os.path.join(base_dir, 'config', 'app_config.yaml')
        os.makedirs(os.path.dirname(app_config_path), exist_ok=True)
        with open(app_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, allow_unicode=True, sort_keys=False)
