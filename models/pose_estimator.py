"""动物姿态估计器 - 两阶段检测流程

阶段1: YOLO 目标检测 → 定位画面中的猫/狗
阶段2: 姿态估计 → 对检测到的动物区域提取关键点

支持多种模型后端：
- YOLOv8/YOLO11 目标检测（检测猫/狗位置）
- RTMPose-M AP-10K ONNX（动物专用姿态模型，本项目默认）
- YOLO-pose（COCO 人体关键点，仅作兼容回退）

自动下载所需模型，无模型时给出明确提示。
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from utils.logger import get_logger
from utils.constants import KEYPOINTS, KEYPOINT_CONNECTIONS

# COCO 数据集中猫/狗的类别 ID
ANIMAL_CLASS_IDS = {15, 16, 17}  # cat=15, dog=16, horse=17
ANIMAL_CLASS_NAMES = {15: 'cat', 16: 'dog', 17: 'horse'}

# AP-10K 官方关键点顺序（17 个），来源 mmpose ap10k_dataset.py：
# 0 左眼 1 右眼 2 鼻子 3 颈部(withers) 4 尾根 5 左前肩 6 左前肘 7 左前爪
# 8 右前肩 9 右前肘 10 右前爪 11 左后肩 12 左后肘 13 左后爪
# 14 右后肩 15 右后肘 16 右后爪
AP10K_KEYPOINTS = (
    'left_eye', 'right_eye', 'nose', 'withers', 'tail_base',
    'left_front_shoulder', 'left_front_elbow', 'left_front_paw',
    'right_front_shoulder', 'right_front_elbow', 'right_front_paw',
    'left_back_shoulder', 'left_back_elbow', 'left_back_paw',
    'right_back_shoulder', 'right_back_elbow', 'right_back_paw',
)

# AP-10K 索引 → 项目关键点名称。AP-10K 不含耳部关键点，
# 因此 left_ear / right_ear 在该模型下不会输出（特征提取会补零）。
AP10K_KEYPOINT_MAP = {index: name for index, name in enumerate(AP10K_KEYPOINTS)}


class AnimalPoseProcessor:
    """动物姿态处理器 — 两阶段检测：目标检测 → 姿态估计"""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.detector = None          # 阶段1: 目标检测器 (YOLO)
        self.pose_model = None         # 阶段2: 姿态估计模型
        self.use_onnx_pose = False
        self.input_size = (256, 256)
        self.detector_name = "none"
        self.pose_model_name = "none"
        self.logger = get_logger()

        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

        if model_path is None:
            self._init_detector(model_dir)   # 先初始化目标检测器
            self._init_pose_model(model_dir) # 再初始化姿态模型
        else:
            self._load_model(model_path)

    # ==================== 阶段1: 目标检测 ====================

    def _init_detector(self, model_dir: str):
        """初始化目标检测器，用于定位画面中的猫/狗"""
        detector_candidates = [
            ("yolov8n.pt", "YOLOv8n-det"),
            ("yolo11n.pt", "YOLO11n-det"),
            ("yolov8s.pt", "YOLOv8s-det"),
        ]

        for filename, name in detector_candidates:
            # 搜索路径: 项目根目录 → models/
            for search_dir in [os.path.dirname(model_dir), model_dir]:
                path = os.path.join(search_dir, filename)
                if os.path.exists(path):
                    if self._load_yolo_detector(path, name):
                        return

        # 尝试自动下载 yolov8n.pt（仅 ~6MB），测试环境跳过
        if not os.environ.get('PYTEST_CURRENT_TEST'):
            self._auto_download_detector(model_dir)

    def _load_yolo_detector(self, path: str, name: str) -> bool:
        """加载 YOLO 目标检测模型"""
        try:
            from ultralytics import YOLO
            self.detector = YOLO(path)
            self.detector_name = name
            self.logger.info(f"Loaded object detector: {name}")
            print(f"[OK] 目标检测模型已加载: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Detector load failed ({path}): {e}")
            return False

    def _auto_download_detector(self, model_dir: str):
        """自动下载轻量级目标检测模型"""
        try:
            from ultralytics import YOLO
            save_path = os.path.join(model_dir, "yolov8n.pt")
            self.logger.info("Auto-downloading YOLOv8n detector...")
            print("[INFO] 正在下载目标检测模型 (yolov8n.pt, 约6MB)，请稍候...")
            self.detector = YOLO("yolov8n.pt")  # ultralytics 自动从 hub 下载
            self.detector_name = "YOLOv8n-det (downloaded)"
            self.logger.info("Detector auto-downloaded successfully")
            print("[OK] 目标检测模型下载完成")
        except Exception as e:
            self.logger.warning(f"Auto-download failed: {e}")
            self.detector = None
            print("[WARN] 无法加载/下载目标检测模型，将使用基础模式")

    def _detect_animals(self, frame: np.ndarray) -> List[Dict]:
        """在帧中检测猫/狗等动物，返回检测结果列表

        Returns:
            [
                {'box': (x1, y1, x2, y2), 'class_id': int, 'class_name': str, 'confidence': float},
                ...
            ]
        """
        if self.detector is None:
            return []

        try:
            # 0.35 兼顾远距离/侧身动物：阈值过高会漏检低置信度目标，
            # 过低则容易把背景误当动物。类别已限定为猫/狗/马，误报可控。
            results = self.detector(frame, verbose=False, conf=0.35, classes=list(ANIMAL_CLASS_IDS))
            detections = []

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = ANIMAL_CLASS_NAMES.get(cls_id, f'animal_{cls_id}')
                        detections.append({
                            'box': tuple(xyxy.astype(int)),
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'confidence': conf,
                        })

            # 按置信度排序，取最佳结果
            detections.sort(key=lambda d: d['confidence'], reverse=True)
            return detections

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []

    # ==================== 阶段2: 姿态估计 ====================

    def _init_pose_model(self, model_dir: str):
        """按优先级尝试加载姿态估计模型

        RTMPose-M AP-10K 是动物专用模型（训练集含猫/狗等 54 种动物），
        优先使用；YOLO-pose 为人体关键点模型，仅在缺少动物模型时回退。
        """
        candidates = [
            (os.path.join(model_dir, 'rtmpose_ap10k.onnx'), "RTMPose-M AP-10K", True),
            (os.path.join(os.path.dirname(model_dir), 'yolo11n-pose.pt'), "YOLO11n-pose", False),
            (os.path.join(model_dir, 'yolo11n-pose.pt'), "YOLO11n-pose", False),
            (os.path.join(model_dir, 'yolov8n-pose.pt'), "YOLOv8n-pose", False),
            (os.path.join(os.path.dirname(model_dir), 'yolov8n-pose.pt'), "YOLOv8n-pose", False),
        ]

        for path, name, is_onnx in candidates:
            if os.path.exists(path):
                if is_onnx:
                    if self._load_onnx_model(path):
                        return
                else:
                    if self._load_yolo_pose(path, name):
                        return

        self.logger.warning("No pose model available; using geometry-based estimation")
        print("[INFO] 未找到姿态估计模型，将使用基于几何特征的行为分析模式")

    def _load_yolo_pose(self, model_path: str, name: str) -> bool:
        """加载 YOLO 姿态估计模型"""
        try:
            from ultralytics import YOLO
            self.pose_model = YOLO(model_path)
            self.use_onnx_pose = False
            self.pose_model_name = name
            self.logger.info(f"Loaded pose model: {name}")
            print(f"[OK] 姿态估计模型已加载: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Pose model load failed: {e}")
            return False

    def _load_onnx_model(self, model_path: str) -> bool:
        """加载 ONNX 姿态模型"""
        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.pose_model = ort.InferenceSession(model_path, providers=providers)
            self.use_onnx_pose = True
            self.pose_model_name = "RTMPose-M AP-10K"

            # 从模型输入形状读取尺寸（本模型为 [N, 3, 256, 256]）
            input_shape = self.pose_model.get_inputs()[0].shape
            if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                self.input_size = (input_shape[3], input_shape[2])

            self.logger.info(f"Loaded ONNX pose model: {model_path} (input {self.input_size})")
            print(f"[OK] RTMPose-M AP-10K 姿态模型已加载 (输入 {self.input_size[0]}x{self.input_size[1]})")
            return True
        except Exception as e:
            self.logger.error(f"ONNX pose model failed: {e}")
            return False

    def _estimate_pose_in_region(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Dict[str, Tuple[float, float, float]]:
        """在指定区域内进行姿态估计"""
        x1, y1, x2, y2 = box
        h_frame, w_frame = frame.shape[:2]

        # 裁剪区域：与 MMPose GetBBoxCenterScale(padding=1.25) 一致，
        # 检测框四周各外扩 12.5%，让四肢/尾巴完整落入输入区域。
        pad_x = int(round((x2 - x1) * 0.125))
        pad_y = int(round((y2 - y1) * 0.125))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w_frame, x2 + pad_x)
        y2 = min(h_frame, y2 + pad_y)

        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return {}

        if self.use_onnx_pose and self.pose_model is not None:
            kpts = self._estimate_onnx(region, x2 - x1, y2 - y1)
        elif self.pose_model is not None:
            kpts = self._estimate_yolo(region, x2 - x1, y2 - y1)
        else:
            kpts = self._estimate_geometry_based(region, x2 - x1, y2 - y1)

        # 将局部坐标映射回全局坐标
        global_kpts = {}
        for name, (x, y, conf) in kpts.items():
            global_kpts[name] = (x + x1, y + y1, conf)

        return global_kpts

    def _estimate_onnx(self, frame: np.ndarray, img_w: int, img_h: int) -> Dict:
        """RTMPose ONNX 推理（MMPose 预处理：BGR→RGB + ImageNet 归一化）"""
        try:
            input_name = self.pose_model.get_inputs()[0].name
            input_w, input_h = self.input_size

            resized = cv2.resize(frame, (input_w, input_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_img = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            input_img = ((input_img - mean) / std).transpose(2, 0, 1)[np.newaxis, ...]

            outputs = self.pose_model.run(None, {input_name: input_img})

            if len(outputs) >= 2:
                return self._decode_simcc(outputs[0][0], outputs[1][0], img_w, img_h)
            return self._process_onnx_outputs(outputs, img_w, img_h)
        except Exception as e:
            self.logger.error(f"ONNX error: {e}")
            return self._estimate_geometry_based(frame, img_w, img_h)

    def _estimate_yolo(self, frame: np.ndarray, img_w: int, img_h: int) -> Dict:
        """YOLO 姿态模型推理"""
        try:
            results = self.pose_model(frame, verbose=False)
            if results and len(results) > 0:
                keypoints = results[0].keypoints
                if keypoints is not None and len(keypoints) > 0:
                    kpts_xy = keypoints[0].xy.cpu().numpy()
                    kpts_conf = keypoints[0].conf.cpu().numpy() if keypoints[0].conf is not None else None
                    return self._map_yolo_to_animal(kpts_xy, kpts_conf, img_w, img_h)
        except Exception as e:
            self.logger.error(f"YOLO pose error: {e}")
        return self._estimate_geometry_based(frame, img_w, img_h)

    def _estimate_geometry_based(self, frame: np.ndarray, img_w: int, img_h: int) -> Dict:
        """基于几何特征的关键点估算（无姿态模型时的回退方案）"""
        cx, cy = img_w / 2, img_h / 2
        result = {}
        offsets = {
            'left_eye': (-img_w*0.12, -img_h*0.25), 'right_eye': (img_w*0.12, -img_h*0.25),
            'nose': (0, -img_h*0.18), 'left_ear': (-img_w*0.28, -img_h*0.35), 'right_ear': (img_w*0.28, -img_h*0.35),
            'left_front_shoulder': (-img_w*0.3, -img_h*0.05), 'right_front_shoulder': (img_w*0.3, -img_h*0.05),
            'left_front_elbow': (-img_w*0.35, img_h*0.15), 'right_front_elbow': (img_w*0.35, img_h*0.15),
            'left_front_paw': (-img_w*0.3, img_h*0.4), 'right_front_paw': (img_w*0.3, img_h*0.4),
            'left_back_shoulder': (-img_w*0.25, img_h*0.15), 'right_back_shoulder': (img_w*0.25, img_h*0.15),
            'left_back_elbow': (-img_w*0.3, img_h*0.35), 'right_back_elbow': (img_w*0.3, img_h*0.35),
            'left_back_paw': (-img_w*0.25, img_h*0.55), 'right_back_paw': (img_w*0.25, img_h*0.55),
        }
        for name, (dx, dy) in offsets.items():
            result[name] = (max(0, min(cx + dx, img_w)), max(0, min(cy + dy, img_h)), 0.5)
        return result

    # ==================== SimCC 解码 & 映射 ====================

    def _decode_simcc(self, pred_x: np.ndarray, pred_y: np.ndarray,
                      img_w: int, img_h: int) -> Dict[str, Tuple[float, float, float]]:
        """SimCC 输出解码为关键点坐标

        pred_x / pred_y 形状为 [17, S]，S = 输入边长 × simcc_split_ratio(2)。
        按 AP-10K 官方索引映射到项目关键点名；AP-10K 没有耳部关键点，
        因此不会输出 left_ear / right_ear（特征提取时补零）。

        置信度取 SimCC 峰值（模型原始分数，非概率），裁剪到 0~1
        供 UI 与规则分类器统一使用。
        """
        result: Dict[str, Tuple[float, float, float]] = {}
        num_kpts = min(pred_x.shape[0], pred_y.shape[0])

        split_x = pred_x.shape[1] / self.input_size[0]
        split_y = pred_y.shape[1] / self.input_size[1]
        scale_x = img_w / self.input_size[0]
        scale_y = img_h / self.input_size[1]

        for index in range(num_kpts):
            name = AP10K_KEYPOINT_MAP.get(index)
            if name is None:
                continue
            x_idx = int(np.argmax(pred_x[index]))
            y_idx = int(np.argmax(pred_y[index]))
            peak = (float(np.max(pred_x[index])) + float(np.max(pred_y[index]))) / 2.0
            x = float(np.clip(x_idx / split_x * scale_x, 0.0, img_w))
            y = float(np.clip(y_idx / split_y * scale_y, 0.0, img_h))
            result[name] = (x, y, float(np.clip(peak, 0.0, 1.0)))

        return result

    def _process_onnx_outputs(self, outputs: list, img_w: int, img_h: int) -> Dict:
        result = {}
        try:
            kpts = outputs[0][0] if len(outputs[0].shape) == 3 else outputs[0]
            sx, sy = img_w / self.input_size[0], img_h / self.input_size[1]
            for i, name in enumerate(KEYPOINTS):
                if i < len(kpts) and len(kpts[i]) >= 2:
                    result[name] = (float(kpts[i][0]) * sx, float(kpts[i][1]) * sy,
                                    float(kpts[i][2]) if len(kpts[i]) > 2 else 0.8)
        except Exception as e:
            self.logger.error(f"ONNX output processing error: {e}")
        return result or self._estimate_geometry_based(np.zeros((img_h, img_w, 3)), img_w, img_h)

    def _map_yolo_to_animal(self, yolo_kpts, yolo_confs, img_w: int, img_h: int) -> Dict:
        """COCO 人体关键点 → 动物关键点映射"""
        yolo_names = ['nose','left_eye','right_eye','left_ear','right_ear',
                       'left_shoulder','right_shoulder','left_elbow','right_elbow',
                       'left_wrist','right_wrist','left_hip','right_hip',
                       'left_knee','right_knee','left_ankle','right_ankle']
        mapped = {}
        try:
            pts = yolo_kpts[0] if len(yolo_kpts.shape) > 2 else yolo_kpts
            confs = yolo_confs[0] if yolo_confs is not None and len(yolo_confs.shape) > 1 else (yolo_confs if yolo_confs is not None else np.ones(17))
            for i, n in enumerate(yolo_names):
                if i < len(pts) and len(pts[i]) >= 2:
                    mapped[n] = (float(pts[i][0]), float(pts[i][1]), float(confs[i]) if i < len(confs) else 0.5)
        except Exception as e:
            self.logger.error(f"Keypoint mapping error: {e}")

        def g(name, default): return mapped.get(name, default)
        c = lambda *a: g(*a, (img_w/2, img_h/2, 0.3))
        return {
            'left_eye': c('left_eye'), 'right_eye': c('right_eye'), 'nose': c('nose'),
            'left_ear': c('left_ear'), 'right_ear': c('right_ear'),
            'left_front_shoulder': c('left_shoulder'), 'right_front_shoulder': c('right_shoulder'),
            'left_front_elbow': c('left_elbow'), 'right_front_elbow': c('right_elbow'),
            'left_front_paw': c('left_wrist'), 'right_front_paw': c('right_wrist'),
            'left_back_shoulder': c('left_hip'), 'right_back_shoulder': c('right_hip'),
            'left_back_elbow': c('left_knee'), 'right_back_elbow': c('right_knee'),
            'left_back_paw': c('left_ankle'), 'right_back_paw': c('right_ankle'),
        }

    # ==================== 主入口 ====================

    def estimate(self, frame: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """主入口：两阶段检测

        流程: 目标检测(找猫/狗) → 裁剪区域 → 姿态估计 → 返回关键点
        """
        h, w = frame.shape[:2]

        # 阶段1: 检测画面中是否有动物
        animals = self._detect_animals(frame)

        if animals:
            # 找到动物 → 在最佳检测框内做姿态估计
            best = animals[0]
            box = best['box']
            self._last_detection = best  # 保存供 UI 绘制边框使用
            kpts = self._estimate_pose_in_region(frame, box)
            if kpts:
                return kpts

        # 无检测到动物或姿态估计失败 → 返回空关键点（不再生成假数据）
        self._last_detection = None
        return {}

    @property
    def last_detection(self) -> Optional[Dict]:
        """最后一次目标检测结果（供 UI 绘制检测框）"""
        return getattr(self, '_last_detection', None)

    # ==================== 特征提取 ====================

    def extract_pose_features(self, keypoints: Dict[str, Tuple[float, float, float]]) -> np.ndarray:
        """从关键点提取特征向量 (51维坐标 + 30维骨骼 = 81维)"""
        features = []
        for name in KEYPOINTS:
            if name in keypoints:
                features.extend(keypoints[name])
            else:
                features.extend([0.0, 0.0, 0.0])
        for conn_start, conn_end in KEYPOINT_CONNECTIONS:
            if conn_start in keypoints and conn_end in keypoints:
                x1, y1, _ = keypoints[conn_start]
                x2, y2, _ = keypoints[conn_end]
                features.extend([np.sqrt((x2-x1)**2 + (y2-y1)**2), np.arctan2(y2-y1, x2-x1)])
            else:
                features.extend([0.0, 0.0])
        return np.array(features, dtype=np.float32)

    def get_model_info(self) -> Dict:
        return {
            'detector': self.detector_name,
            'pose_model': self.pose_model_name,
            'use_onnx': self.use_onnx_pose,
            'input_size': self.input_size,
            'num_keypoints': len(KEYPOINTS),
        }


def create_pose_estimator(device: str = 'cpu'):
    processor = AnimalPoseProcessor(device=device)
    return None, processor
