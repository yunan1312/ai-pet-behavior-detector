"""AP-10K 姿态模型解码与兼容性测试"""

import numpy as np
import pytest

from models.behavior_classifier import RuleBasedBehaviorClassifier
from models.pose_estimator import AP10K_KEYPOINT_MAP, AnimalPoseProcessor


def _make_processor() -> AnimalPoseProcessor:
    """不加载模型，仅构造用于解码测试的处理器实例"""
    processor = AnimalPoseProcessor.__new__(AnimalPoseProcessor)
    processor.input_size = (256, 256)
    return processor


def _synthetic_simcc():
    """构造每个关键点峰值位置已知的 SimCC 输出"""
    pred_x = np.zeros((17, 512), dtype=np.float32)
    pred_y = np.zeros((17, 512), dtype=np.float32)
    for index in range(17):
        pred_x[index, 2 * index] = 0.9
        pred_y[index, 2 * index + 1] = 0.8
    return pred_x, pred_y


# AP-10K 官方顺序：0 左眼 1 右眼 2 鼻 3 颈 4 尾根 5-7 左前肢 8-10 右前肢
# 11-13 左后肢 14-16 右后肢
def test_decode_simcc_uses_official_ap10k_order():
    processor = _make_processor()
    pred_x, pred_y = _synthetic_simcc()

    keypoints = processor._decode_simcc(pred_x, pred_y, 256, 256)

    expected_names = set(AP10K_KEYPOINT_MAP.values())
    assert set(keypoints) == expected_names
    assert len(keypoints) == 17

    # 索引 0 左眼：(2*0/2, (2*0+1)/2) = (0, 0.5)
    assert keypoints['left_eye'] == pytest.approx((0.0, 0.5, 0.85))
    # 索引 2 鼻：(2, 2.5)
    assert keypoints['nose'] == pytest.approx((2.0, 2.5, 0.85))
    # 索引 3 颈/背线：(3, 3.5)
    assert keypoints['withers'] == pytest.approx((3.0, 3.5, 0.85))
    # 索引 4 尾根：(4, 4.5)
    assert keypoints['tail_base'] == pytest.approx((4.0, 4.5, 0.85))
    # 索引 16 右后爪：(16, 16.5)
    assert keypoints['right_back_paw'] == pytest.approx((16.0, 16.5, 0.85))

    # AP-10K 没有耳部关键点，不应输出假数据
    assert 'left_ear' not in keypoints
    assert 'right_ear' not in keypoints


def test_decode_simcc_scales_to_region_size():
    processor = _make_processor()
    pred_x, pred_y = _synthetic_simcc()

    keypoints = processor._decode_simcc(pred_x, pred_y, 512, 384)

    # 输入 256 → 区域 512x384，坐标按比例放大
    assert keypoints['nose'] == pytest.approx((4.0, 3.75, 0.85))
    assert keypoints['right_back_paw'] == pytest.approx((32.0, 24.75, 0.85))


def test_feature_vector_stays_81_dim_without_ears():
    processor = _make_processor()
    pred_x, pred_y = _synthetic_simcc()
    keypoints = processor._decode_simcc(pred_x, pred_y, 256, 256)

    features = processor.extract_pose_features(keypoints)

    assert features.shape == (81,)
    # 缺失的耳部关键点在特征中补零（第 4、5 个关键点，各 3 维）
    assert np.all(features[9:15] == 0.0)


def test_onnx_preprocess_converts_bgr_to_rgb():
    processor = _make_processor()

    class _FakeInput:
        name = 'input'

    class _FakeSession:
        def __init__(self):
            self.fed = None

        def get_inputs(self):
            return [_FakeInput()]

        def run(self, _outputs, feed):
            self.fed = feed['input']
            pred_x, pred_y = _synthetic_simcc()
            return [pred_x[np.newaxis, ...], pred_y[np.newaxis, ...]]

    session = _FakeSession()
    processor.pose_model = session

    # 纯红色帧（OpenCV BGR 顺序：(0, 0, 255)）
    frame = np.full((32, 32, 3), (0, 0, 255), dtype=np.uint8)
    processor._estimate_onnx(frame, 32, 32)

    assert session.fed is not None
    assert session.fed.shape == (1, 3, 256, 256)
    # RGB 通道 0 应为红色，归一化后约为 2.25；通道 2 应为蓝色，约为 -1.80
    assert session.fed[0, 0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)
    assert session.fed[0, 2, 0, 0] == pytest.approx((0.0 - 0.406) / 0.225, abs=1e-4)


class TestClassifierWithoutEars:
    """AP-10K 无耳部关键点时分类器不应误判"""

    @pytest.fixture
    def keypoints_without_ears(self):
        return {
            'left_eye': (280, 100, 0.9),
            'right_eye': (320, 100, 0.9),
            'nose': (300, 100, 0.9),
            'withers': (300, 100, 0.9),
            'tail_base': (500, 130, 0.9),
            'left_front_shoulder': (240, 140, 0.9),
            'right_front_shoulder': (360, 140, 0.9),
            'left_front_elbow': (240, 240, 0.9),
            'right_front_elbow': (360, 240, 0.9),
            'left_front_paw': (240, 340, 0.9),
            'right_front_paw': (360, 340, 0.9),
            'left_back_shoulder': (220, 150, 0.9),
            'right_back_shoulder': (380, 150, 0.9),
            'left_back_elbow': (220, 260, 0.9),
            'right_back_elbow': (380, 260, 0.9),
            'left_back_paw': (220, 450, 0.9),
            'right_back_paw': (380, 450, 0.9),
        }

    def test_body_height_is_normalized_by_body_length(self, keypoints_without_ears):
        classifier = RuleBasedBehaviorClassifier()
        keypoints = keypoints_without_ears

        height = classifier._body_height(keypoints)

        # 背线 (300,100) → 尾根 (500,130) 为身体尺度，爪尖到背线的垂直距离除以它
        scale = ((500 - 300) ** 2 + (130 - 100) ** 2) ** 0.5
        assert height == pytest.approx(350 / scale)
        # 归一化后与分辨率无关：整体放大 2 倍结果不变
        doubled = {
            name: (x * 2, y * 2, conf)
            for name, (x, y, conf) in keypoints.items()
        }
        assert classifier._body_height(doubled) == pytest.approx(height)

    def test_airplane_ears_not_triggered_without_ears(self, keypoints_without_ears):
        classifier = RuleBasedBehaviorClassifier()

        behavior, _confidence = classifier._classify_from_keypoints(keypoints_without_ears)

        assert behavior != 'airplane_ears'

def test_letterbox_preserves_aspect_ratio():
    processor = _make_processor()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    letterboxed, ratio, pad_x, pad_y = processor._letterbox(frame, 256, 256)

    assert letterboxed.shape == (256, 256, 3)
    assert ratio == pytest.approx(256 / 200)
    assert pad_x == 0
    assert pad_y == 64


def test_decode_simcc_compensates_letterbox_padding():
    processor = _make_processor()
    pred_x = np.zeros((17, 512), dtype=np.float32)
    pred_y = np.zeros((17, 512), dtype=np.float32)
    pred_x[:, 256] = 0.5  # x_input = 128
    pred_y[:, 256] = 0.5  # y_input = 128

    keypoints = processor._decode_simcc(
        pred_x, pred_y, 256, 256, scale=2.0, pad_x=16, pad_y=16
    )

    assert keypoints['nose'][0] == pytest.approx((128 - 16) / 2.0)
    assert keypoints['nose'][1] == pytest.approx((128 - 16) / 2.0)


def test_geometry_fallback_returns_no_fake_keypoints():
    """没有姿态模型时不应编造关键点"""
    processor = _make_processor()
    assert processor._estimate_geometry_based(np.zeros((64, 64, 3), np.uint8), 64, 64) == {}


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.values, dtype=np.float32)

    def __getitem__(self, index):
        return self


class _FakeBox:
    def __init__(self, box, confidence, class_id=15):
        self.xyxy = [_FakeTensor(box)]
        self.conf = [confidence]
        self.cls = [class_id]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, _frame, **_kwargs):
        return [_FakeResult(self.boxes)]


def test_detection_keeps_main_subject_and_drops_tiny_boxes():
    """大而可信的主体优先，远处微小目标不应抢占姿态估计区域"""
    processor = _make_processor()
    processor.detector = _FakeDetector([
        _FakeBox((100, 100, 140, 140), 0.95),   # 0.5% 面积，过滤
        _FakeBox((50, 50, 400, 450), 0.30),     # 51% 面积
        _FakeBox((300, 300, 380, 380), 0.90),   # 2% 面积
    ])

    detections = processor._detect_animals(np.zeros((480, 640, 3), np.uint8))

    assert len(detections) == 2
    assert detections[0]['box'] == (50, 50, 400, 450)
    assert all(d['area_ratio'] >= 0.01 for d in detections)
