"""合成姿态关键点：用于验证尺度不变性、帧率不变性与姿态判别"""

KEYPOINT_NAMES = (
    'nose', 'left_eye', 'right_eye', 'withers', 'tail_base',
    'left_front_shoulder', 'right_front_shoulder',
    'left_front_elbow', 'right_front_elbow',
    'left_front_paw', 'right_front_paw',
    'left_back_shoulder', 'right_back_shoulder',
    'left_back_elbow', 'right_back_elbow',
    'left_back_paw', 'right_back_paw',
)


def _to_keypoints(points):
    return {name: (float(x), float(y), 0.9) for name, (x, y) in points.items()}


def _transform(keypoints, scale=1.0, dx=0.0, dy=0.0):
    return {
        name: (x * scale + dx, y * scale + dy, conf)
        for name, (x, y, conf) in keypoints.items()
    }


# 侧视站立：背线水平，四条腿撑地
_STANDING = {
    'nose': (260, 205),
    'left_eye': (285, 195),
    'right_eye': (295, 195),
    'withers': (300, 200),
    'tail_base': (500, 215),
    'left_front_shoulder': (305, 210),
    'right_front_shoulder': (315, 210),
    'left_front_elbow': (305, 300),
    'right_front_elbow': (315, 300),
    'left_front_paw': (305, 400),
    'right_front_paw': (315, 400),
    'left_back_shoulder': (480, 220),
    'right_back_shoulder': (495, 222),
    'left_back_elbow': (480, 310),
    'right_back_elbow': (495, 310),
    'left_back_paw': (480, 400),
    'right_back_paw': (495, 400),
}

# 坐着：背线接近竖直，尾根贴地
_SITTING = {
    'nose': (300, 215),
    'left_eye': (292, 208),
    'right_eye': (308, 208),
    'withers': (300, 250),
    'tail_base': (312, 430),
    'left_front_shoulder': (300, 265),
    'right_front_shoulder': (320, 265),
    'left_front_elbow': (300, 355),
    'right_front_elbow': (320, 355),
    'left_front_paw': (300, 450),
    'right_front_paw': (320, 450),
    'left_back_shoulder': (300, 420),
    'right_back_shoulder': (320, 420),
    'left_back_elbow': (296, 435),
    'right_back_elbow': (316, 435),
    'left_back_paw': (296, 445),
    'right_back_paw': (316, 445),
}

# 趴卧：整条背线贴近地面
_LYING = {
    'nose': (265, 405),
    'left_eye': (285, 398),
    'right_eye': (295, 398),
    'withers': (300, 390),
    'tail_base': (520, 400),
    'left_front_shoulder': (305, 400),
    'right_front_shoulder': (315, 400),
    'left_front_elbow': (305, 408),
    'right_front_elbow': (315, 408),
    'left_front_paw': (305, 413),
    'right_front_paw': (315, 413),
    'left_back_shoulder': (500, 402),
    'right_back_shoulder': (512, 403),
    'left_back_elbow': (498, 410),
    'right_back_elbow': (510, 410),
    'left_back_paw': (496, 415),
    'right_back_paw': (510, 415),
}


# 狮身人面趴姿：胸口抬起、后躯贴地，背线向后下方倾斜
_SPHINX = {
    'nose': (250, 245),
    'left_eye': (270, 240),
    'right_eye': (284, 240),
    'withers': (300, 260),
    'tail_base': (520, 340),
    'left_front_shoulder': (305, 275),
    'right_front_shoulder': (315, 275),
    'left_front_elbow': (305, 345),
    'right_front_elbow': (315, 345),
    'left_front_paw': (305, 415),
    'right_front_paw': (315, 415),
    'left_back_shoulder': (490, 350),
    'right_back_shoulder': (505, 350),
    'left_back_elbow': (495, 390),
    'right_back_elbow': (510, 390),
    'left_back_paw': (495, 415),
    'right_back_paw': (510, 415),
}


def standing_keypoints(scale=1.0, dx=0.0, dy=0.0):
    return _transform(_to_keypoints(_STANDING), scale, dx, dy)


def sitting_keypoints(scale=1.0, dx=0.0, dy=0.0):
    return _transform(_to_keypoints(_SITTING), scale, dx, dy)


def lying_keypoints(scale=1.0, dx=0.0, dy=0.0):
    keypoints = _transform(_to_keypoints(_LYING), scale, dx, dy)
    return keypoints


def sphinx_keypoints(scale=1.0, dx=0.0, dy=0.0):
    return _transform(_to_keypoints(_SPHINX), scale, dx, dy)


def grooming_keypoints(scale=1.0, dx=0.0, dy=0.0):
    """坐着的口鼻贴住前爪"""
    keypoints = sitting_keypoints(scale, dx, dy)
    keypoints['nose'] = (302 * scale + dx, 448 * scale + dy, 0.9)
    return keypoints


def bbox_for(keypoints, pad=20.0):
    xs = [x for x, _y, _c in keypoints.values()]
    ys = [y for _x, y, _c in keypoints.values()]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
