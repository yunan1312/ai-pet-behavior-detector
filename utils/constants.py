"""公共常量定义，消除各模块间的重复代码"""

# 行为中文显示名称映射
BEHAVIOR_DISPLAY_NAMES = {
    'stretching': '伸懒腰',
    'head_bunting': '蹭物体',
    'tail_raising': '尾巴竖起',
    'belly_exposure': '露肚皮',
    'slow_blinking': '缓慢眨眼',
    'airplane_ears': '飞机耳',
    'walking': '行走',
    'running': '奔跑',
    'sitting': '坐着',
    'standing': '站立',
    'lying': '趴卧',
    'sleeping': '睡觉',
    'grooming': '理毛',
    'hunting': '捕猎',
    'playing': '玩耍',
    'tail_lashing': '尾巴甩动',
    'kneading': '踩奶',
    'idle': '发呆',
}

# 行为带图标的显示名称
BEHAVIOR_DISPLAY_NAMES_ICONS = {
    'stretching': '伸懒腰',
    'head_bunting': '蹭物体',
    'tail_raising': '尾巴竖起',
    'belly_exposure': '露肚皮',
    'slow_blinking': '缓慢眨眼',
    'airplane_ears': '飞机耳',
    'walking': '行走',
    'running': '奔跑',
    'sitting': '坐着',
    'standing': '站立',
    'lying': '趴卧',
    'sleeping': '睡觉',
    'grooming': '理毛',
    'hunting': '捕猎',
    'playing': '玩耍',
    'tail_lashing': '尾巴甩动',
    'kneading': '踩奶',
    'idle': '发呆',
}

# 行为到情绪的映射
BEHAVIOR_TO_EMOTION = {
    'stretching': 'relaxed',
    'head_bunting': 'affectionate',
    'tail_raising': 'happy',
    'belly_exposure': 'affectionate',
    'slow_blinking': 'relaxed',
    'airplane_ears': 'nervous',
    'walking': 'neutral',
    'running': 'playful',
    'sitting': 'relaxed',
    'standing': 'neutral',
    'lying': 'relaxed',
    'sleeping': 'sleepy',
    'grooming': 'relaxed',
    'hunting': 'hungry',
    'playing': 'playful',
    'tail_lashing': 'agitated',
    'kneading': 'relaxed',
    'idle': 'neutral',
}

# 所有支持的行为列表
BEHAVIORS = [
    'stretching', 'head_bunting', 'tail_raising', 'belly_exposure',
    'slow_blinking', 'airplane_ears', 'walking', 'running',
    'sitting', 'standing', 'lying', 'sleeping', 'grooming',
    'hunting', 'playing', 'tail_lashing', 'kneading', 'idle'
]

# 姿态关键点定义（UI 与模型共享，避免重复维护）
KEYPOINTS = [
    'left_eye', 'right_eye', 'nose',
    'left_ear', 'right_ear',
    'left_front_shoulder', 'right_front_shoulder',
    'left_front_elbow', 'right_front_elbow',
    'left_front_paw', 'right_front_paw',
    'left_back_shoulder', 'right_back_shoulder',
    'left_back_elbow', 'right_back_elbow',
    'left_back_paw', 'right_back_paw'
]

KEYPOINT_CONNECTIONS = [
    ('left_eye', 'right_eye'),
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
    ('left_front_shoulder', 'right_front_shoulder'),
    ('left_front_shoulder', 'left_front_elbow'),
    ('right_front_shoulder', 'right_front_elbow'),
    ('left_front_elbow', 'left_front_paw'),
    ('right_front_elbow', 'right_front_paw'),
    ('left_back_shoulder', 'right_back_shoulder'),
    ('left_back_shoulder', 'left_back_elbow'),
    ('right_back_shoulder', 'right_back_elbow'),
    ('left_back_elbow', 'left_back_paw'),
    ('right_back_elbow', 'right_back_paw'),
]

# 行为对应情绪的中文名称
EMOTION_DISPLAY_NAMES = {
    'happy': '开心',
    'excited': '兴奋',
    'curious': '好奇',
    'affectionate': '亲昵',
    'relaxed': '放松',
    'playful': '活泼',
    'hungry': '饥饿',
    'sleepy': '困倦',
    'nervous': '紧张',
    'agitated': '烦躁',
    'neutral': '平静',
}

# 应用版本
APP_VERSION = "2.1.0"
APP_NAME = "宠物行为智能分析软件"
APP_NAME_EN = "PetBehaviorAnalyzer"
