import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
SESSION_FOLDER = os.path.join(BASE_DIR, 'data', 'sessions')
GOLD_STANDARD_FOLDER = os.path.join(BASE_DIR, 'data', 'gold_standards')
REPORT_FOLDER = os.path.join(BASE_DIR, 'static', 'reports')

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

MAX_CONTENT_LENGTH = 100 * 1024 * 1024

MEDIAPIPE_CONFIDENCE = 0.5

TOP_K_DEVIATIONS = 3
ANGLE_TOLERANCE_DEFAULT = 15.0

JOINT_DEFINITIONS = {
    'left_shoulder': {
        'landmarks': (23, 11, 13),
        'weight': 1.0,
        'label': 'Left Shoulder',
    },
    'right_shoulder': {
        'landmarks': (24, 12, 14),
        'weight': 1.0,
        'label': 'Right Shoulder',
    },
    'left_elbow': {
        'landmarks': (11, 13, 15),
        'weight': 0.7,
        'label': 'Left Elbow',
    },
    'right_elbow': {
        'landmarks': (12, 14, 16),
        'weight': 0.7,
        'label': 'Right Elbow',
    },
    'left_hip': {
        'landmarks': (11, 23, 25),
        'weight': 1.3,
        'label': 'Left Hip',
    },
    'right_hip': {
        'landmarks': (12, 24, 26),
        'weight': 1.3,
        'label': 'Right Hip',
    },
    'left_knee': {
        'landmarks': (23, 25, 27),
        'weight': 1.5,
        'label': 'Left Knee',
    },
    'right_knee': {
        'landmarks': (24, 26, 28),
        'weight': 1.5,
        'label': 'Right Knee',
    },
    'left_ankle': {
        'landmarks': (25, 27, 31),
        'weight': 1.0,
        'label': 'Left Ankle',
    },
    'right_ankle': {
        'landmarks': (26, 28, 32),
        'weight': 1.0,
        'label': 'Right Ankle',
    },
}

SCORING = {
    'excellent_threshold': 5.0,
    'good_threshold': 10.0,
    'fair_threshold': 20.0,
    'max_error': 45.0,
}

REP_DETECTION_MIN_FRAMES = 10
REP_DETECTION_PROMINENCE = 0.3

for d in [UPLOAD_FOLDER, SESSION_FOLDER, GOLD_STANDARD_FOLDER, REPORT_FOLDER]:
    os.makedirs(d, exist_ok=True)
