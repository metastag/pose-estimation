import json
import os
from config import GOLD_STANDARD_FOLDER, ANGLE_TOLERANCE_DEFAULT


EXERCISES = {
    'squat': {
        'name': 'Squat',
        'description': 'A fundamental lower-body exercise targeting quads, glutes, and hamstrings.',
        'category': 'Lower Body',
        'difficulty': 'Intermediate',
        'target_angles': {
            'left_hip': 90.0,
            'right_hip': 90.0,
            'left_knee': 90.0,
            'right_knee': 90.0,
            'left_ankle': 90.0,
            'right_ankle': 90.0,
            'left_shoulder': 90.0,
            'right_shoulder': 90.0,
            'left_elbow': 170.0,
            'right_elbow': 170.0,
            'trunk_lean': 10.0,
        },
        'tolerances': {
            'left_hip': 15.0,
            'right_hip': 15.0,
            'left_knee': 15.0,
            'right_knee': 15.0,
            'left_ankle': 20.0,
            'right_ankle': 20.0,
            'left_shoulder': 20.0,
            'right_shoulder': 20.0,
            'left_elbow': 25.0,
            'right_elbow': 25.0,
            'trunk_lean': 15.0,
        },
        'primary_joint': 'left_knee',
        'key_joints': [
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'trunk_lean',
        ],
        'rep_direction': 'min',
        'standing_angles': {
            'left_hip': 170.0,
            'right_hip': 170.0,
            'left_knee': 170.0,
            'right_knee': 170.0,
            'left_ankle': 170.0,
            'right_ankle': 170.0,
            'left_shoulder': 10.0,
            'right_shoulder': 10.0,
            'left_elbow': 170.0,
            'right_elbow': 170.0,
            'trunk_lean': 5.0,
        },
    },
    'shoulder_flexion': {
        'name': 'Shoulder Flexion',
        'description': 'Raising the arm forward and upward to improve shoulder mobility and strength.',
        'category': 'Upper Body',
        'difficulty': 'Beginner',
        'target_angles': {
            'left_shoulder': 170.0,
            'right_shoulder': 170.0,
            'left_elbow': 175.0,
            'right_elbow': 175.0,
            'left_hip': 170.0,
            'right_hip': 170.0,
            'left_knee': 170.0,
            'right_knee': 170.0,
            'left_ankle': 170.0,
            'right_ankle': 170.0,
            'trunk_lean': 5.0,
        },
        'tolerances': {
            'left_shoulder': 15.0,
            'right_shoulder': 15.0,
            'left_elbow': 20.0,
            'right_elbow': 20.0,
            'left_hip': 15.0,
            'right_hip': 15.0,
            'left_knee': 20.0,
            'right_knee': 20.0,
            'left_ankle': 20.0,
            'right_ankle': 20.0,
            'trunk_lean': 10.0,
        },
        'primary_joint': 'left_shoulder',
        'key_joints': [
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'trunk_lean',
        ],
        'rep_direction': 'max',
        'standing_angles': {
            'left_shoulder': 10.0,
            'right_shoulder': 10.0,
            'left_elbow': 170.0,
            'right_elbow': 170.0,
            'left_hip': 170.0,
            'right_hip': 170.0,
            'left_knee': 170.0,
            'right_knee': 170.0,
            'left_ankle': 170.0,
            'right_ankle': 170.0,
            'trunk_lean': 5.0,
        },
    },
    'forward_lunge': {
        'name': 'Forward Lunge',
        'description': 'A unilateral lower-body exercise improving balance, quads, and glute strength.',
        'category': 'Lower Body',
        'difficulty': 'Intermediate',
        'target_angles': {
            'left_hip': 90.0,
            'right_hip': 90.0,
            'left_knee': 90.0,
            'right_knee': 90.0,
            'left_ankle': 90.0,
            'right_ankle': 80.0,
            'left_shoulder': 10.0,
            'right_shoulder': 10.0,
            'left_elbow': 170.0,
            'right_elbow': 170.0,
            'trunk_lean': 10.0,
        },
        'tolerances': {
            'left_hip': 15.0,
            'right_hip': 15.0,
            'left_knee': 15.0,
            'right_knee': 15.0,
            'left_ankle': 20.0,
            'right_ankle': 20.0,
            'left_shoulder': 25.0,
            'right_shoulder': 25.0,
            'left_elbow': 25.0,
            'right_elbow': 25.0,
            'trunk_lean': 15.0,
        },
        'primary_joint': 'left_knee',
        'key_joints': [
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'trunk_lean',
        ],
        'rep_direction': 'min',
        'standing_angles': {
            'left_hip': 170.0,
            'right_hip': 170.0,
            'left_knee': 170.0,
            'right_knee': 170.0,
            'left_ankle': 170.0,
            'right_ankle': 170.0,
            'left_shoulder': 10.0,
            'right_shoulder': 10.0,
            'left_elbow': 170.0,
            'right_elbow': 170.0,
            'trunk_lean': 5.0,
        },
    },
    'side_leg_raise': {
        'name': 'Side Leg Raise',
        'description': 'An abduction exercise strengthening the hip abductors and improving lateral stability.',
        'category': 'Lower Body',
        'difficulty': 'Beginner',
        'target_angles': {
            'left_hip': 135.0,
            'right_hip': 135.0,
            'left_knee': 175.0,
            'right_knee': 175.0,
            'left_ankle': 170.0,
            'right_ankle': 170.0,
            'left_shoulder': 10.0,
            'right_shoulder': 10.0,
            'left_elbow': 170.0,
            'right_elbow': 170.0,
            'trunk_lean': 8.0,
        },
        'tolerances': {
            'left_hip': 15.0,
            'right_hip': 15.0,
            'left_knee': 15.0,
            'right_knee': 15.0,
            'left_ankle': 20.0,
            'right_ankle': 20.0,
            'left_shoulder': 25.0,
            'right_shoulder': 25.0,
            'left_elbow': 25.0,
            'right_elbow': 25.0,
            'trunk_lean': 12.0,
        },
        'primary_joint': 'left_hip',
        'key_joints': [
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'trunk_lean',
        ],
        'rep_direction': 'max',
        'standing_angles': {
            'left_hip': 170.0,
            'right_hip': 170.0,
            'left_knee': 170.0,
            'right_knee': 170.0,
            'left_ankle': 170.0,
            'right_ankle': 170.0,
            'left_shoulder': 10.0,
            'right_shoulder': 10.0,
            'left_elbow': 170.0,
            'right_elbow': 170.0,
            'trunk_lean': 5.0,
        },
    },
    'standing_hamstring_stretch': {
        'name': 'Standing Hamstring Stretch',
        'description': 'A flexibility exercise targeting the hamstrings and lower back.',
        'category': 'Flexibility',
        'difficulty': 'Beginner',
        'target_angles': {
            'left_hip': 90.0,
            'right_hip': 120.0,
            'left_knee': 170.0,
            'right_knee': 170.0,
            'left_ankle': 170.0,
            'right_ankle': 170.0,
            'left_shoulder': 10.0,
            'right_shoulder': 10.0,
            'left_elbow': 170.0,
            'right_elbow': 170.0,
            'trunk_lean': 45.0,
        },
        'tolerances': {
            'left_hip': 20.0,
            'right_hip': 20.0,
            'left_knee': 15.0,
            'right_knee': 15.0,
            'left_ankle': 20.0,
            'right_ankle': 20.0,
            'left_shoulder': 25.0,
            'right_shoulder': 25.0,
            'left_elbow': 25.0,
            'right_elbow': 25.0,
            'trunk_lean': 20.0,
        },
        'primary_joint': 'left_hip',
        'key_joints': [
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'trunk_lean',
        ],
        'rep_direction': 'min',
        'standing_angles': {
            'left_hip': 170.0,
            'right_hip': 170.0,
            'left_knee': 170.0,
            'right_knee': 170.0,
            'left_ankle': 170.0,
            'right_ankle': 170.0,
            'left_shoulder': 10.0,
            'right_shoulder': 10.0,
            'left_elbow': 170.0,
            'right_elbow': 170.0,
            'trunk_lean': 5.0,
        },
    },
}


def get_exercise(exercise_id):
    return EXERCISES.get(exercise_id)


def list_exercises():
    return [
        {
            'id': eid,
            'name': ex['name'],
            'description': ex['description'],
            'category': ex['category'],
            'difficulty': ex['difficulty'],
        }
        for eid, ex in EXERCISES.items()
    ]


def get_target_angles(exercise_id):
    exercise = get_exercise(exercise_id)
    if not exercise:
        return None
    return exercise['target_angles']


def get_tolerances(exercise_id):
    exercise = get_exercise(exercise_id)
    if not exercise:
        return {}
    return exercise.get('tolerances', {k: ANGLE_TOLERANCE_DEFAULT for k in exercise['target_angles']})


def save_custom_gold_standard(exercise_id, angles_data):
    filepath = os.path.join(GOLD_STANDARD_FOLDER, f"{exercise_id}_gold.json")
    with open(filepath, 'w') as f:
        json.dump(angles_data, f, indent=2)


def load_custom_gold_standard(exercise_id):
    filepath = os.path.join(GOLD_STANDARD_FOLDER, f"{exercise_id}_gold.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None
