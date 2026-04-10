FEEDBACK_LIBRARY = {
    'left_shoulder': {
        'too_high': 'Lower your left arm slightly - avoid overextending.',
        'too_low': 'Raise your left arm higher toward the target position.',
        'lateral_shift': 'Keep your left arm aligned with your shoulder.',
    },
    'right_shoulder': {
        'too_high': 'Lower your right arm slightly - avoid overextending.',
        'too_low': 'Raise your right arm higher toward the target position.',
        'lateral_shift': 'Keep your right arm aligned with your shoulder.',
    },
    'left_elbow': {
        'too_bent': 'Straighten your left arm more.',
        'too_straight': 'Allow a slight bend in your left elbow for comfort.',
    },
    'right_elbow': {
        'too_bent': 'Straighten your right arm more.',
        'too_straight': 'Allow a slight bend in your right elbow for comfort.',
    },
    'left_hip': {
        'too_flexed': 'Stand up taller on your left side - avoid leaning too far forward.',
        'too_extended': 'Bend forward slightly more from your left hip.',
        'lateral_shift': 'Keep your hips level and centered.',
    },
    'right_hip': {
        'too_flexed': 'Stand up taller on your right side - avoid leaning too far forward.',
        'too_extended': 'Bend forward slightly more from your right hip.',
        'lateral_shift': 'Keep your hips level and centered.',
    },
    'left_knee': {
        'too_bent': 'Straighten your left knee slightly - do not go too deep.',
        'too_straight': 'Bend your left knee more toward the target angle.',
        'valgus': 'Prevent your left knee from caving inward - track over your toes.',
    },
    'right_knee': {
        'too_bent': 'Straighten your right knee slightly - do not go too deep.',
        'too_straight': 'Bend your right knee more toward the target angle.',
        'valgus': 'Prevent your right knee from caving inward - track over your toes.',
    },
    'left_ankle': {
        'too_flexed': 'Ease off the stretch on your left ankle slightly.',
        'too_extended': 'Allow your left ankle to flex more naturally.',
    },
    'right_ankle': {
        'too_flexed': 'Ease off the stretch on your right ankle slightly.',
        'too_extended': 'Allow your right ankle to flex more naturally.',
    },
    'trunk_lean': {
        'too_forward': 'Lift your chest and stand taller - avoid excessive forward lean.',
        'too_upright': 'Allow a slight forward lean as required by the exercise.',
        'lateral_shift': 'Keep your torso centered - avoid leaning to one side.',
    },
}

HIP_DOMINANT_JOINTS = {'left_hip', 'right_hip'}
KNEE_DOMINANT_JOINTS = {'left_knee', 'right_knee'}
SHOULDER_DOMINANT_JOINTS = {'left_shoulder', 'right_shoulder'}
ELBOW_JOINTS = {'left_elbow', 'right_elbow'}
ANKLE_JOINTS = {'left_ankle', 'right_ankle'}


def _determine_direction(joint_name, patient_angle, target_angle):
    if joint_name == 'trunk_lean':
        if patient_angle > target_angle + 5:
            return 'too_forward'
        elif patient_angle < target_angle - 5:
            return 'too_upright'
        return None

    if joint_name in HIP_DOMINANT_JOINTS:
        if patient_angle < target_angle - 5:
            return 'too_flexed'
        elif patient_angle > target_angle + 5:
            return 'too_extended'
        return None

    if joint_name in KNEE_DOMINANT_JOINTS:
        if patient_angle < target_angle - 5:
            return 'too_bent'
        elif patient_angle > target_angle + 5:
            return 'too_straight'
        return None

    if joint_name in SHOULDER_DOMINANT_JOINTS:
        if patient_angle > target_angle + 5:
            return 'too_high'
        elif patient_angle < target_angle - 5:
            return 'too_low'
        return None

    if joint_name in ELBOW_JOINTS:
        if patient_angle < target_angle - 5:
            return 'too_bent'
        elif patient_angle > target_angle + 5:
            return 'too_straight'
        return None

    if joint_name in ANKLE_JOINTS:
        if patient_angle < target_angle - 5:
            return 'too_flexed'
        elif patient_angle > target_angle + 5:
            return 'too_extended'
        return None

    return None


def generate_feedback(deviations, target_angles):
    feedback_list = []
    for dev in deviations:
        joint = dev['joint']
        error = dev['error']
        patient_angle = dev['patient_angle']
        target_angle = dev['target_angle']

        direction = _determine_direction(joint, patient_angle, target_angle)

        joint_feedback = FEEDBACK_LIBRARY.get(joint, {})
        message = joint_feedback.get(direction) if direction else None

        if not message:
            if error > 0:
                message = f"Adjust your {joint.replace('_', ' ')} - you are {error:.1f} deg off the target."
            else:
                message = f"Your {joint.replace('_', ' ')} is well aligned."

        severity = _classify_severity(error)
        feedback_list.append({
            'joint': joint,
            'error': round(error, 1),
            'direction': direction,
            'message': message,
            'severity': severity,
            'patient_angle': round(patient_angle, 1),
            'target_angle': round(target_angle, 1),
        })

    return feedback_list


def _classify_severity(error):
    abs_error = abs(error)
    if abs_error <= 5:
        return 'excellent'
    elif abs_error <= 10:
        return 'good'
    elif abs_error <= 20:
        return 'fair'
    else:
        return 'poor'
