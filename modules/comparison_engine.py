import numpy as np
from scipy.signal import find_peaks
from config import JOINT_DEFINITIONS, TOP_K_DEVIATIONS, SCORING
from modules.gold_standard import get_exercise, get_target_angles, get_tolerances
from modules.feedback_engine import generate_feedback


class ComparisonResult:
    def __init__(self):
        self.frame_scores = []
        self.frame_errors = []
        self.frame_feedback = []
        self.overall_score = 0.0
        self.repetitions = []
        self.rep_scores = []
        self.joint_avg_errors = {}
        self.summary_feedback = []
        self.total_frames = 0
        self.detected_frames = 0

    def to_dict(self):
        return {
            'overall_score': round(self.overall_score, 1),
            'total_frames': self.total_frames,
            'detected_frames': self.detected_frames,
            'num_reps': len(self.repetitions),
            'rep_scores': [round(s, 1) for s in self.rep_scores],
            'joint_avg_errors': {k: round(v, 1) for k, v in self.joint_avg_errors.items()},
            'summary_feedback': self.summary_feedback,
            'frame_scores': [round(s, 1) for s in self.frame_scores],
        }


class ComparisonEngine:
    def __init__(self):
        self.max_error = SCORING['max_error']

    def compute_weighted_frame_score(self, errors, exercise_id):
        exercise = get_exercise(exercise_id)
        if not exercise:
            return 0.0

        key_joints = set(exercise.get('key_joints', []))
        total_weighted_error = 0.0
        total_weight = 0.0

        for joint_name, error in errors.items():
            if joint_name == 'frame':
                continue

            joint_def = JOINT_DEFINITIONS.get(joint_name, {})
            base_weight = joint_def.get('weight', 1.0)

            if joint_name in key_joints:
                weight = base_weight * 1.5
            elif joint_name == 'trunk_lean':
                weight = 1.5
            else:
                weight = base_weight * 0.5

            total_weighted_error += abs(error) * weight
            total_weight += weight

        if total_weight == 0:
            return 100.0

        avg_weighted_error = total_weighted_error / total_weight
        score = max(0, 100 * (1 - avg_weighted_error / self.max_error))
        return score

    def compare_frame(self, patient_angles, exercise_id):
        target = get_target_angles(exercise_id)
        tolerances = get_tolerances(exercise_id)

        if not target:
            return {}, 0.0

        errors = {}
        for joint_name, target_angle in target.items():
            patient_angle = patient_angles.get(joint_name)
            if patient_angle is None:
                continue

            tolerance = tolerances.get(joint_name, 15.0)
            error = patient_angle - target_angle

            if abs(error) <= tolerance:
                error = error * 0.3
            else:
                error = (abs(error) - tolerance) * np.sign(error) + error * 0.3 * np.sign(error)
                error = np.sign(error) * min(abs(error), self.max_error)

            errors[joint_name] = error

        score = self.compute_weighted_frame_score(errors, exercise_id)
        return errors, score

    def identify_deviations(self, errors, exercise_id, k=TOP_K_DEVIATIONS):
        exercise = get_exercise(exercise_id)
        target = get_target_angles(exercise_id)

        sorted_errors = sorted(
            [(j, e) for j, e in errors.items() if j != 'frame'],
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_k = sorted_errors[:k]
        deviations = []
        for joint_name, error in top_k:
            if abs(error) < 3:
                continue
            deviations.append({
                'joint': joint_name,
                'error': abs(error),
                'patient_angle': 0,
                'target_angle': target.get(joint_name, 0),
                'label': JOINT_DEFINITIONS.get(joint_name, {}).get('label', joint_name),
            })

        return deviations

    def detect_repetitions(self, all_angles, exercise_id):
        exercise = get_exercise(exercise_id)
        if not exercise:
            return []

        primary_joint = exercise.get('primary_joint', 'left_knee')
        direction = exercise.get('rep_direction', 'min')

        values = [a.get(primary_joint, 0) for a in all_angles if primary_joint in a]
        if not values:
            return []

        values = np.array(values)

        if direction == 'min':
            peaks, _ = find_peaks(-values, prominence=20, distance=10)
        else:
            peaks, _ = find_peaks(values, prominence=20, distance=10)

        reps = []
        for i, peak_idx in enumerate(peaks):
            start = 0 if i == 0 else (peaks[i - 1] + peak_idx) // 2
            end = len(values) - 1 if i == len(peaks) - 1 else (peak_idx + peaks[i + 1]) // 2
            reps.append({
                'rep_number': i + 1,
                'peak_frame': int(peak_idx),
                'start_frame': int(start),
                'end_frame': int(end),
            })

        return reps

    def analyze_video(self, video_data, exercise_id):
        all_angles = video_data['angles']
        result = ComparisonResult()
        result.total_frames = video_data['total_frames']
        result.detected_frames = video_data['detected_frames']

        target = get_target_angles(exercise_id)
        if not target:
            return result

        joint_error_accumulator = {}
        joint_count = {}

        for frame_angles in all_angles:
            errors, score = self.compare_frame(frame_angles, exercise_id)
            result.frame_scores.append(score)
            result.frame_errors.append(errors)

            for joint_name, error in errors.items():
                if joint_name == 'frame':
                    continue
                joint_error_accumulator[joint_name] = joint_error_accumulator.get(joint_name, 0) + abs(error)
                joint_count[joint_name] = joint_count.get(joint_name, 0) + 1

        for joint_name in joint_error_accumulator:
            if joint_count[joint_name] > 0:
                result.joint_avg_errors[joint_name] = joint_error_accumulator[joint_name] / joint_count[joint_name]

        result.repetitions = self.detect_repetitions(all_angles, exercise_id)

        if result.repetitions:
            for rep in result.repetitions:
                start = rep['start_frame']
                end = rep['end_frame']
                rep_scores = result.frame_scores[start:end + 1]
                if rep_scores:
                    result.rep_scores.append(np.mean(rep_scores))
                else:
                    result.rep_scores.append(0.0)
        else:
            result.rep_scores = [np.mean(result.frame_scores)] if result.frame_scores else [0.0]

        if result.frame_scores:
            result.overall_score = np.mean(result.frame_scores)

        avg_errors_for_feedback = {}
        for joint_name, avg_error in result.joint_avg_errors.items():
            avg_errors_for_feedback[joint_name] = avg_error

        sorted_joints = sorted(
            avg_errors_for_feedback.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        top_deviations = []
        for joint_name, avg_error in sorted_joints[:TOP_K_DEVIATIONS]:
            if avg_error < 3:
                continue
            top_deviations.append({
                'joint': joint_name,
                'error': avg_error,
                'patient_angle': 0,
                'target_angle': target.get(joint_name, 0),
            })

        result.summary_feedback = generate_feedback(top_deviations, target)

        return result
