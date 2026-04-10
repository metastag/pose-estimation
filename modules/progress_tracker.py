import json
import os
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from config import SESSION_FOLDER


class ProgressTracker:
    def __init__(self):
        os.makedirs(SESSION_FOLDER, exist_ok=True)

    _session_counter = 0

    def save_session(self, patient_id, exercise_id, analysis_result, video_filename=None):
        ProgressTracker._session_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_id = f"{patient_id}_{exercise_id}_{timestamp}_{ProgressTracker._session_counter}"

        session_data = {
            'session_id': session_id,
            'patient_id': patient_id,
            'exercise_id': exercise_id,
            'timestamp': datetime.now().isoformat(),
            'video_filename': video_filename,
            'overall_score': round(analysis_result.overall_score, 1),
            'num_reps': len(analysis_result.repetitions),
            'rep_scores': [round(s, 1) for s in analysis_result.rep_scores],
            'joint_avg_errors': {k: round(v, 1) for k, v in analysis_result.joint_avg_errors.items()},
            'summary_feedback': analysis_result.summary_feedback,
            'total_frames': analysis_result.total_frames,
            'detected_frames': analysis_result.detected_frames,
        }

        patient_folder = os.path.join(SESSION_FOLDER, patient_id)
        os.makedirs(patient_folder, exist_ok=True)

        filepath = os.path.join(patient_folder, f"{session_id}.json")
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

        return session_id

    def get_sessions(self, patient_id, exercise_id=None):
        patient_folder = os.path.join(SESSION_FOLDER, patient_id)
        if not os.path.exists(patient_folder):
            return []

        sessions = []
        for filename in sorted(os.listdir(patient_folder)):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(patient_folder, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            if exercise_id and data.get('exercise_id') != exercise_id:
                continue
            sessions.append(data)

        return sessions

    def get_progress_data(self, patient_id, exercise_id):
        sessions = self.get_sessions(patient_id, exercise_id)
        if not sessions:
            return None

        dates = []
        scores = []
        rep_counts = []
        joint_trends = {}

        for session in sessions:
            dates.append(session['timestamp'][:10])
            scores.append(session['overall_score'])
            rep_counts.append(session['num_reps'])

            for joint, error in session.get('joint_avg_errors', {}).items():
                if joint not in joint_trends:
                    joint_trends[joint] = []
                joint_trends[joint].append(error)

        trend_info = self._compute_trend(scores)

        return {
            'patient_id': patient_id,
            'exercise_id': exercise_id,
            'dates': dates,
            'scores': scores,
            'rep_counts': rep_counts,
            'joint_trends': joint_trends,
            'trend': trend_info,
            'total_sessions': len(sessions),
            'latest_score': scores[-1] if scores else 0,
            'best_score': max(scores) if scores else 0,
        }

    def _compute_trend(self, scores):
        if len(scores) < 2:
            return {
                'direction': 'insufficient_data',
                'slope': 0,
                'improvement_pct': 0,
                'description': 'Need at least 2 sessions to compute a trend.',
            }

        X = np.arange(len(scores)).reshape(-1, 1)
        y = np.array(scores)

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]

        if slope > 1:
            direction = 'improving'
        elif slope < -1:
            direction = 'declining'
        else:
            direction = 'stable'

        improvement_pct = ((scores[-1] - scores[0]) / max(scores[0], 1)) * 100

        return {
            'direction': direction,
            'slope': round(slope, 2),
            'improvement_pct': round(improvement_pct, 1),
            'description': self._trend_description(direction, improvement_pct),
        }

    @staticmethod
    def _trend_description(direction, improvement_pct):
        if direction == 'improving':
            return f"Your form is improving - scores have increased by {improvement_pct:.1f}% over your sessions."
        elif direction == 'declining':
            return f"Your scores have decreased by {abs(improvement_pct):.1f}%. Consider reviewing your form with a therapist."
        else:
            return "Your performance is stable. Keep up the consistent effort!"

    def get_all_patient_exercises(self, patient_id):
        sessions = self.get_sessions(patient_id)
        exercises = {}
        for session in sessions:
            eid = session.get('exercise_id')
            if eid not in exercises:
                exercises[eid] = {
                    'exercise_id': eid,
                    'session_count': 0,
                    'latest_score': 0,
                    'latest_date': '',
                }
            exercises[eid]['session_count'] += 1
            exercises[eid]['latest_score'] = session['overall_score']
            exercises[eid]['latest_date'] = session['timestamp'][:10]

        return list(exercises.values())

    def get_latest_session(self, patient_id, exercise_id):
        sessions = self.get_sessions(patient_id, exercise_id)
        return sessions[-1] if sessions else None
