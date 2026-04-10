import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from config import UPLOAD_FOLDER, SESSION_FOLDER, REPORT_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
from modules.pose_estimator import PoseEstimator
from modules.comparison_engine import ComparisonEngine
from modules.gold_standard import get_exercise, list_exercises, get_target_angles
from modules.progress_tracker import ProgressTracker
from modules.report_generator import ReportGenerator


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

_pose_estimator = None
comparison_engine = ComparisonEngine()
progress_tracker = ProgressTracker()
report_generator = ReportGenerator()


def get_pose_estimator():
    global _pose_estimator
    if _pose_estimator is None:
        _pose_estimator = PoseEstimator()
    return _pose_estimator


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/exercises')
def exercises_page():
    return render_template('exercises.html')


@app.route('/analyze')
def analyze_page():
    return render_template('analyze.html')


@app.route('/progress')
def progress_page():
    return render_template('progress.html')


@app.route('/api/exercises', methods=['GET'])
def api_list_exercises():
    exercises = list_exercises()
    return jsonify({'exercises': exercises, 'count': len(exercises)})


@app.route('/api/exercises/<exercise_id>', methods=['GET'])
def api_get_exercise(exercise_id):
    exercise = get_exercise(exercise_id)
    if not exercise:
        return jsonify({'error': 'Exercise not found'}), 404
    return jsonify({'id': exercise_id, **exercise})


@app.route('/api/exercises/<exercise_id>/target', methods=['GET'])
def api_get_target(exercise_id):
    target = get_target_angles(exercise_id)
    if target is None:
        return jsonify({'error': 'Exercise not found'}), 404
    return jsonify({'exercise_id': exercise_id, 'target_angles': target})


@app.route('/api/analyze', methods=['POST'])
def api_analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    exercise_id = request.form.get('exercise_id')
    if not exercise_id:
        return jsonify({'error': 'Exercise ID is required'}), 400

    exercise = get_exercise(exercise_id)
    if not exercise:
        return jsonify({'error': 'Invalid exercise ID'}), 400

    patient_id = request.form.get('patient_id', 'default_patient')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{patient_id}_{exercise_id}_{timestamp}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        video_data = get_pose_estimator().process_video(filepath)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

    if not video_data['angles']:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'No pose detected in the video. Please ensure the person is clearly visible.'}), 400

    analysis_result = comparison_engine.analyze_video(video_data, exercise_id)

    session_id = progress_tracker.save_session(
        patient_id=patient_id,
        exercise_id=exercise_id,
        analysis_result=analysis_result,
        video_filename=filename,
    )

    result_dict = analysis_result.to_dict()
    result_dict['session_id'] = session_id
    result_dict['patient_id'] = patient_id
    result_dict['exercise_id'] = exercise_id
    result_dict['exercise_name'] = exercise['name']

    return jsonify(result_dict)


@app.route('/api/progress/<patient_id>', methods=['GET'])
def api_patient_progress(patient_id):
    exercise_id = request.args.get('exercise_id')
    exercises = progress_tracker.get_all_patient_exercises(patient_id)
    return jsonify({
        'patient_id': patient_id,
        'exercises': exercises,
        'total_exercises': len(exercises),
    })


@app.route('/api/progress/<patient_id>/<exercise_id>', methods=['GET'])
def api_exercise_progress(patient_id, exercise_id):
    progress_data = progress_tracker.get_progress_data(patient_id, exercise_id)
    if not progress_data:
        return jsonify({'error': 'No data found for this patient/exercise combination'}), 404
    return jsonify(progress_data)


@app.route('/api/progress/<patient_id>/<exercise_id>/sessions', methods=['GET'])
def api_session_history(patient_id, exercise_id):
    sessions = progress_tracker.get_sessions(patient_id, exercise_id)
    return jsonify({
        'patient_id': patient_id,
        'exercise_id': exercise_id,
        'sessions': sessions,
        'count': len(sessions),
    })


@app.route('/api/report/<patient_id>/<exercise_id>', methods=['GET'])
def api_generate_report(patient_id, exercise_id):
    pdf_path = report_generator.generate_pdf_report(patient_id, exercise_id)
    if not pdf_path:
        return jsonify({'error': 'No data available to generate report'}), 404

    exercise = get_exercise(exercise_id)
    exercise_name = exercise['name'] if exercise else exercise_id
    download_name = f"PT_Report_{exercise_name.replace(' ', '_')}_{patient_id}.pdf"

    return send_file(pdf_path, as_attachment=True, download_name=download_name)


@app.route('/api/sessions/<patient_id>', methods=['GET'])
def api_all_sessions(patient_id):
    sessions = progress_tracker.get_sessions(patient_id)
    return jsonify({
        'patient_id': patient_id,
        'sessions': sessions,
        'count': len(sessions),
    })


@app.route('/api/sessions/<patient_id>/<exercise_id>/latest', methods=['GET'])
def api_latest_session(patient_id, exercise_id):
    session = progress_tracker.get_latest_session(patient_id, exercise_id)
    if not session:
        return jsonify({'error': 'No sessions found'}), 404
    return jsonify(session)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
