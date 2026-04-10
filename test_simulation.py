import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.comparison_engine import ComparisonEngine
from modules.gold_standard import get_exercise, get_target_angles
from modules.progress_tracker import ProgressTracker
from modules.report_generator import ReportGenerator
from modules.feedback_engine import generate_feedback


def simulate_session(patient_id, exercise_id, score_modifier=0.0):
    exercise = get_exercise(exercise_id)
    if not exercise:
        print(f"Unknown exercise: {exercise_id}")
        return None

    target_angles = exercise['target_angles']
    standing_angles = exercise['standing_angles']

    num_frames = 90
    fps = 30.0
    all_angles = []

    for f in range(num_frames):
        t = f / num_frames
        cycle = np.sin(2 * np.pi * t * 2)

        if cycle < 0:
            progress = 0
        else:
            progress = cycle

        frame_angles = {}
        for joint_name, standing_angle in standing_angles.items():
            target = target_angles.get(joint_name, standing_angle)
            angle = standing_angle + (target - standing_angle) * progress
            angle += np.random.normal(0, 3 + abs(score_modifier))
            angle = max(0, min(180, angle))
            frame_angles[joint_name] = angle

        frame_angles['frame'] = f
        all_angles.append(frame_angles)

    video_data = {
        'angles': all_angles,
        'fps': fps,
        'total_frames': num_frames,
        'detected_frames': num_frames,
    }

    engine = ComparisonEngine()
    result = engine.analyze_video(video_data, exercise_id)

    tracker = ProgressTracker()
    session_id = tracker.save_session(
        patient_id=patient_id,
        exercise_id=exercise_id,
        analysis_result=result,
        video_filename=None,
    )

    return result


def main():
    patient_id = 'test_patient'
    exercise_id = 'squat'

    print("=" * 60)
    print("PT Progress Tracker - End-to-End Simulation Test")
    print("=" * 60)

    print(f"\nPatient: {patient_id}")
    print(f"Exercise: {exercise_id}")

    print("\n--- Simulating 5 sessions with improving form ---")
    for i in range(5):
        noise = 8.0 - (i * 1.5)
        result = simulate_session(patient_id, exercise_id, score_modifier=noise)
        print(f"\nSession {i+1}:")
        print(f"  Overall Score: {result.overall_score:.1f}")
        print(f"  Reps Detected: {len(result.repetitions)}")
        if result.rep_scores:
            print(f"  Rep Scores: {[f'{s:.1f}' for s in result.rep_scores]}")
        if result.summary_feedback:
            print(f"  Top Feedback:")
            for fb in result.summary_feedback[:2]:
                print(f"    - {fb['joint'].replace('_',' ').title()}: {fb['message'][:60]}...")

    print("\n--- Checking Progress ---")
    tracker = ProgressTracker()
    progress = tracker.get_progress_data(patient_id, exercise_id)
    if progress:
        print(f"  Total Sessions: {progress['total_sessions']}")
        print(f"  Latest Score: {progress['latest_score']:.1f}")
        print(f"  Best Score: {progress['best_score']:.1f}")
        print(f"  Trend: {progress['trend']['direction']} ({progress['trend']['improvement_pct']:.1f}%)")
        print(f"  Description: {progress['trend']['description']}")
    else:
        print("  No progress data found!")
        return

    print("\n--- Generating PDF Report ---")
    report_gen = ReportGenerator()
    pdf_path = report_gen.generate_pdf_report(patient_id, exercise_id)
    if pdf_path:
        print(f"  Report generated: {pdf_path}")
        print(f"  File size: {os.path.getsize(pdf_path)} bytes")
    else:
        print("  Failed to generate report!")

    print("\n--- Testing All Exercises ---")
    for eid in ['squat', 'shoulder_flexion', 'forward_lunge', 'side_leg_raise', 'standing_hamstring_stretch']:
        result = simulate_session(f'{patient_id}_2', eid, score_modifier=3.0)
        if result:
            print(f"  {eid}: score={result.overall_score:.1f}, reps={len(result.repetitions)}")

    print("\n" + "=" * 60)
    print("All simulation tests completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
