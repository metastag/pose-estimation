import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from fpdf import FPDF
from config import REPORT_FOLDER, JOINT_DEFINITIONS
from modules.gold_standard import get_exercise
from modules.progress_tracker import ProgressTracker


def _sanitize_text(text):
    if not text:
        return text
    replacements = {
        '\u2014': '--',
        '\u2013': '-',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2026': '...',
        '\u00b0': ' deg',
        '\u2265': '>=',
        '\u2264': '<=',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


class ReportGenerator:
    def __init__(self):
        os.makedirs(REPORT_FOLDER, exist_ok=True)
        self.tracker = ProgressTracker()

    def _generate_progress_chart(self, progress_data, output_path):
        dates = progress_data['dates']
        scores = progress_data['scores']

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

        ax1 = axes[0]
        ax1.plot(range(len(scores)), scores, 'b-o', linewidth=2, markersize=8, label='Overall Score')
        if len(scores) >= 2:
            X = np.arange(len(scores)).reshape(-1, 1)
            y = np.array(scores)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X, y)
            trend = model.predict(X)
            ax1.plot(range(len(scores)), trend, 'r--', linewidth=1.5, alpha=0.7, label='Trend')
        ax1.set_ylabel('Score (0-100)', fontsize=12)
        ax1.set_title('Exercise Form Score Over Sessions', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.set_xticks(range(len(dates)))
        ax1.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        joint_trends = progress_data.get('joint_trends', {})
        if joint_trends:
            joints = list(joint_trends.keys())[:6]
            joint_labels = [JOINT_DEFINITIONS.get(j, {}).get('label', j.replace('_', ' ').title()) for j in joints]
            latest_errors = [joint_trends[j][-1] for j in joints]
            colors = ['#e74c3c' if e > 15 else '#f39c12' if e > 8 else '#2ecc71' for e in latest_errors]
            ax2.barh(range(len(joints)), latest_errors, color=colors)
            ax2.set_yticks(range(len(joints)))
            ax2.set_yticklabels(joint_labels, fontsize=9)
            ax2.set_xlabel('Average Error (degrees)', fontsize=10)
            ax2.set_title('Joint Error Breakdown (Latest Session)', fontsize=12, fontweight='bold')
            ax2.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Warning')
            ax2.axvline(x=15, color='red', linestyle='--', alpha=0.5, label='Critical')
            ax2.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    def _generate_score_breakdown_chart(self, session_data, output_path):
        rep_scores = session_data.get('rep_scores', [])
        if not rep_scores:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#e74c3c' if s < 40 else '#f39c12' if s < 70 else '#2ecc71' for s in rep_scores]
        ax.bar(range(1, len(rep_scores) + 1), rep_scores, color=colors)
        ax.set_xlabel('Repetition', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Repetition Score Breakdown', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    def generate_pdf_report(self, patient_id, exercise_id):
        progress_data = self.tracker.get_progress_data(patient_id, exercise_id)
        if not progress_data:
            return None

        exercise = get_exercise(exercise_id)
        exercise_name = exercise['name'] if exercise else exercise_id

        report_folder = os.path.join(REPORT_FOLDER, patient_id)
        os.makedirs(report_folder, exist_ok=True)

        progress_chart_path = os.path.join(report_folder, f'{exercise_id}_progress.png')
        breakdown_chart_path = os.path.join(report_folder, f'{exercise_id}_breakdown.png')
        pdf_path = os.path.join(report_folder, f'{exercise_id}_report.pdf')

        self._generate_progress_chart(progress_data, progress_chart_path)

        latest_session = self.tracker.get_latest_session(patient_id, exercise_id)
        if latest_session:
            self._generate_score_breakdown_chart(latest_session, breakdown_chart_path)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 24)
        pdf.cell(0, 15, 'Physical Therapy', ln=True, align='C')
        pdf.cell(0, 10, 'Progress Report', ln=True, align='C')
        pdf.ln(5)

        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', ln=True, align='C')
        pdf.ln(10)

        pdf.set_draw_color(41, 128, 185)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)

        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Patient Summary', ln=True)
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 7, f'Patient ID: {_sanitize_text(patient_id)}', ln=True)
        pdf.cell(0, 7, f'Exercise: {_sanitize_text(exercise_name)}', ln=True)
        pdf.cell(0, 7, f'Total Sessions: {progress_data["total_sessions"]}', ln=True)
        pdf.cell(0, 7, f'Latest Score: {progress_data["latest_score"]:.1f}/100', ln=True)
        pdf.cell(0, 7, f'Best Score: {progress_data["best_score"]:.1f}/100', ln=True)
        pdf.ln(5)

        trend = progress_data.get('trend', {})
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Trend Analysis', ln=True)
        pdf.set_font('Helvetica', '', 11)
        direction = trend.get('direction', 'N/A')
        direction_text = {'improving': 'Improving', 'declining': 'Declining', 'stable': 'Stable', 'insufficient_data': 'Insufficient Data'}.get(direction, direction)
        pdf.cell(0, 7, f'Direction: {direction_text}', ln=True)
        pdf.cell(0, 7, f'Slope: {trend.get("slope", 0):.2f} points/session', ln=True)
        pdf.cell(0, 7, f'Improvement: {trend.get("improvement_pct", 0):.1f}%', ln=True)
        pdf.ln(3)
        pdf.set_font('Helvetica', 'I', 10)
        pdf.multi_cell(0, 6, _sanitize_text(trend.get('description', '')))
        pdf.ln(5)

        if os.path.exists(progress_chart_path):
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, 'Progress Over Time', ln=True)
            pdf.image(progress_chart_path, x=10, w=190)
            pdf.ln(5)

        if latest_session and os.path.exists(breakdown_chart_path):
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, 'Latest Session Breakdown', ln=True)
            pdf.set_font('Helvetica', '', 11)
            pdf.cell(0, 7, f'Session Date: {latest_session["timestamp"][:10]}', ln=True)
            pdf.cell(0, 7, f'Overall Score: {latest_session["overall_score"]:.1f}/100', ln=True)
            pdf.cell(0, 7, f'Repetitions Detected: {latest_session["num_reps"]}', ln=True)
            pdf.ln(5)
            pdf.image(breakdown_chart_path, x=15, w=180)
            pdf.ln(5)

        if latest_session:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Joint Error Analysis', ln=True)
            pdf.set_font('Helvetica', '', 10)
            joint_errors = latest_session.get('joint_avg_errors', {})
            for joint, error in sorted(joint_errors.items(), key=lambda x: x[1], reverse=True):
                label = JOINT_DEFINITIONS.get(joint, {}).get('label', joint.replace('_', ' ').title())
                severity = 'Excellent' if error <= 5 else 'Good' if error <= 10 else 'Fair' if error <= 20 else 'Needs Work'
                pdf.cell(0, 6, f'  {label}: {error:.1f} deg ({severity})', ln=True)
            pdf.ln(5)

        feedback = latest_session.get('summary_feedback', []) if latest_session else []
        if feedback:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Feedback & Recommendations', ln=True)
            pdf.set_font('Helvetica', '', 10)
            for fb in feedback:
                joint_label = fb.get('joint', '').replace('_', ' ').title()
                message = _sanitize_text(fb.get('message', ''))
                severity = fb.get('severity', '').title()
                pdf.multi_cell(0, 6, _sanitize_text(f'  [{severity}] {joint_label}: {message}'))
                pdf.ln(2)

        pdf.ln(10)
        pdf.set_draw_color(41, 128, 185)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.cell(0, 5, 'This report is generated by the PT Progress Tracker AI system.', ln=True, align='C')
        pdf.cell(0, 5, 'It is intended as a supplementary tool and does not replace professional medical advice.', ln=True, align='C')

        pdf.output(pdf_path)
        return pdf_path
