# PT Progress Tracker

AI-powered physical therapy rehabilitation tracking. Upload a video of a patient performing an exercise, and the system will analyze their form, score it, detect repetitions, provide feedback, and track progress over time.

---

## Usage

### 1. Select an Exercise

Go to the Exercises page to see the 5 supported exercises:
- Squat
- Shoulder Flexion
- Forward Lunge
- Side Leg Raise
- Standing Hamstring Stretch

### 2. Upload a Video

Go to Analyze, select the exercise being performed, enter a patient ID, and upload a video file (mp4, avi, mov, webm, or mkv).

### 3. View Results

The system will display:
- Overall score (0–100)
- Repetition count
- Joint-by-joint error breakdown
- Human-readable feedback

### 4. Track Progress

Visit the Progress page to see:
- Score trends over time
- Linear regression trend analysis (improving/declining/stable)
- Detailed charts

### 5. Download Reports

Generate PDF reports with charts for any patient/exercise combination.

---

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_CONTENT_LENGTH` | 100 MB | Max video upload size |
| `MEDIAPIPE_CONFIDENCE` | 0.5 | AI model confidence threshold |
| `ANGLE_TOLERANCE_DEFAULT` | 15° | Default joint tolerance |

---

## Dependencies

- **flask** — Web framework
- **mediapipe** — AI pose estimation model
- **opencv-python** — Video processing
- **numpy** — Math operations
- **scipy** — Peak detection for rep counting
- **scikit-learn** — Linear regression for trends
- **matplotlib** — Chart generation
- **fpdf2** — PDF report generation

Full list in `requirements.txt`.

