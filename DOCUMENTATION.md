# PT Progress Tracker — Project Documentation

## What Is This Project?

PT Progress Tracker is a web application that helps physical therapy patients and therapists track exercise form and progress over time. You upload a video of a patient doing an exercise, and the system:

1. Uses an AI model to detect the person's body positions in each frame
2. Calculates joint angles (like how bent the knee is)
3. Compares those angles against clinical "gold standard" templates
4. Scores the exercise form on a 0–100 scale
5. Detects how many repetitions were performed
6. Gives human-readable feedback ("Straighten your right knee slightly")
7. Tracks progress across multiple sessions and generates PDF reports

---

## Project Structure

```
pttracker/
├── app.py                    # Main Flask web server (entry point)
├── config.py                 # All configuration settings
├── requirements.txt          # Python dependencies
├── test_simulation.py        # End-to-end test script
│
├── modules/
│   ├── pose_estimator.py     # AI pose detection from video
│   ├── gold_standard.py      # Ideal angle templates for each exercise
│   ├── comparison_engine.py  # Scoring, error calculation, rep detection
│   ├── feedback_engine.py    # Turns errors into readable feedback
│   ├── progress_tracker.py   # Saves sessions + trend analysis
│   └── report_generator.py   # PDF report with charts
│
├── templates/                # HTML pages (Flask/Jinja2)
│   ├── base.html              # Shared layout
│   ├── index.html             # Home page
│   ├── exercises.html         # Exercise browser
│   ├── analyze.html           # Upload + results page
│   └── progress.html          # Progress charts (Plotly)
│
├── static/
│   ├── css/style.css          # All styles
│   ├── uploads/               # Uploaded videos
│   └── reports/               # Generated PDFs + chart images
│
└── data/
    ├── sessions/              # Session data (JSON per patient)
    └── gold_standards/        # Custom gold standard overrides (future)
```

---

## How It All Works — Step by Step

### Step 1: Upload a Video

The user visits the `/analyze` page and uploads a video file (mp4, avi, mov, webm, or mkv, up to 100 MB) along with selecting which exercise is being performed.

### Step 2: AI Detects Body Landmarks

The `PoseEstimator` module (in `modules/pose_estimator.py`) processes every frame of the video:

1. It reads each frame using OpenCV
2. Converts the image from BGR to RGB color format
3. Feeds the image into **Google's MediaPipe Pose Landmarker** — a pre-trained deep learning model
4. The model outputs **33 body landmark points** in 3D (x, y, z) — things like shoulders, elbows, knees, ankles, etc.

> **What is MediaPipe Pose Landmarker?**
> It's a neural network made by Google that's already trained to find human body positions in images. Think of it like a very smart tool that looks at a photo and says "the left knee is at pixel (340, 520)" for 33 different body points. This project uses the "heavy" version which is the most accurate but slowest variant.

### Step 3: Calculate Joint Angles

From those 33 landmarks, the system calculates **11 joint angles** per frame:

- **10 body joints** (left and right): shoulder, elbow, hip, knee, ankle
- **1 trunk lean** — how far forward the upper body is leaning

The angle at each joint is calculated using basic vector math: if you have three points A–B–C, the angle at B is computed using the dot product of vectors BA and BC. This is the standard geometric angle formula.

### Step 4: Compare Against Gold Standards

The `ComparisonEngine` (in `modules/comparison_engine.py`) compares the patient's angles against the "gold standard" — a set of ideal angles defined by clinicians for each exercise.

**How the comparison works:**
- For each joint, it calculates the **error** = patient's angle − ideal angle
- If the error is within a **tolerance window** (e.g., ±15°), the error is reduced by 70% — being close enough is treated as nearly correct
- If the error is outside the tolerance, the excess is counted more heavily
- Errors are capped at 45° maximum

**How scoring works:**
- Each joint has a **weight** (knees = 1.5, hips = 1.3, shoulders/ankles = 1.0, elbows = 0.7)
- **Key joints** for the exercise get a 1.5× multiplier (e.g., knees are key for squats)
- Non-key joints get a 0.5× multiplier
- Trunk lean always has weight 1.5 (safety-critical)
- Final score = `100 × (1 − weighted_average_error / 45)`, minimum 0

### Step 5: Detect Repetitions

The system automatically counts how many times the exercise was performed using **peak detection** from `scipy.signal.find_peaks`:

- For exercises like squats and lunges, it finds **valleys** (the knee bends to a minimum angle)
- For exercises like shoulder flexion, it finds **peaks** (the shoulder raises to a maximum angle)
- It requires at least 10 frames between reps and a prominence of 20° to avoid counting small movements as reps

### Step 6: Generate Feedback

The `FeedbackEngine` (in `modules/feedback_engine.py`) translates joint errors into plain-language advice:

- It determines the **direction** of each error (too bent, too straight, too far forward, etc.)
- Looks up a pre-written feedback message from a library
- Assigns a severity level: excellent (≤5°), good (≤10°), fair (≤20°), poor (>20°)
- Reports the top 3 biggest deviations

Example: *"Straighten your right knee slightly — do not go too deep"*

### Step 7: Track Progress Over Time

The `ProgressTracker` (in `modules/progress_tracker.py`) saves each session's results as a JSON file and analyzes trends:

- Uses **linear regression** (from scikit-learn) to fit a trend line across sessions
- If the slope is > 1: trend is "improving"
- If the slope is < −1: trend is "declining"
- Otherwise: trend is "stable"
- Also calculates percentage improvement from first to last session

### Step 8: Generate Reports

The `ReportGenerator` (in `modules/report_generator.py`) creates PDF reports with:
- Patient and exercise information
- Overall score and trend analysis
- Progress chart (scores over sessions)
- Joint error breakdown chart
- Per-repetition score chart
- Detailed joint error table
- Feedback messages

---

## The AI Model — Deep Dive

### What Model Is Used?

The project uses **Google's MediaPipe Pose Landmarker** model, specifically the **"heavy"** variant. The model file is `pose_landmarker_heavy.task`.

### Where Does the Model Come From?

The model is **pre-trained by Google** and downloaded separately. It must be placed at `~/.mediapipe/pose_landmarker.task`. This project does **not** train any neural network — it only uses the model for inference (making predictions).

### What Does the Model Do?

The model takes a single RGB image and outputs **33 body landmarks**, each with (x, y, z) coordinates:

| Index | Landmark | Index | Landmark |
|-------|----------|-------|----------|
| 0 | Nose | 17-22 | Hand/finger points |
| 11 | Left shoulder | 12 | Right shoulder |
| 13 | Left elbow | 14 | Right elbow |
| 15 | Left wrist | 16 | Right wrist |
| 23 | Left hip | 24 | Right hip |
| 25 | Left knee | 26 | Right knee |
| 27 | Left ankle | 28 | Right ankle |
| 29-30 | Heel points | 31-32 | Foot index points |

### How Does the Model Run?

```python
# Simplified version of what happens in pose_estimator.py:

import mediapipe as mp

# 1. Load the model
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="~/.mediapipe/pose_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    min_tracking_confidence=0.5
)
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

# 2. For each video frame:
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
result = landmarker.detect_for_video(mp_image, timestamp_ms)

# 3. result.pose_landmarks contains the 33 landmark points
```

Key details:
- The model runs in **VIDEO mode** — this means it processes frames sequentially with timestamps, which gives it temporal context (it can use information from previous frames to be more accurate)
- The **confidence threshold** is 0.5 — if the model is less than 50% confident about tracking a landmark, it may drop it
- The landmarker is **re-created for every video** — this is required because VIDEO mode maintains internal state per stream

### How Are Landmarks Turned Into Angles?

The project defines joints as triples of landmark indices. For example, the left knee angle uses landmarks: left hip (23) → left knee (25) → left ankle (27). The angle at the knee is calculated:

```
vector_1 = hip - knee      (from knee toward hip)
vector_2 = ankle - knee    (from knee toward ankle)
angle = arccos(dot(vector_1, vector_2) / (|vector_1| × |vector_2|))
```

This gives the angle in degrees at the knee joint.

---

## Supported Exercises

| Exercise | Category | Difficulty | Primary Joint | What It Watches For |
|----------|----------|------------|---------------|---------------------|
| **Squat** | Lower Body | Intermediate | Knee (min) | Hip=90°, Knee=90°, Trunk lean ≤10° |
| **Shoulder Flexion** | Upper Body | Beginner | Shoulder (max) | Shoulder=170°, Trunk lean ≤5° |
| **Forward Lunge** | Lower Body | Intermediate | Knee (min) | Hip=90°, Knee=90°, Trunk lean ≤10° |
| **Side Leg Raise** | Lower Body | Beginner | Hip (max) | Hip=135°, Trunk lean ≤8° |
| **Standing Hamstring Stretch** | Flexibility | Beginner | Hip (min) | Hip=90°/120°, Trunk lean ≤45° |

Each exercise defines:
- **Target angles** — ideal joint angles at the peak of the movement
- **Tolerances** — how many degrees off is acceptable per joint
- **Standing angles** — expected angles at rest position
- **Key joints** — the most important joints for scoring (weighted higher)

---

## API Endpoints

| Route | Method | What It Does |
|-------|--------|--------------|
| `/` | GET | Home page |
| `/exercises` | GET | Browse exercises |
| `/analyze` | GET | Video upload page |
| `/progress` | GET | Progress dashboard |
| `/api/exercises` | GET | List all exercises (JSON) |
| `/api/exercises/<id>` | GET | Get exercise details |
| `/api/exercises/<id>/target` | GET | Get target angles for exercise |
| `/api/analyze` | POST | Upload video and run analysis |
| `/api/progress/<patient_id>` | GET | All exercises for a patient |
| `/api/progress/<patient_id>/<exercise_id>` | GET | Progress data with trend |
| `/api/progress/<patient_id>/<exercise_id>/sessions` | GET | Session history |
| `/api/report/<patient_id>/<exercise_id>` | GET | Download PDF report |
| `/api/sessions/<patient_id>` | GET | All sessions for patient |
| `/api/sessions/<patient_id>/<exercise_id>/latest` | GET | Latest session |

### Upload and Analyze (the main endpoint)

```
POST /api/analyze
Content-Type: multipart/form-data

Fields:
  video: the video file
  exercise_id: one of "squat", "shoulder_flexion", "forward_lunge", "side_leg_raise", "standing_hamstring_stretch"
  patient_id: identifier for the patient

Response: JSON with score, rep_count, feedback, joint_errors, etc.
```

---

## Key Configuration (config.py)

| Setting | Value | What It Means |
|---------|-------|---------------|
| `MAX_CONTENT_LENGTH` | 100 MB | Maximum video upload size |
| `MEDIAPIPE_CONFIDENCE` | 0.5 | How confident the AI needs to be to track a landmark (0–1) |
| `TOP_K_DEVIATIONS` | 3 | How many problem areas to report in feedback |
| `ANGLE_TOLERANCE_DEFAULT` | 15° | Default acceptable deviation from ideal angle |
| Scoring: `max_error` | 45° | Maximum error that contributes to score (larger errors are capped) |
| Scoring: excellent | ≤5° | Joint is very close to ideal |
| Scoring: good | ≤10° | Joint is reasonably close |
| Scoring: fair | ≤20° | Joint is somewhat off |
| Scoring: poor | >20° | Joint is significantly off |
| Joint weights | varies | Knees=1.5, Hips=1.3, Shoulders/Ankles=1.0, Elbows=0.7 |

---

## Dependencies

| Package | What It's Used For |
|---------|-------------------|
| **flask** | Web server — routes, pages, API, file uploads |
| **mediapipe** | The AI pose detection model (core of the system) |
| **opencv-python** | Reading video files frame by frame, drawing skeleton overlay |
| **numpy** | Math operations for angle calculations |
| **scipy** | Peak detection for counting exercise repetitions |
| **scikit-learn** | Linear regression for progress trend analysis |
| **matplotlib** | Generating charts for PDF reports |
| **fpdf2** | Creating PDF reports |
| **plotly** | Interactive charts in the web browser (loaded via CDN) |
| **pandas** | Listed in requirements but currently unused |

---

## Data Storage

Session data is stored as flat JSON files — there is no database:

```
data/sessions/
├── patient_001/
│   └── squat_20240101_120000.json
└── test_patient/
    ├── squat_20240101_100000.json
    ├── squat_20240102_100000.json
    └── ...
```

Each JSON file contains: patient ID, exercise ID, overall score, rep count, frame-by-frame joint angles, errors, feedback, and timestamps.

---

## Running the Project

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the MediaPipe model:**
   The `pose_landmarker_heavy.task` model file must be downloaded from Google and placed at `~/.mediapipe/pose_landmarker.task`.

3. **Run the Flask app:**
   ```bash
   python app.py
   ```

4. **Open in browser:** Go to `http://localhost:5000`

5. **Run tests:**
   ```bash
   python test_simulation.py
   ```

---

## How the Pieces Connect

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  User Uploads │────▶│  PoseEstimator    │────▶│ ComparisonEngine   │
│  Video        │     │  (AI Model)       │     │ (Scoring + Reps)   │
└─────────────┘     └──────────────────┘     └────────┬──────────┘
                                                       │
                            ┌──────────────────────────┤
                            ▼                          ▼
                   ┌──────────────────┐     ┌───────────────────┐
                   │ FeedbackEngine    │     │ ProgressTracker    │
                   │ (Readable Tips)   │     │ (Save + Trend)    │
                   └──────────────────┘     └────────┬──────────┘
                                                     │
                                                     ▼
                                           ┌───────────────────┐
                                           │ ReportGenerator    │
                                           │ (PDF + Charts)     │
                                           └───────────────────┘
```

**Flow in plain English:**

1. You give it a video of someone exercising
2. The AI model looks at each frame and finds where all the body parts are
3. It measures the angles at each joint
4. It compares those angles to what they should be for that exercise
5. It calculates a score and counts repetitions
6. It tells you what needs improvement in plain language
7. It saves the results and tracks whether the patient is improving over time
8. It can generate a PDF report with charts
