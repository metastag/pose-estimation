import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from config import MEDIAPIPE_CONFIDENCE, JOINT_DEFINITIONS


MODEL_PATH = None


def _find_model():
    global MODEL_PATH
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        return MODEL_PATH

    candidates = [
        os.path.expanduser('~/.mediapipe/pose_landmarker.task'),
        'pose_landmarker.task',
    ]

    for path in candidates:
        if os.path.exists(path):
            MODEL_PATH = path
            return path

    raise FileNotFoundError(
        "Pose landmarker model not found. Download from: "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task "
        "Place it at ~/.mediapipe/pose_landmarker.task"
    )


class PoseEstimator:
    def __init__(self):
        model_path = _find_model()
        self._model_path = model_path
        self._create_landmarker()

    def _create_landmarker(self):
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.VIDEO,
            min_tracking_confidence=MEDIAPIPE_CONFIDENCE,
        )
        self.landmarker = PoseLandmarker.create_from_options(self.options)

    @staticmethod
    def angle_between_points(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    @staticmethod
    def _landmark_to_coords(landmark):
        return [landmark.x, landmark.y, landmark.z]

    def compute_trunk_lean(self, landmarks):
        left_shoulder = np.array(self._landmark_to_coords(landmarks[11]))
        right_shoulder = np.array(self._landmark_to_coords(landmarks[12]))
        left_hip = np.array(self._landmark_to_coords(landmarks[23]))
        right_hip = np.array(self._landmark_to_coords(landmarks[24]))

        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        mid_hip = (left_hip + right_hip) / 2.0

        torso_vec = mid_shoulder - mid_hip
        vertical = np.array([0, -1, 0])

        cosine = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) * np.linalg.norm(vertical) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def compute_joint_angles(self, landmarks):
        angles = {}
        for joint_name, joint_def in JOINT_DEFINITIONS.items():
            a_idx, b_idx, c_idx = joint_def['landmarks']
            a = self._landmark_to_coords(landmarks[a_idx])
            b = self._landmark_to_coords(landmarks[b_idx])
            c = self._landmark_to_coords(landmarks[c_idx])
            angles[joint_name] = self.angle_between_points(a, b, c)
        angles['trunk_lean'] = self.compute_trunk_lean(landmarks)
        return angles

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.landmarker.close()
        self._create_landmarker()

        all_angles = []
        annotated_frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(frame_idx * 1000 / fps)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm_list = result.pose_landmarks[0]
                if len(lm_list) >= 33:
                    angles = self.compute_joint_angles(lm_list)
                    angles['frame'] = frame_idx
                    all_angles.append(angles)

                    annotated = frame.copy()
                    annotated = self._draw_skeleton(annotated, lm_list, w, h)
                    annotated_frames.append(annotated)

            frame_idx += 1

        cap.release()
        return {
            'angles': all_angles,
            'annotated_frames': annotated_frames,
            'fps': fps,
            'total_frames': frame_count,
            'detected_frames': len(all_angles),
        }

    def _draw_skeleton(self, frame, landmarks, width, height):
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
            (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32),
            (15, 17), (16, 18), (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                pt1 = (int(start.x * width), int(start.y * height))
                pt2 = (int(end.x * width), int(end.y * height))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

        for i, lm in enumerate(landmarks):
            pt = (int(lm.x * width), int(lm.y * height))
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)

        return frame

    def close(self):
        self.landmarker.close()
