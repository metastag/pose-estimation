import os
import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_CONFIDENCE, JOINT_DEFINITIONS


MODEL_DIR = os.path.expanduser('~/.pttracker')
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, 'pttracker-pose.pt')
PRETRAINED_MODEL = 'yolov8n-pose.pt'

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _find_model():
    if os.path.exists(CUSTOM_MODEL_PATH):
        return CUSTOM_MODEL_PATH
    return PRETRAINED_MODEL


def _estimate_foot_point(knee, ankle, keypoints):
    """
    COCO keypoints (17 points) don't include foot/toe sub-points like MediaPipe's
    33-point scheme. Estimate the foot position using the person's facing direction
    to approximate a reasonable ankle angle.
    """
    nose = np.array(keypoints[0][:2])
    mid_hip = (np.array(keypoints[11][:2]) + np.array(keypoints[12][:2])) / 2.0
    facing = nose - mid_hip
    facing_norm = np.linalg.norm(facing)
    if facing_norm < 1e-6:
        facing = np.array([1.0, 0.0])
    else:
        facing = facing / facing_norm

    shin = np.array(ankle) - np.array(knee)
    shin_len = np.linalg.norm(shin)
    foot_length = shin_len * 0.4

    foot_point = np.array(ankle) + facing * foot_length
    return foot_point.tolist()


class PoseEstimator:
    def __init__(self):
        model_path = _find_model()
        self.model = YOLO(model_path)
        self.confidence = YOLO_CONFIDENCE

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

    def _extract_keypoints(self, result):
        """Extract keypoints for the most prominent person from YOLO results."""
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        keypoints_data = result.keypoints.data
        if keypoints_data.shape[0] == 0:
            return None

        if result.boxes is not None and len(result.boxes) > 1:
            areas = []
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                areas.append((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            best_idx = int(np.argmax(areas))
        else:
            best_idx = 0

        kpts = keypoints_data[best_idx].cpu().numpy()
        valid = kpts[:, 2] >= self.confidence
        if not valid.any():
            return None

        return kpts

    def compute_trunk_lean(self, keypoints):
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]

        mid_shoulder = (np.array(left_shoulder) + np.array(right_shoulder)) / 2.0
        mid_hip = (np.array(left_hip) + np.array(right_hip)) / 2.0

        torso_vec = mid_shoulder - mid_hip
        vertical = np.array([0, -1])

        cosine = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) * np.linalg.norm(vertical) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def compute_joint_angles(self, keypoints):
        angles = {}
        for joint_name, joint_def in JOINT_DEFINITIONS.items():
            a_idx, b_idx, c_idx = joint_def['landmarks']

            a = keypoints[a_idx][:2].tolist()
            b = keypoints[b_idx][:2].tolist()

            if c_idx == -1:
                c = _estimate_foot_point(keypoints[a_idx][:2], keypoints[b_idx][:2], keypoints)
            else:
                c = keypoints[c_idx][:2].tolist()

            angles[joint_name] = self.angle_between_points(a, b, c)

        angles['trunk_lean'] = self.compute_trunk_lean(keypoints)
        return angles

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_angles = []
        annotated_frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False, conf=self.confidence)

            if results and len(results) > 0:
                keypoints = self._extract_keypoints(results[0])

                if keypoints is not None:
                    major_joints = [5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
                    detected_major = sum(1 for i in major_joints if keypoints[i][2] >= self.confidence)

                    if detected_major >= 6:
                        angles = self.compute_joint_angles(keypoints)
                        angles['frame'] = frame_idx
                        all_angles.append(angles)

                        annotated = frame.copy()
                        annotated = self._draw_skeleton(annotated, keypoints)
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

    def _draw_skeleton(self, frame, keypoints):
        for start_idx, end_idx in COCO_SKELETON:
            if (keypoints[start_idx][2] >= self.confidence and
                    keypoints[end_idx][2] >= self.confidence):
                pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

        for i in range(len(keypoints)):
            if keypoints[i][2] >= self.confidence:
                pt = (int(keypoints[i][0]), int(keypoints[i][1]))
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)

        return frame

    def close(self):
        pass
