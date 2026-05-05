import cv2
import mediapipe as mp
import numpy as np
import math
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GazeTracker:
    def __init__(self):
        # Initialize Face Landmarker using the Tasks API
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"'{model_path}' not found. Please run 'python download_model.py' first.")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # 3D model points of a generic face
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Baseline pose
        self.baseline_angles = None
        
        # Calibration properties
        self.is_calibrated = False
        self.calibration_samples = []

    def process_frame(self, frame):
        # Convert the frame to MediaPipe Image object
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face landmarks
        detection_result = self.detector.detect(mp_image)
        
        is_looking = False
        angles = (0, 0, 0)
        annotated_image = frame.copy()

        if detection_result.face_landmarks:
            # We only care about the first face detected
            face_landmarks = detection_result.face_landmarks[0]
            
            # Calculate head pose
            angles, nose_2d = self.calculate_head_pose(annotated_image, face_landmarks)
            
            if not self.is_calibrated:
                # Collect samples for calibration
                self.calibration_samples.append(angles)
                cv2.putText(annotated_image, f"Calibrating... {len(self.calibration_samples)}/30", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if len(self.calibration_samples) >= 30:
                    self.baseline_angles = np.mean(self.calibration_samples, axis=0)
                    self.is_calibrated = True
                    print(f"Calibration finished. Baseline: {self.baseline_angles}")
            else:
                # Check if looking at screen
                pitch, yaw, roll = angles
                base_pitch, base_yaw, base_roll = self.baseline_angles
                
                # Thresholds in degrees
                pitch_diff = abs(pitch - base_pitch)
                yaw_diff = abs(yaw - base_yaw)
                
                # If within 15 degrees, consider it looking at the screen
                if pitch_diff < 15 and yaw_diff < 15:
                    is_looking = True
                
                # Draw text
                text = "LOOKING AT SCREEN" if is_looking else "AWAY"
                color = (0, 255, 0) if is_looking else (0, 0, 255)
                cv2.putText(annotated_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(annotated_image, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw a simple point for the nose to show tracking is active
            cv2.circle(annotated_image, (int(nose_2d[0]), int(nose_2d[1])), 5, (0, 255, 0), -1)

        return annotated_image, is_looking, angles

    def calculate_head_pose(self, image, face_landmarks):
        img_h, img_w, img_c = image.shape
        face_2d = []

        # Get relevant landmarks (indices for Nose, Chin, Eyes, Mouth)
        # Using standard FaceMesh indices
        landmark_indices = [1, 152, 33, 263, 61, 291]

        for idx in landmark_indices:
            lm = face_landmarks[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])

        # Convert to NumPy arrays
        face_2d = np.array(face_2d, dtype=np.float64)

        # Camera internals
        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1] ])

        # Distance Matrix
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(self.model_points, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Get the angles in degrees
        pitch = angles[0] * 360 
        yaw = angles[1] * 360 
        roll = angles[2] * 360 

        nose_2d = face_2d[0]
        return (pitch, yaw, roll), nose_2d

if __name__ == "__main__":
    import os
    cap = cv2.VideoCapture(0)
    tracker = GazeTracker()
    
    print("Please look directly at the camera for calibration.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        annotated_image, is_looking, angles = tracker.process_frame(frame)
        
        cv2.imshow('Gaze Tracker', annotated_image)
        if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
            break
            
    cap.release()
    cv2.destroyAllWindows()
