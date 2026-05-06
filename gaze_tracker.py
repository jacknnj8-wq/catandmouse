import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Facial landmark indices used for solvePnP head-pose estimation
# (Nose tip, Chin, Left-eye corner, Right-eye corner, Left-mouth, Right-mouth)
POSE_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

# Matching 3-D reference points on a generic face model (millimetres)
FACE_3D_MODEL = np.array([
    ( 0.0,    0.0,    0.0),   # Nose tip
    ( 0.0, -330.0,  -65.0),   # Chin
    (-225.0,  170.0, -135.0), # Left eye outer corner
    ( 225.0,  170.0, -135.0), # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    ( 150.0, -150.0, -125.0), # Right mouth corner
], dtype=np.float64)

# Gaussian spread (degrees).  Tune these to widen/narrow the "looking" zone.
YAW_SIGMA   = 12.0   # left-right sensitivity
PITCH_SIGMA = 10.0   # up-down sensitivity

# Number of grid rows / cols drawn on the face bounding box
GRID_ROWS = 5
GRID_COLS = 5

# Minimum confidence to be considered "looking at this screen"
LOOKING_THRESHOLD = 0.35

# How many calibration frames to collect
CALIB_FRAMES = 40

# ---------------------------------------------------------------------------

class GazeTracker:
    """
    Tracks head pose from a webcam frame and returns a continuous confidence
    score (0.0 – 1.0) representing how directly the user faces this camera.

    Visual output
    -------------
    * 5×5 spatial grid drawn over the face bounding box, with each cell
      tinted according to how much of the face mesh falls inside it.
    * A pose direction arrow from the nose tip.
    * Confidence bar along the left edge.
    """

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"'{model_path}' not found. Run 'python download_model.py' first.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.baseline_angles: np.ndarray | None = None
        self.is_calibrated  = False
        self.calibration_samples: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray):
        """
        Returns
        -------
        annotated_image : np.ndarray   BGR frame with overlays
        confidence      : float        0.0 – 1.0  (replaces binary is_looking)
        angles          : tuple        (pitch, yaw, roll) in degrees
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        annotated = frame.copy()
        confidence = 0.0
        angles     = (0.0, 0.0, 0.0)

        if not result.face_landmarks:
            cv2.putText(annotated, "No face detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return annotated, confidence, angles

        landmarks = result.face_landmarks[0]
        img_h, img_w = frame.shape[:2]

        angles, nose_px = self._head_pose(annotated, landmarks, img_w, img_h)
        pitch, yaw, roll = angles

        # ---------- calibration ----------
        if not self.is_calibrated:
            self.calibration_samples.append(angles)
            pct = int(len(self.calibration_samples) / CALIB_FRAMES * 100)
            cv2.putText(annotated,
                        f"Calibrating – look straight at camera  {pct}%",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            self._draw_grid(annotated, landmarks, img_w, img_h, fill=0.0)
            if len(self.calibration_samples) >= CALIB_FRAMES:
                self.baseline_angles = np.mean(self.calibration_samples, axis=0)
                self.is_calibrated   = True
                print(f"[GazeTracker] Calibrated. Baseline: {self.baseline_angles}")
            return annotated, 0.0, angles

        # ---------- confidence scoring ----------
        base_pitch, base_yaw, _ = self.baseline_angles
        d_yaw   = yaw   - base_yaw
        d_pitch = pitch - base_pitch
        confidence = float(np.exp(
            -0.5 * ((d_yaw / YAW_SIGMA) ** 2 + (d_pitch / PITCH_SIGMA) ** 2)
        ))

        # ---------- overlays ----------
        self._draw_grid(annotated, landmarks, img_w, img_h, fill=confidence)
        self._draw_pose_arrow(annotated, nose_px, yaw, pitch, img_w, img_h)
        self._draw_confidence_bar(annotated, confidence, img_h)

        label  = f"Confidence: {confidence:.2f}  Yaw:{yaw:+.1f}  Pitch:{pitch:+.1f}"
        color  = (0, 255, 0) if confidence >= LOOKING_THRESHOLD else (0, 100, 255)
        status = "FOCUSED" if confidence >= LOOKING_THRESHOLD else "away"
        cv2.putText(annotated, status, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(annotated, label, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        return annotated, confidence, angles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _head_pose(self, image, landmarks, img_w, img_h):
        """Run solvePnP on the 6 key landmarks and return Euler angles."""
        face_2d = np.array([
            [int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)]
            for i in POSE_LANDMARK_IDX
        ], dtype=np.float64)

        focal = img_w
        cam_matrix = np.array([
            [focal,     0, img_w / 2],
            [0,     focal, img_h / 2],
            [0,         0,         1]
        ], dtype=np.float64)
        dist = np.zeros((4, 1))

        ok, rvec, _ = cv2.solvePnP(FACE_3D_MODEL, face_2d, cam_matrix, dist)
        rmat, _    = cv2.Rodrigues(rvec)
        euler, *_  = cv2.RQDecomp3x3(rmat)

        pitch = euler[0] * 360
        yaw   = euler[1] * 360
        roll  = euler[2] * 360
        nose_px = face_2d[0]
        return (pitch, yaw, roll), nose_px

    def _draw_grid(self, img, landmarks, img_w, img_h, fill: float):
        """
        Draw a GRID_ROWS × GRID_COLS grid over the face bounding box.
        Each cell is filled with an alpha-blend proportional to `fill`
        (confidence) AND to how many face-mesh points lie within it –
        giving the 'spatial grid surface' effect.
        """
        xs = [lm.x * img_w for lm in landmarks]
        ys = [lm.y * img_h for lm in landmarks]
        x0, y0 = max(0, int(min(xs)) - 10), max(0, int(min(ys)) - 10)
        x1, y1 = min(img_w, int(max(xs)) + 10), min(img_h, int(max(ys)) + 10)
        if x1 <= x0 or y1 <= y0:
            return

        cell_w = (x1 - x0) / GRID_COLS
        cell_h = (y1 - y0) / GRID_ROWS

        # Count landmarks per cell for density map
        density = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        for lm in landmarks:
            px, py = lm.x * img_w, lm.y * img_h
            col = int((px - x0) / cell_w)
            row = int((py - y0) / cell_h)
            if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                density[row, col] += 1
        if density.max() > 0:
            density /= density.max()

        overlay = img.copy()
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cx0 = int(x0 + c * cell_w)
                cy0 = int(y0 + r * cell_h)
                cx1 = int(cx0 + cell_w)
                cy1 = int(cy0 + cell_h)

                # Cell colour: green tint scaled by fill × density
                intensity = fill * density[r, c]
                cell_color = (
                    int(20  * intensity),
                    int(220 * intensity),
                    int(60  * intensity),
                )
                cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), cell_color, -1)
                cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1),
                              (80, 200, 80), 1)

        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    def _draw_pose_arrow(self, img, nose_px, yaw, pitch, img_w, img_h):
        """Draw a 3-D pose direction arrow from the nose tip."""
        arrow_len = 80
        dx = int(arrow_len * np.sin(np.radians(yaw)))
        dy = int(-arrow_len * np.sin(np.radians(pitch)))
        tip = (int(nose_px[0]) + dx, int(nose_px[1]) + dy)
        cv2.arrowedLine(img, (int(nose_px[0]), int(nose_px[1])), tip,
                        (0, 255, 255), 2, tipLength=0.25)

    def _draw_confidence_bar(self, img, confidence, img_h):
        """Vertical confidence bar on the right edge."""
        bar_x   = img.shape[1] - 30
        bar_top = 20
        bar_bot = img_h - 20
        bar_h   = bar_bot - bar_top
        fill_h  = int(bar_h * confidence)

        cv2.rectangle(img, (bar_x, bar_top), (bar_x + 18, bar_bot),
                      (50, 50, 50), -1)
        if fill_h > 0:
            green = int(255 * confidence)
            red   = int(255 * (1 - confidence))
            cv2.rectangle(img,
                          (bar_x, bar_bot - fill_h),
                          (bar_x + 18, bar_bot),
                          (0, green, red), -1)
        cv2.rectangle(img, (bar_x, bar_top), (bar_x + 18, bar_bot),
                      (150, 150, 150), 1)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    source = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open camera"); sys.exit(1)

    tracker = GazeTracker()
    print("Look straight at the camera for calibration.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        out, conf, angles = tracker.process_frame(frame)
        cv2.imshow("Gaze Tracker", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
