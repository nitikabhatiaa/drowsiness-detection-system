"""
detection.py — Core computer-vision logic for the drowsiness detection system.

Responsibilities:
  • Detect faces in a grayscale frame using Haar Cascade.
  • Detect eyes within each face region.
  • Maintain a rolling closed-eye frame counter.
  • Return a structured result that app.py can act on.

Design note
-----------
Eyes are counted per face.  A driver is considered "drowsy" when BOTH eyes
are absent for ≥ CLOSED_FRAMES_THRESHOLD consecutive frames.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Tuneable parameters
# ──────────────────────────────────────────────────────────────────────────────
CLOSED_FRAMES_THRESHOLD = 20    # consecutive frames without eyes → drowsy alert
MIN_EYE_AREA_RATIO      = 0.01  # eye region must be > 1 % of face area (noise filter)


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FaceResult:
    """Detection result for a single detected face."""
    face_rect : Tuple[int, int, int, int]  # (x, y, w, h)
    eye_rects : List[Tuple[int, int, int, int]] = field(default_factory=list)
    eyes_open : bool = True                # True → ≥1 eye detected inside face


@dataclass
class DetectionResult:
    """Aggregated result returned by process_frame()."""
    faces          : List[FaceResult] = field(default_factory=list)
    closed_frames  : int  = 0          # running counter passed back from caller
    is_drowsy      : bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Core detection logic
# ──────────────────────────────────────────────────────────────────────────────
class DrowsinessDetector:
    """
    Stateful detector that accumulates a closed-eye frame count
    and triggers a drowsiness alert when the threshold is exceeded.

    Usage
    -----
        detector = DrowsinessDetector(face_cascade, eye_cascade)
        result   = detector.process_frame(frame)
    """

    def __init__(self, face_cascade: cv2.CascadeClassifier,
                       eye_cascade:  cv2.CascadeClassifier,
                       threshold:    int = CLOSED_FRAMES_THRESHOLD):
        self.face_cascade = face_cascade
        self.eye_cascade  = eye_cascade
        self.threshold    = threshold
        self._closed_frames = 0  # internal running counter

    # ── public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Analyse a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR image from cv2.VideoCapture.

        Returns
        -------
        DetectionResult
            faces        — list of detected faces with their eye sub-detections
            closed_frames — current consecutive-closed-frames count
            is_drowsy    — True when threshold exceeded
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improve contrast in varying light

        faces = self._detect_faces(gray)
        result = DetectionResult(faces=faces, closed_frames=self._closed_frames)

        if not faces:
            # No face → reset counter (driver may have looked away)
            self._closed_frames = 0
            result.closed_frames = 0
            return result

        # Use the first (primary) face to drive the drowsiness counter
        primary_face = faces[0]

        if primary_face.eyes_open:
            # Eyes visible → reset streak
            self._closed_frames = 0
        else:
            # Eyes not detected inside the face region → increment streak
            self._closed_frames += 1

        result.closed_frames = self._closed_frames
        result.is_drowsy     = self._closed_frames >= self.threshold
        return result

    def reset(self):
        """Manually reset the closed-frames counter (e.g. on key press)."""
        self._closed_frames = 0

    # ── private helpers ────────────────────────────────────────────────────────

    def _detect_faces(self, gray: np.ndarray) -> List[FaceResult]:
        """Run face detection and then eye detection per face."""
        detected = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        results: List[FaceResult] = []
        if not isinstance(detected, np.ndarray) or len(detected) == 0:
            return results

        for (fx, fy, fw, fh) in detected:
            face_area = fw * fh
            # Restrict eye search to the upper 60 % of the face (below the forehead)
            eye_roi_gray = gray[fy : fy + int(fh * 0.6), fx : fx + fw]

            eyes = self.eye_cascade.detectMultiScale(
                eye_roi_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(20, 20),
            )

            # Filter out detections that are too small relative to the face
            valid_eyes = []
            if isinstance(eyes, np.ndarray) and len(eyes) > 0:
                for (ex, ey, ew, eh) in eyes:
                    if (ew * eh) / face_area >= MIN_EYE_AREA_RATIO:
                        # Convert coords back to full-frame space
                        valid_eyes.append((fx + ex, fy + ey, ew, eh))

            results.append(FaceResult(
                face_rect=(fx, fy, fw, fh),
                eye_rects=valid_eyes,
                eyes_open=len(valid_eyes) > 0,
            ))

        return results
