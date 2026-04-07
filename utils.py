"""
utils.py — Shared utility helpers for the drowsiness detection system.
Loads Haar Cascade classifiers and provides small helper functions.
"""

import os
import cv2


def get_model_path(filename: str) -> str:
    """Return the absolute path to a model file in the models/ directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models", filename)


def load_cascades():
    """
    Load and return the face and eye Haar Cascade classifiers.

    Returns
    -------
    face_cascade : cv2.CascadeClassifier
    eye_cascade  : cv2.CascadeClassifier
    """
    face_path = get_model_path("haarcascade_frontalface_default.xml")
    eye_path  = get_model_path("haarcascade_eye.xml")

    if not os.path.exists(face_path):
        raise FileNotFoundError(f"Face cascade not found: {face_path}")
    if not os.path.exists(eye_path):
        raise FileNotFoundError(f"Eye cascade not found: {eye_path}")

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade  = cv2.CascadeClassifier(eye_path)

    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade classifier.")
    if eye_cascade.empty():
        raise RuntimeError("Failed to load eye cascade classifier.")

    return face_cascade, eye_cascade


def draw_text(frame, text: str, position: tuple, color=(0, 0, 255), scale=1.2, thickness=2):
    """
    Draw bold text on a frame for maximum visibility.

    Parameters
    ----------
    frame     : numpy array (BGR image)
    text      : string to draw
    position  : (x, y) top-left corner of the text
    color     : BGR color tuple
    scale     : font scale factor
    thickness : line thickness
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw a dark shadow for contrast
    shadow_offset = 2
    sx, sy = position[0] + shadow_offset, position[1] + shadow_offset
    cv2.putText(frame, text, (sx, sy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Draw the actual text
    cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)
