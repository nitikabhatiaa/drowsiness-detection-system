"""
alert.py — Alert system for the drowsiness detection application.

Handles:
  1. Visual overlay warnings on the video frame.
  2. Audible beep alarm using pygame (falls back gracefully if unavailable).
"""

import threading
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Internal state
# ──────────────────────────────────────────────────────────────────────────────
_alarm_playing = False   # True while the alarm thread is alive
_alarm_lock    = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Alarm sound
# ──────────────────────────────────────────────────────────────────────────────
def _alarm_thread_fn(duration_ms: int = 800, frequency_hz: int = 1000):
    """Target function run in a daemon thread to play a beep sound."""
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        # Generate a simple sine-wave beep
        import numpy as np
        sample_rate = 44100
        t = np.linspace(0, duration_ms / 1000.0, int(sample_rate * duration_ms / 1000.0), endpoint=False)
        wave = (32767 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.int16)
        # pygame needs stereo-like shape
        stereo = np.column_stack([wave, wave])
        sound = pygame.sndarray.make_sound(stereo)
        sound.play()
        pygame.time.wait(duration_ms + 100)
    except Exception:
        # If pygame is unavailable or the display is headless, skip audio silently
        pass
    finally:
        global _alarm_playing
        with _alarm_lock:
            _alarm_playing = False


def play_alarm():
    """
    Trigger a non-blocking alarm beep.
    If an alarm is already playing this call is a no-op (no stacking).
    """
    global _alarm_playing
    with _alarm_lock:
        if _alarm_playing:
            return
        _alarm_playing = True

    t = threading.Thread(target=_alarm_thread_fn, daemon=True)
    t.start()


# ──────────────────────────────────────────────────────────────────────────────
# Visual warning overlay
# ──────────────────────────────────────────────────────────────────────────────
def display_warning(frame, message: str = "DROWSINESS ALERT!", blink_flag: bool = True):
    """
    Draw a bold red warning banner across the top of the frame.

    Parameters
    ----------
    frame      : numpy array — the current BGR video frame (modified in place)
    message    : warning text to show
    blink_flag : pass alternating True/False each frame to create a blink effect
    """
    if not blink_flag:
        return  # skip every other frame for a blinking effect

    h, w = frame.shape[:2]

    # Semi-transparent red banner
    overlay = frame.copy()
    banner_h = 70
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Warning text
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.4
    thickness  = 3
    text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)
    text_x = (w - text_size[0]) // 2
    text_y = (banner_h + text_size[1]) // 2

    # Shadow
    cv2.putText(frame, message, (text_x + 2, text_y + 2), font, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Bright yellow text for maximum contrast
    cv2.putText(frame, message, (text_x, text_y), font, font_scale,
                (0, 255, 255), thickness, cv2.LINE_AA)
