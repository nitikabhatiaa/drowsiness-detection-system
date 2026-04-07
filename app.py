"""
app.py — Entry point for the Real-Time Driver Drowsiness Detection System.

Run with:
    python app.py

Controls:
    Q  — quit
    R  — reset drowsiness counter
"""

import sys
import cv2

# Allow running from the project root without installing the package
sys.path.insert(0, __file__.replace("app.py", ""))

from src.detection import DrowsinessDetector
from src.alert     import play_alarm, display_warning
from src.utils     import load_cascades, draw_text


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0      # 0 = default webcam
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
DROWSY_THRESHOLD    = 20     # consecutive closed-eye frames before alert


def main():
    print("=" * 55)
    print("  Real-Time Driver Drowsiness Detection System")
    print("=" * 55)
    print("  Controls: Q → quit  |  R → reset counter")
    print("=" * 55)

    # ── Load cascade classifiers ──────────────────────────────────────────────
    try:
        face_cascade, eye_cascade = load_cascades()
        print("[OK] Cascade classifiers loaded.")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        print("        Make sure a webcam is connected and not in use.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f"[OK] Camera opened ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} px).")

    # ── Initialise detector ───────────────────────────────────────────────────
    detector   = DrowsinessDetector(face_cascade, eye_cascade, DROWSY_THRESHOLD)
    frame_num  = 0   # used to alternate blink state every 15 frames

    print("[OK] Starting detection loop. Press Q to quit.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame. Retrying…")
            continue

        frame_num += 1
        result     = detector.process_frame(frame)

        # ── Draw face & eye boxes ────────────────────────────────────────────
        for face in result.faces:
            fx, fy, fw, fh = face.face_rect
            # Face rectangle — green when awake, red when drowsy
            box_color = (0, 0, 220) if result.is_drowsy else (0, 200, 0)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)

            # Eye rectangles — cyan
            for (ex, ey, ew, eh) in face.eye_rects:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 2)

        # ── HUD: status bar ──────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Background strip at the bottom
        cv2.rectangle(frame, (0, h - 50), (w, h), (30, 30, 30), -1)

        # Closed-frames counter
        counter_text  = f"Closed frames: {result.closed_frames}/{DROWSY_THRESHOLD}"
        counter_color = (0, 200, 0)  # green
        if result.closed_frames > DROWSY_THRESHOLD * 0.5:
            counter_color = (0, 165, 255)  # orange
        if result.is_drowsy:
            counter_color = (0, 0, 255)   # red
        draw_text(frame, counter_text, (10, h - 15), color=counter_color,
                  scale=0.7, thickness=2)

        # Status label (right-aligned)
        if not result.faces:
            status_text  = "No face detected"
            status_color = (180, 180, 180)
        elif result.faces[0].eyes_open:
            status_text  = "AWAKE"
            status_color = (0, 220, 0)
        else:
            status_text  = "Eyes closed…"
            status_color = (0, 120, 255)

        ts, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        draw_text(frame, status_text, (w - ts[0] - 15, h - 15),
                  color=status_color, scale=0.7, thickness=2)

        # ── Drowsiness alert ─────────────────────────────────────────────────
        if result.is_drowsy:
            blink_flag = (frame_num // 15) % 2 == 0   # blink every ~0.5 s at 30 fps
            display_warning(frame, "DROWSINESS ALERT!", blink_flag=blink_flag)
            play_alarm()

        # ── Show frame ───────────────────────────────────────────────────────
        cv2.imshow("Driver Drowsiness Detection", frame)

        # ── Keyboard controls ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("\n[INFO] Quit signal received. Exiting…")
            break
        elif key == ord("r") or key == ord("R"):
            detector.reset()
            print("[INFO] Counter reset.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    main()
