"""
=============================================================
  FACE ENROLLMENT SCRIPT — No-Stop Attendance System
  Captures face images at multiple angles for a new employee
=============================================================

USAGE:
  python enroll_face.py

REQUIREMENTS:
  pip install opencv-python insightface onnxruntime numpy

OUTPUT:
  known_faces/<name>/  — folder with all captured images
"""

import cv2
import os
import time
import numpy as np

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR = "known_faces"          # root folder for all enrolled faces
IMAGES_PER_ANGLE = 5                # how many shots to take per angle
COUNTDOWN_SECONDS = 3               # countdown before capturing each angle
CAMERA_INDEX = 0                    # 0 = default webcam

# Angles to capture with on-screen instructions
ANGLES = [
    ("front",       "Look STRAIGHT at the camera",        (0, 255, 120)),
    ("left",        "Turn face SLIGHTLY LEFT",             (0, 200, 255)),
    ("right",       "Turn face SLIGHTLY RIGHT",            (255, 200, 0)),
    ("up",          "Tilt face SLIGHTLY UP",               (200, 100, 255)),
    ("down",        "Tilt face SLIGHTLY DOWN",             (255, 100, 100)),
    ("front_2",     "Look STRAIGHT again (lighting vary)", (0, 255, 180)),
]

# ─────────────────────────────────────────────
#  FACE DETECTOR (OpenCV Haar — no heavy model needed for enrollment)
# ─────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(frame):
    """Returns (x, y, w, h) of largest face, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # pick largest face
    return max(faces, key=lambda f: f[2] * f[3])

def draw_face_box(frame, face, color=(0, 255, 120), label=""):
    x, y, w, h = face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    if label:
        cv2.rectangle(frame, (x, y - 26), (x + w, y), color, -1)
        cv2.putText(frame, label, (x + 4, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

def draw_ui(frame, instruction, color, countdown=None,
            captured=0, total=0, angle_idx=0, total_angles=0, name=""):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)
    cv2.putText(frame, f"Enrolling: {name}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Angle {angle_idx+1}/{total_angles}",
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

    # Progress bar top
    bar_w = int((angle_idx / total_angles) * w)
    cv2.rectangle(frame, (0, 58), (bar_w, 62), color, -1)

    # Instruction banner
    cv2.rectangle(frame, (0, h - 80), (w, h - 40), (15, 15, 15), -1)
    text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    tx = (w - text_size[0]) // 2
    cv2.putText(frame, instruction, (tx, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Shot counter
    cv2.rectangle(frame, (0, h - 40), (w, h), (10, 10, 10), -1)
    shots_text = f"Shots captured: {captured}/{total}"
    cv2.putText(frame, shots_text, (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # Countdown circle
    if countdown is not None:
        cx, cy, r = w - 60, h - 60, 28
        cv2.circle(frame, (cx, cy), r, color, -1)
        cv2.putText(frame, str(countdown), (cx - 9, cy + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Guide oval (face alignment helper)
    oval_cx, oval_cy = w // 2, h // 2 + 10
    cv2.ellipse(frame, (oval_cx, oval_cy), (90, 115), 0, 0, 360, color, 2)

def capture_angle(cap, angle_name, instruction, color,
                  save_dir, images_per_angle, angle_idx, total_angles, name):
    """Runs the countdown + capture loop for one angle."""
    captured = 0
    saved_paths = []

    # ── Countdown phase ─────────────────────────────
    for secs in range(COUNTDOWN_SECONDS, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)  # mirror

            face = detect_face(frame)
            if face is not None:
                draw_face_box(frame, face, color, "Detected ✓")

            draw_ui(frame, instruction, color,
                    countdown=secs, captured=captured,
                    total=images_per_angle, angle_idx=angle_idx,
                    total_angles=total_angles, name=name)
            cv2.imshow("Face Enrollment", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                return saved_paths

    # ── Capture phase ────────────────────────────────
    while captured < images_per_angle:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        face = detect_face(frame)
        display = frame.copy()

        if face is not None:
            draw_face_box(display, face, color, f"Capturing {captured+1}/{images_per_angle}")

            # Save the cropped face with padding
            x, y, w_f, h_f = face
            pad = 30
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w_f + pad)
            y2 = min(frame.shape[0], y + h_f + pad)
            face_crop = frame[y1:y2, x1:x2]

            filename = os.path.join(save_dir, f"{angle_name}_{captured+1:02d}.jpg")
            cv2.imwrite(filename, face_crop)
            saved_paths.append(filename)
            captured += 1

            # Flash effect
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]),
                          (255, 255, 255), -1)
            cv2.addWeighted(flash, 0.3, display, 0.7, 0, display)

            time.sleep(0.25)  # small delay between shots
        else:
            cv2.putText(display, "No face detected — move closer",
                        (30, display.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)

        draw_ui(display, instruction, color,
                countdown=None, captured=captured,
                total=images_per_angle, angle_idx=angle_idx,
                total_angles=total_angles, name=name)
        cv2.imshow("Face Enrollment", display)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    return saved_paths


def show_completion(cap, name, total_saved):
    """Show a completion screen for 3 seconds."""
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 30, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        lines = [
            (f"Enrollment complete!", 0.9, (0, 255, 120)),
            (f"Employee: {name}", 0.65, (200, 255, 200)),
            (f"{total_saved} images saved to known_faces/{name}/", 0.55, (160, 200, 160)),
            ("Ready for attendance system.", 0.5, (120, 180, 120)),
        ]
        y_start = h // 2 - 70
        for i, (text, scale, color) in enumerate(lines):
            ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
            tx = (w - ts[0]) // 2
            cv2.putText(frame, text, (tx, y_start + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

        cv2.imshow("Face Enrollment", frame)
        cv2.waitKey(30)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  FACE ENROLLMENT — No-Stop Attendance System")
    print("="*55)
    name = input("\n  Enter employee name (e.g. Aditi_Sharma): ").strip()
    if not name:
        print("  [ERROR] Name cannot be empty.")
        return

    # Sanitize name for folder
    name = name.replace(" ", "_")
    save_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  Saving images to: {save_dir}/")
    print(f"  Angles to capture: {len(ANGLES)}")
    print(f"  Images per angle:  {IMAGES_PER_ANGLE}")
    print(f"  Total images:      {len(ANGLES) * IMAGES_PER_ANGLE}")
    print("\n  Opening camera... press Q at any time to quit.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera. Check CAMERA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    all_saved = []

    for idx, (angle_name, instruction, color) in enumerate(ANGLES):
        print(f"  [{idx+1}/{len(ANGLES)}] Angle: {angle_name} — {instruction}")

        # Between angles: show "get ready" screen
        if idx > 0:
            ready_deadline = time.time() + 2.0
            while time.time() < ready_deadline:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                draw_ui(frame, f"NEXT: {instruction}", (200, 200, 200),
                        countdown=None, captured=0, total=IMAGES_PER_ANGLE,
                        angle_idx=idx, total_angles=len(ANGLES), name=name)
                h, w = frame.shape[:2]
                cv2.putText(frame, "Get ready...", (w//2 - 80, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 50), 2)
                cv2.imshow("Face Enrollment", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        saved = capture_angle(
            cap, angle_name, instruction, color,
            save_dir, IMAGES_PER_ANGLE, idx, len(ANGLES), name
        )
        all_saved.extend(saved)
        print(f"     Saved {len(saved)} images.")

    show_completion(cap, name, len(all_saved))

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  ✓ Done! {len(all_saved)} images saved.")
    print(f"  ✓ Folder: {os.path.abspath(save_dir)}")
    print("\n  Next steps:")
    print("  1. Run this script for each employee")
    print("  2. Start attendance_engine.py — it will auto-load all enrolled faces")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()