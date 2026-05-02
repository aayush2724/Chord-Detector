"""
collect_data.py — Guitar Chord Landmark Data Collector
=======================================================
Uses OpenCV + MediaPipe Hands to capture hand landmarks from webcam.
Press a chord key to record the current 63 landmark values (21 pts × x,y,z)
along with the chord label into data/chord_data.csv.

Key Bindings:
  a  → Am
  c  → C
  d  → D
  e  → E
  m  → Em
  f  → F
  g  → G
  b  → Bm
  s  → F#m
  7  → Fmaj7
  h  → A  (A major)
  j  → Dm (D minor)
  SPACE → Toggle continuous recording (hold chord + hold space)
  q  → Quit

Usage:
  python collect_data.py

CSV saved to: data/chord_data.csv
Each row: label, x0,y0,z0, x1,y1,z1, ... x20,y20,z20  (1 + 63 = 64 columns)
"""

import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "chord_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Chord key mappings ─────────────────────────────────────────────────────────
CHORD_KEYS = {
    ord('a'): 'Am',
    ord('c'): 'C',
    ord('d'): 'D',
    ord('e'): 'E',
    ord('m'): 'Em',
    ord('f'): 'F',
    ord('g'): 'G',
    ord('b'): 'Bm',
    ord('s'): 'F#m',
    ord('7'): 'Fmaj7',
    ord('h'): 'A',
    ord('j'): 'Dm',
}

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    'primary':   (255, 180, 0),    # amber
    'accent':    (0, 220, 180),    # teal
    'danger':    (0, 80, 255),     # red (BGR)
    'white':     (255, 255, 255),
    'black':     (0, 0, 0),
    'overlay':   (20, 20, 20),
}

# ── CSV header ─────────────────────────────────────────────────────────────────
CSV_HEADER = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]


def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist (index 0) and scale by hand span.
    This makes the model position- and scale-invariant.
    Returns a flat list of 63 floats.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    # Translate: wrist to origin
    pts -= pts[0]
    # Scale: divide by the distance between wrist and middle-finger MCP (landmark 9)
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten().tolist()


def draw_landmarks_custom(frame, hand_landmarks, h, w):
    """Draw the 21-point skeleton with a custom style."""
    connections = mp.solutions.hands.HAND_CONNECTIONS
    pts = {}
    for idx, lm in enumerate(hand_landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts[idx] = (cx, cy)

    # Draw connections
    for conn in connections:
        pt1, pt2 = pts[conn[0]], pts[conn[1]]
        cv2.line(frame, pt1, pt2, COLORS['accent'], 2, cv2.LINE_AA)

    # Draw landmark dots
    for idx, pt in pts.items():
        color = COLORS['primary'] if idx == 0 else COLORS['white']
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, COLORS['black'], 1, cv2.LINE_AA)


def draw_hud(frame, chord_label, recording, sample_count, counts_per_chord, fps):
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ───────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "Guitar Chord Data Collector", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['primary'], 2, cv2.LINE_AA)

    # FPS + total samples
    info = f"FPS: {fps:.0f}  |  Total Samples: {sample_count}"
    cv2.putText(frame, info, (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1, cv2.LINE_AA)

    # ── Current chord label box ────────────────────────────────────────────────
    if chord_label:
        rec_color = (0, 80, 220) if recording else (50, 50, 50)
        cv2.rectangle(frame, (w - 200, 10), (w - 10, 65), rec_color, -1)
        cv2.rectangle(frame, (w - 200, 10), (w - 10, 65), COLORS['primary'], 1)
        status = "● REC" if recording else "  HOLD"
        cv2.putText(frame, status, (w - 185, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['white'], 1, cv2.LINE_AA)
        cv2.putText(frame, chord_label, (w - 185, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['primary'], 2, cv2.LINE_AA)

    # ── Key legend (bottom-left panel) ────────────────────────────────────────
    panel_x, panel_y = 10, h - 20 - len(CHORD_KEYS) * 22 - 30
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (panel_x - 5, panel_y - 20),
                  (panel_x + 190, panel_y + len(CHORD_KEYS) * 22 + 10), (10, 10, 30), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "Keys:", (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['accent'], 1, cv2.LINE_AA)

    key_names = {
        'a': 'Am',  'c': 'C',  'd': 'D',  'e': 'E',  'm': 'Em',
        'f': 'F',   'g': 'G',  'b': 'Bm', 's': 'F#m','7': 'Fmaj7',
        'h': 'A',   'j': 'Dm',
    }
    for i, (k, v) in enumerate(key_names.items()):
        count = counts_per_chord.get(v, 0)
        bar_len = min(int(count / 5), 60)
        label_color = COLORS['primary'] if v == chord_label else COLORS['white']
        line = f"[{k}] {v:<6} {count:>3}"
        cv2.putText(frame, line, (panel_x, panel_y + 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)
        cv2.rectangle(frame,
                      (panel_x + 120, panel_y + 8 + i * 22),
                      (panel_x + 120 + bar_len, panel_y + 18 + i * 22),
                      COLORS['accent'], -1)

    # ── Bottom hint bar ────────────────────────────────────────────────────────
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (0, h - 28), (w, h), (10, 10, 30), -1)
    cv2.addWeighted(overlay3, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "SPACE: continuous record  |  q: quit  |  r: reset last chord",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS['white'], 1, cv2.LINE_AA)

    # ── No hand detected warning ───────────────────────────────────────────────
    return frame


def get_sample_counts(csv_path):
    counts = {}
    if not os.path.exists(csv_path):
        return counts, 0
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        total = 0
        for row in reader:
            if row:
                label = row[0]
                counts[label] = counts.get(label, 0) + 1
                total += 1
    return counts, total


def main():
    # ── Init CSV ───────────────────────────────────────────────────────────────
    write_header = not os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(CSV_HEADER)

    # ── Init MediaPipe ─────────────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    # ── Init webcam ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("\n" + "="*60)
    print("  Guitar Chord Data Collector")
    print("="*60)
    print("  Press chord keys to record a sample.")
    print("  SPACE = toggle continuous recording")
    print("  q = quit\n")

    chord_label = None
    continuous_record = False
    counts_per_chord, total_samples = get_sample_counts(CSV_PATH)

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── FPS ───────────────────────────────────────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── MediaPipe ─────────────────────────────────────────────────────────
        results = hands.process(rgb)
        hand_detected = results.multi_hand_landmarks is not None

        recorded_this_frame = False

        if hand_detected:
            hand_lms = results.multi_hand_landmarks[0]
            draw_landmarks_custom(frame, hand_lms, h, w)

            if continuous_record and chord_label:
                flat = normalize_landmarks(hand_lms.landmark)
                writer.writerow([chord_label] + flat)
                csv_file.flush()
                counts_per_chord[chord_label] = counts_per_chord.get(chord_label, 0) + 1
                total_samples += 1
                recorded_this_frame = True
        else:
            # "No hand" warning
            cv2.putText(frame, "⚠  No hand detected", (w // 2 - 140, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['danger'], 2, cv2.LINE_AA)

        # ── Recording flash indicator ──────────────────────────────────────────
        if recorded_this_frame:
            cv2.circle(frame, (w - 20, 90), 8, (0, 0, 255), -1)

        # ── Draw HUD ──────────────────────────────────────────────────────────
        draw_hud(frame, chord_label, continuous_record and hand_detected,
                 total_samples, counts_per_chord, fps)

        cv2.imshow("Chord Data Collector", frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Quitting.")
            break

        elif key == ord(' '):
            continuous_record = not continuous_record
            status = "ON" if continuous_record else "OFF"
            print(f"[INFO] Continuous recording: {status}")

        elif key == ord('r'):
            # Remove last recorded row
            if chord_label and counts_per_chord.get(chord_label, 0) > 0:
                # Rewrite CSV minus last row of current chord
                csv_file.close()
                with open(CSV_PATH, 'r') as f:
                    rows = list(csv.reader(f))
                # Find last row matching this chord (from end)
                for i in range(len(rows) - 1, 0, -1):
                    if rows[i] and rows[i][0] == chord_label:
                        rows.pop(i)
                        counts_per_chord[chord_label] -= 1
                        total_samples -= 1
                        print(f"[INFO] Removed 1 sample of {chord_label}")
                        break
                with open(CSV_PATH, 'w', newline='') as f:
                    csv.writer(f).writerows(rows)
                csv_file = open(CSV_PATH, 'a', newline='')
                writer = csv.writer(csv_file)

        elif key in CHORD_KEYS:
            chord_label = CHORD_KEYS[key]
            # Single-shot record if hand detected
            if hand_detected:
                flat = normalize_landmarks(results.multi_hand_landmarks[0].landmark)
                writer.writerow([chord_label] + flat)
                csv_file.flush()
                counts_per_chord[chord_label] = counts_per_chord.get(chord_label, 0) + 1
                total_samples += 1
                print(f"[SAVED] {chord_label} (total {chord_label}: {counts_per_chord[chord_label]})")
            else:
                print(f"[WARN]  No hand detected — chord set to {chord_label}, no sample saved")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    csv_file.close()
    hands.close()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("  Session Summary")
    print("="*60)
    for chord, cnt in sorted(counts_per_chord.items()):
        bar = "█" * min(cnt, 40)
        print(f"  {chord:<8} {cnt:>4}  {bar}")
    print(f"\n  Total samples: {total_samples}")
    print(f"  Saved to: {CSV_PATH}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
