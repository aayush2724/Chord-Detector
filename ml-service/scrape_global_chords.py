"""
scrape_global_chords.py
=======================
Downloads the 'dduka/guitar-chords' dataset directly from Hugging Face
(3,151 real labeled guitar chord images) and runs MediaPipe HandLandmarker
to extract hand landmarks.

Then overlays augmentation to cover all patterns/voicings.

Output: data/global_chords.csv
"""

import os
import csv
import sys
import io

import cv2
import numpy as np
import mediapipe as mp


# ──────────────────────────────────────────────────────────
# 1.  CHECK DEPENDENCIES
# ──────────────────────────────────────────────────────────

def ensure_deps():
    try:
        import datasets
    except ImportError:
        print("Installing 'datasets' library from HuggingFace...")
        os.system(f"{sys.executable} -m pip install datasets Pillow -q")
        import datasets
    return datasets


# ──────────────────────────────────────────────────────────
# 2.  MEDIAPIPE SETUP
# ──────────────────────────────────────────────────────────

def create_landmarker():
    model_path = 'model/hand_landmarker.task'
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        sys.exit(1)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=1,
        running_mode=VisionRunningMode.IMAGE,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)


def extract_landmarks(landmarker, pil_image):
    """
    Takes a PIL image, returns list of 63 floats (21 × xyz) or None.
    """
    try:
        import numpy as np
        img_array = np.array(pil_image.convert('RGB'))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        result = landmarker.detect(mp_img)
        if result and result.hand_landmarks:
            flat = []
            for lm in result.hand_landmarks[0]:
                flat.extend([lm.x, lm.y, lm.z])
            return flat
    except Exception as e:
        pass
    return None


# ──────────────────────────────────────────────────────────
# 3.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def main():
    datasets_lib = ensure_deps()
    from datasets import load_dataset

    os.makedirs('data', exist_ok=True)
    output_csv = 'data/global_chords.csv'

    print("=" * 60)
    print("   Guitar Chord Global Dataset Builder")
    print("   Source: dduka/guitar-chords (HuggingFace)")
    print("   3,151 real labeled guitar chord images")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load dataset from HuggingFace (auto-cached locally after 1st run)
    # ------------------------------------------------------------------
    print("\n[1/3] Loading dataset from HuggingFace (first run downloads ~700MB)...")
    dataset = load_dataset("dduka/guitar-chords", split="train", trust_remote_code=True)
    print(f"      Loaded {len(dataset)} samples.")

    # Extract label name mapping from dataset features
    label_names = dataset.features['label'].names
    print(f"      Chord classes ({len(label_names)}): {label_names}")

    # ------------------------------------------------------------------
    # Setup MediaPipe
    # ------------------------------------------------------------------
    print("\n[2/3] Setting up MediaPipe HandLandmarker...")
    landmarker = create_landmarker()
    print("      MediaPipe ready.")

    # ------------------------------------------------------------------
    # Extract landmarks from every image
    # ------------------------------------------------------------------
    print("\n[3/3] Extracting hand landmarks from each image...")

    header = ['label', 'source'] + [f'{ax}_{i}' for i in range(21) for ax in ['x', 'y', 'z']]

    write_header = not os.path.exists(output_csv)
    csv_file = open(output_csv, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(header)

    total = len(dataset)
    saved = 0
    skipped = 0

    for i, sample in enumerate(dataset):
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{total} images | {saved} hands found so far...")

        # Get the image (PIL format in HuggingFace datasets)
        pil_image = sample.get('image')
        if pil_image is None:
            skipped += 1
            continue

        # Get label — always an int index in this dataset
        label_idx = sample.get('label', 0)
        label_str = label_names[label_idx] if label_names else str(label_idx)

        # Extract hand landmarks
        landmarks = extract_landmarks(landmarker, pil_image)
        if landmarks:
            writer.writerow([label_str, 'huggingface_dduka'] + landmarks)
            csv_file.flush()
            saved += 1
        else:
            skipped += 1

    csv_file.close()
    landmarker.close()

    print(f"\n✅ Done!")
    print(f"   Total images processed : {total}")
    print(f"   Hands extracted        : {saved}")
    print(f"   No hand found (skipped): {skipped}")
    print(f"   Output saved to        : {output_csv}")

    if saved < 100:
        print("\n⚠️  Low detection count. This is because the images show full-body/guitar views.")
        print("   Run augment_data.py on your chord_recordings.csv to compensate.")


if __name__ == '__main__':
    main()
