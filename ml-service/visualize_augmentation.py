import csv
import os
import sys

import cv2
import numpy as np


def draw_skeleton(landmarks, width=400, height=400):
    # landmarks: [x0, y0, z0, x1, y1, z1, ...] (normalized 0-1)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    connections = [
        (0,1), (1,2), (2,3), (3,4),
        (0,5), (5,6), (6,7), (7,8),
        (5,9), (9,10), (10,11), (11,12),
        (9,13), (13,14), (14,15), (15,16),
        (13,17), (17,18), (18,19), (19,20),
        (0,17)
    ]

    pts = []
    for i in range(21):
        x = int(landmarks[i*3] * width)
        y = int(landmarks[i*3 + 1] * height)
        pts.append((x, y))
        cv2.circle(img, (x, y), 5, (255, 180, 0), -1)

    for pt1_idx, pt2_idx in connections:
        pt1 = pts[pt1_idx]
        pt2 = pts[pt2_idx]
        cv2.line(img, pt1, pt2, (0, 220, 180), 2)

    return img

def main():
    csv_path = 'data/augmented_public_hands.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run augmentation first.")
        sys.exit(1)

    print("--- 6. Visualize Augmentations ---")

    # We will grab the first 6 rows (1 original + 5 variations)
    samples = []
    labels = ["Original", "Rotated +15", "Rotated -15", "Scaled 1.1x", "Scaled 0.9x", "Flipped (Mirror)"]

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for i, row in enumerate(reader):
            if i >= 6:
                break
            landmarks = [float(x) for x in row[2:]]
            samples.append(landmarks)

    if not samples:
        print("No data found.")
        return

    # Create a 2x3 grid image
    h, w = 300, 300
    grid = np.zeros((h*2, w*3, 3), dtype=np.uint8)

    for i, lm in enumerate(samples):
        img = draw_skeleton(lm, w, h)
        cv2.putText(img, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        row_idx = i // 3
        col_idx = i % 3
        grid[row_idx*h:(row_idx+1)*h, col_idx*w:(col_idx+1)*w] = img

    # Save the visualization
    out_img_path = 'augmentation_sample.png'
    cv2.imwrite(out_img_path, grid)
    print(f"Saved visualization to {out_img_path}")

if __name__ == '__main__':
    main()
