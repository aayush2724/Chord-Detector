import csv
import sys
import os
import numpy as np
import argparse

def rotate_point(x, y, cx, cy, angle_deg):
    angle_rad = np.radians(angle_deg)
    # MediaPipe aspect ratio is assumed to be 1:1 for simplicity, though actual might vary.
    # For robust 2D rotation, we rotate around (cx, cy)
    x_new = cx + (x - cx) * np.cos(angle_rad) - (y - cy) * np.sin(angle_rad)
    y_new = cy + (x - cx) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad)
    return x_new, y_new

def apply_augmentations(landmarks):
    """
    Takes 63 flat landmarks [x0, y0, z0, x1, y1, z1, ...]
    Returns 5 augmented lists.
    """
    variations = []
    
    # 1. Base copy
    variations.append(landmarks.copy())
    
    # Extract points
    pts = np.array(landmarks).reshape(-1, 3)
    wrist_x, wrist_y = pts[0][0], pts[0][1]
    
    # 2. Rotate +15 deg
    pts_rot1 = pts.copy()
    for i in range(21):
        x, y = rotate_point(pts_rot1[i][0], pts_rot1[i][1], wrist_x, wrist_y, 15)
        pts_rot1[i][0], pts_rot1[i][1] = x, y
    variations.append(pts_rot1.flatten().tolist())
    
    # 3. Rotate -15 deg
    pts_rot2 = pts.copy()
    for i in range(21):
        x, y = rotate_point(pts_rot2[i][0], pts_rot2[i][1], wrist_x, wrist_y, -15)
        pts_rot2[i][0], pts_rot2[i][1] = x, y
    variations.append(pts_rot2.flatten().tolist())
    
    # 4. Scale 1.1x and Add Noise
    pts_scale = pts.copy()
    scale_factor = 1.1
    for i in range(21):
        # Scale relative to wrist
        pts_scale[i][0] = wrist_x + (pts_scale[i][0] - wrist_x) * scale_factor
        pts_scale[i][1] = wrist_y + (pts_scale[i][1] - wrist_y) * scale_factor
        # Add noise
        pts_scale[i][0] += np.random.normal(0, 0.01)
        pts_scale[i][1] += np.random.normal(0, 0.01)
    variations.append(pts_scale.flatten().tolist())
    
    # 5. Scale 0.9x and Add Noise
    pts_scale2 = pts.copy()
    scale_factor = 0.9
    for i in range(21):
        pts_scale2[i][0] = wrist_x + (pts_scale2[i][0] - wrist_x) * scale_factor
        pts_scale2[i][1] = wrist_y + (pts_scale2[i][1] - wrist_y) * scale_factor
        # Add noise
        pts_scale2[i][0] += np.random.normal(0, 0.01)
        pts_scale2[i][1] += np.random.normal(0, 0.01)
    variations.append(pts_scale2.flatten().tolist())
    
    # 6. Flip (Mirror Left-Right)
    pts_flip = pts.copy()
    for i in range(21):
        pts_flip[i][0] = 1.0 - pts_flip[i][0]
    variations.append(pts_flip.flatten().tolist())
    
    return variations

def main():
    parser = argparse.ArgumentParser(description="Augment hand landmark CSV data.")
    parser.add_argument("input_csv", help="Path to input CSV")
    args = parser.parse_args()

    input_path = args.input_csv
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        sys.exit(1)
        
    output_path = input_path.replace('.csv', '')
    if "public_hands" in input_path:
        output_path = input_path.replace('public_hands.csv', 'augmented_public_hands.csv')
    else:
        output_path = output_path + '_augmented.csv'
        
    print(f"--- 2. Data Augmentation Pipeline ---")
    print(f"Reading {input_path}...")
    
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        header = next(reader)
        writer.writerow(header)
        
        row_count = 0
        aug_count = 0
        
        for row in reader:
            row_count += 1
            label = row[0]
            angle = row[1]
            landmarks = [float(x) for x in row[2:]]
            
            variations = apply_augmentations(landmarks)
            for var in variations:
                writer.writerow([label, angle] + var)
                aug_count += 1
                
    print(f"Processed {row_count} original samples.")
    print(f"Generated {aug_count} augmented samples.")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
