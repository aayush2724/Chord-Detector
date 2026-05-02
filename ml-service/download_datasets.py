import os
import urllib.request
import zipfile
import csv
import cv2
import mediapipe as mp
import time

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    if percent % 10 == 0:
        print(f"\rDownloading Hagrid subsample: {percent}%", end="")

def download_and_extract():
    url = 'https://sc.link/AO5l'
    zip_path = 'data/hagrid_subsample.zip'
    extract_path = 'data/hagrid'

    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists(extract_path):
        if not os.path.exists(zip_path):
            print("Downloading Hagrid dataset (this might take a few minutes for ~2GB)...")
            urllib.request.urlretrieve(url, zip_path, reporthook=download_progress)
            print("\nDownload complete.")
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")
    
    return extract_path

def extract_landmarks(image_dir, max_images=2000):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = 'model/hand_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=1,
        running_mode=VisionRunningMode.IMAGE
    )

    all_landmarks = []
    
    print(f"Finding images in {image_dir}...")
    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                
    print(f"Found {len(image_paths)} images. Processing up to {max_images}...")
    
    # Process subset to save time
    image_paths = image_paths[:max_images]
    
    processed = 0
    detected = 0
    
    with HandLandmarker.create_from_options(options) as landmarker:
        for img_path in image_paths:
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{len(image_paths)} images. Found {detected} hands so far.")
                
            img = cv2.imread(img_path)
            if img is None: continue
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            
            result = landmarker.detect(mp_image)
            if result and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    landmarks_flat = []
                    for lm in hand_landmarks:
                        landmarks_flat.extend([lm.x, lm.y, lm.z])
                    all_landmarks.append(landmarks_flat)
                    detected += 1

    print(f"Finished processing. Successfully extracted {detected} hand samples.")
    return all_landmarks

def save_to_csv(landmarks, output_path):
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Add angle column to match the new collect_data format
        header = ['label', 'angle'] + [f'{axis}_{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)
        
        for lm in landmarks:
            # label="Background", angle="unknown"
            row = ['Background', 'unknown'] + lm
            writer.writerow(row)
            
    print("Save complete.")

def main():
    print("--- 1. Download Public Datasets ---")
    extract_path = download_and_extract()
    landmarks = extract_landmarks(extract_path, max_images=2000)
    
    if landmarks:
        save_to_csv(landmarks, 'data/public_hands.csv')
    else:
        print("No landmarks extracted. Something went wrong.")

if __name__ == '__main__':
    main()
