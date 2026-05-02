import csv
import sys
import os
from collections import Counter

def analyze_csv(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    print(f"\n--- Statistics for {os.path.basename(filepath)} ---")
    
    total_samples = 0
    label_counts = Counter()
    angle_counts = Counter()
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        has_angle = 'angle' in header
        
        for row in reader:
            total_samples += 1
            label = row[0]
            label_counts[label] += 1
            if has_angle:
                angle = row[1]
                angle_counts[angle] += 1
                
    print(f"Total Samples: {total_samples}")
    print("\nClass Balance:")
    for label, count in label_counts.most_common():
        percentage = (count / total_samples) * 100
        print(f"  - {label}: {count} ({percentage:.1f}%)")
        
    if has_angle:
        print("\nAngle Distribution:")
        for angle, count in angle_counts.most_common():
            percentage = (count / total_samples) * 100
            print(f"  - {angle}: {count} ({percentage:.1f}%)")

def main():
    print("--- 5. Dataset Statistics ---")
    datasets = [
        'data/public_hands.csv',
        'data/augmented_public_hands.csv',
        'data/chord_recordings.csv',
        'data/training_data_final.csv'
    ]
    
    analyzed_any = False
    for path in datasets:
        if os.path.exists(path):
            analyze_csv(path)
            analyzed_any = True
            
    if not analyzed_any:
        print("No datasets found to analyze. Please run the data collection and generation scripts first.")

if __name__ == '__main__':
    main()
