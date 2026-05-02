import os
import csv
import sys

def main():
    augmented_public_csv = 'data/augmented_public_hands.csv'
    chord_recordings_csv = 'data/chord_recordings.csv'
    final_output_csv = 'data/training_data_final.csv'

    print("--- 4. Merge & Prepare Final Training Data ---")

    # Check existence
    has_public = os.path.exists(augmented_public_csv)
    has_chords = os.path.exists(chord_recordings_csv)

    if not has_public and not has_chords:
        print("Error: Neither augmented_public_hands.csv nor chord_recordings.csv exist.")
        sys.exit(1)

    print(f"Loading datasets...")
    
    # Extract headers and data
    headers = None
    merged_data = []
    
    # 1. Load Public Hands (Background)
    if has_public:
        with open(augmented_public_csv, 'r') as f:
            reader = csv.reader(f)
            public_headers = next(reader)
            if not headers: headers = public_headers
            
            count = 0
            for row in reader:
                merged_data.append(row)
                count += 1
        print(f"Loaded {count} public background samples.")
    else:
        print(f"Warning: {augmented_public_csv} not found. Skipping public data.")

    # 2. Load Chord Recordings
    if has_chords:
        with open(chord_recordings_csv, 'r') as f:
            reader = csv.reader(f)
            chord_headers = next(reader)
            if not headers: headers = chord_headers
            
            # Simple check to ensure schemas match
            if len(chord_headers) != len(headers):
                print("Error: CSV schemas do not match between datasets!")
                sys.exit(1)
                
            count = 0
            for row in reader:
                merged_data.append(row)
                count += 1
        print(f"Loaded {count} chord recordings.")
    else:
        print(f"Warning: {chord_recordings_csv} not found. Skipping chord recordings.")

    # 3. Write Final File
    print(f"Writing {len(merged_data)} total samples to {final_output_csv}...")
    with open(final_output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(merged_data)
        
    print("Merge complete!")

if __name__ == '__main__':
    main()
