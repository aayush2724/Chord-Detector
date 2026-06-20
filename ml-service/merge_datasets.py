"""
merge_datasets.py
=================
Merges all available CSVs into a single training_data_final.csv.

Sources (all optional — uses whatever exists):
  - data/augmented_public_hands.csv   (background / non-chord hands)
  - data/global_chords.csv            (scraped labeled guitar chords)
  - data/chord_recordings.csv         (your personal camera recordings)
"""

import csv
import os
import random
import sys
from collections import Counter

# Cap Background samples to reduce class imbalance
MAX_BACKGROUND_SAMPLES = 500
RANDOM_SEED = 42


def load_csv(path, normalize_cols):
    """Load a CSV and ensure it has the right columns. Returns (header, rows)."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                rows.append(row)
    return header, rows


def normalize_row(row, src_header, tgt_header):
    """
    Map a source row to the target header, filling missing columns with 'unknown'.
    """
    src_map = {col: i for i, col in enumerate(src_header)}
    out = []
    for col in tgt_header:
        if col in src_map:
            out.append(row[src_map[col]])
        else:
            out.append('unknown')
    return out


def main():
    # ── canonical header ──────────────────────────────────────────────
    COORD_COLS = [f'{ax}_{i}' for i in range(21) for ax in ['x', 'y', 'z']]
    TARGET_HEADER = ['label', 'source'] + COORD_COLS

    sources = [
        ('data/augmented_public_hands.csv', 'augmented_public'),
        ('data/global_chords.csv',          'global_scrape'),
        ('data/chord_recordings.csv',        'personal_recording'),
    ]

    output_csv = 'data/training_data_final.csv'

    print("=" * 55)
    print("   Merge Datasets → training_data_final.csv")
    print("=" * 55)

    found_any = False
    all_rows = []

    for path, source_tag in sources:
        if not os.path.exists(path):
            print(f"  ⚠  Skipping (not found): {path}")
            continue

        found_any = True
        header, rows = load_csv(path, TARGET_HEADER)

        # Inject source column if missing
        if 'source' not in header:
            header = header[:1] + ['source'] + header[1:]
            rows = [r[:1] + [source_tag] + r[1:] for r in rows]

        # Add 'angle' column under a 'source' label if it has angle but not source
        normalized = []
        for row in rows:
            normalized.append(normalize_row(row, header, TARGET_HEADER))

        all_rows.extend(normalized)
        print(f"  ✓  {os.path.basename(path)}: {len(rows)} samples loaded")

    if not found_any:
        print("\nError: No datasets found at all. Run the data collection scripts first.")
        sys.exit(1)

    # ── Undersample Background class ──────────────────────────────────
    background_rows = [r for r in all_rows if r[0].strip().lower() == 'background']
    other_rows      = [r for r in all_rows if r[0].strip().lower() != 'background']

    if len(background_rows) > MAX_BACKGROUND_SAMPLES:
        random.seed(RANDOM_SEED)
        background_rows = random.sample(background_rows, MAX_BACKGROUND_SAMPLES)
        print(f"\n  ✂  Background undersampled: {len(background_rows)} samples kept (max {MAX_BACKGROUND_SAMPLES})")

    all_rows = other_rows + background_rows

    # ── Write final CSV ───────────────────────────────────────────────
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(TARGET_HEADER)
        writer.writerows(all_rows)

    # ── Summary ───────────────────────────────────────────────────────
    label_counts = Counter(row[0] for row in all_rows)

    print(f"\n  Total samples : {len(all_rows)}")
    print(f"  Unique classes: {len(label_counts)}")
    print("\n  Class breakdown:")
    for label, count in label_counts.most_common():
        print(f"    {label:<20} {count}")

    print(f"\n  ✅ Saved → {output_csv}")


if __name__ == '__main__':
    main()
