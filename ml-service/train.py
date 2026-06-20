"""
train.py — Guitar Chord Classifier Training
============================================
Loads landmark data from data/chord_data.csv, trains a Random Forest
and an MLP classifier, evaluates both, and saves the best model to
model/chord_model.pkl along with the label encoder.

Usage:
  python train.py [--data data/chord_data.csv] [--model model/chord_model.pkl]

Output:
  model/chord_model.pkl   — pickled dict: { 'model': clf, 'encoder': le, 'classes': [...] }
  Prints accuracy, classification report, and confusion matrix.
"""

import argparse
import csv
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA  = os.path.join(SCRIPT_DIR, "data", "chord_data.csv")
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "model", "chord_model.pkl")


def load_data(csv_path: str):
    """Load CSV and return (X, y) as numpy arrays."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    X, y = [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        has_source = len(header) > 1 and header[1] == 'source'

        for row in reader:
            if has_source:
                if len(row) < 65:
                    continue
                label = row[0]
                features = [float(v) for v in row[2:65]]
            else:
                if len(row) < 64:
                    continue
                label = row[0]
                features = [float(v) for v in row[1:64]]

            y.append(label)
            X.append(features)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y


def print_banner(text: str):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def train(data_path: str, model_path: str):
    print_banner("Guitar Chord Classifier — Training")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading data from: {data_path}")
    X, y = load_data(data_path)
    print(f"[INFO] Total samples : {len(X)}")
    print(f"[INFO] Features      : {X.shape[1]}")

    unique, counts = np.unique(y, return_counts=True)
    print(f"[INFO] Chord classes : {list(unique)}")
    print("\n  Samples per chord:")
    for chord, cnt in zip(unique, counts):
        bar = "█" * min(cnt // 2, 40)
        print(f"    {chord:<10} {cnt:>4}  {bar}")

    if len(unique) < 2:
        raise ValueError("Need at least 2 chord classes to train. Collect more data first.")

    min_samples = int(counts.min())
    if min_samples < 10:
        print(f"\n[WARN] Some chords have very few samples ({min_samples}). "
              "Consider collecting at least 50 samples per chord.")

    # ── Encode labels ──────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Train/test split ───────────────────────────────────────────────────────
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
    )
    print(f"\n[INFO] Train: {len(X_train)} | Test: {len(X_test)} ({test_size*100:.0f}% split)")

    # ── Define classifiers ─────────────────────────────────────────────────────
    classifiers = {
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1,
                random_state=42,
            ))
        ]),
        "MLP Neural Net": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ))
        ]),
        "Hist Gradient Boosting": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            ))
        ]),
    }

    best_name = None
    best_acc  = 0.0
    best_clf  = None

    for name, pipe in classifiers.items():
        print(f"\n[TRAIN] {name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Test accuracy: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc  = acc
            best_name = name
            best_clf  = pipe

    print_banner(f"Best model: {best_name}  ({best_acc*100:.2f}% accuracy)")

    # ── Full report on best model ──────────────────────────────────────────────
    y_pred_best = best_clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    header = "         " + "  ".join(f"{c:<6}" for c in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{le.classes_[i]:<8} " + "  ".join(f"{v:<6}" for v in row)
        print(row_str)

    # ── Cross-validation score ─────────────────────────────────────────────────
    print("\n[INFO] 5-fold CV on full dataset ...")
    cv_scores = cross_val_score(best_clf, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # ── Save model ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    payload = {
        'model':   best_clf,
        'encoder': le,
        'classes': list(le.classes_),
        'n_features': X.shape[1],
        'model_name': best_name,
        'accuracy': best_acc,
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'trained_at': datetime.now().isoformat(),
        'n_samples': len(X),
    }
    with open(model_path, 'wb') as f:
        pickle.dump(payload, f)

    # Save a timestamped copy for version history
    model_dir = os.path.dirname(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(model_dir, f"chord_model_{timestamp}.pkl")
    with open(versioned_path, 'wb') as f:
        pickle.dump(payload, f)

    print(f"\n[SAVED] Model written to: {model_path}")
    print(f"  Versioned copy: {versioned_path}")
    print(f"  Classes: {list(le.classes_)}")
    print_banner("Done! Run main.py to serve the model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train guitar chord classifier")
    parser.add_argument('--data',  default=DEFAULT_DATA,  help="Path to CSV data file")
    parser.add_argument('--model', default=DEFAULT_MODEL, help="Output model path (.pkl)")
    args = parser.parse_args()
    train(args.data, args.model)
