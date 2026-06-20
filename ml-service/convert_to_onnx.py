"""
convert_to_onnx.py — Train a compact model and convert to ONNX for browser inference.
=====================================================================================
Trains a lightweight RandomForest that fits within ONNX protobuf limits,
then converts and verifies.

Usage:
  python convert_to_onnx.py [--data data/chord_data.csv] [--output model/chord_model.onnx]
"""

import argparse
import csv
import os
import pickle
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "data", "chord_data.csv")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "model", "chord_model.onnx")


def load_data(csv_path: str):
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
    return np.array(X, dtype=np.float32), np.array(y)


def convert(data_path: str, output_path: str):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    print(f"Loading data from: {data_path}")
    X, y = load_data(data_path)
    print(f"Samples: {len(X)} | Features: {X.shape[1]}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    print(f"Classes: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("\nTraining compact RandomForest (100 trees, depth=15)...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42,
        ))
    ])
    pipe.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"Train accuracy: {train_acc*100:.1f}%")
    print(f"Test accuracy:  {test_acc*100:.1f}%")

    print("\nConverting to ONNX...")
    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nSaved: {output_path} ({size_kb:.0f} KB)")

    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    test_input = X_test[:5].astype(np.float32)
    result = session.run(None, {"float_input": test_input})
    pred_indices = np.argmax(result[0], axis=1)
    pred_labels = [classes[i] for i in pred_indices]
    true_labels = [classes[i] for i in y_test[:5]]
    print("\nVerification (first 5 test samples):")
    for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
        mark = "✓" if pred == true else "✗"
        print(f"  {mark} Predicted: {pred:<12} Actual: {true}")

    # Also save metadata for the frontend
    meta = {
        'classes': classes,
        'n_features': int(X.shape[1]),
        'model_type': 'RandomForest_100trees_depth15',
        'accuracy': float(test_acc),
    }
    meta_path = output_path.replace('.onnx', '_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"\nMetadata saved: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to ONNX for browser")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.data, args.output)
