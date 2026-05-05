# Guitar Chord Detector 🎸

A real-time guitar chord recognition system that uses computer vision to detect hand landmarks and classify the chord being played. 

## Features
- **Real-time Detection:** Uses Google's MediaPipe framework to track 21 hand landmarks directly in the browser.
- **Accurate Classification:** A custom-trained machine learning model (Scikit-Learn) classifies the 3D landmark data into one of 15 guitar chords (A, Am, B, Bm, C, Cm, D, Dm, E, Em, F, Fm, G, Gm, Background).
- **FastAPI ML Service:** A robust Python backend exposing the trained model via a `/predict` REST endpoint.
- **Node.js Proxy:** An Express backend that acts as an API gateway for handling CORS and routing.
- **React Frontend:** A responsive, dark-themed UI built with Vite and React, featuring live webcam streaming, a skeletal hand overlay, and animated chord predictions.

## Architecture

1. **`frontend/`** (React + Vite)
   - Captures video using `getUserMedia`.
   - Uses `@mediapipe/tasks-vision` to extract a 63-dimensional feature vector (x,y,z for 21 points).
   - Draws a live skeletal overlay of the hand on an HTML Canvas.
   - Throttles requests and posts the landmark array to the Node proxy.
2. **`backend/`** (Node.js + Express)
   - A lightweight server running on port `3001` that handles CORS.
   - Proxies the `POST /api/predict` requests directly to the ML Service.
3. **`ml-service/`** (Python + FastAPI)
   - Collects, augments, and merges data (using custom scripts and HuggingFace datasets).
   - Trains multiple classifiers (Random Forest, MLP, HistGradientBoosting) and pickles the highest-accuracy model.
   - Hosts the `POST /predict` endpoint using FastAPI on port `8001`.

## Setup & Running Locally

To run the entire stack locally, you will need three terminal instances.

### 1. Start the ML Service
```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

### 2. Start the Node Proxy
```bash
cd backend
npm install
npm start
```
*(Runs on `http://localhost:3001`)*

### 3. Start the React Frontend
```bash
cd frontend
npm install
npm run dev
```
*(Runs on `http://localhost:5173`)*

Open your browser to `http://localhost:5173`, allow camera access, and start playing chords!

## Data Pipeline
To retrain the model or add new chords:
1. `collect_data.py`: Record your own hands playing chords.
2. `download_datasets.py`: Fetch background hand data to prevent false positives.
3. `merge_datasets.py`: Compiles all CSV sources and artificially balances the classes.
4. `train.py`: Fits the classifiers and saves `chord_model.pkl`.
