import React, { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import axios from 'axios';

const SMOOTHING_WINDOW = 5;
const CONFIDENCE_THRESHOLD = 0.6;

function smoothPredictions(buffer) {
  if (buffer.length === 0) return null;
  if (buffer.length === 1) return buffer[0];

  const counts = {};
  for (const pred of buffer) {
    if (!pred.chord || pred.chord === 'Background') continue;
    counts[pred.chord] = (counts[pred.chord] || 0) + pred.confidence;
  }

  if (Object.keys(counts).length === 0) return buffer[buffer.length - 1];

  const bestChord = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  const matching = buffer.filter((p) => p.chord === bestChord);
  const avgConfidence = matching.reduce((s, p) => s + p.confidence, 0) / matching.length;

  return {
    chord: bestChord,
    confidence: avgConfidence,
    all_probs: buffer[buffer.length - 1].all_probs,
    model_loaded: true,
  };
}

const Camera = ({ videoRef, onLandmarks, onPrediction, onStatusChange }) => {
  const requestRef = useRef();
  const landmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastPostTimeRef = useRef(0);
  const predictionBufferRef = useRef([]);

  const [isModelLoaded, setIsModelLoaded] = useState(false);

  useEffect(() => {
    let active = true;

    const initializeMediaPipe = async () => {
      onStatusChange('Loading ML Model...');
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/model/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });

        if (!active) return;
        landmarkerRef.current = landmarker;
        setIsModelLoaded(true);
        onStatusChange('Model Loaded. Waiting for camera...');
        
        startCamera();
      } catch (err) {
        console.error("Error loading MediaPipe model:", err);
        onStatusChange('Error loading ML Model.');
      }
    };

    const startCamera = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
          });
          if (videoRef.current && active) {
            videoRef.current.srcObject = stream;
            videoRef.current.addEventListener('loadeddata', predictWebcam);
            onStatusChange('Looking for hand...');
          }
        } catch (err) {
          console.error("Camera error:", err);
          onStatusChange('Camera access denied or unavailable.');
        }
      }
    };

    initializeMediaPipe();

    return () => {
      active = false;
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
      }
    };
  }, []);

  const predictWebcam = async () => {
    if (!videoRef.current || !landmarkerRef.current) return;
    const video = videoRef.current;

    let startTimeMs = performance.now();
    if (lastVideoTimeRef.current !== video.currentTime) {
      lastVideoTimeRef.current = video.currentTime;
      
      const results = landmarkerRef.current.detectForVideo(video, startTimeMs);
      
      if (results.landmarks && results.landmarks.length > 0) {
        const handLandmarks = results.landmarks[0];
        onLandmarks(handLandmarks);
        
        // Extract 63 floats
        const features = [];
        for (let i = 0; i < 21; i++) {
          features.push(handLandmarks[i].x, handLandmarks[i].y, handLandmarks[i].z);
        }

        // Throttle API calls to every 300ms
        if (startTimeMs - lastPostTimeRef.current > 300) {
          lastPostTimeRef.current = startTimeMs;
          sendPredictionRequest(features);
        }
      } else {
        onLandmarks(null);
        onPrediction(null);
        onStatusChange('Looking for hand...');
      }
    }
    
    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  const sendPredictionRequest = async (landmarks) => {
    try {
      const res = await axios.post('/api/predict', { landmarks }, { timeout: 2000 });
      const raw = res.data;

      predictionBufferRef.current.push(raw);
      if (predictionBufferRef.current.length > SMOOTHING_WINDOW) {
        predictionBufferRef.current.shift();
      }

      const smoothed = smoothPredictions(predictionBufferRef.current);
      onPrediction(smoothed);

      if (smoothed.chord && smoothed.confidence > CONFIDENCE_THRESHOLD && smoothed.chord !== 'Background') {
        onStatusChange(`Chord: ${smoothed.chord}`);
      } else {
        onStatusChange('Hand detected!');
      }
    } catch (err) {
      console.error("API error:", err);
    }
  };

  return (
    <video
      ref={videoRef}
      className="camera-video"
      autoPlay
      playsInline
      muted
    />
  );
};

export default Camera;
