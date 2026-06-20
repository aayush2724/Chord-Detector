import { useEffect, useRef } from 'react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import axios from 'axios';
import { loadModel as loadOnnxModel, predict as onnxPredict, isLoaded as onnxLoaded } from '../lib/onnxInference';

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

const sendPredictionRequest = async (landmarks) => {
  try {
    const res = await axios.post('/api/predict', { landmarks }, { timeout: 2000 });
    return res.data;
  } catch (err) {
    console.error("API error:", err);
    return null;
  }
};

const Camera = ({ videoRef, onLandmarks, onPrediction, onStatusChange }) => {
  const requestRef = useRef();
  const landmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastPostTimeRef = useRef(0);
  const predictionBufferRef = useRef([]);
  const inferenceModeRef = useRef('api');
  const videoStreamRef = useRef(null);
  const predictRef = useRef(null);

  useEffect(() => {
    predictRef.current = async () => {
      if (!videoRef.current || !landmarkerRef.current) return;
      const video = videoRef.current;

      const startTimeMs = performance.now();
      if (lastVideoTimeRef.current !== video.currentTime) {
        lastVideoTimeRef.current = video.currentTime;

        const results = landmarkerRef.current.detectForVideo(video, startTimeMs);
        const hands = results.landmarks || [];

        if (hands.length > 0) {
          onLandmarks(hands);

          const throttleMs = inferenceModeRef.current === 'onnx' ? 100 : 300;
          if (startTimeMs - lastPostTimeRef.current > throttleMs) {
            lastPostTimeRef.current = startTimeMs;

            let bestRaw = null;
            for (const handLandmarks of hands) {
              const features = [];
              for (let i = 0; i < 21; i++) {
                features.push(handLandmarks[i].x, handLandmarks[i].y, handLandmarks[i].z);
              }

              let raw;
              if (inferenceModeRef.current === 'onnx' && onnxLoaded()) {
                raw = onnxPredict(features);
              } else {
                raw = await sendPredictionRequest(features);
              }

              if (raw && (!bestRaw || raw.confidence > bestRaw.confidence)) {
                bestRaw = raw;
              }
            }

            if (bestRaw) {
              predictionBufferRef.current.push(bestRaw);
              if (predictionBufferRef.current.length > SMOOTHING_WINDOW) {
                predictionBufferRef.current.shift();
              }

              const smoothed = smoothPredictions(predictionBufferRef.current);
              onPrediction(smoothed);

              if (smoothed.chord && smoothed.confidence > CONFIDENCE_THRESHOLD && smoothed.chord !== 'Background') {
                onStatusChange(`Chord: ${smoothed.chord}`);
              } else {
                onStatusChange(hands.length > 1 ? '2 hands detected!' : 'Hand detected!');
              }
            }
          }
        } else {
          onLandmarks(null);
          onPrediction(null);
          onStatusChange('Looking for hand...');
        }
      }

      requestRef.current = requestAnimationFrame(predictRef.current);
    };
  });

  useEffect(() => {
    let active = true;

    const initializeMediaPipe = async () => {
      onStatusChange('Loading hand tracking model...');
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
          numHands: 2
        });

        if (!active) return;
        landmarkerRef.current = landmarker;

        onStatusChange('Loading chord classifier...');
        const onnxResult = await loadOnnxModel();
        if (onnxResult.loaded) {
          inferenceModeRef.current = 'onnx';
          onStatusChange('Running locally. Ready!');
        } else {
          inferenceModeRef.current = 'api';
          onStatusChange('Local model unavailable. Using API...');
        }

        startCamera();
      } catch (err) {
        console.error("Error loading models:", err);
        onStatusChange('Error loading models.');
      }
    };

    const startCamera = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
          });
          videoStreamRef.current = stream;
          if (videoRef.current && active) {
            videoRef.current.srcObject = stream;
            videoRef.current.addEventListener('loadeddata', () => {
              predictRef.current?.();
            });
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
      const stream = videoStreamRef.current;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
