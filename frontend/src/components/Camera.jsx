import React, { useRef, useEffect, useState } from 'react';
import { Camera as MediaPipeCamera } from '@mediapipe/camera_utils';
import { Hands } from '@mediapipe/hands';
import axios from 'axios';
import HandSkeleton from './HandSkeleton';

// Backend URL - Change this to production URL when deploying
const BACKEND_URL = 'http://localhost:3001/api';

const Camera = ({ onChordDetected, setBackendStatus }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Smooth predictions
  const historyRef = useRef([]);

  useEffect(() => {
    let camera;

    const initializeMediaPipe = async () => {
      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });

      hands.onResults(handleResults);

      if (videoRef.current) {
        camera = new MediaPipeCamera(videoRef.current, {
          onFrame: async () => {
            if (videoRef.current) {
              await hands.send({ image: videoRef.current });
            }
          },
          width: 1280,
          height: 720,
        });
        camera.start();
      }
    };

    initializeMediaPipe();

    return () => {
      if (camera) {
        camera.stop();
      }
    };
  }, []);

  const normalizeLandmarks = (landmarks) => {
    // Exact same logic as Python's normalize_landmarks
    const wrist = landmarks[0];
    
    // Calculate distance between wrist and middle finger MCP (landmark 9)
    const mcp = landmarks[9];
    const dx = mcp.x - wrist.x;
    const dy = mcp.y - wrist.y;
    const dz = mcp.z - wrist.z;
    const scale = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;

    const flat = [];
    for (let i = 0; i < landmarks.length; i++) {
      flat.push((landmarks[i].x - wrist.x) / scale);
      flat.push((landmarks[i].y - wrist.y) / scale);
      flat.push((landmarks[i].z - wrist.z) / scale);
    }
    return flat;
  };

  const handleResults = async (res) => {
    setResults(res);
    setIsLoading(false);

    if (canvasRef.current && videoRef.current) {
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
    }

    if (res.multiHandLandmarks && res.multiHandLandmarks.length > 0) {
      const landmarks = res.multiHandLandmarks[0];
      const flatLandmarks = normalizeLandmarks(landmarks);

      try {
        const response = await axios.post(`${BACKEND_URL}/predict`, {
          landmarks: flatLandmarks
        });
        
        setBackendStatus('connected');

        // Smoothing logic: Majority vote of last 5 frames
        const pred = response.data;
        if (pred.confidence > 0.6) {
          historyRef.current.push(pred.chord);
          if (historyRef.current.length > 5) {
            historyRef.current.shift();
          }

          const counts = {};
          let maxCount = 0;
          let majorityChord = null;

          historyRef.current.forEach(c => {
            counts[c] = (counts[c] || 0) + 1;
            if (counts[c] > maxCount) {
              maxCount = counts[c];
              majorityChord = c;
            }
          });

          if (maxCount >= 3) {
            onChordDetected({ chord: majorityChord, confidence: pred.confidence });
          }
        } else {
          historyRef.current = [];
          onChordDetected(null);
        }

      } catch (error) {
        console.error("Backend prediction error", error);
        setBackendStatus('error');
      }
    } else {
      historyRef.current = [];
      onChordDetected(null);
    }
  };

  return (
    <div className="camera-wrapper">
      {isLoading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Initializing Camera & Hand Tracking...</p>
        </div>
      )}
      
      <video
        ref={videoRef}
        className="video-feed"
        autoPlay
        playsInline
      ></video>
      
      <canvas
        ref={canvasRef}
        className="canvas-overlay"
      ></canvas>

      <HandSkeleton canvasRef={canvasRef} results={results} />
    </div>
  );
};

export default Camera;
