import React, { useEffect, useRef } from 'react';
import { HAND_CONNECTIONS } from '@mediapipe/hands';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';

const HandSkeleton = ({ canvasRef, results }) => {
  useEffect(() => {
    if (!canvasRef.current || !results) return;

    const canvasCtx = canvasRef.current.getContext('2d');
    const { width, height } = canvasRef.current;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, width, height);

    if (results.multiHandLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        // Draw elegant connections (teal)
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
          color: '#00dcb4',
          lineWidth: 4,
        });
        
        // Draw landmark nodes (amber and white)
        drawLandmarks(canvasCtx, landmarks, {
          color: (data) => (data.index === 0 ? '#ffb400' : '#ffffff'),
          fillColor: '#000000',
          lineWidth: 2,
          radius: 5,
        });
      }
    }
    canvasCtx.restore();
  }, [results, canvasRef]);

  return null; // This component just handles drawing to the provided canvas
};

export default HandSkeleton;
