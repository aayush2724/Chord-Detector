import { useEffect, useRef } from 'react';
import { AudioChordDetector } from '../lib/audioChordDetector';

const SMOOTHING_WINDOW = 5;
const CONFIDENCE_THRESHOLD = 0.5;

function smoothPredictions(buffer) {
  if (buffer.length === 0) return null;
  if (buffer.length === 1) return buffer[0];

  const counts = {};
  for (const pred of buffer) {
    if (!pred.chord || pred.chord === 'N/C') continue;
    counts[pred.chord] = (counts[pred.chord] || 0) + pred.confidence;
  }

  if (Object.keys(counts).length === 0) return buffer[buffer.length - 1];

  const bestChord = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  const matching = buffer.filter((p) => p.chord === bestChord);
  const avgConfidence = matching.reduce((s, p) => s + p.confidence, 0) / matching.length;

  return {
    chord: bestChord,
    confidence: avgConfidence,
    chroma: buffer[buffer.length - 1].chroma,
    model_loaded: true,
    source: 'audio',
  };
}

export default function AudioDetector({ onPrediction, onStatusChange }) {
  const detectorRef = useRef(null);
  const bufferRef = useRef([]);

  useEffect(() => {
    const detector = new AudioChordDetector();
    detectorRef.current = detector;

    detector.onChord = (result) => {
      if (!result) {
        onPrediction(null);
        onStatusChange('Listening...');
        return;
      }

      bufferRef.current.push(result);
      if (bufferRef.current.length > SMOOTHING_WINDOW) {
        bufferRef.current.shift();
      }

      const smoothed = smoothPredictions(bufferRef.current);
      onPrediction(smoothed);

      if (smoothed.chord && smoothed.confidence > CONFIDENCE_THRESHOLD && smoothed.chord !== 'N/C') {
        onStatusChange(`Chord: ${smoothed.chord}`);
      } else {
        onStatusChange('Listening...');
      }
    };

    detector.start().then(() => {
      onStatusChange('Listening...');
    }).catch((err) => {
      console.error('Microphone error:', err);
      onStatusChange('Microphone access denied.');
    });

    return () => detector.stop();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return null;
}
