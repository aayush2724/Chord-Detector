import React, { useState, useRef } from 'react';
import Camera from './components/Camera';
import HandSkeleton from './components/HandSkeleton';
import ChordOverlay from './components/ChordOverlay';
import ChordDiagram from './components/ChordDiagram';
import './index.css';

function App() {
  const [landmarks, setLandmarks] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState('Initializing...');
  const videoRef = useRef(null);

  const activeChord = prediction?.chord && prediction.confidence > 0.6 && prediction.chord !== 'Background'
    ? prediction.chord
    : null;

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Guitar Chord Detector</h1>
        <div className="status-indicator">
          <span className="pulse-dot"></span>
          {status}
        </div>
      </header>

      <main className="video-container">
        <Camera 
          videoRef={videoRef}
          onLandmarks={setLandmarks} 
          onPrediction={setPrediction}
          onStatusChange={setStatus}
        />
        <HandSkeleton landmarks={landmarks} videoRef={videoRef} />
        <ChordOverlay prediction={prediction} />
        {activeChord && (
          <div className="chord-diagram-container">
            <ChordDiagram chord={activeChord} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
