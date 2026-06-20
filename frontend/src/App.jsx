import { useState, useRef } from 'react';
import Camera from './components/Camera';
import HandSkeleton from './components/HandSkeleton';
import ChordOverlay from './components/ChordOverlay';
import ChordDiagram from './components/ChordDiagram';
import AudioDetector from './components/AudioDetector';
import './index.css';

function App() {
  const [landmarks, setLandmarks] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState('Initializing...');
  const [mode, setMode] = useState('camera');
  const videoRef = useRef(null);

  const activeChord = prediction?.chord && prediction.confidence > 0.6 &&
    prediction.chord !== 'Background' && prediction.chord !== 'N/C'
    ? prediction.chord
    : null;

  const handleModeSwitch = (newMode) => {
    setMode(newMode);
    setPrediction(null);
    setLandmarks(null);
    setStatus('Initializing...');
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Guitar Chord Detector</h1>
        <div className="header-controls">
          <div className="mode-toggle">
            <button
              className={`mode-btn ${mode === 'camera' ? 'active' : ''}`}
              onClick={() => handleModeSwitch('camera')}
            >
              Camera
            </button>
            <button
              className={`mode-btn ${mode === 'audio' ? 'active' : ''}`}
              onClick={() => handleModeSwitch('audio')}
            >
              Microphone
            </button>
          </div>
          <div className="status-indicator">
            <span className="pulse-dot"></span>
            {status}
          </div>
        </div>
      </header>

      <main className="video-container">
        {mode === 'camera' ? (
          <>
            <Camera
              videoRef={videoRef}
              onLandmarks={setLandmarks}
              onPrediction={setPrediction}
              onStatusChange={setStatus}
            />
            <HandSkeleton landmarks={landmarks} videoRef={videoRef} />
          </>
        ) : (
          <AudioDetector
            onPrediction={setPrediction}
            onStatusChange={setStatus}
          />
        )}
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
