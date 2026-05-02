import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Camera from './components/Camera';
import ChordOverlay from './components/ChordOverlay';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [backendStatus, setBackendStatus] = useState('connecting'); // connecting, connected, error
  const [supportedChords, setSupportedChords] = useState([]);

  useEffect(() => {
    // Fetch supported chords on load
    const fetchClasses = async () => {
      try {
        const res = await axios.get('http://localhost:3001/api/classes');
        if (res.data && res.data.classes) {
          setSupportedChords(res.data.classes);
        }
      } catch (err) {
        console.warn("Could not fetch supported chords", err);
      }
    };
    fetchClasses();
  }, []);

  const handleChordDetected = (data) => {
    setPrediction(data);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header glass-panel">
        <div className="brand">🎸 ChordSense</div>
        <div className="status-badge">
          <div className={`status-dot ${backendStatus}`}></div>
          {backendStatus === 'connected' && 'Live Engine Active'}
          {backendStatus === 'connecting' && 'Connecting to Backend...'}
          {backendStatus === 'error' && 'Backend Unavailable'}
        </div>
      </header>

      {/* Main Stage */}
      <main className="main-stage">
        <Camera 
          onChordDetected={handleChordDetected} 
          setBackendStatus={setBackendStatus}
        />
        
        <ChordOverlay 
          predictedChord={prediction?.chord} 
          confidence={prediction?.confidence} 
        />
      </main>

      {/* Footer / Info Strip */}
      <div className="chords-strip glass-panel">
        {supportedChords.length > 0 ? (
          supportedChords.map(c => (
            <div 
              key={c} 
              className={`chord-pill ${prediction?.chord === c ? 'active' : ''}`}
            >
              {c}
            </div>
          ))
        ) : (
          <div className="chord-pill">Loading model classes...</div>
        )}
      </div>
    </div>
  );
}

export default App;
