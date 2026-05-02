import React from 'react';

const ChordOverlay = ({ predictedChord, confidence }) => {
  const isVisible = predictedChord && confidence > 0.6;
  const displayConfidence = Math.round((confidence || 0) * 100);

  return (
    <div className={`chord-popup-container ${isVisible ? 'visible' : 'hidden'}`}>
      <div className="chord-card">
        <div className="chord-label">{predictedChord || '...'}</div>
        <div className="confidence-bar-container">
          <div 
            className="confidence-bar" 
            style={{ width: `${displayConfidence}%` }} 
          />
        </div>
        <div className="confidence-text">{displayConfidence}% Match</div>
      </div>
    </div>
  );
};

export default ChordOverlay;
