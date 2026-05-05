import React from 'react';

const ChordOverlay = ({ prediction }) => {
  const isVisible = prediction && prediction.confidence > 0.6 && prediction.chord !== 'Background';
  
  return (
    <div className={`chord-overlay-container ${isVisible ? 'visible' : 'hidden'}`}>
      {isVisible && (
        <div className="chord-badge">
          <div className="chord-name">{prediction.chord}</div>
          <div className="chord-confidence">
            {Math.round(prediction.confidence * 100)}% Match
          </div>
        </div>
      )}
    </div>
  );
};

export default ChordOverlay;
