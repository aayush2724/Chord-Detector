import React from 'react';

const CHORD_DATA = {
  'A':   { frets: [2, 2, 2, 0, 0, 0], fingers: [2, 1, 3, 0, 0, 0], baseFret: 1 },
  'Am':  { frets: [2, 2, 1, 0, 0, 0], fingers: [2, 3, 1, 0, 0, 0], baseFret: 1 },
  'B':   { frets: [-1, 4, 4, 4, 2, 2], fingers: [0, 2, 3, 4, 1, 1], baseFret: 2 },
  'Bm':  { frets: [-1, 2, 4, 4, 3, 2], fingers: [0, 1, 3, 4, 2, 1], baseFret: 2 },
  'C':   { frets: [-1, 3, 2, 0, 1, 0], fingers: [0, 3, 2, 0, 1, 0], baseFret: 1 },
  'Cm':  { frets: [-1, 3, 5, 5, 4, 3], fingers: [0, 1, 3, 4, 2, 1], baseFret: 3 },
  'D':   { frets: [-1, -1, 0, 2, 3, 2], fingers: [0, 0, 0, 1, 3, 2], baseFret: 1 },
  'Dm':  { frets: [-1, -1, 0, 2, 3, 1], fingers: [0, 0, 0, 2, 3, 1], baseFret: 1 },
  'E':   { frets: [0, 2, 2, 1, 0, 0], fingers: [0, 2, 3, 1, 0, 0], baseFret: 1 },
  'Em':  { frets: [0, 2, 2, 0, 0, 0], fingers: [0, 2, 3, 0, 0, 0], baseFret: 1 },
  'F':   { frets: [1, 3, 3, 2, 1, 1], fingers: [1, 3, 4, 2, 1, 1], baseFret: 1 },
  'Fm':  { frets: [1, 3, 3, 1, 1, 1], fingers: [1, 3, 4, 1, 1, 1], baseFret: 1 },
  'G':   { frets: [3, 2, 0, 0, 0, 3], fingers: [2, 1, 0, 0, 0, 3], baseFret: 1 },
  'Gm':  { frets: [3, 5, 5, 3, 3, 3], fingers: [1, 3, 4, 1, 1, 1], baseFret: 3 },
};

const STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e'];

export default function ChordDiagram({ chord }) {
  if (!chord || !CHORD_DATA[chord]) return null;

  const { frets, fingers, baseFret } = CHORD_DATA[chord];
  const displayFrets = frets.map((f) => (f === -1 ? -1 : f - baseFret + 1));

  return (
    <div className="chord-diagram">
      <div className="chord-diagram-title">{chord}</div>
      <svg viewBox="-30 -10 120 95" className="chord-svg">
        {/* Nut (thick line at top if baseFret is 1) */}
        {baseFret === 1 && (
          <line x1="0" y1="0" x2="100" y2="0" stroke="#fff" strokeWidth="3" />
        )}

        {/* Frets */}
        {[1, 2, 3, 4].map((fret) => (
          <line
            key={fret}
            x1="0"
            y1={fret * 18}
            x2="100"
            y2={fret * 18}
            stroke="#555"
            strokeWidth="1"
          />
        ))}

        {/* Strings */}
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <line
            key={i}
            x1={i * 20}
            y1="0"
            x2={i * 20}
            y2="72"
            stroke="#888"
            strokeWidth="1"
          />
        ))}

        {/* Finger positions */}
        {displayFrets.map((fret, i) => {
          if (fret === -1) {
            return (
              <text
                key={i}
                x={i * 20}
                y="-3"
                textAnchor="middle"
                fill="#ff5555"
                fontSize="10"
                fontWeight="bold"
              >
                X
              </text>
            );
          }
          if (fret === 0) {
            return (
              <circle
                key={i}
                cx={i * 20}
                cy="-3"
                r="4"
                fill="none"
                stroke="#00ffcc"
                strokeWidth="1.5"
              />
            );
          }
          return (
            <React.Fragment key={i}>
              <circle
                cx={i * 20}
                cy={fret * 18 - 9}
                r="6"
                fill="#00ffcc"
              />
              {fingers[i] > 0 && (
                <text
                  x={i * 20}
                  y={fret * 18 - 6}
                  textAnchor="middle"
                  fill="#000"
                  fontSize="8"
                  fontWeight="bold"
                >
                  {fingers[i]}
                </text>
              )}
            </React.Fragment>
          );
        })}

        {/* Base fret indicator */}
        {baseFret > 1 && (
          <text x="-20" y="20" fill="#888" fontSize="8" textAnchor="end">
            {baseFret}fr
          </text>
        )}

        {/* String names */}
        {STRING_NAMES.map((name, i) => (
          <text
            key={name}
            x={i * 20}
            y="85"
            textAnchor="middle"
            fill="#666"
            fontSize="7"
          >
            {name}
          </text>
        ))}
      </svg>
    </div>
  );
}
