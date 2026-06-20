import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ChordDiagram from '../components/ChordDiagram';

describe('ChordDiagram', () => {
  it('renders nothing for unknown chord', () => {
    const { container } = render(<ChordDiagram chord="Xm9" />);
    expect(container.querySelector('.chord-diagram')).toBeNull();
  });

  it('renders nothing for null chord', () => {
    const { container } = render(<ChordDiagram chord={null} />);
    expect(container.querySelector('.chord-diagram')).toBeNull();
  });

  it('renders diagram for known chord', () => {
    const { container } = render(<ChordDiagram chord="Am" />);
    expect(screen.getByText('Am', { selector: '.chord-diagram-title' })).toBeInTheDocument();
    expect(container.querySelector('.chord-svg')).toBeInTheDocument();
  });

  it('renders all 14 supported chords without crashing', () => {
    const chords = ['A', 'Am', 'B', 'Bm', 'C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm'];
    for (const chord of chords) {
      const { container, unmount } = render(<ChordDiagram chord={chord} />);
      expect(container.querySelector('.chord-diagram')).toBeInTheDocument();
      unmount();
    }
  });
});
