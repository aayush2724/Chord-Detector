import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ChordOverlay from '../components/ChordOverlay';

describe('ChordOverlay', () => {
  it('renders nothing when prediction is null', () => {
    const { container } = render(<ChordOverlay prediction={null} />);
    expect(container.querySelector('.chord-badge')).toBeNull();
  });

  it('renders nothing when confidence is below threshold', () => {
    const { container } = render(
      <ChordOverlay prediction={{ chord: 'Am', confidence: 0.3, all_probs: {} }} />
    );
    expect(container.querySelector('.chord-badge')).toBeNull();
  });

  it('renders nothing for Background class', () => {
    const { container } = render(
      <ChordOverlay prediction={{ chord: 'Background', confidence: 0.9, all_probs: {} }} />
    );
    expect(container.querySelector('.chord-badge')).toBeNull();
  });

  it('shows chord name when confidence is above threshold', () => {
    render(
      <ChordOverlay prediction={{ chord: 'Am', confidence: 0.85, all_probs: {} }} />
    );
    expect(screen.getByText('Am')).toBeInTheDocument();
    expect(screen.getByText('85% Match')).toBeInTheDocument();
  });

  it('applies visible class when showing chord', () => {
    const { container } = render(
      <ChordOverlay prediction={{ chord: 'G', confidence: 0.7, all_probs: {} }} />
    );
    expect(container.querySelector('.visible')).toBeInTheDocument();
  });

  it('applies hidden class when not showing chord', () => {
    const { container } = render(
      <ChordOverlay prediction={{ chord: 'Background', confidence: 0.5, all_probs: {} }} />
    );
    expect(container.querySelector('.hidden')).toBeInTheDocument();
  });
});
