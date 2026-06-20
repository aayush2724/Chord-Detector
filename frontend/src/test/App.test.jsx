import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from '../App';

vi.mock('@mediapipe/tasks-vision', () => ({
  HandLandmarker: { createFromOptions: vi.fn() },
  FilesetResolver: { forVisionTasks: vi.fn() },
}));

vi.mock('axios', () => ({
  default: { post: vi.fn().mockResolvedValue({ data: {} }) },
}));

beforeEach(() => {
  Object.defineProperty(navigator, 'mediaDevices', {
    value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }) },
    writable: true,
  });
});

describe('App', () => {
  it('renders the header', () => {
    render(<App />);
    expect(screen.getByText('Guitar Chord Detector')).toBeInTheDocument();
  });

  it('shows a status indicator', () => {
    render(<App />);
    expect(document.querySelector('.status-indicator')).toBeInTheDocument();
    expect(document.querySelector('.pulse-dot')).toBeInTheDocument();
  });

  it('renders the video element', () => {
    render(<App />);
    expect(document.querySelector('video')).toBeInTheDocument();
  });
});
