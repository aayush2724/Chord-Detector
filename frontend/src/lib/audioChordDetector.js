const CHORD_PROFILES = {
  'A':   [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
  'Am':  [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
  'B':   [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
  'Bm':  [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
  'C':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
  'Cm':  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
  'D':   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
  'Dm':  [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
  'E':   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
  'Em':  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
  'F':   [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
  'Fm':  [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
  'G':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
  'Gm':  [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
};

function freqToMidi(freq) {
  return 69 + 12 * Math.log2(freq / 440);
}

function chromaCorrelation(chroma, profile) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < 12; i++) {
    dot += chroma[i] * profile[i];
    normA += chroma[i] * chroma[i];
    normB += profile[i] * profile[i];
  }
  return normA && normB ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
}

export class AudioChordDetector {
  constructor() {
    this.audioContext = null;
    this.analyser = null;
    this.source = null;
    this.stream = null;
    this.fftSize = 4096;
    this.running = false;
    this.onChord = null;
    this._loop = this._loop.bind(this);
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = this.fftSize;
    this.analyser.smoothingTimeConstant = 0.3;

    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.source.connect(this.analyser);

    this.running = true;
    requestAnimationFrame(this._loop);
  }

  stop() {
    this.running = false;
    if (this.source) this.source.disconnect();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.audioContext) this.audioContext.close();
  }

  _loop() {
    if (!this.running) return;

    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);
    this.analyser.getFloatFrequencyData(dataArray);

    const sampleRate = this.audioContext.sampleRate;
    const binWidth = sampleRate / this.fftSize;

    const chroma = new Float32Array(12);
    let totalEnergy = 0;

    for (let i = 0; i < bufferLength; i++) {
      const freq = i * binWidth;
      if (freq < 60 || freq > 2000) continue;

      const power = Math.pow(10, dataArray[i] / 10);
      totalEnergy += power;

      const midi = freqToMidi(freq);
      const noteIdx = Math.round(midi) % 12;
      const cents = midi - Math.floor(midi);
      const weight = 1 - Math.abs(cents - 0.5) * 2;
      chroma[noteIdx] += power * Math.max(weight, 0.3);
    }

    if (totalEnergy < 1000) {
      if (this.onChord) this.onChord(null);
      requestAnimationFrame(this._loop);
      return;
    }

    const maxChroma = Math.max(...chroma);
    if (maxChroma > 0) {
      for (let i = 0; i < 12; i++) chroma[i] /= maxChroma;
    }

    let bestChord = 'N/C';
    let bestScore = -1;
    for (const [chord, profile] of Object.entries(CHORD_PROFILES)) {
      const score = chromaCorrelation(chroma, profile);
      if (score > bestScore) {
        bestScore = score;
        bestChord = chord;
      }
    }

    const chromaArr = Array.from(chroma);
    if (this.onChord) {
      this.onChord({
        chord: bestChord,
        confidence: Math.max(0, Math.min(1, bestScore)),
        chroma: chromaArr,
        source: 'audio',
      });
    }

    requestAnimationFrame(this._loop);
  }
}
