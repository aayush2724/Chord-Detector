// server.js — Express entry point
const express    = require('express');
const cors       = require('cors');
const morgan     = require('morgan');
const rateLimit  = require('express-rate-limit');
require('dotenv').config();

const chordRouter = require('./routes/chord');

const app  = express();
const PORT = process.env.PORT || 3001;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '1mb' }));
app.use(morgan('dev'));

// Rate-limit: max 60 requests / 10 seconds per IP (covers 30fps with headroom)
const limiter = rateLimit({
  windowMs: 10_000,
  max: 60,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, slow down.' },
});
app.use('/api', limiter);

// ── Routes ────────────────────────────────────────────────────────────────────
app.use('/api', chordRouter);

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'chord-detector-backend' });
});

app.use((_req, res) => res.status(404).json({ error: 'Not found' }));

// Global error handler
app.use((err, _req, res, _next) => {
  console.error('[ERROR]', err.message);
  res.status(err.status || 500).json({ error: err.message || 'Internal server error' });
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🎸 Chord Detector Backend running on http://localhost:${PORT}`);
  console.log(`   ML Service URL: ${process.env.ML_SERVICE_URL || 'http://localhost:8001'}\n`);
});
