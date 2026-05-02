// routes/chord.js — Proxy route to FastAPI ML service
const express = require('express');
const axios   = require('axios');
const router  = express.Router();

const ML_URL = process.env.ML_SERVICE_URL || 'http://localhost:8001';

/**
 * POST /api/predict
 * Body: { landmarks: [63 floats] }
 * Returns: { chord, confidence, all_probs }
 */
router.post('/predict', async (req, res, next) => {
  const { landmarks } = req.body;

  // Basic validation
  if (!Array.isArray(landmarks) || landmarks.length !== 63) {
    return res.status(400).json({
      error: `Invalid payload: expected landmarks array of length 63, got ${landmarks?.length ?? 'none'}`,
    });
  }

  try {
    const response = await axios.post(
      `${ML_URL}/predict`,
      { landmarks },
      { timeout: 3000, headers: { 'Content-Type': 'application/json' } }
    );
    return res.json(response.data);
  } catch (err) {
    if (err.response) {
      // FastAPI returned an error
      return res.status(err.response.status).json(err.response.data);
    }
    if (err.code === 'ECONNREFUSED' || err.code === 'ENOTFOUND') {
      return res.status(503).json({
        error: 'ML service unavailable. Make sure FastAPI is running on port 8001.',
      });
    }
    if (err.code === 'ETIMEDOUT' || err.code === 'ECONNABORTED') {
      return res.status(504).json({ error: 'ML service timed out.' });
    }
    next(err);
  }
});

/**
 * GET /api/classes
 * Returns list of chord classes from the ML service.
 */
router.get('/classes', async (req, res, next) => {
  try {
    const response = await axios.get(`${ML_URL}/classes`, { timeout: 2000 });
    return res.json(response.data);
  } catch (err) {
    next(err);
  }
});

/**
 * GET /api/health
 * Proxy health check to ML service.
 */
router.get('/ml-health', async (req, res) => {
  try {
    const response = await axios.get(`${ML_URL}/health`, { timeout: 2000 });
    return res.json({ backend: 'ok', ml_service: response.data });
  } catch {
    return res.json({ backend: 'ok', ml_service: 'unreachable' });
  }
});

module.exports = router;
