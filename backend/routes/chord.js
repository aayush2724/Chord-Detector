const express = require('express');
const axios = require('axios');
const router = express.Router();

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8001';

router.post('/predict', async (req, res) => {
  try {
    const { landmarks } = req.body;
    
    if (!landmarks || !Array.isArray(landmarks) || landmarks.length !== 63) {
      return res.status(400).json({ error: 'Expected 63 landmark floats' });
    }

    const response = await axios.post(`${FASTAPI_URL}/predict`, { landmarks });
    
    res.json(response.data);
  } catch (error) {
    console.error('Error proxying to FastAPI:', error.message);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: 'Internal Server Error when contacting ML service' });
    }
  }
});

module.exports = router;
