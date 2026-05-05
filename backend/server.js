require('dotenv').config();
const express = require('express');
const cors = require('cors');
const chordRoutes = require('./routes/chord');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api', chordRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: 'Node.js backend is running' });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
