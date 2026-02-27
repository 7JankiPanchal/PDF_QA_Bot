const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json());

// Storage for uploaded PDFs with file size limit (10MB)
const upload = multer({
  dest: "uploads/",
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Accept only PDF files
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed'));
    }
  }
});

// Route: Upload PDF
app.post("/upload", upload.single("file"), async (req, res) => {
  let filePath = null;

  try {
    // Check if file was uploaded
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded. Use form field name 'file'." });
    }

    // Build absolute file path
    filePath = path.join(__dirname, req.file.path);

    // Verify file exists on disk
    if (!fs.existsSync(filePath)) {
      return res.status(500).json({ error: "File upload failed - file not found on disk" });
    }

    console.log(`Processing PDF: ${req.file.originalname} (${req.file.size} bytes)`);

    // Send PDF to Python service for processing
    const response = await axios.post("http://localhost:5000/process-pdf", {
      filePath: filePath,
    }, {
      timeout: 60000 // 60 second timeout
    });

    res.json({
      message: "PDF uploaded & processed successfully!",
      filename: req.file.originalname,
      size: req.file.size
    });
  } catch (err) {
    // Clean up uploaded file on error
    if (filePath && fs.existsSync(filePath)) {
      try {
        fs.unlinkSync(filePath);
        console.log(`Cleaned up file after error: ${filePath}`);
      } catch (cleanupErr) {
        console.error(`Failed to cleanup file: ${cleanupErr.message}`);
      }
    }

    // Determine error type and send appropriate response
    if (err.code === 'ECONNREFUSED') {
      console.error("RAG service not available");
      return res.status(503).json({
        error: "RAG service unavailable",
        details: "Please ensure the Python service is running on port 5000"
      });
    }

    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({
        error: "File too large",
        details: "Maximum file size is 10MB"
      });
    }

    const details = err.response?.data || err.message;
    console.error("Upload processing failed:", details);
    res.status(500).json({ error: "PDF processing failed", details });
  }
});

// Route: Ask Question
app.post("/ask", async (req, res) => {
  const { question } = req.body;
  try {
    const response = await axios.post("http://localhost:5000/ask", {
      question,
    });

    res.json({ answer: response.data.answer });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: "Error answering question" });
  }
});

app.post("/summarize", async (req, res) => {
  try {
    const response = await axios.post("http://localhost:5000/summarize", req.body || {});
    res.json({ summary: response.data.summary });
  } catch (err) {
    const details = err.response?.data || err.message;
    console.error("Summarization failed:", details);
    res.status(500).json({ error: "Error summarizing PDF", details });
  }
});

// Route: Generate Smart Question Suggestions
app.post("/generate-suggestions", async (req, res) => {
  try {
    const response = await axios.post("http://localhost:5000/suggest-questions", {}, {
      timeout: 10000
    });
    res.json({ suggestions: response.data.suggestions || [] });
  } catch (err) {
    console.error("Suggestion generation failed:", err.message);
    res.json({ suggestions: [] }); // Fail gracefully
  }
});

// Global error handler for multer errors
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({
        error: "File too large",
        details: "Maximum file size is 10MB"
      });
    }
    return res.status(400).json({ error: err.message });
  } else if (err) {
    return res.status(400).json({ error: err.message });
  }
  next();
});

app.listen(4000, () => console.log("Backend running on http://localhost:4000"));
