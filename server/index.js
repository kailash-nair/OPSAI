// index.js — SQLite-backed API
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const { v4: uuid } = require("uuid");

const db = require("./db");

const PORT = process.env.PORT || 5001;
const ORIGIN = process.env.WEB_ORIGIN || "http://localhost:5173";
const AGENT_MODE = (process.env.AGENT_MODE || "mock").toLowerCase(); // "mock" or "python"
const PYTHON_EXE = process.env.PYTHON_EXE || (process.platform === "win32" ? "python" : "python3");
const AGENT_MAIN = process.env.AGENT_MAIN || path.join(__dirname, "..", "main.py");

const app = express();
app.use(cors({ origin: ORIGIN }));
app.use(express.json({ limit: "50mb" }));

// uploads + tmp
const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
app.use("/uploads", express.static(uploadsDir));
const tmpDir = path.join(__dirname, "tmp");
if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });

// ---------- NEW: centralize transcript file location ----------
function transcriptPathFor(id) {
  return path.join(tmpDir, `transcript_${id}.txt`);
}

// multer
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, uploadsDir),
  filename: (_, file, cb) => {
    const ext = path.extname(file.originalname);
    const base = path.basename(file.originalname, ext).replace(/\s+/g, "_");
    cb(null, `${base}_${uuid()}${ext}`);
  },
});
const upload = multer({ storage });

// prepared statements
const insertStmt = db.prepare(`
  INSERT INTO meetings (
    id, title, original_file_url, original_file_path, file_name, file_size,
    processing_method, validate_summary, status, progress, created_date, created_by, markdown_output
  ) VALUES (
    @id, @title, @original_file_url, @original_file_path, @file_name, @file_size,
    @processing_method, @validate_summary, @status, @progress, @created_date, @created_by, @markdown_output
  )
`);
const listDescStmt = db.prepare(`SELECT * FROM meetings ORDER BY datetime(created_date) DESC LIMIT ?`);
const listAscStmt  = db.prepare(`SELECT * FROM meetings ORDER BY datetime(created_date) ASC  LIMIT ?`);
const getStmt      = db.prepare(`SELECT * FROM meetings WHERE id = ?`);
const progStmt     = db.prepare(`UPDATE meetings SET progress=? WHERE id=?`);
const doneStmt     = db.prepare(`UPDATE meetings SET status=?, progress=?, markdown_output=? WHERE id=?`);
const restartStmt  = db.prepare(`UPDATE meetings SET status='processing', progress=0, markdown_output='' WHERE id=?`);

// helpers
function newMeeting(body) {
  const id = uuid();
  const now = new Date().toISOString(); // unchanged
  return {
    id,
    title: body.title || "Untitled",
    original_file_url: body.original_file_url || "",
    original_file_path: body.original_file_path || "",
    file_name: body.file_name || "",
    file_size: Number(body.file_size || 0),
    processing_method: body.processing_method || "openai_whisper",
    validate_summary: body.validate_summary !== false ? 1 : 0, // Default to enabled
    status: "processing",
    progress: 0,
    created_date: now,
    created_by: "You",
    markdown_output: "" // always defined
  };
}

// upload
app.post("/api/files", upload.single("file"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file" });
  const file_url = `http://localhost:${PORT}/uploads/${req.file.filename}`;
  const file_path = req.file.path;
  res.json({ file_url, file_path });
});

// create meeting
app.post("/api/meetings", (req, res) => {
  const m = newMeeting(req.body || {});
  insertStmt.run(m);
  const created = getStmt.get(m.id);
  res.json(created);

  if (AGENT_MODE === "python") {
    startAgentPython(m.id).catch(err => markFailed(m.id, String(err)));
  } else {
    startAgentMock(m.id);
  }
});

// list
app.get("/api/meetings", (req, res) => {
  const sort = req.query.sort || "-created_date";
  const limit = Number(req.query.limit || 100);
  const rows = (sort === "-created_date" ? listDescStmt : listAscStmt).all(limit);
  res.json(rows);
});

// get one
app.get("/api/meetings/:id", (req, res) => {
  const row = getStmt.get(req.params.id);
  if (!row) return res.status(404).json({ error: "Not found" });
  res.json(row);
});

// ---------- NEW: split summary/transcript fetch ----------
app.get("/api/meetings/:id/summary", (req, res) => {
  const row = getStmt.get(req.params.id);
  if (!row) return res.status(404).json({ error: "Not found" });
  return res.json({ markdown: row.markdown_output || "" });
});

app.get("/api/meetings/:id/transcript", (req, res) => {
  const id = req.params.id;
  const p = transcriptPathFor(id);
  if (!fs.existsSync(p)) return res.status(404).json({ error: "Transcript not available yet" });
  try {
    const text = fs.readFileSync(p, "utf-8");
    return res.json({ text });
  } catch {
    return res.status(500).json({ error: "Failed to read transcript" });
  }
});

// restart
app.post("/api/meetings/:id/restart", (req, res) => {
  const row = getStmt.get(req.params.id);
  if (!row) return res.status(404).json({ error: "Not found" });
  restartStmt.run(req.params.id);
  if (AGENT_MODE === "python") {
    startAgentPython(req.params.id).catch(err => markFailed(req.params.id, String(err)));
  } else {
    startAgentMock(req.params.id);
  }
  res.json({ ok: true });
});

// ---- MOCK processing ----
function startAgentMock(id) {
  let step = 0;
  const timer = setInterval(() => {
    step += 1;
    const prog = Math.min(100, step * 12);
    try { progStmt.run(prog, id); } catch {}
    if (prog >= 100) {
      clearInterval(timer);
      const row = getStmt.get(id);
      const summaryMd =
        `# Operations Meeting Summary\n\n` +
        `**Date:** ${row.created_date.slice(0, 10)}\n` +
        `**Attendees:** Rahul, Meera, Akash\n\n` +
        `**Purpose:** Demo summary generated by mock server\n\n` +
        `## Issues Raised\n\n` +
        `### Example Issue\n` +
        `**Discussion Highlights**\n- Highlight A\n- Highlight B\n\n` +
        `**Decisions**\n- Proceed with demo\n\n` +
        `**Action Items**\n- **Share real transcript with agent** — **Owner:** Design (Deadline: by EOW)\n`;

      // ---------- NEW: write a demo transcript file ----------
      const transcriptText =
        `Speaker 1: Good morning everyone, let's begin the operations meeting.\n` +
        `Speaker 2: Production backlog is down 12% this week.\n` +
        `Speaker 3: Logistics confirmed two pending deliveries for tomorrow.\n` +
        `...`;
      try { fs.writeFileSync(transcriptPathFor(id), transcriptText, "utf-8"); } catch {}

      // keep summary in DB (unchanged)
      doneStmt.run("completed", 100, summaryMd, id);
    }
  }, 800);
}

// ---- Python agent processing ----
async function startAgentPython(id) {
  const row = getStmt.get(id);
  if (!row) throw new Error("Meeting not found");
  if (!row.original_file_path || !fs.existsSync(row.original_file_path))
    throw new Error("original_file_path missing or not found");
  if (!fs.existsSync(AGENT_MAIN))
    throw new Error(`AGENT_MAIN not found: ${AGENT_MAIN}`);

  const outSummary = path.join(tmpDir, `summary_${id}.md`);
  const outTranscript = transcriptPathFor(id);
  const date = row.created_date.slice(0, 10);
  const attendees = process.env.AGENT_ATTENDEES || "Ops Team";

  let tick = 0;
  const ticker = setInterval(() => {
    tick = Math.min(95, tick + 2);
    try { progStmt.run(tick, id); } catch {}
  }, 1000);

  // ---------- NEW: auto-detect dual-output support ----------
  let supportsDual = false;
  try {
    const py = fs.readFileSync(AGENT_MAIN, "utf-8");
    supportsDual = py.includes("--out-summary") && py.includes("--out-transcript");
  } catch {}

  // Determine validation flag
  const validateFlag = row.validate_summary ? "--validate" : "--no-validate";

  const args = supportsDual
    ? [
        AGENT_MAIN,
        "--input", row.original_file_path,
        "--date", date,
        "--attendees", attendees,
        "--out-summary", outSummary,
        "--out-transcript", outTranscript,
        validateFlag
      ]
    : [
        AGENT_MAIN,
        "--input", row.original_file_path,
        "--date", date,
        "--attendees", attendees,
        "--out", outSummary // legacy (summary only)
      ];

  const child = spawn(PYTHON_EXE, args, { env: process.env });
  let stderr = "";
  child.stderr.on("data", d => stderr += d.toString());
  child.on("close", code => {
    clearInterval(ticker);
    if (code !== 0) {
      doneStmt.run("failed", 100, "", id);
      console.error("Python agent failed:", stderr);
      return;
    }
    try {
      const md = fs.readFileSync(outSummary, "utf-8");
      doneStmt.run("completed", 100, md || "", id);
      // transcript file already written by python when supportsDual==true
    } catch (e) {
      doneStmt.run("failed", 100, "", id);
      console.error("Failed to read agent output:", e);
    }
  });
}

function markFailed(id, why) {
  try { doneStmt.run("failed", 100, "", id); } catch {}
  console.error("Marked failed:", why);
}

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
