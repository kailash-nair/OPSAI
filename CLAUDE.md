# CLAUDE.md - AI Assistant Guide for OPSAI

## Project Overview

**OPSAI** is an AI-powered meeting transcription and summarization system designed to convert Malayalam-language operations meetings (typically from Microsoft Teams recordings) into structured, actionable English summaries. The application uses an agent-based architecture with multiple configurable speech-to-text backends, LLM-based summarization, and a modern React + Express full-stack architecture.

**Author**: Kailash Nair
**License**: MIT

---

## Quick Start Commands

```bash
# Full stack development (recommended)
npm run start:agent          # Launches both server (5001) and web (5173)

# Individual services
cd server && npm run dev     # Express API with nodemon
cd web && npm run dev        # Vite React dev server

# Python agent standalone testing
python main.py \
  --input "recording.mp4" \
  --date "2025-08-10" \
  --attendees "Rahul; Meera; Akash" \
  --out-summary "summary.md" \
  --out-transcript "transcript.txt" \
  --stt openai

# Build production
cd web && npm run build

# Linting
cd web && npm run lint
```

---

## Repository Structure

```
OPSAI/
├── main.py                 # Python AI Agent (core processing engine)
├── start.js                # Cross-platform launcher
├── package.json            # Root package (orchestrator)
├── requirements.txt        # Python dependencies
├── open_opsai.bat          # Windows launcher
├── debug_opsai.bat         # Windows debug launcher
│
├── server/                 # Node.js/Express Backend
│   ├── index.js            # REST API server
│   ├── db.js               # SQLite database setup
│   ├── .env                # Server configuration
│   ├── opsai.db            # SQLite database
│   ├── uploads/            # Uploaded video files
│   └── tmp/                # Temporary transcript/summary files
│
└── web/                    # React + Vite Frontend
    ├── src/
    │   ├── main.jsx        # React entry point with router
    │   ├── Layout.jsx      # Main layout with sidebar
    │   ├── config.js       # API base URL config
    │   ├── pages/          # Page components
    │   │   ├── Upload.jsx      # File upload interface
    │   │   ├── Dashboard.jsx   # Processing queue
    │   │   └── Archive.jsx     # Meeting history
    │   ├── components/
    │   │   ├── ui/         # shadcn/ui primitives
    │   │   ├── dashboard/  # Dashboard components
    │   │   └── archive/    # Archive components
    │   ├── entities/       # API clients
    │   ├── integrations/   # File upload utilities
    │   └── hooks/          # React hooks
    ├── .env                # Frontend config
    └── vite.config.js      # Vite configuration
```

---

## Tech Stack

### Python Agent (`main.py`)
- **Core**: Python 3, Pydantic (data validation)
- **Audio**: FFmpeg, soundfile, librosa, numpy
- **Speech-to-Text**: OpenAI Whisper API, Faster-Whisper, HuggingFace models
- **LLM**: OpenAI API (gpt-4o-mini, gpt-4o)
- **ML**: transformers, torch, torchaudio

### Backend (`server/`)
- **Framework**: Express.js 5.1.0
- **Database**: SQLite via better-sqlite3
- **File Upload**: Multer
- **Dev**: nodemon for auto-reload

### Frontend (`web/`)
- **Framework**: React 19.1.1
- **Build**: Vite 7.1.2
- **Styling**: Tailwind CSS 4.x with glassmorphism design
- **UI**: shadcn/ui (Radix UI primitives)
- **Icons**: lucide-react
- **Router**: react-router-dom 6.x

---

## Architecture Flow

```
Browser (React SPA on :5173)
    ↓
Express API (:5001)
    ├── POST /api/files → Upload video
    ├── POST /api/meetings → Start processing
    ├── GET /api/meetings → List meetings
    └── GET /api/meetings/:id/summary → Get result
    ↓
Python Agent (subprocess)
    ├── Extract audio (FFmpeg)
    ├── Transcribe (STT backend)
    ├── Polish transcript (LLM)
    ├── Tag departments
    ├── Summarize issues (LLM)
    └── Format markdown
    ↓
SQLite Database + File outputs
```

---

## Key Files & Their Purposes

| File | Purpose |
|------|---------|
| `main.py` | Core AI agent with 6-stage pipeline: audio extraction, transcription, register transformation, department tagging, summarization, markdown formatting |
| `server/index.js` | REST API with file upload, meeting CRUD, process spawning |
| `server/db.js` | SQLite schema setup with `meetings` table |
| `web/src/pages/Upload.jsx` | Drag-drop file upload with STT backend selection |
| `web/src/pages/Dashboard.jsx` | Real-time processing queue with progress bars |
| `web/src/pages/Archive.jsx` | Meeting history with search and filtering |
| `web/src/entities/Meeting.js` | API client for meeting operations |

---

## Configuration

### Server Environment (`server/.env`)

```env
PORT=5001
WEB_ORIGIN=http://localhost:5173
AGENT_MODE=python              # "python" or "mock" for testing
PYTHON_EXE=python              # or "py" on Windows
AGENT_MAIN=../main.py
AGENT_ATTENDEES=Rahul; Meera; Akash

# OpenAI / LLM
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Speech-to-Text Backend
STT_BACKEND=hf_wav2vec2_ml     # openai|faster_whisper|hf_whisper_ml|hf_wav2vec2_ml
FW_MODEL=large-v3              # for faster-whisper
HF_WHISPER_ML_MODEL=vrclc/Whisper-medium-Malayalam
HF_W2V2_ML_MODEL=gvs/wav2vec2-large-xlsr-malayalam
```

### Frontend Environment (`web/.env`)

```env
VITE_API_BASE=http://localhost:5001
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/files` | Upload video file (multipart) |
| `POST` | `/api/meetings` | Create meeting and start processing |
| `GET` | `/api/meetings` | List meetings (supports `sort`, `limit` query params) |
| `GET` | `/api/meetings/:id` | Get single meeting details |
| `GET` | `/api/meetings/:id/summary` | Get markdown summary |
| `GET` | `/api/meetings/:id/transcript` | Get full English transcript |
| `POST` | `/api/meetings/:id/restart` | Reprocess a meeting |

---

## Database Schema

**Table: `meetings`**

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `title` | TEXT | Meeting title |
| `file_name` | TEXT | Original filename |
| `file_size` | INTEGER | File size in bytes |
| `original_file_path` | TEXT | Path to uploaded file |
| `processing_method` | TEXT | STT backend used |
| `status` | TEXT | pending/processing/completed/failed |
| `progress` | INTEGER | 0-100 progress |
| `created_date` | TEXT | ISO timestamp |
| `created_by` | TEXT | User identifier |
| `markdown_output` | TEXT | Final summary content |

---

## Python Agent Pipeline

The agent (`main.py`) follows a Task/Tool/Agent pattern with 6 processing stages:

1. **AudioExtractor**: FFmpeg extracts WAV audio from video
2. **Transcriber**: Speech-to-text with 4 backend options:
   - `openai`: Cloud-based Whisper API
   - `faster_whisper`: Local Faster-Whisper
   - `hf_whisper_ml`: HuggingFace Whisper fine-tuned for Malayalam
   - `hf_wav2vec2_ml`: HuggingFace Wav2Vec2 for Malayalam
3. **RegisterTransformer**: LLM polishes raw transcript into business English
4. **DeptCueTagger**: Tags sentences with departments using Malayalam XLM-R embeddings
5. **IssueSummarizer**: Extracts issues, highlights, decisions, action items via LLM
6. **MarkdownFormatter**: Renders structured Pydantic models as Markdown

**Departments tracked**: Installation, Production, Design, Estimation, Quality, Logistics, Stores

---

## Development Guidelines

### Code Style

- **Frontend**: ESLint with React hooks rules, JSX files
- **Backend**: Standard Node.js/Express patterns
- **Python**: Pydantic models for data validation, type hints

### Component Organization

- UI primitives go in `web/src/components/ui/`
- Page-specific components go in `web/src/components/{page}/`
- API clients go in `web/src/entities/`
- Shared utilities go in `web/src/lib/`

### Adding New Features

1. For new API endpoints: Add to `server/index.js`
2. For new pages: Create in `web/src/pages/` and add route in `main.jsx`
3. For new AI tools: Add Pydantic models and tool classes in `main.py`
4. For UI components: Use shadcn/ui patterns in `web/src/components/ui/`

### Testing the Agent

Use mock mode for frontend development without Python dependencies:

```env
# In server/.env
AGENT_MODE=mock
```

---

## Common Tasks

### Adding a New STT Backend

1. Add environment variables in `server/.env`
2. Implement transcription logic in `Transcriber` class in `main.py`
3. Update the backend selection switch statement
4. Add option to frontend `Upload.jsx` processing method selector

### Modifying the Summary Format

1. Update Pydantic models (`ActionItem`, `IssueSummary`, `MeetingSummary`) in `main.py`
2. Modify `IssueSummarizer` prompt if needed
3. Update `MarkdownFormatter` output template

### Adding New UI Components

```bash
# shadcn/ui components are in web/src/components/ui/
# Add new primitives following the existing pattern
cd web
npx shadcn@latest add [component-name]
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Python agent not starting | Check `PYTHON_EXE` in server/.env (use `py` on Windows) |
| CORS errors | Verify `WEB_ORIGIN` matches frontend URL |
| Database locked | Ensure no other processes accessing opsai.db |
| STT failures | Check API keys and model availability |

### Debug Mode

```bash
# Run with visible logs
AGENT_MODE=python npm run dev  # in server/

# Or use Windows debug launcher
debug_opsai.bat
```

---

## Design System

The frontend uses a glassmorphism aesthetic:

- **Background gradient**: `#1a1d3d` → `#2e2240` → `#4a2f52`
- **Glass panels**: `rgba(30, 27, 46, 0.3)` with `backdrop-filter: blur(25px)`
- **CSS variables**: `--glass-bg`, `--text-primary`, etc.
- **Component library**: shadcn/ui with New York style

---

## Dependencies Installation

```bash
# Root dependencies
npm install

# Server dependencies
cd server && npm install

# Web dependencies
cd web && npm install

# Python dependencies
pip install -r requirements.txt
```

---

## Important Notes for AI Assistants

1. **Don't modify** shadcn/ui primitives in `web/src/components/ui/` unless necessary
2. **Preserve** the glassmorphism design aesthetic when adding UI elements
3. **Use Pydantic models** for any new data structures in the Python agent
4. **Test with mock mode** first when modifying the processing pipeline
5. **Check both outputs** (summary.md and transcript.txt) when modifying the agent
6. **SQLite uses WAL mode** - be mindful of concurrent access
7. **Express 5.x** is used - async error handling differs from v4
8. **React 19** is used - check for any deprecated patterns
9. **The agent pipeline is sequential** - tools depend on previous outputs via `$ctx.key` placeholders
