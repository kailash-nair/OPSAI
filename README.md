# OPSAI

**AI-Powered Meeting Transcription & Summarization for Malayalam Operations Meetings**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Node.js 18+](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org)
[![React 19](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev)

OPSAI transforms Malayalam-language operations meetings (from Microsoft Teams recordings or other video sources) into structured, actionable English summaries. It features an agent-based architecture with multiple speech-to-text backends, LLM-powered summarization, and a modern glassmorphism UI.

---

## Features

- **Multi-backend Speech-to-Text**: Choose from OpenAI Whisper API, Faster-Whisper, or HuggingFace models optimized for Malayalam
- **Intelligent Summarization**: GPT-4o extracts issues, decisions, action items, and highlights
- **Department Tagging**: Automatically categorizes content by department (Installation, Production, Design, etc.)
- **Real-time Processing Dashboard**: Track transcription progress with live updates
- **Meeting Archive**: Search, filter, and browse historical meeting summaries
- **Modern UI**: Glassmorphism design with drag-and-drop file upload

---

## Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+
- **FFmpeg** (for audio extraction)
- **OpenAI API key** (for transcription and summarization)

### Installation

```bash
# Clone the repository
git clone https://github.com/kailash-nair/OPSAI.git
cd OPSAI

# Install Node.js dependencies
npm install
cd server && npm install && cd ..
cd web && npm install && cd ..

# Install Python dependencies
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` in both `server/` and `web/` directories, then add your OpenAI API key.

### Run the Application

```bash
# Start both server and web app
npm run start:agent

# Or run individually
cd server && npm run dev     # API server on port 5001
cd web && npm run dev        # Web app on port 5173
```

Open http://localhost:5173 in your browser.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser (React SPA :5173)                    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Express API Server (:5001)                    │
│  POST /api/files      │  Upload video                           │
│  POST /api/meetings   │  Start processing                       │
│  GET  /api/meetings   │  List meetings                          │
│  GET  /api/meetings/:id/summary  │  Get result                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Python AI Agent (main.py)                    │
│  1. AudioExtractor     │  FFmpeg extracts WAV from video        │
│  2. Transcriber        │  STT backend converts speech to text   │
│  3. RegisterTransformer│  LLM polishes transcript               │
│  4. DeptCueTagger      │  Tags content by department            │
│  5. IssueSummarizer    │  Extracts issues & action items        │
│  6. MarkdownFormatter  │  Renders final summary                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SQLite Database + Files                       │
│  opsai.db  │  uploads/  │  tmp/                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
OPSAI/
├── main.py                 # Python AI Agent (core processing)
├── start.js                # Cross-platform launcher
├── package.json            # Root package (orchestrator)
├── requirements.txt        # Python dependencies
├── open_opsai.bat          # Windows launcher
├── debug_opsai.bat         # Windows debug launcher
│
├── server/                 # Express.js Backend
│   ├── index.js            # REST API server
│   ├── db.js               # SQLite database setup
│   ├── .env                # Server configuration
│   ├── opsai.db            # SQLite database
│   ├── uploads/            # Uploaded video files
│   └── tmp/                # Temporary processing files
│
└── web/                    # React + Vite Frontend
    ├── src/
    │   ├── main.jsx        # Entry point with router
    │   ├── Layout.jsx      # Main layout with sidebar
    │   ├── config.js       # API configuration
    │   ├── pages/
    │   │   ├── Upload.jsx      # File upload interface
    │   │   ├── Dashboard.jsx   # Processing queue
    │   │   └── Archive.jsx     # Meeting history
    │   ├── components/
    │   │   ├── ui/         # shadcn/ui primitives
    │   │   ├── dashboard/  # Dashboard components
    │   │   └── archive/    # Archive components
    │   ├── entities/       # API clients
    │   └── hooks/          # React hooks
    └── vite.config.js      # Vite configuration
```

---

## Speech-to-Text Backends

OPSAI supports multiple STT backends for flexibility:

| Backend | Description | Best For |
|---------|-------------|----------|
| `openai` | OpenAI Whisper API | Highest accuracy, cloud-based |
| `faster_whisper` | Local Faster-Whisper | Privacy, offline use |
| `hf_whisper_ml` | HuggingFace Whisper (Malayalam) | Malayalam optimization |
| `hf_wav2vec2_ml` | HuggingFace Wav2Vec2 (Malayalam) | Malayalam optimization |

Configure via `STT_BACKEND` in `server/.env`.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/files` | Upload video file (multipart/form-data) |
| `POST` | `/api/meetings` | Create meeting and start processing |
| `GET` | `/api/meetings` | List meetings (`sort`, `limit` params) |
| `GET` | `/api/meetings/:id` | Get meeting details |
| `GET` | `/api/meetings/:id/summary` | Get markdown summary |
| `GET` | `/api/meetings/:id/transcript` | Get full transcript |
| `POST` | `/api/meetings/:id/restart` | Reprocess a meeting |

---

## CLI Usage

Run the Python agent directly for batch processing:

```bash
python main.py \
  --input "meeting_recording.mp4" \
  --date "2025-08-10" \
  --attendees "Rahul; Meera; Akash" \
  --out-summary "summary.md" \
  --out-transcript "transcript.txt" \
  --stt openai
```

---

## Development

### Running in Mock Mode

For frontend development without Python/API dependencies:

```env
# In server/.env
AGENT_MODE=mock
```

### Building for Production

```bash
cd web && npm run build
```

### Linting

```bash
cd web && npm run lint
```

### Adding UI Components

This project uses [shadcn/ui](https://ui.shadcn.com/):

```bash
cd web
npx shadcn@latest add [component-name]
```

---

## Tech Stack

### Python Agent
- Python 3.10+, Pydantic, FFmpeg
- OpenAI API, Faster-Whisper, HuggingFace Transformers
- PyTorch, librosa, soundfile

### Backend
- Express.js 5.1, SQLite (better-sqlite3), Multer

### Frontend
- React 19, Vite 7, Tailwind CSS 4
- shadcn/ui (Radix UI), lucide-react, react-router-dom 6

---

## Database Schema

**`meetings` table:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT | UUID (primary key) |
| `title` | TEXT | Meeting title |
| `file_name` | TEXT | Original filename |
| `file_size` | INTEGER | File size in bytes |
| `original_file_path` | TEXT | Path to uploaded file |
| `processing_method` | TEXT | STT backend used |
| `status` | TEXT | pending/processing/completed/failed |
| `progress` | INTEGER | 0-100 progress percentage |
| `created_date` | TEXT | ISO timestamp |
| `created_by` | TEXT | User identifier |
| `markdown_output` | TEXT | Final summary content |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Python agent not starting | Check `PYTHON_EXE` in server/.env (use `py` on Windows) |
| CORS errors | Verify `WEB_ORIGIN` matches your frontend URL |
| Database locked | Ensure no other processes are accessing opsai.db |
| STT failures | Verify API keys and model availability |
| FFmpeg not found | Install FFmpeg and ensure it's in your PATH |

### Debug Mode

```bash
# Run server with visible logs
cd server && AGENT_MODE=python npm run dev

# Windows
debug_opsai.bat
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- **Python**: Type hints, Pydantic models for data validation
- **JavaScript**: ESLint with React hooks rules
- **UI**: Follow glassmorphism design patterns

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Kailash Nair**

---

## Acknowledgments

- [OpenAI Whisper](https://openai.com/research/whisper) for speech recognition
- [shadcn/ui](https://ui.shadcn.com/) for UI components
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for local transcription
- HuggingFace for Malayalam language models
