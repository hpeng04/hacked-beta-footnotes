# TimeStamp

Upload a lecture, get a searchable timestamped transcript, Q&A with links that seek the video, and a colored topic timeline. Hackathon project, team of 3, zero budget.

## Stack

- Backend: Python, FastAPI (`backend/app`)
- Transcription: `faster-whisper` (local), audio via ffmpeg
- Embeddings: `sentence-transformers` `all-MiniLM-L6-v2` (local)
- Vector DB: Chroma, persisted in `backend/data/chroma`
- LLM: Gemini Flash free tier; Groq or Ollama as backup
- Frontend: React + Vite (`frontend/src`)

## Layout and ownership

- `backend/app/schemas.py`: shared contract. Don't change it without telling the team.
- `backend/app/ingestion/`: Person 1. Upload, transcribe, chunk, store.
- `backend/app/search/`: Person 2. Retrieval, Q&A, timeline.
- `frontend/src/components/`: Person 3. Upload, VideoPlayer, TimelineBar, SearchPanel.
- `backend/data/mock/`: mock data for `lecture_id=demo`. Keep it working until the real pipeline replaces it.

## Run

```bash
cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/uvicorn app.main:app --reload
```

```bash
cd frontend && npm install && npm run dev
```

## Rules

- Everything must run free: local models, free-tier LLM only.
- All LLM calls go through `backend/app/search/llm.py`, so the provider can be swapped in one place.
- Cache expensive results (transcripts, chunks, timelines) in `backend/data/cache` and reuse them.
- Timestamps are seconds as floats everywhere.
- Never commit `.env`, uploads or the Chroma DB.
- Keep it simple: hackathon code, no extra infra (no queues, auth or Docker) unless needed for the demo.
- Follow conventional git commit/pr/branch naming standards
