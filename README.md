# TimeStamp

Upload a lecture and get a timestamped transcript you can search and ask questions about ("when did he explain deadlock?"). Answers link to the right moment in the video.

## Stack (all free)

| Piece | Tool |
|---|---|
| Transcription | `faster-whisper` (local) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local) |
| Vector DB | Chroma or FAISS (in-process) |
| Q&A and timeline LLM | Gemini or Groq free tier, Ollama as backup |

## Project structure

```
backend/
  app/
    main.py        FastAPI app
    schemas.py     shared contract
    mock.py        mock data loader
    ingestion/     Person 1: upload, transcribe, chunk, store
    search/        Person 2: retrieval, Q&A, timeline
  data/mock/       mock transcript and timeline for lecture "demo"
  scripts/         preprocess.py for the demo lectures
frontend/
  src/
    api.js         backend calls
    components/    Person 3: Upload, VideoPlayer, TimelineBar, SearchPanel
```

Every endpoint already answers with mock data for `lecture_id=demo`. Replace the `TODO`s with the real pipeline.

## Run

Backend:

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Tests

All from `backend/`. No test needs a model download or an API key unless marked.

| Suite | Command | What it covers |
|---|---|---|
| Everything fast | `.venv/bin/python -m pytest` | all suites below |
| Unit | `.venv/bin/python -m pytest --ignore=tests/integration` | each module alone, neighbours faked |
| Top-down integration | `.venv/bin/python -m pytest tests/integration/test_top_down.py` | starts at the API with everything below stubbed, then swaps in one real module per level |
| Big bang integration | `.venv/bin/python -m pytest tests/integration/test_big_bang.py` | the whole backend at once: real ffmpeg, Chroma, cache and scripts; only the models and the LLM are stubbed |
| Real models | `.venv/bin/python -m pytest -m models` | big bang with the real Whisper and embedding models (slow, downloads them) |

The integration tests need `ffmpeg`; tests that use it are skipped without it.

## Demo lectures

Transcripts and timelines of the demo lectures are committed in `backend/data/cache`. The videos and the vector DB are not.

After cloning or pulling new transcripts, rebuild the vector DB (no Whisper run):

```bash
cd backend && .venv/bin/python -m scripts.preprocess --reindex
```

Put each demo video in `backend/data/uploads/` as `<lecture_id>.mp4`; it is then served at `/media/<lecture_id>.mp4`.

To add a demo lecture (needs `ffmpeg`; transcription is slow on CPU):

```bash
cd backend && .venv/bin/python -m scripts.preprocess path/to/lecture.mp4 LECTURE_ID --model small
```

```bash
cd backend && .venv/bin/python -m scripts.build_timelines LECTURE_ID
```

Then commit `data/cache/LECTURE_ID.transcript.json` and `data/cache/LECTURE_ID.timeline.json`. Give demo lectures a named id such as `os-lecture`: cache files of live uploads (ids of 8 hex characters) are git-ignored.

Before the demo, on the demo machine, download and load the models once so the first live upload doesn't wait on a download:

```bash
cd backend && .venv/bin/python -m scripts.preprocess --warmup
```

Live uploads use `WHISPER_MODEL` from `.env` (`base` in `.env.example`); `--model` overrides it for one run of the script.

## Team task division

### Person 1 — Ingestion

- Upload endpoint and audio extraction (ffmpeg)
- Whisper transcription
- Chunk by timestamp, embed, store in the vector DB
- Pre-process and cache the 2–3 demo lectures
- Owns integration and the demo script once ingestion is done

### Person 2 — Search & LLM

- Semantic search endpoint
- Q&A (RAG) returning answers with timestamps
- Topic timeline: one LLM call that splits the transcript into sections
- Cache timeline results for the demo lectures

### Person 3 — Frontend

- Upload UI
- Video player with seek-to-timestamp
- Search and Q&A panel with clickable timestamp links
- Colored topic timeline bar
- Demo polish

## Shared contract

Agree on this in the first hour so nobody is blocked.

Chunk schema:

```json
{ "lecture_id": "string", "start": 0.0, "end": 0.0, "text": "string" }
```

API:

| Endpoint | Owner | Returns |
|---|---|---|
| `POST /upload` | Person 1 | `lecture_id`, `video_url` |
| `GET /status?lecture_id=` | Person 1 | `lecture_id`, `status` (`processing`, `ready` or `failed`), `error` |
| `GET /transcript?lecture_id=` | Person 1 | all chunks |
| `GET /search?lecture_id=&q=` | Person 2 | matching chunks |
| `POST /ask` | Person 2 | answer plus the chunks it cites |
| `GET /timeline?lecture_id=` | Person 2 | sections as `{title, start, end}` |

`POST /upload` returns at once and transcribes in the background. Poll `GET /status` until it says `ready` before loading the transcript, timeline or search; on `failed`, show `error`.

Person 1 hands over one hand-made transcript JSON right away, so Persons 2 and 3 can build against mock data.

## Demo plan

- Demo on the cached lectures.
- Show live upload with a short clip only, since transcription is slow.
- Keep a second LLM provider key or Ollama ready in case of rate limits.
