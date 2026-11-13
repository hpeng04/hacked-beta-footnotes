# LectureLens

Upload a lecture and get a timestamped transcript you can search and ask questions about ("when did he explain deadlock?"). Answers link to the right moment in the video.

## Stack (all free)

| Piece | Tool |
|---|---|
| Transcription | `faster-whisper` (local) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local) |
| Vector DB | Chroma or FAISS (in-process) |
| Q&A and timeline LLM | Gemini or Groq free tier, Ollama as backup |

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
| `POST /upload` | Person 1 | `lecture_id` |
| `GET /search?lecture_id=&q=` | Person 2 | matching chunks |
| `POST /ask` | Person 2 | answer plus the chunks it cites |
| `GET /timeline?lecture_id=` | Person 2 | sections as `{title, start, end}` |

Person 1 hands over one hand-made transcript JSON right away, so Persons 2 and 3 can build against mock data.

## Demo plan

- Demo on the cached lectures.
- Show live upload with a short clip only, since transcription is slow.
- Keep a second LLM provider key or Ollama ready in case of rate limits.
