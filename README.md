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

API test page (no build): open `frontend/test.html` in a browser with the backend running. It exercises every endpoint.

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

Then commit `data/cache/LECTURE_ID.transcript.json`, `data/cache/LECTURE_ID.segments.json`, and `data/cache/LECTURE_ID.timeline.json`. Give demo lectures a named id such as `os-lecture`: cache files of live uploads (ids of 8 hex characters) are git-ignored.

Search uses merged chunks for context. Whisper runs locally with `word_timestamps=True`; its versioned `*.segments.json` cache preserves word alignment. Answers and topic timelines use sentence excerpts built from those words, including sentences spanning Whisper segments. Silence and long unpunctuated speech also bound excerpts. Word timing is a model estimate, not frame-exact visual alignment.

For an older recording with missing word timing, its first question or timeline request recovers alignment from the saved video once, which can take time on CPU. This leaves its searchable transcript and vector index unchanged. Portable transcripts without a recording retain their available segment/chunk timing; the app does not interpolate word times from text.

The LLM selects stable excerpt IDs. The backend validates those IDs and supplies the stored float timestamps; it does not move citations using keyword overlap. Invalid replies and provider failures produce a retry message without a transcript dump. Nearby citations from a continuous passage are combined into one replay point: starts must be within 10 seconds of the first citation, with no gap over two seconds between cited excerpts. Consecutive answer sentences sharing that passage display one timestamp. Separate passages keep separate links, and replay uses the original float start time.

Timeline caches record a format version and a fingerprint of their excerpts. Old caches and changed timing are regenerated automatically when the transcript is available. Long timelines use multiple requests, each with at most 24,000 transcript characters, to retain sentence anchors within free-tier prompt limits. Results are cached only after all requests succeed. `scripts.build_timelines` also upgrades stale caches.

The section-membership foundation in `app/search/sections.py` stores a separate `*.sections.json` index for future topic segmentation. It preserves sentence text, float timing, and the existing stable sentence IDs. Each topic section has a title, a membership-derived ID, bounds derived from its sentences, and an ordered list of sentence IDs. Every sentence must belong to exactly one section; sections must be ordered and cannot overlap. Silence between sections is allowed. Retitling a section preserves its ID; changing its membership changes its ID.

The segmentation stage calls `build_index(lecture_id, sentences, drafts)` with `SectionDraft(title=..., sentence_ids=[...])` proposals, then `save_index(index, segmentation=settings)`. Settings must include nonempty `algorithm` and `revision` strings plus every parameter that affects segmentation (such as model revision, window size, and similarity threshold). `load_index(lecture_id, current_sentences, segmentation=settings)` returns `None` for missing, corrupt, or stale caches, signaling that the caller must regenerate them. Cache validity includes the current sentence snapshot, sentence-building version, segmentation settings, and storage format version. Writes are atomic and invalid replacements preserve the previous cache.

Topic detection is available through `app/search/segmentation.py` and a preprocessing command:

```bash
cd backend
.venv/bin/python -m scripts.build_sections LECTURE_ID
# Use only local embeddings, with titles extracted from section text:
.venv/bin/python -m scripts.build_sections LECTURE_ID --semantic-only
# Retry refinement or regenerate a cached result:
.venv/bin/python -m scripts.build_sections LECTURE_ID --force
```

The detector embeds sentences with the existing MiniLM model, compares mean embeddings of neighboring windows, and proposes local similarity minima as topic boundaries. Defaults are a three-sentence window, cosine threshold 0.55, at least two sentences per section, and at least 12 seconds per section. These are heuristic defaults, adjustable through `--window`, `--threshold`, `--min-sentences`, and `--min-seconds`. Continuous topics are not split at fixed time intervals; overlapping speech cannot form a section boundary.

The LLM receives bounded context around candidates and can retain, move to a nearby allowed sentence, or remove each boundary. It must return valid sentence IDs in order, including the opening section. All sentences are then assigned to contiguous groups and validated using the storage layer. Requests are capped at 24,000 characters by default, and batching never creates section boundaries of its own.

Missing or invalid refinement results retain the semantic boundaries and short descriptions extracted from the section text. That fallback is cached to avoid repeatedly spending model calls; use `--force` to retry refinement. Embedding failures raise an error and preserve the previous cache. Cache fingerprints include the detector settings, embedding model/revision, and configured LLM providers/models; concurrent requests for the same lecture share one generation. Use `segmentation.build(lecture_id, options=Options(...))` from Python to configure context size, prompt budget, or an updated embedding revision.

Q&A and the timeline now share this section index. The answer model still receives sentence evidence and returns validated sentence IDs. The API preserves all supporting excerpts in `sources` and `sentences`, while `citations` contains one display link per cited section across the whole answer. Each link includes `section_id`, `title`, `start`, and `end`; its replay target is the earliest cited supporting sentence within the section, not necessarily the section start. Distinct sections retain separate links even when they are close together. If membership is unavailable or mismatches the evidence snapshot, proximity grouping supplies display links while preserving the evidence.

The frontend renders the answer text followed by a list of clickable `mm:ss - description` rows without transcript blocks. Generic generated titles such as `Part 1` are replaced with short descriptions extracted from the section text. The timeline uses the same section titles and starts, extending each displayed band through the silence before the following section. Existing timeline generation is retained as a fallback when the section index is unavailable.

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
- Topic timeline: select sentence boundaries with bounded LLM requests
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
| `POST /ask` | Person 2 | `answer`, cited `sources`, and `sentences` as `{text, sources}` |
| `GET /timeline?lecture_id=` | Person 2 | sections as `{title, start, end}` |

`POST /upload` returns at once and transcribes in the background. Poll `GET /status` until it says `ready` before loading the transcript, timeline or search; on `failed`, show `error`.

`answer` remains a readable string with timestamp labels. `citations` supplies deduplicated, titled replay links; clients should render those links and seek to each citation's `start` without rounding. `sentences` retains the original evidence associations. The top-level `sources` contains only cited excerpts, not every retrieval candidate.

Person 1 hands over one hand-made transcript JSON right away, so Persons 2 and 3 can build against mock data.

## Demo plan

- Demo on the cached lectures.
- Show live upload with a short clip only, since transcription is slow.
- Keep a second LLM provider key or Ollama ready in case of rate limits.
