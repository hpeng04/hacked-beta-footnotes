# TimeStamp

Upload a lecture and get a timestamped transcript you can search and ask questions about ("when did he explain deadlock?"). Answers link to the right moment in the video.

## Stack (all free)

| Piece | Tool |
|---|---|
| Backend | Python, FastAPI |
| Transcription | Groq hosted Whisper (free tier) when a key is set, local `faster-whisper` otherwise or on failure; audio via ffmpeg |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local) |
| Vector DB | Chroma (in-process, persisted in `backend/data/chroma`) |
| Q&A, timeline and topic LLM | Groq free tier (default), Ollama as backup; Gemini also supported |
| Frontend | React + Vite |

## Project structure

```
backend/
  app/
    main.py        FastAPI app, serves uploads at /media
    config.py      paths and .env settings
    schemas.py     shared contract
    mock.py        mock data loader
    ingestion/     Person 1
      router.py      /upload, /status, /transcript
      pipeline.py    transcribe -> chunk -> store, status and transcript cache
      transcribe.py  ffmpeg audio, Groq Whisper, local faster-whisper fallback
      timing.py      word timing and sentence excerpts
      chunking.py    merge segments into search chunks
      store.py       embeddings and Chroma
    search/        Person 2
      router.py        /search, /ask, /timeline
      llm.py           the only place that calls an LLM; Q&A prompts and citation parsing
      retrieval.py     vector search
      timeline.py      LLM topic timeline and its cache
      segmentation.py  topic detection (embeddings + LLM refinement)
      sections.py      section index storage (*.sections.json)
      titles.py        fallback section titles
      presentation.py  section-grouped citations for /ask and /timeline
  data/
    mock/          mock transcript and timeline for lecture "demo", sample lectures in samples/
    cache/         transcripts, segments, timelines, sections, upload status
    uploads/       videos (not committed)
    chroma/        vector DB (not committed)
  scripts/         preprocess, build_timelines, build_sections, live_llm_test
  tests/           unit and integration tests
frontend/
  test.html        standalone API test page
  src/
    App.jsx        workspace layout, upload status polling
    api.js         backend calls
    components/    Person 3: Upload, VideoPlayer (custom controls + TimelineBar), SearchPanel (Q&A)
```

`lecture_id=demo` and the sample lectures in `backend/data/mock/samples` are served from mock data; uploaded and pre-processed lectures use the real pipeline.

## Run

Needs Python 3.12, Node 20 and `ffmpeg`.

Backend:

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/uvicorn app.main:app --reload
```

Set `LLM_API_KEY` in `backend/.env` (free Groq key). The same key enables Groq Whisper; without a key, transcription runs locally and the LLM falls back to Ollama. `.env` overrides shell variables.

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend reads the backend URL from `VITE_API_URL` (`frontend/.env`, default `http://localhost:8000`).

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
| Live LLM | `.venv/bin/python -m pytest -m live` | the providers configured in `backend/.env` |
| Ollama | `.venv/bin/python -m pytest -m ollama` | a local Ollama server |

The integration tests need `ffmpeg`; tests that use it are skipped without it.

`.venv/bin/python -m scripts.live_llm_test` runs the real LLM on the sample lectures and checks the citations, without writing the cache.

CI (`.github/workflows/ci.yml`) runs `ruff` (errors only) and the fast tests for the backend, and `npm run build` for the frontend.

## Demo lectures

Transcripts, segments and timelines of pre-processed demo lectures live in `backend/data/cache` and are meant to be committed. The videos and the vector DB are not.

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

Before the demo, on the demo machine, download and load the models once so the first live upload doesn't wait on a download:

```bash
cd backend && .venv/bin/python -m scripts.preprocess --warmup
```

Live uploads use `WHISPER_MODEL` from `.env` (`base` in `.env.example`); `--model` overrides it for one run of the script.

## How it works

### Transcription and timing

With a Groq key, audio is compressed to 32 kbps mono MP3 and sent to Groq's hosted Whisper (`GROQ_WHISPER_MODEL`, about 14 MB per hour, 25 MB limit). Without a key, over the limit or on any Groq failure, local `faster-whisper` runs batched (`WHISPER_BATCH_SIZE`, `WHISPER_BEAM_SIZE`).

Search uses merged chunks for context. Both paths return word timestamps; the versioned `*.segments.json` cache preserves word alignment. Answers and topic timelines use sentence excerpts built from those words, including sentences spanning Whisper segments. Silence and long unpunctuated speech also bound excerpts. Word timing is a model estimate, not frame-exact visual alignment.

For an older recording with missing word timing, its first question or timeline request recovers alignment from the saved video once, which can take time on CPU. This leaves its searchable transcript and vector index unchanged. Portable transcripts without a recording retain their available segment/chunk timing; the app does not interpolate word times from text.

### Answers and citations

The LLM selects stable excerpt IDs. The backend validates those IDs and supplies the stored float timestamps; it does not move citations using keyword overlap. Invalid replies and provider failures produce a retry message without a transcript dump. Nearby citations from a continuous passage are combined into one replay point: starts must be within 10 seconds of the first citation, with no gap over two seconds between cited excerpts. Consecutive answer sentences sharing that passage display one timestamp. Separate passages keep separate links, and replay uses the original float start time.

### Timeline

Timeline caches record a format version and a fingerprint of their excerpts. Old caches and changed timing are regenerated automatically when the transcript is available. Long timelines use multiple requests, each with at most 24,000 transcript characters, to retain sentence anchors within free-tier prompt limits. Results are cached only after all requests succeed. `scripts.build_timelines` also upgrades stale caches.

### Topic sections

`app/search/sections.py` stores a separate `*.sections.json` index of topic sections. It preserves sentence text, float timing, and the existing stable sentence IDs. Each topic section has a title, a membership-derived ID, bounds derived from its sentences, and an ordered list of sentence IDs. Every sentence must belong to exactly one section; sections must be ordered and cannot overlap. Silence between sections is allowed. Retitling a section preserves its ID; changing its membership changes its ID.

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

Refinement degrades per decision: an invalid decision keeps its semantic boundary, an unusable title falls back to a short description extracted from the section text, and a failed request only affects its own batch. A result with failed requests is cached as a fallback and served; the app retries its refinement in the background (at most once a minute per lecture, reusing the stored boundaries), and `--force` retries immediately. Embedding failures raise an error and preserve the previous cache. Cache fingerprints include the detector settings, embedding model/revision, and configured LLM providers/models; concurrent requests for the same lecture share one generation. Use `segmentation.build(lecture_id, options=Options(...))` from Python to configure context size, prompt budget, or an updated embedding revision.

### Section-grouped answers

Q&A and the timeline share this section index. The answer model still receives sentence evidence and returns validated sentence IDs. The API preserves all supporting excerpts in `sources` and `sentences`, while `citations` contains one display link per cited section across the whole answer. Each link includes `id`, `section_id`, `title`, `start`, and `end`; its replay target is the earliest cited supporting sentence within the section, not necessarily the section start. Distinct sections retain separate links even when they are close together. If membership is unavailable or mismatches the evidence snapshot, proximity grouping supplies display links while preserving the evidence; those links have `section_id: null` and an `id` derived from the passage.

Each entry in `sentences` also carries `citation_ids`, the `id`s of the `citations` that support it, in list order. The frontend renders the answer text with a small numbered marker after each sentence, followed by a numbered list of rows (clickable `mm:ss–mm:ss` chip, then the section title) without transcript blocks. Generic generated titles such as `Part 1` are replaced with short descriptions extracted from the section text. The timeline uses the same section titles and starts, extending each displayed band through the silence before the following section. Requests never build sections: `/ask` and `/timeline` read the cached index and start a background build when it is missing, so until it exists `/ask` uses proximity grouping and `/timeline` serves the existing timeline (cached or generated). The app accepts a cached index built with any options, so `build_sections` output (including `--semantic-only`) is served as is.

## Team ownership

### Person 1 — Ingestion

- Upload, status and transcript endpoints; audio extraction (ffmpeg)
- Whisper transcription (Groq, local fallback) with word timing
- Chunk by timestamp, embed, store in Chroma
- `scripts.preprocess` for demo lectures; integration tests

### Person 2 — Search & LLM

- Semantic search endpoint
- Q&A (RAG) with validated, section-grouped citations
- Topic timeline and topic-section detection, both cached
- LLM provider layer with backup and rate-limit retries

### Person 3 — Frontend

- Upload UI with processing status
- Custom video player with the colored topic timeline as its seek bar
- Q&A panel with numbered citations that seek the video
- Dark, token-based theme

## Shared contract

Defined in `backend/app/schemas.py`. Chunk schema:

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
| `POST /ask` (`lecture_id`, `question`) | Person 2 | `answer`, cited `sources`, `sentences` as `{text, sources, citation_ids}`, `citations` as `{id, section_id, title, start, end}` |
| `GET /timeline?lecture_id=` | Person 2 | sections as `{title, start, end}` |
| `GET /media/<file>` | — | the uploaded video |

`POST /upload` returns at once and transcribes in the background. Poll `GET /status` until it says `ready` before loading the transcript, timeline or search; on `failed`, show `error`.

`answer` remains a readable string ending with one `[mm:ss Title]` label per cited section (plain `[mm:ss]` when sections are unavailable). `citations` supplies deduplicated, titled replay links; clients should render those links and seek to each citation's `start` without rounding. `sentences` retains the original evidence associations. The top-level `sources` contains only cited excerpts, not every retrieval candidate.

## Demo plan

- Demo on the cached lectures.
- Show live upload with a short clip only, since transcription is slow.
- Run `scripts.preprocess --warmup` beforehand.
- Keep Ollama running as the backup LLM in case of rate limits.
