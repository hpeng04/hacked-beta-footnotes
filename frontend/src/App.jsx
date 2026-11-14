import { useEffect, useRef, useState } from "react";
import { API_URL, status as fetchStatus } from "./api.js";
import Upload from "./components/Upload.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";
import SearchPanel from "./components/SearchPanel.jsx";

const DEMO = { lecture_id: "demo", video_url: "", ready: true };
const POLL_MS = 2000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Static preview of what an answer looks like, shown before anything is uploaded.
const EXAMPLE_TOPICS = [61, 34, 109, 36, 30];

function Example() {
  return (
    <figure className="example" aria-label="Example answer">
      <figcaption>Example</figcaption>
      <div className="example-strip" aria-hidden="true">
        {EXAMPLE_TOPICS.map((length, i) => (
          <span key={i} style={{ flexGrow: length, "--color": `var(--topic-${i + 1})` }} />
        ))}
      </div>
      <p className="qa-question">Why do two threads corrupt a shared counter?</p>
      <p className="qa-answer">
        Both threads read the same value before either writes it back,<span className="citation-marker">1</span>{" "}
        so one update is lost. A lock makes the read and the write a single step.<span className="citation-marker">2</span>
      </p>
      <ol className="footnotes">
        <li>
          <span className="citation-number">1</span>
          <span className="timestamp-chip">0:18–0:44</span>
          <span className="citation-title">Race conditions</span>
        </li>
        <li>
          <span className="citation-number">2</span>
          <span className="timestamp-chip">1:05–1:31</span>
          <span className="citation-title">Locks</span>
        </li>
      </ol>
    </figure>
  );
}

export default function App() {
  // "demo" is served from the backend's mock data and has no video file yet.
  // ready: false while an upload is still being transcribed.
  const [lecture, setLecture] = useState(DEMO);
  const videoRef = useRef(null);

  // The backend transcribes in the background: poll until it is done.
  // This lives here, not in Upload, because Upload remounts once a video is set.
  useEffect(() => {
    if (lecture.ready || lecture.error) return;
    let live = true;
    (async () => {
      try {
        for (;;) {
          const { status, error } = await fetchStatus(lecture.lecture_id);
          if (!live) return;
          if (status === "ready") {
            setLecture((l) => ({ ...l, ready: true }));
            return;
          }
          if (status === "failed") throw new Error(`Transcription failed: ${error || "unknown error"}`);
          await sleep(POLL_MS);
          if (!live) return;
        }
      } catch (err) {
        if (live) setLecture((l) => ({ ...l, error: err.message }));
      }
    })();
    return () => { live = false; };
  }, [lecture.lecture_id, lecture.ready, lecture.error]);

  const seek = (seconds) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = seconds;
    videoRef.current.play();
  };

  const hasVideo = Boolean(lecture.video_url);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor">
              <path d="M3 1.8v10.4a.6.6 0 0 0 .92.5l8.1-5.2a.6.6 0 0 0 0-1L3.92 1.3A.6.6 0 0 0 3 1.8Z" />
            </svg>
          </span>
          TimeStamp
        </div>
        {hasVideo && (
          <div className="recording-actions">
            <Upload onUploaded={setLecture} buttonLabel="Change recording" />
            <button type="button" className="btn btn-quiet" onClick={() => setLecture(DEMO)}>
              Clear
            </button>
          </div>
        )}
      </header>

      {hasVideo ? (
        <main className="workspace">
          <section className="lesson" aria-labelledby="recording-heading">
            <div className="lesson-head">
              <h1 id="recording-heading">{lecture.filename || "Lecture recording"}</h1>
              {!lecture.ready && !lecture.error && (
                <span className="pill" role="status">
                  <span className="spinner" aria-hidden="true" />
                  Transcribing
                </span>
              )}
            </div>
            <VideoPlayer
              ref={videoRef}
              src={API_URL + lecture.video_url}
              lectureId={lecture.lecture_id}
              ready={lecture.ready}
            />
          </section>

          <aside className="notes" aria-label="Lecture questions and answers">
            {lecture.ready ? (
              <SearchPanel key={lecture.lecture_id} lectureId={lecture.lecture_id} onSeek={seek} />
            ) : (
              <section className="panel">
                <h2 className="panel-head">Ask about this lecture</h2>
                <div className="panel-body">
                  {lecture.error ? (
                    <p className="error" role="alert">{lecture.error}</p>
                  ) : (
                    <p className="panel-hint">
                      Transcribing. This can take a few minutes, and you can watch while it works.
                    </p>
                  )}
                </div>
              </section>
            )}
          </aside>
        </main>
      ) : (
        <main className="landing">
          <div className="landing-copy">
            <h1>
              Find the <mark>minute</mark> it was explained.
            </h1>
            <p className="lede">
              Upload a lecture and ask about it. Every answer cites the moment it was said, and one
              click replays it.
            </p>
            <Upload onUploaded={setLecture} dropzone />
          </div>
          <Example />
        </main>
      )}
    </div>
  );
}
