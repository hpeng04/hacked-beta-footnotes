import { useEffect, useRef, useState } from "react";
import { API_URL, status as fetchStatus } from "./api.js";
import Upload from "./components/Upload.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";
import SearchPanel from "./components/SearchPanel.jsx";

const DEMO = { lecture_id: "demo", video_url: "", ready: true };
const POLL_MS = 2000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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
  return (
    <main className="timestamp-app">
      <header className="timestamp-header">
        <h1 className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <path d="M3 1.8v10.4a.6.6 0 0 0 .92.5l8.1-5.2a.6.6 0 0 0 0-1L3.92 1.3A.6.6 0 0 0 3 1.8Z" />
            </svg>
          </span>
          TimeStamp
        </h1>
        {lecture.video_url && (
          <div className="recording-actions">
            <Upload onUploaded={setLecture} buttonLabel="Change recording" />
            <button type="button" className="btn btn-quiet" onClick={() => setLecture(DEMO)}>
              Clear recording
            </button>
          </div>
        )}
      </header>

      <div className="timestamp-workspace">
        <section className="timestamp-lesson" aria-labelledby="recording-heading">
          <h2 id="recording-heading">
            {lecture.filename || "Lecture recording"}
          </h2>
          <div className="timestamp-player">
            {lecture.video_url ? (
              <VideoPlayer
                ref={videoRef}
                src={API_URL + lecture.video_url}
                lectureId={lecture.lecture_id}
                ready={lecture.ready}
              />
            ) : (
              <div className="timestamp-upload">
                <span className="upload-icon" aria-hidden="true">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 16V4m0 0L7 9m5-5 5 5M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2" />
                  </svg>
                </span>
                <h3>Upload a lecture recording</h3>
                <p>Video or audio. Transcribed on this machine.</p>
                <Upload onUploaded={setLecture} primary />
              </div>
            )}
          </div>
        </section>

        <aside className="timestamp-qa" aria-label="Lecture questions and answers">
          {lecture.video_url && !lecture.ready ? (
            <section className="panel">
              <h2>Ask about this lecture</h2>
              {lecture.error ? (
                <p className="error" role="alert">{lecture.error}</p>
              ) : (
                <p className="panel-status" role="status">
                  <span className="spinner" aria-hidden="true" />
                  Transcribing… this can take a few minutes
                </p>
              )}
            </section>
          ) : lecture.video_url ? (
            <SearchPanel
              key={lecture.lecture_id}
              lectureId={lecture.lecture_id}
              onSeek={seek}
            />
          ) : (
            <section className="panel">
              <h2>Ask about this lecture</h2>
              <p className="panel-hint">Upload a recording to start asking questions.</p>
            </section>
          )}
        </aside>
      </div>
    </main>
  );
}
