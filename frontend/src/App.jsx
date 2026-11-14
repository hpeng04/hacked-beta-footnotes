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
        <h1>TimeStamp</h1>
        {lecture.video_url && (
          <div className="recording-actions">
            <Upload onUploaded={setLecture} buttonLabel="Change recording" />
            <button
              type="button"
              onClick={() => setLecture(DEMO)}
            >
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
                <h3>Upload a lecture recording</h3>
                <Upload onUploaded={setLecture} />
              </div>
            )}
          </div>
        </section>

        <aside className="timestamp-qa" aria-label="Lecture questions and answers">
          {lecture.video_url && !lecture.ready ? (
            <section className="panel">
              <h2>Ask about this lecture</h2>
              <p className={lecture.error ? "error" : undefined} role="status">
                {lecture.error || "Transcribing… this can take a few minutes"}
              </p>
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
              <p>Open a recording to start asking questions.</p>
            </section>
          )}
        </aside>
      </div>
    </main>
  );
}