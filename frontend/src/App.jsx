import { useRef, useState } from "react";
import { API_URL } from "./api.js";
import Upload from "./components/Upload.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";
import SearchPanel from "./components/SearchPanel.jsx";

export default function App() {
  const [lecture, setLecture] = useState({
    lecture_id: "demo",
    video_url: "",
  });
  const videoRef = useRef(null);

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
              onClick={() => setLecture({ lecture_id: "demo", video_url: "" })}
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
          {lecture.video_url ? (
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