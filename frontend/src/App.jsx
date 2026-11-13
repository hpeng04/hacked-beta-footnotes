import { useRef, useState } from "react";
import { API_URL } from "./api.js";
import Upload from "./components/Upload.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";
import TimelineBar from "./components/TimelineBar.jsx";
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
    <main>
      <header className="app-header">
        <h1>TimeStamp</h1>
        {lecture.video_url && <Upload onUploaded={setLecture} />}
      </header>

      <div className="layout">
        {lecture.video_url ? (
          <SearchPanel
            key={lecture.lecture_id}
            lectureId={lecture.lecture_id}
            onSeek={seek}
          />
        ) : (
          <section className="panel">
            <h2>Ask about this lecture</h2>
            <p>Upload a recording to get started.</p>
          </section>
        )}

        <section className="lecture-pane">
          {lecture.video_url ? (
            <VideoPlayer ref={videoRef} src={API_URL + lecture.video_url} />
          ) : (
            <div className="video-empty">
              <strong>Upload a lecture recording</strong>
              <span>Choose a video to open it in TimeStamp.</span>
              <Upload onUploaded={setLecture} />
            </div>
          )}
        </section>
      </div>
    </main>
  );
}