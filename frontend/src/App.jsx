import { useRef, useState } from "react";
import { API_URL } from "./api.js";
import Upload from "./components/Upload.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";
import TimelineBar from "./components/TimelineBar.jsx";
import SearchPanel from "./components/SearchPanel.jsx";

export default function App() {
  // "demo" is served from the backend's mock data and has no video file yet.
  const [lecture, setLecture] = useState({ lecture_id: "demo", video_url: "" });
  const videoRef = useRef(null);

  const seek = (seconds) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = seconds;
    videoRef.current.play();
  };

  return (
    <main>
      <h1>LectureLens</h1>
      <Upload onUploaded={setLecture} />
      <div className="layout">
        <section>
          <VideoPlayer ref={videoRef} src={lecture.video_url && API_URL + lecture.video_url} />
          <TimelineBar lectureId={lecture.lecture_id} onSeek={seek} />
        </section>
        <SearchPanel key={lecture.lecture_id} lectureId={lecture.lecture_id} onSeek={seek} />
      </div>
    </main>
  );
}
