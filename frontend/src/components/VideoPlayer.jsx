import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { timeline } from "../api.js";
import TimelineBar, { colorOf, fmt, sectionAt } from "./TimelineBar.jsx";

const RATES = [0.75, 1, 1.25, 1.5, 2];

const Icon = ({ d }) => (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" aria-hidden="true">
    <path d={d} />
  </svg>
);
const ICONS = {
  play: "M8 5v14l11-7z",
  pause: "M6 5h4v14H6zm8 0h4v14h-4z",
  volume: "M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.5 4.5 0 0 0 2.5-4z",
  muted: "M3 9v6h4l5 5V4L7 9H3zm18.3-.9-1.4-1.4L17 9.6l-2.9-2.9-1.4 1.4 2.9 2.9-2.9 2.9 1.4 1.4 2.9-2.9 2.9 2.9 1.4-1.4-2.9-2.9z",
  fullscreen: "M5 5h6v2H7v4H5zm8 0h6v6h-2V7h-4zM5 13h2v4h4v2H5zm12 0h2v6h-6v-2h4z",
};

// Custom player. The forwarded ref is the <video> element, so callers can seek and play it.
// ready: the lecture's timeline exists, so the seek bar can show its topic sections.
const VideoPlayer = forwardRef(function VideoPlayer({ src, lectureId, ready }, ref) {
  const video = useRef(null);
  const box = useRef(null);
  useImperativeHandle(ref, () => video.current);

  const [sections, setSections] = useState([]);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);
  const [rate, setRate] = useState(1);

  useEffect(() => {
    setSections([]);
    if (!ready) return;
    let live = true;
    timeline(lectureId).then((s) => live && setSections(s)).catch(() => {});
    return () => { live = false; };
  }, [lectureId, ready]);

  const toggle = () => (video.current.paused ? video.current.play() : video.current.pause());
  const seek = (t) => {
    video.current.currentTime = t;
    setTime(t);
  };
  const toggleFullscreen = () =>
    document.fullscreenElement ? document.exitFullscreen() : box.current.requestFullscreen();

  const onKeyDown = (e) => {
    if (e.target.closest("button, select, input")) return;
    if (e.key === " " || e.key === "k") toggle();
    else if (e.key === "ArrowLeft") seek(Math.max(0, time - 5));
    else if (e.key === "ArrowRight") seek(Math.min(duration, time + 5));
    else if (e.key === "f") toggleFullscreen();
    else if (e.key === "m") video.current.muted = !video.current.muted;
    else return;
    e.preventDefault();
  };

  // The timeline can be a little shorter or longer than the media; scale to whichever is longer.
  const total = Math.max(Number.isFinite(duration) ? duration : 0, ...sections.map((s) => s.end));
  const section = sectionAt(sections, time);
  const silent = muted || volume === 0;

  return (
    <>
    <div ref={box} className="player" onKeyDown={onKeyDown}>
      <video
        ref={video}
        src={src || undefined}
        className="video"
        tabIndex={0}
        onClick={toggle}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={(e) => setTime(e.target.currentTime)}
        onDurationChange={(e) => setDuration(e.target.duration)}
        onVolumeChange={(e) => {
          setVolume(e.target.volume);
          setMuted(e.target.muted);
        }}
        onRateChange={(e) => setRate(e.target.playbackRate)}
      />
      <div className="player-controls">
        <TimelineBar sections={sections} duration={total} currentTime={time} onSeek={seek} />
        <div className="player-row">
          <button type="button" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
            <Icon d={playing ? ICONS.pause : ICONS.play} />
          </button>
          <span className="player-time">
            {fmt(time)} / {fmt(total)}
          </span>
          {section && <span className="player-section">{section.title}</span>}
          <span className="player-spacer" />
          <button
            type="button"
            onClick={() => { video.current.muted = !video.current.muted; }}
            aria-label={silent ? "Unmute" : "Mute"}
          >
            <Icon d={silent ? ICONS.muted : ICONS.volume} />
          </button>
          <input
            className="player-volume"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={silent ? 0 : volume}
            aria-label="Volume"
            onChange={(e) => {
              video.current.volume = Number(e.target.value);
              video.current.muted = false;
            }}
          />
          <select
            value={rate}
            aria-label="Playback speed"
            onChange={(e) => { video.current.playbackRate = Number(e.target.value); }}
          >
            {RATES.map((r) => (
              <option key={r} value={r}>{r}×</option>
            ))}
          </select>
          <button type="button" onClick={toggleFullscreen} aria-label="Fullscreen">
            <Icon d={ICONS.fullscreen} />
          </button>
        </div>
      </div>
    </div>
    <section className="topics" aria-labelledby="topics-heading">
      <h2 id="topics-heading">Topics</h2>
      {sections.length ? (
        <ol>
          {sections.map((s, i) => (
            <li key={s.start}>
              <button
                type="button"
                className={s === section ? "topic is-current" : "topic"}
                aria-current={s === section ? "true" : undefined}
                style={{ "--color": colorOf(i) }}
                onClick={() => {
                  seek(s.start);
                  video.current.play();
                }}
              >
                <span className="topic-time">{fmt(s.start)}</span>
                <span className="topic-title">{s.title}</span>
              </button>
            </li>
          ))}
        </ol>
      ) : (
        <p className="topics-empty">
          {ready ? "No topics for this recording yet." : "Topics appear here once the transcript is ready."}
        </p>
      )}
    </section>
    </>
  );
});

export default VideoPlayer;
