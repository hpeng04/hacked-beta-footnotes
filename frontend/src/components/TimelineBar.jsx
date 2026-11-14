import { useRef, useState } from "react";

const TOPIC_COLORS = 6; // --topic-1 … --topic-6 in styles.css

// The color of the i-th topic section, shared by the seek bar and the topic list.
export const colorOf = (i) => `var(--topic-${(i % TOPIC_COLORS) + 1})`;

export const fmt = (s) => {
  const t = Math.max(0, Math.floor(s || 0));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const ss = String(t % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
};

export const sectionAt = (sections, t) => sections.find((s) => t >= s.start && t < s.end);

// The player's seek bar: one colored segment per topic section.
// Click or drag to seek; hover shows the time and topic under the cursor.
export default function TimelineBar({ sections, duration, currentTime, onSeek }) {
  const track = useRef(null);
  const dragging = useRef(false);
  const [hover, setHover] = useState(null); // { ratio, time }

  // Before the timeline is ready the bar is a single neutral segment.
  const segments = sections.length ? sections : [{ title: "", start: 0, end: duration }];
  const pct = (t) => (duration ? Math.min(100, Math.max(0, (t / duration) * 100)) : 0);

  const at = (e) => {
    const box = track.current.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
    return { ratio, time: ratio * duration };
  };

  const onPointerDown = (e) => {
    if (!duration) return;
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    onSeek(at(e).time);
  };
  const onPointerMove = (e) => {
    if (!duration) return;
    const pos = at(e);
    setHover(pos);
    if (dragging.current) onSeek(pos.time);
  };
  const onPointerUp = () => { dragging.current = false; };

  const onKeyDown = (e) => {
    const step = { ArrowLeft: -5, ArrowRight: 5 }[e.key];
    if (!step) return;
    e.preventDefault();
    e.stopPropagation();
    onSeek(Math.min(duration, Math.max(0, currentTime + step)));
  };

  const current = sectionAt(sections, currentTime);
  const hovered = hover && sectionAt(sections, hover.time);

  return (
    <div
      ref={track}
      className="timeline"
      role="slider"
      tabIndex={0}
      aria-label="Seek"
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(currentTime)}
      aria-valuetext={`${fmt(currentTime)} of ${fmt(duration)}${current ? `, ${current.title}` : ""}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onPointerLeave={() => setHover(null)}
      onKeyDown={onKeyDown}
    >
      <div className="timeline-track">
        {segments.map((s, i) => {
          const color = sections.length ? colorOf(i) : "var(--stage-idle)";
          const played = s.end > s.start ? (currentTime - s.start) / (s.end - s.start) : 0;
          return (
            <span
              key={s.start}
              className="timeline-segment"
              style={{ left: `${pct(s.start)}%`, width: `${pct(s.end) - pct(s.start)}%`, "--color": color }}
            >
              <span style={{ width: `${Math.min(1, Math.max(0, played)) * 100}%` }} />
            </span>
          );
        })}
        <span className="timeline-thumb" style={{ left: `${pct(currentTime)}%` }} />
      </div>
      {hover && (
        <span className="timeline-tip" style={{ left: `${hover.ratio * 100}%` }}>
          {hovered && <strong>{hovered.title}</strong>}
          {fmt(hover.time)}
        </span>
      )}
    </div>
  );
}
