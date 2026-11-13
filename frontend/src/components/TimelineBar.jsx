import { useEffect, useState } from "react";
import { timeline } from "../api.js";

const COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759", "#76b7b2"];

export default function TimelineBar({ lectureId, onSeek }) {
  const [sections, setSections] = useState([]);

  useEffect(() => {
    timeline(lectureId).then(setSections).catch(() => setSections([]));
  }, [lectureId]);

  const total = Math.max(0, ...sections.map((s) => s.end));

  return (
    <div className="timeline">
      {sections.map((s, i) => (
        <button
          key={s.start}
          title={s.title}
          onClick={() => onSeek(s.start)}
          style={{ width: `${((s.end - s.start) / total) * 100}%`, background: COLORS[i % COLORS.length] }}
        >
          {s.title}
        </button>
      ))}
    </div>
  );
}
