import { useState } from "react";
import { ask } from "../api.js";

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function SearchPanel({ lectureId, onSeek }) {
  const [question, setQuestion] = useState("");
  const [askedQuestion, setAskedQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    try {
      setResult(await ask(lectureId, question));
      setAskedQuestion(question);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <section className="panel">
      <h2>Ask about this lecture</h2>
      <p>Click a timestamp in an answer to replay that part of the lecture.</p>
      <form onSubmit={submit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about this recording…"
        />
        <button>Ask</button>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <div className="answer-note">
            <p className="answer-question">{askedQuestion}</p>
            <p>{result.answer}</p>
          </div>

          {result.sources.length > 0 && (
            <h3 className="source-heading">From the transcript</h3>
          )}

          <ul>
            {result.sources.map((c) => (
              <li key={c.start}>
                <a href="#" onClick={(e) => { e.preventDefault(); onSeek(c.start); }}>Replay at {fmt(c.start)}</a> {c.text}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
