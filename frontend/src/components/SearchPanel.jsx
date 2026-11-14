import { useState } from "react";
import { ask } from "../api.js";

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function SearchPanel({ lectureId, onSeek }) {
  const [question, setQuestion] = useState("");
  const [askedQuestion, setAskedQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;
    setError("");
    setLoading(true);
    try {
      setResult(await ask(lectureId, q));
      setAskedQuestion(q);
      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <h2>Ask about this lecture</h2>
      <div className="panel-body">
        {!result && !loading && !error && (
          <p className="panel-hint">
            Answers link to timestamps. Click one to replay that part of the lecture.
          </p>
        )}
        {result && (
          <>
            <div className="answer-note">
              <p className="answer-question">{askedQuestion}</p>
              <p>{result.answer}</p>
            </div>

            {result.sources.length > 0 && (
              <>
                <h3 className="source-heading">From the transcript</h3>
                <ul className="sources">
                  {result.sources.map((c) => (
                    <li key={c.start}>
                      <button
                        type="button"
                        className="timestamp-chip"
                        aria-label={`Replay at ${fmt(c.start)}`}
                        onClick={() => onSeek(c.start)}
                      >
                        <span aria-hidden="true">▶</span> {fmt(c.start)}
                      </button>
                      <span>{c.text}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        )}
        {loading && (
          <p className="panel-status" role="status">
            <span className="spinner" aria-hidden="true" /> Finding the answer…
          </p>
        )}
        {error && <p className="error" role="alert">{error}</p>}
      </div>
      <form className="ask-form" onSubmit={submit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question…"
          aria-label="Question about this recording"
        />
        <button className="btn btn-primary" disabled={loading || !question.trim()}>
          Ask
        </button>
      </form>
    </section>
  );
}
