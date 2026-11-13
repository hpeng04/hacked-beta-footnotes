import { useState } from "react";
import { ask } from "../api.js";

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function SearchPanel({ lectureId, onSeek }) {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    try {
      setResult(await ask(lectureId, question));
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <section className="panel">
      <form onSubmit={submit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="When did he explain deadlock?"
        />
        <button>Ask</button>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <p>{result.answer}</p>
          <ul>
            {result.sources.map((c) => (
              <li key={c.start}>
                <a href="#" onClick={(e) => { e.preventDefault(); onSeek(c.start); }}>{fmt(c.start)}</a> {c.text}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
