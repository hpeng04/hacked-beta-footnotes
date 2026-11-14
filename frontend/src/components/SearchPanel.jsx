import { useState } from "react";
import { ask } from "../api.js";
import { fmt } from "./TimelineBar.jsx";

function AnswerText({ answer, sources, sentences, citations, onSeek }) {
  if (citations != null) {
    return <>
      {sentences?.length ? sentences.map(s => s.text).join(" ") : answer}
      {citations.map(citation => <span key={citation.section_id}>
        {" "}<button type="button" className="timestamp-chip answer-timestamp"
          aria-label={`Replay ${citation.title} at ${fmt(citation.start)}`}
          onClick={() => onSeek(citation.start)}>
          {citation.title} [{fmt(citation.start)}]
        </button>
      </span>)}
    </>;
  }
  if (sentences?.length) {
    return sentences.map((sentence, index) => (
      <span key={index}>
        {index > 0 && " "}{sentence.text}
        {sentence.sources.map((source, sourceIndex) => (
          <span key={sourceIndex}>
            {" "}<button
              type="button"
              className="timestamp-chip answer-timestamp"
              aria-label={`Replay at ${fmt(source.start)}`}
              onClick={() => onSeek(source.start)}
            >
              [{fmt(source.start)}]
            </button>
          </span>
        ))}
      </span>
    ));
  }
  const byTimestamp = new Map(sources.map((source) => [`[${fmt(source.start)}]`, source]));
  return answer.split(/(\[(?:\d+:)?\d+:\d{2}\])/g).map((part, index) => {
    const source = byTimestamp.get(part);
    if (!source) return part;
    return (
      <button
        key={index}
        type="button"
        className="timestamp-chip answer-timestamp"
        aria-label={`Replay at ${fmt(source.start)}`}
        onClick={() => onSeek(source.start)}
      >
        {part}
      </button>
    );
  });
}

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
          <div className="answer-note">
            <p className="answer-question">{askedQuestion}</p>
            <p><AnswerText answer={result.answer} sources={result.sources} sentences={result.sentences} citations={result.citations} onSeek={onSeek} /></p>
          </div>
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
