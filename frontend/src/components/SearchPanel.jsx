import { useEffect, useRef, useState } from "react";
import { ask } from "../api.js";
import { fmt } from "./TimelineBar.jsx";

function AnswerText({ answer, sources, sentences, citations, onSeek }) {
  if (citations != null) {
    const numbers = new Map(citations.map((citation, index) => [citation.id, index + 1]));
    return <>
      <p className="qa-answer">
        {sentences?.length ? sentences.map((sentence, index) => (
          <span key={index} className="cited">
            {index > 0 && " "}{sentence.text}
            {(sentence.citation_ids ?? []).filter(id => numbers.has(id)).map(id => {
              const citation = citations[numbers.get(id) - 1];
              return <button key={id} type="button" className="citation-marker"
                aria-label={`Source ${numbers.get(id)}: replay ${citation.title} at ${fmt(citation.start)}`}
                onClick={() => onSeek(citation.start)}>
                {numbers.get(id)}
              </button>;
            })}
          </span>
        )) : answer}
      </p>
      {citations.length > 0 && <ol className="footnotes">
        {citations.map((citation, index) => <li key={citation.id}>
          <span className="citation-number" aria-hidden="true">{index + 1}</span>
          <button type="button" className="timestamp-chip"
            aria-label={`Replay ${citation.title} from ${fmt(citation.start)} to ${fmt(citation.end)}`}
            onClick={() => onSeek(citation.start)}>
            {fmt(citation.start)}–{fmt(citation.end)}
          </button>
          <span className="citation-title">{citation.title}</span>
        </li>)}
      </ol>}
    </>;
  }
  const chip = (source, key, label = `[${fmt(source.start)}]`) => (
    <button
      key={key}
      type="button"
      className="timestamp-chip"
      aria-label={`Replay at ${fmt(source.start)}`}
      onClick={() => onSeek(source.start)}
    >
      {label}
    </button>
  );
  if (sentences?.length) {
    return (
      <p className="qa-answer">
        {sentences.map((sentence, index) => (
          <span key={index}>
            {index > 0 && " "}{sentence.text}
            {sentence.sources.map((source, sourceIndex) => (
              <span key={sourceIndex}> {chip(source)}</span>
            ))}
          </span>
        ))}
      </p>
    );
  }
  const byTimestamp = new Map(sources.map((source) => [`[${fmt(source.start)}]`, source]));
  return (
    <p className="qa-answer">
      {answer.split(/(\[(?:\d+:)?\d+:\d{2}\])/g).map((part, index) => {
        const source = byTimestamp.get(part);
        return source ? chip(source, index, part) : part;
      })}
    </p>
  );
}

export default function SearchPanel({ lectureId, onSeek }) {
  const [question, setQuestion] = useState("");
  const [thread, setThread] = useState([]); // [{ question, result }], oldest first
  const [pending, setPending] = useState(""); // the question being answered
  const [error, setError] = useState("");
  const body = useRef(null);
  const latest = useRef(null);

  // Bring the newest question to the top of the panel without scrolling the page.
  useEffect(() => {
    if (latest.current) body.current.scrollTop = latest.current.offsetTop;
  }, [thread.length, pending]);

  const submit = async (e) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || pending) return;
    setError("");
    setPending(q);
    setQuestion("");
    try {
      const result = await ask(lectureId, q);
      setThread((t) => [...t, { question: q, result }]);
    } catch (err) {
      setError(err.message);
      setQuestion(q); // so the question can be sent again
    } finally {
      setPending("");
    }
  };

  return (
    <section className="panel">
      <h2 className="panel-head">Ask about this lecture</h2>
      <div className="panel-body" ref={body}>
        {!thread.length && !pending && !error && (
          <p className="panel-hint">
            Answers cite the moment it was said. Select a timestamp to replay that part.
          </p>
        )}
        {thread.map((item, index) => (
          <article
            key={index}
            className="qa"
            ref={!pending && index === thread.length - 1 ? latest : undefined}
          >
            <h3 className="qa-question">{item.question}</h3>
            <AnswerText {...item.result} onSeek={onSeek} />
          </article>
        ))}
        {pending && (
          <article className="qa" ref={latest}>
            <h3 className="qa-question">{pending}</h3>
            <p className="panel-status" role="status">
              <span className="spinner" aria-hidden="true" /> Finding the answer…
            </p>
          </article>
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
        <button className="btn btn-primary" disabled={Boolean(pending) || !question.trim()}>
          Ask
        </button>
      </form>
    </section>
  );
}
