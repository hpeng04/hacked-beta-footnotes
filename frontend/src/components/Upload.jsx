import { useEffect, useRef, useState } from "react";
import { upload } from "../api.js";

// dropzone: the large first-run target that also accepts a dragged file.
export default function Upload({ onUploaded, buttonLabel = "Choose a file", dropzone = false }) {
  const fileInputRef = useRef(null);
  const [message, setMessage] = useState("");
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const run = useRef(0); // bumped by every new upload and on unmount, so stale uploads are ignored

  useEffect(() => () => { run.current += 1; }, []);

  const start = async (file) => {
    if (!file) return;
    if (!/^(video|audio)\//.test(file.type)) {
      setFailed(true);
      setMessage("That file isn't video or audio.");
      return;
    }
    const id = ++run.current;
    const current = () => run.current === id;
    setFailed(false);
    setBusy(true);
    setMessage("Uploading…");
    try {
      const lecture = await upload(file);
      if (!current()) return;
      // The backend transcribes in the background; App polls until it is ready.
      // ready: false lets the video play while it is transcribed.
      onUploaded({ ...lecture, filename: file.name, ready: false });
      setMessage("");
    } catch (err) {
      if (!current()) return;
      setFailed(true);
      setMessage(err.message);
    } finally {
      if (current()) setBusy(false);
    }
  };

  const onChange = (e) => {
    const file = e.target.files[0];
    e.target.value = ""; // so picking the same file again still fires onChange
    start(file);
  };

  const picker = (
    <>
      <button
        type="button"
        className={dropzone ? "btn btn-primary" : "btn"}
        disabled={busy}
        onClick={() => fileInputRef.current?.click()}
      >
        {buttonLabel}
      </button>
      <input ref={fileInputRef} type="file" accept="video/*,audio/*" onChange={onChange} hidden />
      {message && (
        <span className={failed ? "upload-status error" : "upload-status"} role="status">
          {busy && <span className="spinner" aria-hidden="true" />}
          {message}
        </span>
      )}
    </>
  );

  if (!dropzone) return <div className="upload">{picker}</div>;

  return (
    <div
      className={over ? "dropzone is-over" : "dropzone"}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!busy) start(e.dataTransfer.files[0]);
      }}
    >
      <p className="dropzone-title">Drop a recording here</p>
      <p className="dropzone-hint">Video or audio</p>
      {picker}
    </div>
  );
}
