import { useEffect, useRef, useState } from "react";
import { upload } from "../api.js";

export default function Upload({ onUploaded, buttonLabel = "Choose a file", primary = false }) {
  const fileInputRef = useRef(null);
  const [message, setMessage] = useState("");
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const run = useRef(0); // bumped by every new upload and on unmount, so stale uploads are ignored

  useEffect(() => () => { run.current += 1; }, []);

  const onChange = async (e) => {
    const file = e.target.files[0];
    e.target.value = ""; // so picking the same file again still fires onChange
    if (!file) return;
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

  return (
    <div className="upload">
      <button
        type="button"
        className={primary ? "btn btn-primary" : "btn"}
        disabled={busy}
        onClick={() => fileInputRef.current?.click()}
      >
        {buttonLabel}
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,audio/*"
        onChange={onChange}
        hidden
      />
      {message && (
        <span className={failed ? "upload-status error" : "upload-status"} role="status">
          {message}
        </span>
      )}
    </div>
  );
}
