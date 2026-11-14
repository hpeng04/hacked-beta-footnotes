import { useEffect, useRef, useState } from "react";
import { upload } from "../api.js";

export default function Upload({ onUploaded, buttonLabel }) {
  const fileInputRef = useRef(null);
  const [message, setMessage] = useState("");
  const [failed, setFailed] = useState(false);
  const run = useRef(0); // bumped by every new upload and on unmount, so stale uploads are ignored

  useEffect(() => () => { run.current += 1; }, []);

  const onChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const id = ++run.current;
    const current = () => run.current === id;
    setFailed(false);
    setMessage("Uploading…");
    try {
      const lecture = await upload(file);
      if (!current()) return;
      // The backend transcribes in the background; App polls until it is ready.
      // ready: false lets the video play while it is transcribed.
      onUploaded({ ...lecture, filename: file.name, ready: false });
      setMessage("Uploaded");
    } catch (err) {
      if (!current()) return;
      setFailed(true);
      setMessage(err.message);
    }
  };

  return (
    <p>
      {buttonLabel && (
        <button type="button" onClick={() => fileInputRef.current?.click()}>
          {buttonLabel}
        </button>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,audio/*"
        onChange={onChange}
        hidden={Boolean(buttonLabel)}
      />
      {message && <span className={failed ? "error" : undefined} role="status"> {message}</span>}
    </p>
  );
}
