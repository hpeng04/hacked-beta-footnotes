import { useEffect, useRef, useState } from "react";
import { status as fetchStatus, upload } from "../api.js";

const POLL_MS = 2000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export default function Upload({ onUploaded }) {
  const [message, setMessage] = useState("");
  const [failed, setFailed] = useState(false);
  const run = useRef(0); // bumped by every new upload and on unmount, so stale polls stop

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
      onUploaded({ ...lecture, ready: false }); // the video can play while it is transcribed
      setMessage("Transcribing… this can take a few minutes");
      // The backend transcribes in the background: poll until it is done.
      for (;;) {
        const { status, error } = await fetchStatus(lecture.lecture_id);
        if (!current()) return;
        if (status === "ready") break;
        if (status === "failed") throw new Error(`Transcription failed: ${error || "unknown error"}`);
        await sleep(POLL_MS);
        if (!current()) return;
      }
      onUploaded({ ...lecture, ready: true });
      setMessage("Ready");
    } catch (err) {
      if (!current()) return;
      setFailed(true);
      setMessage(err.message);
    }
  };

  return (
    <p>
      <input type="file" accept="video/*,audio/*" onChange={onChange} />{" "}
      <span className={failed ? "error" : undefined} role="status">{message}</span>
    </p>
  );
}
