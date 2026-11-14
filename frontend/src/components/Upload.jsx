import { useRef, useState } from "react";
import { upload } from "../api.js";

export default function Upload({ onUploaded, buttonLabel }) {
  const fileInputRef = useRef(null);
  const [status, setStatus] = useState("");

  const onChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setStatus("Uploading…");
    try {
      const uploadedLecture = await upload(file);
      onUploaded({ ...uploadedLecture, filename: file.name });
      onUploaded({ ...uploadedLecture, filename: file.name });
      setStatus("Uploaded");
    } catch (err) {
      setStatus(err.message);
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
      {status && <span role="status"> {status}</span>}
    </p>
  );
}
