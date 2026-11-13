import { useState } from "react";
import { upload } from "../api.js";

export default function Upload({ onUploaded }) {
  const [status, setStatus] = useState("");

  const onChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setStatus("Uploading…");
    try {
      onUploaded(await upload(file));
      setStatus("Uploaded");
    } catch (err) {
      setStatus(err.message);
    }
  };

  return (
    <p>
      <input type="file" accept="video/*,audio/*" onChange={onChange} /> {status}
    </p>
  );
}
