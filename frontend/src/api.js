export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(API_URL + path, options);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export function upload(file) {
  const body = new FormData();
  body.append("file", file);
  return request("/upload", { method: "POST", body });
}

// { lecture_id, status: "processing" | "ready" | "failed", error }
export const status = (lectureId) => request(`/status?lecture_id=${lectureId}`);

export const search = (lectureId, q) =>
  request(`/search?lecture_id=${lectureId}&q=${encodeURIComponent(q)}`);

export const ask = (lectureId, question) =>
  request("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lecture_id: lectureId, question }),
  });

export const timeline = (lectureId) => request(`/timeline?lecture_id=${lectureId}`);

export const transcript = (lectureId) => request(`/transcript?lecture_id=${lectureId}`);
