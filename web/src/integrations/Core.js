import { API_BASE } from "@/config";

export async function UploadFile({ file }) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/files`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}
