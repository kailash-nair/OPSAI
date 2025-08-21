import { API_BASE } from "@/config";

export const Meeting = {
  async create(payload) {
    const res = await fetch(`${API_BASE}/api/meetings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`Create meeting failed: ${res.status}`);
    return res.json();
  },

  async list(sort = "-created_date", limit = 100) {
    const url = new URL(`${API_BASE}/api/meetings`);
    url.searchParams.set("sort", sort);
    url.searchParams.set("limit", String(limit));
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`List meetings failed: ${res.status}`);
    return res.json();
  },

  async get(id) {
    const res = await fetch(`${API_BASE}/api/meetings/${id}`);
    if (!res.ok) throw new Error(`Get meeting failed: ${res.status}`);
    return res.json();
  },

  async restart(id) {
    const res = await fetch(`${API_BASE}/api/meetings/${id}/restart`, { method: "POST" });
    if (!res.ok) throw new Error(`Restart failed: ${res.status}`);
  },

  // ---------- NEW ----------
  async getSummary(id) {
    const res = await fetch(`${API_BASE}/api/meetings/${id}/summary`);
    if (!res.ok) throw new Error(`Summary fetch failed: ${res.status}`);
    return res.json(); // { markdown }
  },

  async getTranscript(id) {
    const res = await fetch(`${API_BASE}/api/meetings/${id}/transcript`);
    if (!res.ok) throw new Error(`Transcript fetch failed: ${res.status}`);
    return res.json(); // { text }
  },
};
