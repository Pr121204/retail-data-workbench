// Thin fetch wrapper around the FastAPI backend. Every helper resolves to
// {ok, status, data} so callers can surface backend error messages in the UI
// instead of swallowing them.

export const API_BASE = "http://localhost:8000";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (networkError) {
    // fetch only throws on network-level failures (server down, CORS blocked)
    return {
      ok: false,
      status: 0,
      data: { detail: `Network error: is the backend running on ${API_BASE}? (${networkError.message})` },
    };
  }

  let data = null;
  const text = await response.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }

  if (!response.ok) {
    // FastAPI errors look like {detail: "..."} or {detail: [{msg: ...}]}
    let message = `HTTP ${response.status}`;
    if (data && data.detail) {
      if (typeof data.detail === "string") {
        message = data.detail;
      } else if (Array.isArray(data.detail)) {
        message = data.detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
      }
    }
    return { ok: false, status: response.status, data: { detail: message } };
  }

  return { ok: true, status: response.status, data };
}

export async function uploadRuns(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file); // field name "files" matches the backend
  }
  return request("/runs/upload", { method: "POST", body: formData });
}

export async function getRun(runId) {
  return request(`/runs/${runId}`);
}

export async function profileRun(runId) {
  return request(`/runs/${runId}/profile`, { method: "POST" });
}

export async function cleanRun(runId) {
  return request(`/runs/${runId}/clean`, { method: "POST" });
}

export async function getAnalytics(runId) {
  return request(`/runs/${runId}/analytics`);
}

export async function createChatSession(runId) {
  return request(`/runs/${runId}/chat/sessions`, { method: "POST" });
}

export async function createChatTurn(runId, sessionId, question) {
  return request(`/runs/${runId}/chat/sessions/${sessionId}/turns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}
