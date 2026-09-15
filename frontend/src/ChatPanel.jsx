import { useState } from "react";
import { createChatSession, createChatTurn } from "./api.js";

function StatusBadge({ status }) {
  // green for ok, yellow for clarification_needed, red for refused/plan_rejected
  const tone =
    status === "ok"
      ? "badge-green"
      : status === "clarification_needed"
        ? "badge-yellow"
        : "badge-red";
  return <span className={`badge ${tone}`}>{status}</span>;
}

function Turn({ turn }) {
  const answer =
    turn.answer_text ||
    (turn.status === "clarification_needed"
      ? "Could you clarify the question?"
      : null);

  return (
    <div className={`turn turn-${turn.status === "ok" ? "ok" : "warn"}`}>
      <div className="turn-user">
        <span className="bubble bubble-user">{turn.question}</span>
        <StatusBadge status={turn.status} />
      </div>
      {answer && (
        <div className="turn-assistant">
          <span className="bubble bubble-assistant">{answer}</span>
        </div>
      )}
      <details className="evidence">
        <summary>Evidence</summary>
        <div>
          <strong>plan:</strong>
          <pre>{JSON.stringify(turn.plan, null, 2)}</pre>
          <strong>evidence:</strong>
          <pre>{JSON.stringify(turn.evidence, null, 2)}</pre>
        </div>
      </details>
    </div>
  );
}

export default function ChatPanel({ runId, runStatus }) {
  const [sessionId, setSessionId] = useState(null);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState([]);
  const [waiting, setWaiting] = useState(false);
  const [error, setError] = useState(null);

  const ready = runStatus === "cleaned" || runStatus === "validated";

  async function handleStartSession() {
    setWaiting(true);
    setError(null);
    const res = await createChatSession(runId);
    setWaiting(false);
    if (!res.ok) {
      setError(res.data.detail);
      return;
    }
    setSessionId(res.data.session_id);
    setTurns([]);
  }

  async function handleAsk() {
    const q = question.trim();
    if (!q || !sessionId) return;
    setWaiting(true);
    setError(null);
    const res = await createChatTurn(runId, sessionId, q);
    setWaiting(false);
    if (!res.ok) {
      setError(res.data.detail); // surfaced, never swallowed
      return;
    }
    setTurns((prev) => [...prev, res.data]);
    setQuestion("");
  }

  if (!ready) {
    return (
      <section className="panel">
        <h2>3. Chat</h2>
        <p>Available once the run has been cleaned.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>3. Chat</h2>
      {!sessionId ? (
        <button onClick={handleStartSession} disabled={waiting}>
          {waiting ? "Starting…" : "Start Chat Session"}
        </button>
      ) : (
        <p>
          Session <code>{sessionId}</code>
        </p>
      )}
      {error && <p className="error">{error}</p>}

      {sessionId && (
        <>
          <div className="turn-list">
            {turns.map((turn, i) => (
              <Turn key={i} turn={turn} />
            ))}
          </div>
          <div className="ask-row">
            <input
              type="text"
              value={question}
              placeholder='e.g. "Show me orders in the West region"'
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !waiting) handleAsk();
              }}
              disabled={waiting}
            />
            <button onClick={handleAsk} disabled={waiting || !question.trim()}>
              {waiting ? "Asking…" : "Ask"}
            </button>
          </div>
        </>
      )}
    </section>
  );
}
