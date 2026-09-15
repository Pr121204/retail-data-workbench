import { useState } from "react";
import ChatPanel from "./ChatPanel.jsx";
import RunStatusPanel from "./RunStatusPanel.jsx";
import UploadPanel from "./UploadPanel.jsx";

export default function App() {
  const [runId, setRunId] = useState(null);
  const [runStatus, setRunStatus] = useState(null);

  return (
    <main className="app">
      <h1>Retail Data Workbench</h1>
      <UploadPanel onRunCreated={setRunId} />
      {runId && (
        <RunStatusPanel
          runId={runId}
          runStatus={runStatus}
          onRunStatusChange={setRunStatus}
        />
      )}
      {runId && (runStatus === "cleaned" || runStatus === "validated") && (
        <ChatPanel runId={runId} runStatus={runStatus} />
      )}
    </main>
  );
}
