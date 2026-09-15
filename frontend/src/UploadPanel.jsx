import { useState } from "react";
import { uploadRuns } from "./api.js";

export default function UploadPanel({ onRunCreated }) {
  const [files, setFiles] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [uploaded, setUploaded] = useState(null);

  async function handleUpload() {
    if (!files || files.length === 0) {
      setError("Select at least one CSV file first.");
      return;
    }
    setUploading(true);
    setError(null);
    setUploaded(null);
    const res = await uploadRuns(files);
    setUploading(false);
    if (!res.ok) {
      setError(res.data.detail); // show the backend message plainly
      return;
    }
    setUploaded(res.data);
    onRunCreated(res.data.run_id);
  }

  return (
    <section className="panel">
      <h2>1. Upload datasets</h2>
      <input
        type="file"
        multiple
        accept=".csv"
        onChange={(e) => setFiles(Array.from(e.target.files || []))}
        disabled={uploading}
      />
      <button onClick={handleUpload} disabled={uploading}>
        {uploading ? "Uploading…" : "Upload"}
      </button>
      {error && <p className="error">{error}</p>}
      {uploaded && (
        <table>
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Rows</th>
              <th>Columns</th>
            </tr>
          </thead>
          <tbody>
            {uploaded.datasets.map((d) => (
              <tr key={d.name}>
                <td>{d.name}</td>
                <td>{d.row_count}</td>
                <td>{d.column_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
