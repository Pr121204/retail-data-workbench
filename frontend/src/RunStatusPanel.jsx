import { useEffect, useState } from "react";
import { cleanRun, getAnalytics, getRun, profileRun } from "./api.js";

// Order of the pipeline; a step's button is disabled until the run's status
// allows it (mirrors the backend's status checks).
const STEPS = [
  { key: "profile", label: "Profile", requires: "profiling", next: "profiled" },
  { key: "clean", label: "Clean", requires: "profiled", next: "cleaned" },
  { key: "analytics", label: "View Analytics", requires: "cleaned", next: "cleaned" },
];

export default function RunStatusPanel({ runId, runStatus, onRunStatusChange }) {
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [validation, setValidation] = useState(null); // from /clean
  const [analytics, setAnalytics] = useState(null); // from /analytics

  // Refresh status from the server whenever the run changes, so button
  // enablement follows the real Run.status rather than local guesses.
  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    (async () => {
      const res = await getRun(runId);
      if (!cancelled && res.ok) {
        onRunStatusChange(res.data.status);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  async function runStep(step) {
    setBusy(step.key);
    setError(null);
    let res;
    if (step.key === "profile") res = await profileRun(runId);
    else if (step.key === "clean") res = await cleanRun(runId);
    else res = await getAnalytics(runId);

    setBusy(null);
    if (!res.ok) {
      setError(res.data.detail);
      return;
    }
    if (step.key === "clean") {
      setValidation(res.data.results);
      setAnalytics(null);
    } else if (step.key === "analytics") {
      setAnalytics(res.data);
    }
    if (step.key !== "analytics") {
      // Re-check authoritative status after a state-changing step
      const statusRes = await getRun(runId);
      if (statusRes.ok) onRunStatusChange(statusRes.data.status);
    }
  }

  function stepEnabled(step) {
    if (busy) return false; // one step at a time
    const order = ["profiling", "profiled", "cleaned", "validated"];
    return order.indexOf(runStatus) >= order.indexOf(step.requires);
  }

  const validationRows = validation
    ? Object.entries(validation).map(([name, v]) => ({
        name,
        before: v.validation ? v.validation.row_count_before : null,
        after: v.validation ? v.validation.row_count_after : null,
        applied: v.validation ? v.validation.steps_applied : null,
        skipped: v.validation ? v.validation.steps_skipped : null,
        failed: v.validation ? v.validation.steps_failed : null,
      }))
    : [];

  const revenue = analytics && analytics.revenue_by_category && analytics.revenue_by_category.metrics;
  const stockouts =
    analytics && analytics.stockout_and_ageing && analytics.stockout_and_ageing.metrics
      ? analytics.stockout_and_ageing.metrics.stockout_items
      : null;

  return (
    <section className="panel">
      <h2>2. Pipeline</h2>
      <p>
        Run <code>{runId}</code> — status: <strong>{runStatus || "unknown"}</strong>
      </p>
      <div className="button-row">
        {STEPS.map((step) => (
          <button
            key={step.key}
            onClick={() => runStep(step)}
            disabled={!stepEnabled(step)}
          >
            {busy === step.key ? "Working…" : step.label}
          </button>
        ))}
      </div>
      {error && <p className="error">{error}</p>}

      {validationRows.length > 0 && (
        <>
          <h3>Cleaning results (before → after)</h3>
          <table>
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Rows before</th>
                <th>Rows after</th>
                <th>Steps applied</th>
                <th>Skipped</th>
                <th>Failed</th>
              </tr>
            </thead>
            <tbody>
              {validationRows.map((r) => (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  <td>{r.before}</td>
                  <td>{r.after}</td>
                  <td>{r.applied}</td>
                  <td>{r.skipped}</td>
                  <td>{r.failed}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {revenue && (
        <>
          <h3>Revenue by category</h3>
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th>Total revenue</th>
                <th>Orders</th>
                <th>AOV</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(revenue).map(([cat, m]) => (
                <tr key={cat}>
                  <td>{cat}</td>
                  <td>{m.total_revenue}</td>
                  <td>{m.order_count}</td>
                  <td>{m.aov}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {stockouts && stockouts.length > 0 && (
        <>
          <h3>Stockout items</h3>
          <ul>
            {stockouts.map((item, i) => (
              <li key={i}>
                {item.product_id} — {item.product_name || "unknown product"} (store {item.store_id})
              </li>
            ))}
          </ul>
        </>
      )}
      {stockouts && stockouts.length === 0 && <p>No stockout items.</p>}
    </section>
  );
}
