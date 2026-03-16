const form = document.getElementById("hand-upload-form");
const fileInput = document.getElementById("hand-history-file");
const siteSelect = document.getElementById("hand-history-site");
const heroInput = document.getElementById("hero-name");
const submitButton = document.getElementById("analyze-submit");
const reviewStatus = document.getElementById("review-status");
const reviewResults = document.getElementById("review-results");
const summaryGrid = document.getElementById("summary-grid");
const warningBlock = document.getElementById("warning-block");
const warningList = document.getElementById("warning-list");
const decisionList = document.getElementById("decision-list");
const opponentList = document.getElementById("opponent-list");
const modelSource = document.getElementById("model-source");
const MAX_POLL_WAIT_MS = 15 * 60 * 1000;

if (form && fileInput && siteSelect && heroInput) {
  form.addEventListener("submit", async event => {
    event.preventDefault();

    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      renderStatus("Choose a `.txt` hand-history export to start the review.");
      reviewResults.classList.add("hidden");
      return;
    }

    const payload = {
      handHistoryText: await file.text(),
      site: siteSelect.value || "auto",
      heroName: heroInput.value.trim() || null
    };

    setSubmitting(true);
    renderStatus(`Submitting ${file.name} for local review...`);
    reviewResults.classList.add("hidden");

    try {
      const response = await fetch("/api/analyze-hand-history", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });
      const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));

      if (!response.ok) {
        renderStatus(body.error || `Request failed with status ${response.status}.`);
        return;
      }

      if (body.jobId) {
        const statusUrl = body.statusUrl || response.headers.get("Location");
        if (!statusUrl) {
          renderStatus("Server accepted the upload but did not return a job status URL.");
          return;
        }

        renderStatus(jobStatusMessage(file.name, body.status));
        const result = await pollAnalysisJob(file.name, statusUrl, body.pollAfterMs);
        renderResults(file.name, result);
        return;
      }

      renderResults(file.name, body);
    } catch (error) {
      renderStatus(`Request failed: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally {
      setSubmitting(false);
    }
  });
}

function setSubmitting(isSubmitting) {
  submitButton.disabled = isSubmitting;
  submitButton.textContent = isSubmitting ? "Queueing Review..." : "Queue Review";
}

function renderStatus(message) {
  reviewStatus.innerHTML = `
    <p class="card-kicker">Status</p>
    <h3>${escapeHtml(message)}</h3>
    <p class="section-note">
      SICFUN accepts the upload quickly, analyzes it in a background job, and fills this board with hand
      counts, EV gaps, warnings, and opponent notes when the review is ready.
    </p>
  `;
}

async function pollAnalysisJob(fileName, statusUrl, initialPollAfterMs) {
  let pollAfterMs = normalizePollAfterMs(initialPollAfterMs);
  const deadline = Date.now() + MAX_POLL_WAIT_MS;

  for (;;) {
    if (Date.now() >= deadline) {
      throw new Error("Analysis timed out while waiting for the review job to finish.");
    }

    await sleep(pollAfterMs);

    const response = await fetch(statusUrl, {
      headers: {
        "Accept": "application/json"
      }
    });
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error("Review job expired or was purged before the page could load the result. Submit the upload again.");
      }
      throw new Error(body.error || `Status request failed with status ${response.status}.`);
    }

    renderStatus(jobStatusMessage(fileName, body.status));

    if (body.status === "completed") {
      return body.result || {};
    }

    if (body.status === "failed") {
      throw new Error(body.error || "Analysis failed.");
    }

    if (body.status !== "queued" && body.status !== "running") {
      throw new Error(`Unexpected analysis job status: ${body.status || "unknown"}`);
    }

    pollAfterMs = normalizePollAfterMs(body.pollAfterMs);
  }
}

function normalizePollAfterMs(value) {
  const parsed = Number(value || 0);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 750;
  }
  return Math.max(250, Math.min(5000, Math.round(parsed)));
}

function sleep(ms) {
  return new Promise(resolve => window.setTimeout(resolve, ms));
}

function jobStatusMessage(fileName, status) {
  switch (status) {
    case "queued":
      return `Queued ${fileName} for local review...`;
    case "running":
      return `Reviewing ${fileName} in the background...`;
    case "completed":
      return `Completed local review for ${fileName}.`;
    case "failed":
      return `Local review failed for ${fileName}.`;
    default:
      return `Processing ${fileName}...`;
  }
}

function renderResults(fileName, data) {
  renderStatus(
    `Review ready: imported ${formatInteger(data.handsImported)} hand${data.handsImported === 1 ? "" : "s"} from ${fileName}.`
  );

  summaryGrid.innerHTML = [
    summaryCard("Site", data.site, data.heroName ? `Hero: ${escapeHtml(data.heroName)}` : "Hero shown only when one clear name is resolved"),
    summaryCard("Hands", formatInteger(data.handsAnalyzed), `${formatInteger(data.handsSkipped)} skipped`),
    summaryCard("Decisions", formatInteger(data.decisionsAnalyzed), `${formatInteger(data.mistakes)} mistakes flagged`),
    summaryCard("EV Lost", formatSigned(-Math.abs(Number(data.totalEvLost || 0))), "Aggregate avoidable EV gap"),
    summaryCard("Biggest Gap", formatNumber(data.biggestMistakeEv), "Worst single decision"),
    summaryCard("Model", escapeHtml(data.modelSource || "-"), "Loaded for this review")
  ].join("");

  modelSource.textContent = `Model: ${data.modelSource || "-"}`;

  const warnings = Array.isArray(data.warnings) ? data.warnings : [];
  if (warnings.length > 0) {
    warningBlock.classList.remove("hidden");
    warningList.innerHTML = warnings.map(item => `<li>${escapeHtml(item)}</li>`).join("");
  } else {
    warningBlock.classList.add("hidden");
    warningList.innerHTML = "";
  }

  const decisions = Array.isArray(data.decisions) ? data.decisions : [];
  decisionList.innerHTML =
    decisions.length > 0
      ? decisions.map(renderDecisionCard).join("")
      : emptyCard("No analyzable hero decisions were returned for this upload.");

  const opponents = Array.isArray(data.opponents) ? data.opponents : [];
  opponentList.innerHTML =
    opponents.length > 0
      ? opponents.map(renderOpponentCard).join("")
      : emptyCard("No opponent notes were returned from this upload.");

  reviewResults.classList.remove("hidden");
}

function summaryCard(label, value, note) {
  return `
    <article class="summary-card">
      <p class="summary-label">${escapeHtml(label)}</p>
      <p class="summary-value">${escapeHtml(value)}</p>
      <p class="summary-note">${note}</p>
    </article>
  `;
}

function renderDecisionCard(decision) {
  return `
    <article class="decision-card">
      <div class="decision-head">
        <h3 class="decision-title">${escapeHtml(decision.handId)} &middot; ${escapeHtml(decision.street)}</h3>
        <p class="card-meta">Hero ${escapeHtml(decision.heroCards || "cards hidden")}</p>
      </div>
      <p class="decision-meta">
        Actual: ${escapeHtml(decision.actualAction)}<br>
        Recommended: ${escapeHtml(decision.recommendedAction)}
      </p>
      <div class="decision-grid">
        ${decisionRow("Actual EV", formatSigned(decision.actualEv))}
        ${decisionRow("Recommended EV", formatSigned(decision.recommendedEv))}
        ${decisionRow("EV Diff", formatSigned(decision.evDifference))}
        ${decisionRow("Hero Equity", formatPercent(decision.heroEquityMean))}
      </div>
    </article>
  `;
}

function renderOpponentCard(opponent) {
  const hints = Array.isArray(opponent.hints) && opponent.hints.length > 0
    ? opponent.hints.map(hint => escapeHtml(hint)).join("<br>")
    : "No exploit hints returned.";

  return `
    <article class="opponent-card">
      <div class="decision-head">
        <h3 class="decision-title">${escapeHtml(opponent.playerName)}</h3>
        <p class="card-meta">${formatInteger(opponent.handsObserved)} hand${opponent.handsObserved === 1 ? "" : "s"}</p>
      </div>
      <p class="opponent-meta">
        Archetype: ${escapeHtml(opponent.archetype)}<br>
        ${hints}
      </p>
    </article>
  `;
}

function decisionRow(label, value) {
  return `
    <div class="decision-row">
      <strong>${escapeHtml(label)}</strong>
      <span>${value}</span>
    </div>
  `;
}

function emptyCard(message) {
  return `
    <article class="decision-card">
      <p class="decision-meta">${escapeHtml(message)}</p>
    </article>
  `;
}

function formatInteger(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function formatNumber(value) {
  return Number(value || 0).toFixed(2);
}

function formatSigned(value) {
  const number = Number(value || 0);
  return `${number > 0 ? "+" : ""}${number.toFixed(2)}`;
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}
