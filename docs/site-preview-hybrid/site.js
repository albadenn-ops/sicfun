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

const authStatePanel = document.getElementById("auth-state");
const authForm = document.getElementById("auth-form");
const authEmail = document.getElementById("auth-email");
const authPassword = document.getElementById("auth-password");
const authDisplayName = document.getElementById("auth-display-name");
const authLoginButton = document.getElementById("auth-login");
const authRegisterButton = document.getElementById("auth-register");
const googleLoginLink = document.getElementById("google-login-link");
const profileCard = document.getElementById("profile-card");
const profileSummary = document.getElementById("profile-summary");
const profileForm = document.getElementById("profile-form");
const profileDisplayName = document.getElementById("profile-display-name");
const profileHeroName = document.getElementById("profile-hero-name");
const profilePreferredSite = document.getElementById("profile-preferred-site");
const profileTimeZone = document.getElementById("profile-time-zone");
const authLogoutButton = document.getElementById("auth-logout");

const MAX_POLL_WAIT_MS = 15 * 60 * 1000;

let authState = normalizeAuthState({});

void boot();

if (form && fileInput && siteSelect && heroInput) {
  form.addEventListener("submit", async event => {
    event.preventDefault();

    if (requiresPlatformSignIn() && !authState.authenticated) {
      renderStatus("Sign in to queue a review job for this deployment.");
      reviewResults.classList.add("hidden");
      return;
    }

    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      renderStatus("Choose a `.txt` hand-history export to start the review.");
      reviewResults.classList.add("hidden");
      return;
    }

    const payload = {
      handHistoryText: await file.text(),
      site: resolvedUploadSite(),
      heroName: resolvedHeroName()
    };

    setSubmitting(true);
    renderStatus(`Submitting ${file.name} for local review...`);
    reviewResults.classList.add("hidden");

    try {
      const response = await fetch("/api/analyze-hand-history", {
        method: "POST",
        credentials: "same-origin",
        headers: jsonHeaders(true),
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

if (authLoginButton) {
  authLoginButton.addEventListener("click", () => {
    void submitAuth("/api/auth/login", false);
  });
}

if (authRegisterButton) {
  authRegisterButton.addEventListener("click", () => {
    void submitAuth("/api/auth/register", true);
  });
}

if (profileForm) {
  profileForm.addEventListener("submit", event => {
    event.preventDefault();
    void saveProfile();
  });
}

if (authLogoutButton) {
  authLogoutButton.addEventListener("click", () => {
    void logout();
  });
}

async function boot() {
  await refreshAuthState();
  renderAuthFlash();
}

async function refreshAuthState() {
  try {
    const response = await fetch("/api/auth/me", {
      credentials: "same-origin",
      headers: {
        "Accept": "application/json"
      }
    });

    if (!response.ok) {
      throw new Error(`Auth probe failed with status ${response.status}.`);
    }

    const body = await response.json();
    applyAuthState(body);
  } catch (error) {
    authState = normalizeAuthState({});
    updateAccountUi(`Account bootstrap failed: ${error instanceof Error ? error.message : "unknown error"}`);
  }
}

function applyAuthState(data, flashMessage = "") {
  authState = normalizeAuthState(data);
  hydrateUploadDefaults();
  updateUploadAvailability();
  updateAccountUi(flashMessage);
}

function normalizeAuthState(data) {
  return {
    authenticationEnabled: Boolean(data.authenticationEnabled),
    authenticationMode: typeof data.authenticationMode === "string" ? data.authenticationMode : "none",
    authenticated: Boolean(data.authenticated),
    allowLocalRegistration: Boolean(data.allowLocalRegistration),
    providers: Array.isArray(data.providers) ? data.providers : [],
    user: data && typeof data.user === "object" && data.user ? data.user : null,
    csrfToken: typeof data.csrfToken === "string" && data.csrfToken ? data.csrfToken : null
  };
}

function requiresPlatformSignIn() {
  return authState.authenticationMode === "users";
}

function resolvedUploadSite() {
  if (siteSelect.value && siteSelect.value !== "auto") {
    return siteSelect.value;
  }
  if (authState.user && authState.user.preferredSite) {
    return authState.user.preferredSite;
  }
  return "auto";
}

function resolvedHeroName() {
  const explicit = heroInput.value.trim();
  if (explicit) {
    return explicit;
  }
  if (authState.user && authState.user.heroName) {
    return authState.user.heroName;
  }
  return null;
}

function hydrateUploadDefaults() {
  if (!authState.user) {
    return;
  }

  if (heroInput && !heroInput.value.trim() && authState.user.heroName) {
    heroInput.value = authState.user.heroName;
  }

  if (siteSelect && siteSelect.value === "auto" && authState.user.preferredSite) {
    siteSelect.value = authState.user.preferredSite;
  }
}

function updateUploadAvailability() {
  const locked = requiresPlatformSignIn() && !authState.authenticated;
  [fileInput, siteSelect, heroInput].forEach(element => {
    if (element) {
      element.disabled = locked;
    }
  });
  if (submitButton) {
    submitButton.disabled = locked;
    submitButton.textContent = locked ? "Sign In Required" : "Queue Review";
  }
  if (locked) {
    renderStatus("Sign in to queue a review job for this deployment.");
  }
}

function updateAccountUi(message = "") {
  if (!authStatePanel) {
    return;
  }

  const googleProvider = authState.providers.find(provider => provider.id === "google");
  if (googleLoginLink) {
    if (googleProvider && googleProvider.startPath && !authState.authenticated) {
      googleLoginLink.href = googleProvider.startPath;
      googleLoginLink.classList.remove("hidden");
    } else {
      googleLoginLink.classList.add("hidden");
    }
  }

  if (authState.authenticationMode === "none") {
    authStatePanel.innerHTML = accountPanel(
      "Open access",
      "This deployment does not require sign-in.",
      message || "Upload review is open. User accounts are disabled for this instance."
    );
    toggleHidden(authForm, true);
    toggleHidden(profileCard, true);
    return;
  }

  if (authState.authenticationMode === "basic") {
    authStatePanel.innerHTML = accountPanel(
      "Basic auth",
      "HTTP Basic auth is enforced upstream for this deployment.",
      message || "Use the browser auth challenge configured by the operator. The platform-user module is not enabled here."
    );
    toggleHidden(authForm, true);
    toggleHidden(profileCard, true);
    return;
  }

  if (!authState.authenticated || !authState.user) {
    authStatePanel.innerHTML = accountPanel(
      "Sign in required",
      "Queue review jobs behind a platform user session.",
      message || "Use local registration/login or continue with a configured OIDC provider such as Google."
    );
    toggleHidden(authForm, false);
    toggleHidden(profileCard, true);
    if (authRegisterButton) {
      authRegisterButton.disabled = !authState.allowLocalRegistration;
    }
    return;
  }

  authStatePanel.innerHTML = accountPanel(
    "Signed in",
    `${escapeHtml(authState.user.displayName)} is active for this browser.`,
    message || `Email: ${escapeHtml(authState.user.email)}`
  );
  toggleHidden(authForm, true);
  toggleHidden(profileCard, false);

  if (profileSummary) {
    const providers = Array.isArray(authState.user.linkedProviders) && authState.user.linkedProviders.length > 0
      ? authState.user.linkedProviders.join(", ")
      : "local";
    profileSummary.innerHTML = `
      <p class="card-kicker">Profile</p>
      <h3>${escapeHtml(authState.user.displayName)}</h3>
      <p class="section-note">
        ${escapeHtml(authState.user.email)}<br>
        Linked providers: ${escapeHtml(providers)}
      </p>
    `;
  }

  if (profileDisplayName) {
    profileDisplayName.value = authState.user.displayName || "";
  }
  if (profileHeroName) {
    profileHeroName.value = authState.user.heroName || "";
  }
  if (profilePreferredSite) {
    profilePreferredSite.value = authState.user.preferredSite || "";
  }
  if (profileTimeZone) {
    profileTimeZone.value = authState.user.timeZone || "";
  }
}

async function submitAuth(path, includeDisplayName) {
  if (authState.authenticationMode !== "users") {
    return;
  }

  const email = authEmail ? authEmail.value.trim() : "";
  const password = authPassword ? authPassword.value : "";
  const displayName = authDisplayName ? authDisplayName.value.trim() : "";

  if (!email || !password) {
    updateAccountUi("Email and password are required.");
    return;
  }

  const payload = {
    email,
    password
  };
  if (includeDisplayName && displayName) {
    payload.displayName = displayName;
  }

  try {
    const response = await fetch(path, {
      method: "POST",
      credentials: "same-origin",
      headers: jsonHeaders(false),
      body: JSON.stringify(payload)
    });
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));
    if (!response.ok) {
      updateAccountUi(body.error || `Authentication request failed with status ${response.status}.`);
      return;
    }

    if (authPassword) {
      authPassword.value = "";
    }
    applyAuthState(body, includeDisplayName ? "Registration complete." : "Signed in.");
  } catch (error) {
    updateAccountUi(`Authentication request failed: ${error instanceof Error ? error.message : "unknown error"}`);
  }
}

async function saveProfile() {
  if (!authState.authenticated || !authState.csrfToken) {
    updateAccountUi("Sign in before saving a profile.");
    return;
  }

  const payload = {
    displayName: profileDisplayName ? profileDisplayName.value.trim() || null : null,
    heroName: profileHeroName ? profileHeroName.value.trim() || null : null,
    preferredSite: profilePreferredSite ? profilePreferredSite.value || null : null,
    timeZone: profileTimeZone ? profileTimeZone.value.trim() || null : null
  };

  try {
    const response = await fetch("/api/auth/profile", {
      method: "POST",
      credentials: "same-origin",
      headers: jsonHeaders(true),
      body: JSON.stringify(payload)
    });
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));
    if (!response.ok) {
      updateAccountUi(body.error || `Profile save failed with status ${response.status}.`);
      return;
    }

    applyAuthState(body, "Profile saved.");
  } catch (error) {
    updateAccountUi(`Profile save failed: ${error instanceof Error ? error.message : "unknown error"}`);
  }
}

async function logout() {
  if (!authState.authenticated || !authState.csrfToken) {
    return;
  }

  try {
    const response = await fetch("/api/auth/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: jsonHeaders(true),
      body: "{}"
    });
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));
    if (!response.ok) {
      updateAccountUi(body.error || `Sign out failed with status ${response.status}.`);
      return;
    }

    if (authEmail) {
      authEmail.value = "";
    }
    if (authDisplayName) {
      authDisplayName.value = "";
    }
    applyAuthState(body, "Signed out.");
  } catch (error) {
    updateAccountUi(`Sign out failed: ${error instanceof Error ? error.message : "unknown error"}`);
  }
}

function renderAuthFlash() {
  const params = new URLSearchParams(window.location.search);
  const authResult = params.get("auth");
  const authError = params.get("auth_error");

  if (authResult === "success") {
    updateAccountUi("OIDC sign-in completed.");
  } else if (authError) {
    updateAccountUi(`OIDC sign-in failed: ${authError.replaceAll("+", " ")}`);
  } else {
    return;
  }

  params.delete("auth");
  params.delete("auth_error");
  const nextQuery = params.toString();
  const nextUrl = `${window.location.pathname}${nextQuery ? `?${nextQuery}` : ""}${window.location.hash}`;
  window.history.replaceState({}, document.title, nextUrl);
}

function accountPanel(kicker, title, note) {
  return `
    <p class="card-kicker">${escapeHtml(kicker)}</p>
    <h3>${escapeHtml(title)}</h3>
    <p class="section-note">${escapeHtml(note)}</p>
  `;
}

function toggleHidden(element, hidden) {
  if (!element) {
    return;
  }
  element.classList.toggle("hidden", hidden);
}

function jsonHeaders(includeCsrf) {
  const headers = {
    "Content-Type": "application/json"
  };
  if (includeCsrf && authState.csrfToken) {
    headers["X-CSRF-Token"] = authState.csrfToken;
  }
  return headers;
}

function setSubmitting(isSubmitting) {
  if (!submitButton) {
    return;
  }
  if (requiresPlatformSignIn() && !authState.authenticated) {
    submitButton.disabled = true;
    submitButton.textContent = "Sign In Required";
    return;
  }
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
      credentials: "same-origin",
      headers: {
        "Accept": "application/json"
      }
    });
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error("Review job expired, was purged, or is not visible to this user session.");
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
    ? opponent.hints.map(renderHint).join("<br>")
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

function renderHint(hint) {
  if (typeof hint === "string") {
    return escapeHtml(hint);
  }

  const text = escapeHtml(hint && hint.text ? hint.text : "Unknown hint");
  const metrics = Array.isArray(hint && hint.metrics)
    ? `[${hint.metrics.map(formatMetric).join(", ")}]`
    : "";

  return metrics ? `${text}<br><span class="card-meta">${escapeHtml(metrics)}</span>` : text;
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

function formatMetric(value) {
  return Number(value || 0).toFixed(3);
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
