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
const hallForm = document.getElementById("playing-hall-form");
const hallHandsInput = document.getElementById("hall-hands");
const hallTableCountInput = document.getElementById("hall-table-count");
const hallPlayerCountSelect = document.getElementById("hall-player-count");
const hallSeedInput = document.getElementById("hall-seed");
const hallRandomSeedInput = document.getElementById("hall-random-seed");
const hallHeroStyleSelect = document.getElementById("hall-hero-style");
const hallHeroPositionSelect = document.getElementById("hall-hero-position");
const hallGtoModeSelect = document.getElementById("hall-gto-mode");
const hallExplorationRateInput = document.getElementById("hall-exploration-rate");
const hallRaiseSizeInput = document.getElementById("hall-raise-size");
const hallLearnEveryInput = document.getElementById("hall-learn-every");
const hallLearningWindowInput = document.getElementById("hall-learning-window");
const hallBunchingTrialsInput = document.getElementById("hall-bunching-trials");
const hallEquityTrialsInput = document.getElementById("hall-equity-trials");
const hallSaveReviewInput = document.getElementById("hall-save-review");
const hallFullRingInput = document.getElementById("hall-full-ring");
const hallSubmitButton = document.getElementById("hall-submit");
const hallStatus = document.getElementById("hall-status");
const hallResults = document.getElementById("hall-results");
const hallStatusTitle = document.getElementById("hall-status-title");
const hallStatusNote = document.getElementById("hall-status-note");
const hallProgress = document.getElementById("hall-progress");
const hallElapsed = document.getElementById("hall-elapsed");
const hallCancelButton = document.getElementById("hall-cancel");
const hallPresetBar = document.getElementById("hall-preset-bar");
const hallRecentCount = document.getElementById("hall-recent-count");
const hallRecentList = document.getElementById("hall-recent-runs-list");
const hallKpiGrid = document.getElementById("hall-kpi-grid");
const hallChartChips = document.getElementById("hall-chart-chips-over-time");
const hallChartActionDonut = document.getElementById("hall-chart-action-donut");
const hallChartWlt = document.getElementById("hall-chart-wlt");
const hallChartVillainBar = document.getElementById("hall-chart-villain-bar");
const hallChartEquityHist = document.getElementById("hall-chart-equity-hist");
const hallChartMatrix = document.getElementById("hall-chart-matrix");
const hallOutputList = document.getElementById("hall-output-list");

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
const profileSaveButton = document.getElementById("profile-save");

// 16 minutes: 1 minute of slack over the default PLAYING_HALL_TIMEOUT_MS
// (15 min) so the frontend deadline check at the top of the poll loop
// doesn't throw "timed out" the moment the server's own timeout fires.
// Without slack the two deadlines line up exactly, and the queue wait
// between submit and the worker start (which the server measures from)
// means the server can return a terminal status (completed OR timeout-
// failed) microseconds after the frontend's pre-poll deadline check
// has already given up. The user then sees a misleading client-side
// "timed out" instead of the server's actual outcome. ANALYZE jobs use
// a 2-min server timeout so they have 14 min of slack and aren't
// affected.
const MAX_POLL_WAIT_MS = 16 * 60 * 1000;

// Initial cap matches the server's default --maxUploadBytes (2 MiB). The
// server already rejects oversize bodies with 413, but the frontend has no
// way to discover the configured cap until boot, so we bound it client-side
// to spare the user from a multi-second file-read + upload + 413-rejection
// round trip when they pick the wrong file. boot() probes /api/health and
// reassigns this to the server's actual maxUploadBytes so an operator who
// raised the server cap also gets the larger client-side allowance.
let maxUploadFileBytes = 2 * 1024 * 1024;

// Single-shot fetch timeout. Without this, a hung server (or a network
// drop after the TCP handshake) leaves the user staring at a stuck
// "Submitting..." / "Signing in..." indicator until the OS-level TCP
// timeout fires -- which on Windows is ~21s for SYN retries but can be
// many minutes for an idle ESTABLISHED socket. 15 seconds is comfortably
// above the worst-case happy-path latency (auth PBKDF2 + DB write,
// submit + JSON parse, etc.) and far below "user assumes the page is
// broken". Poll loops use a longer timeout via the timeoutMs override
// (see POLL_FETCH_TIMEOUT_MS).
const SINGLE_SHOT_FETCH_TIMEOUT_MS = 15_000;

// Poll-status timeout: more generous than single-shot so a brief server
// stall doesn't kill the loop, but tight enough that a genuinely hung
// fetch fails well before MAX_POLL_WAIT_MS (16 min) is exhausted.
// Without this, an OS-level TCP idle timeout (minutes) could leave one
// poll waiting while the deadline check at the top of the loop never
// gets a chance to re-evaluate.
const POLL_FETCH_TIMEOUT_MS = 30_000;

function fetchWithTimeout(url, options, timeoutMs = SINGLE_SHOT_FETCH_TIMEOUT_MS) {
  // Feature-detect AbortSignal.timeout (Chrome 103+, Firefox 100+, Safari
  // 16+). Older browsers throw TypeError ('AbortSignal.timeout is not a
  // function') the moment we call it -- before fetch even starts -- and
  // describeFetchError wouldn't recognize the message, so every single
  // request would surface a confusing developer-facing error to the user.
  // Fall back to a fetch without timeout (matches the pre-timeout
  // behavior) so the page still works on legacy browsers; modern users
  // keep the timeout they had.
  if (typeof AbortSignal === "undefined" || typeof AbortSignal.timeout !== "function") {
    return fetch(url, options);
  }
  return fetch(url, {
    ...options,
    signal: AbortSignal.timeout(timeoutMs)
  });
}

// Translate a thrown fetch error into a user-facing message. The two cases
// the user benefits from naming explicitly:
//   - AbortSignal.timeout throws DOMException with name='TimeoutError' and
//     a terse "signal timed out" message that reads like a bug to a non-
//     developer. Translate to actionable language.
//   - A network-layer failure (server unreachable, DNS down, offline)
//     throws TypeError "Failed to fetch" in Chrome / "NetworkError when
//     attempting to fetch resource" in Firefox -- match either substring.
// Anything else passes through the original error.message so a server-
// thrown reason (e.g., from pollAnalysisJob's manual throws) still
// surfaces verbatim.
function describeFetchError(error) {
  if (!error) return "unknown error";
  if (error.name === "TimeoutError" || error.name === "AbortError") {
    return "the server did not respond in time -- please try again";
  }
  if (error instanceof TypeError && /fetch|network/i.test(error.message)) {
    return "could not reach the server -- check your connection";
  }
  return error instanceof Error ? error.message : "unknown error";
}

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

    if (file.size > maxUploadFileBytes) {
      // Skip the file.text() + JSON.stringify + upload round trip entirely.
      // For a 50 MB mis-picked file (DB dump, archive, video) this saves the
      // user several seconds of "Submitting..." status before the inevitable
      // 413, AND avoids holding the whole file in the JS heap. Show the
      // received and allowed sizes so the user can decide whether to trim or
      // split rather than guess the cap.
      renderStatus(`File is ${formatFileSize(file.size)}, exceeds the ${formatFileSize(maxUploadFileBytes)} upload limit. Trim or split the hand history and try again.`);
      reviewResults.classList.add("hidden");
      return;
    }

    if (file.size === 0) {
      // The server also rejects empty handHistoryText (post-trim), but a
      // synchronous local check beats round-tripping a 400. Common cause:
      // the user picked the wrong file from a "Save As" template that
      // left only an empty placeholder.
      renderStatus(`${file.name} is empty. Pick a hand-history export with at least one hand.`);
      reviewResults.classList.add("hidden");
      return;
    }

    // Disable the submit button BEFORE the first await -- file.text() is
    // async, so leaving setSubmitting(true) after it would let an
    // impatient double-click fire a second submit handler that races
    // through its own file read, JSON build, and POST. The hall form
    // gets away without this ordering because its payload build is
    // entirely sync (numeric inputs only); the analyze form has to read
    // the file. Use try/finally from this point so setSubmitting(false)
    // still runs if the file read or payload build throws.
    setSubmitting(true);
    renderStatus(`Submitting ${file.name} for local review...`);
    reviewResults.classList.add("hidden");

    try {
      const payload = {
        handHistoryText: await file.text(),
        site: resolvedUploadSite(),
        heroName: resolvedHeroName()
      };

      const response = await fetchWithTimeout("/api/analyze-hand-history", {
        method: "POST",
        credentials: "same-origin",
        headers: jsonHeaders(true),
        body: JSON.stringify(payload)
      });
      const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));

      if (!response.ok) {
        await maybeReauthOn401(response);
        renderStatus(formatErrorMessage(response, body));
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
      renderStatus(`Request failed: ${describeFetchError(error)}`);
    } finally {
      setSubmitting(false);
    }
  });
}

if (hallForm) {
  hallForm.addEventListener("submit", async event => {
    event.preventDefault();

    if (requiresPlatformSignIn() && !authState.authenticated) {
      renderHallStatus("Sign in to launch a playing hall run on this deployment.");
      hallResults.classList.add("hidden");
      return;
    }

    if (!validateHallForm()) {
      renderHallStatus("Fix invalid fields before launching the hall.");
      hallResults.classList.add("hidden");
      return;
    }

    const payload = {
      hands: numericValue(hallHandsInput, 240),
      tableCount: numericValue(hallTableCountInput, 2),
      playerCount: numericValue(hallPlayerCountSelect, 6),
      heroStyle: hallHeroStyleSelect ? hallHeroStyleSelect.value : "adaptive",
      heroPosition: hallHeroPositionSelect ? hallHeroPositionSelect.value : "Button",
      gtoMode: hallGtoModeSelect ? hallGtoModeSelect.value : "exact",
      villainPool: selectedVillainPool(),
      heroExplorationRate: numericValue(hallExplorationRateInput, 0),
      raiseSize: numericValue(hallRaiseSizeInput, 2.5),
      bunchingTrials: numericValue(hallBunchingTrialsInput, 40),
      equityTrials: numericValue(hallEquityTrialsInput, 240),
      learnEveryHands: numericValue(hallLearnEveryInput, 0),
      learningWindowSamples: numericValue(hallLearningWindowInput, 200),
      seed: hallRandomSeedInput && hallRandomSeedInput.checked
        ? Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)
        : numericValue(hallSeedInput, 42),
      saveReviewHandHistory: Boolean(hallSaveReviewInput && hallSaveReviewInput.checked),
      fullRing: Boolean(hallFullRingInput && hallFullRingInput.checked)
    };

    setHallSubmitting(true);
    renderHallStatus("Queueing a local playing hall run...");
    hallResults.classList.add("hidden");

    try {
      const response = await fetchWithTimeout("/api/playing-hall", {
        method: "POST",
        credentials: "same-origin",
        headers: jsonHeaders(true),
        body: JSON.stringify(payload)
      });
      const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));

      if (!response.ok) {
        await maybeReauthOn401(response);
        renderHallStatus(formatErrorMessage(response, body));
        return;
      }

      if (body.jobId) {
        const statusUrl = body.statusUrl || response.headers.get("Location");
        if (!statusUrl) {
          renderHallStatus("Server accepted the hall run but did not return a job status URL.");
          return;
        }

        startHallElapsed(body.jobId);
        renderHallStatus(playingHallJobStatusMessage(body.status));
        const result = await pollPlayingHallJob(statusUrl, body.pollAfterMs);
        renderHallResults(result);
        pushRecentRun(payload, (result && result.summary) || {});
        return;
      }

      renderHallResults(body);
      pushRecentRun(payload, (body && body.summary) || {});
    } catch (error) {
      renderHallStatus(`Playing hall request failed: ${describeFetchError(error)}`);
    } finally {
      setHallSubmitting(false);
      finishHallProgress();
    }
  });
}

if (hallRandomSeedInput) {
  hallRandomSeedInput.addEventListener("change", () => {
    if (hallSeedInput) {
      hallSeedInput.disabled = hallRandomSeedInput.checked;
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

if (authForm) {
  // Without this handler, hitting Enter in the password (or email) field
  // implicitly submits the <form>, which has no `action` or `method` -- so
  // the browser defaults to method=GET against the current URL and appends
  // every form field as a query parameter. The PASSWORD ends up in the URL
  // bar, in browser history, and in any subsequent Referer header. Catch
  // the submit event, prevent the default GET-with-credentials navigation,
  // and route to the login flow (the most common action). The user can
  // still click Register explicitly if that was their intent.
  authForm.addEventListener("submit", event => {
    event.preventDefault();
    if (authState.authenticationMode !== "users") return;
    if (authState.authenticated) return;
    void submitAuth("/api/auth/login", false);
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
  // /api/health is unauthenticated and cheap; do it in parallel with the
  // auth probe so a slow auth-state response doesn't delay the upload cap
  // sync. Both Promises are fire-and-forget on error.
  await Promise.all([refreshAuthState(), probeServerLimits()]);
  renderAuthFlash();
  renderPresetBar();
  renderRecentRuns();
  wireHallValidation();
  syncRandomSeedState();
  mirrorHelpDataToAriaLabel();
}

// Read the server's actual maxUploadBytes from /api/health and adopt it
// as the client-side upload cap. Without this, an operator who raises
// MAX_UPLOAD_BYTES beyond the 2 MiB default would have the frontend still
// refusing files larger than 2 MiB because the cap was hard-coded.
// Falls back silently to the initial 2 MiB if the probe fails -- worst
// case is the frontend stops at 2 MiB even though the server would accept
// more, which is strictly safer than the opposite. Uses the same
// AbortSignal timeout as other single-shot fetches.
async function probeServerLimits() {
  try {
    const response = await fetchWithTimeout("/api/health", {
      credentials: "same-origin",
      headers: { "Accept": "application/json" }
    });
    if (!response.ok) return;
    const body = await response.json();
    const serverMax = Number(body && body.maxUploadBytes);
    if (Number.isFinite(serverMax) && serverMax > 0) {
      maxUploadFileBytes = serverMax;
    }
  } catch (_) {
    // Probe is best-effort. A network blip leaves the 2 MiB default in
    // place, which still allows legitimate hand-history uploads.
  }
}

// The ⓘ help icons use a CSS ::after pseudo-element to render their
// data-help attribute as a hover tooltip. Screen readers don't announce
// pseudo-element generated content, so the data-help text is invisible
// to assistive tech -- a focused help icon is just announced as the
// ℹ symbol with no context. Mirror data-help into aria-label so screen
// readers read the same hint the sighted tooltip shows. tabindex=0 in
// the markup already makes them focusable; this just labels them.
function mirrorHelpDataToAriaLabel() {
  document.querySelectorAll(".help[data-help]").forEach(el => {
    const help = el.getAttribute("data-help");
    if (help && !el.hasAttribute("aria-label")) {
      el.setAttribute("aria-label", help);
    }
  });
}

// Browsers preserve checkbox state across reload, so a user who left the
// random-seed checkbox ticked sees it ticked on reload -- but the change
// listener that disables the seed-number input only runs on user toggle,
// not on initial state. Without this sync the seed input renders as
// editable on reload even though its value is ignored on submit, which
// is misleading. Run once on boot to bring the disabled state in line
// with the checkbox.
function syncRandomSeedState() {
  if (hallRandomSeedInput && hallSeedInput) {
    hallSeedInput.disabled = hallRandomSeedInput.checked;
  }
}

async function refreshAuthState() {
  try {
    const response = await fetchWithTimeout("/api/auth/me", {
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
    updateAccountUi(`Account bootstrap failed: ${describeFetchError(error)}`);
  }
}

// When a state-changing or polled request comes back 401 or 403, the
// frontend's auth state is out of sync with the server's:
//   - 401: session expired / revoked (logged out in another tab) -- the
//     login form needs to reappear so the user knows.
//   - 403: stale CSRF token (re-auth happened in another tab and minted a
//     new csrfToken that this tab hasn't seen yet). Refreshing /api/auth/me
//     pulls the fresh token so a retry succeeds without a full page reload.
// Either way, refresh from /api/auth/me so the UI matches reality.
async function maybeReauthOn401(response) {
  if (!response) return;
  const status = response.status;
  if ((status === 401 || status === 403) && authState.authenticated) {
    await refreshAuthState();
  }
}

// Build a user-facing error message that augments the server's `error` field
// with retry-after info for transient failures the user can re-try. Applies
// to 429 (rate-limited) AND 503 (queue full / drain mode / timed-out worker
// recovery). Both responses carry a Retry-After header (and 429 also includes
// retryAfterSeconds in the JSON body, which is preferred because it's the
// precise server-computed value rather than the JsonHandler's 5-second 503
// fallback). For any other status code, returns the base error unchanged.
function formatErrorMessage(response, body) {
  const fallback = `Request failed with status ${response.status}.`;
  const base = body && typeof body.error === "string" && body.error ? body.error : fallback;
  const isRetryable = response.status === 429 || response.status === 503;
  if (!isRetryable) return base;
  let retrySeconds = body && Number.isFinite(Number(body.retryAfterSeconds))
    ? Number(body.retryAfterSeconds)
    : null;
  if (retrySeconds == null) {
    const headerValue = response.headers.get("Retry-After");
    const parsed = headerValue ? parseInt(headerValue, 10) : NaN;
    retrySeconds = Number.isFinite(parsed) ? parsed : null;
  }
  if (retrySeconds == null || retrySeconds <= 0) return base;
  const unit = retrySeconds === 1 ? "second" : "seconds";
  return `${base} Try again in ${retrySeconds} ${unit}.`;
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
  // hydrateUploadDefaults already prefills siteSelect from
  // authState.user.preferredSite when the user signs in, so whatever the
  // select shows is the user's effective choice. Returning preferredSite
  // here too used to OVERRIDE an explicit 'Auto-detect' pick: if the
  // signed-in user with preferredSite='pokerstars' changed the select to
  // 'Auto-detect', this code path silently sent 'pokerstars' anyway,
  // because siteSelect.value === 'auto' fell through to the preferredSite
  // branch. Trust the select; it already reflects the prefill.
  return siteSelect.value || "auto";
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
  [
    hallHandsInput,
    hallTableCountInput,
    hallPlayerCountSelect,
    hallSeedInput,
    hallRandomSeedInput,
    hallHeroStyleSelect,
    hallHeroPositionSelect,
    hallGtoModeSelect,
    hallExplorationRateInput,
    hallRaiseSizeInput,
    hallLearnEveryInput,
    hallLearningWindowInput,
    hallBunchingTrialsInput,
    hallEquityTrialsInput,
    hallSaveReviewInput,
    hallFullRingInput,
    ...Array.from(document.querySelectorAll('input[name="villain-pool"]'))
  ].forEach(element => {
    if (element) {
      element.disabled = locked;
    }
  });
  if (hallSubmitButton) {
    hallSubmitButton.disabled = locked;
    hallSubmitButton.textContent = locked ? "Sign In Required" : "Run Playing Hall";
  }
  if (locked && hallStatus) {
    renderHallStatus("Sign in to launch a playing hall run on this deployment.");
  }
  // The blanket `element.disabled = locked` above re-enables hallSeedInput
  // on sign-in even when the random-seed checkbox is ticked, which is
  // misleading -- the seed value is ignored on submit but the input
  // appears editable. Re-apply the random-seed → seed-input dependency
  // after the blanket pass so it survives auth-state changes.
  if (!locked) syncRandomSeedState();
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
      const registrationOff = !authState.allowLocalRegistration;
      authRegisterButton.disabled = registrationOff;
      // Tell the user WHY the button is grayed out. Without this they
      // see a dimmed Register and have no signal whether it's a
      // transient lock (slow network mid-fetch) or a deployment-level
      // policy (USER_AUTH_ALLOW_REGISTRATION=false). aria-disabled
      // mirrors disabled so screen readers announce the state too;
      // title gives the hover tooltip on desktop.
      if (registrationOff) {
        authRegisterButton.title = "Registration is disabled on this deployment";
        authRegisterButton.setAttribute("aria-disabled", "true");
      } else {
        authRegisterButton.removeAttribute("title");
        authRegisterButton.removeAttribute("aria-disabled");
      }
    }
    return;
  }

  // accountPanel applies escapeHtml to each field on its own, so we pass raw
  // values here. Pre-escaping inside the template (e.g. escapeHtml(displayName))
  // would double-encode -- `Alice & Bob` becomes `Alice &amp;amp; Bob` after
  // the second pass and renders to the user as `Alice &amp; Bob` instead of
  // `Alice & Bob`. Same fix shape as the modelSource cleanup earlier.
  authStatePanel.innerHTML = accountPanel(
    "Signed in",
    `${authState.user.displayName} is active for this browser.`,
    message || `Email: ${authState.user.email}`
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

  // Disable BOTH auth buttons (login + register) for the duration of the
  // request so a slow network doesn't tempt the user into a double-click
  // that fires the request twice -- a duplicate register racing on the
  // same email surfaces as a confusing "already exists" error, and a
  // duplicate login wastes a PBKDF2 verify per click.
  setAuthButtonsBusy(true);
  try {
    const response = await fetchWithTimeout(path, {
      method: "POST",
      credentials: "same-origin",
      headers: jsonHeaders(false),
      body: JSON.stringify(payload)
    });
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));
    if (!response.ok) {
      // 409 'already signed in' from register/login means the server sees a
      // valid session cookie that the frontend's authState doesn't reflect
      // -- typically because this tab booted before sign-in happened in a
      // sibling tab. Refreshing /api/auth/me pulls the real state so the
      // UI switches from sign-in to signed-in without a manual reload.
      if (response.status === 409) {
        await refreshAuthState();
      }
      updateAccountUi(formatErrorMessage(response, body));
      return;
    }

    if (authPassword) {
      authPassword.value = "";
    }
    applyAuthState(body, includeDisplayName ? "Registration complete." : "Signed in.");
  } catch (error) {
    updateAccountUi(`Authentication request failed: ${describeFetchError(error)}`);
  } finally {
    setAuthButtonsBusy(false);
  }
}

function setAuthButtonsBusy(busy) {
  // Keep updateAccountUi's allow-local-registration gate intact when
  // re-enabling: it stays disabled if registration is off for the
  // deployment, even after the fetch completes. Swap the labels to
  // 'Signing in...' / 'Registering...' for the same in-flight feedback
  // shape the profile-save and logout buttons use; the disabled state
  // alone is too subtle (Tab-key users may not notice).
  if (authLoginButton) {
    authLoginButton.disabled = busy;
    authLoginButton.textContent = busy ? "Signing in..." : "Sign In";
  }
  if (authRegisterButton) {
    authRegisterButton.disabled = busy || !authState.allowLocalRegistration;
    authRegisterButton.textContent = busy ? "Registering..." : "Register";
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

  // Disable the Save button for the duration of the request -- the
  // profile form's <button type="submit"> means hitting Enter in any
  // field also fires saveProfile, so a slow network + impatient retry
  // (click, then Enter, then click again) would otherwise race
  // multiple POSTs against the same CSRF token.
  if (profileSaveButton) {
    profileSaveButton.disabled = true;
    profileSaveButton.textContent = "Saving...";
  }
  try {
    const response = await fetchWithTimeout("/api/auth/profile", {
      method: "POST",
      credentials: "same-origin",
      headers: jsonHeaders(true),
      body: JSON.stringify(payload)
    });
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));
    if (!response.ok) {
      await maybeReauthOn401(response);
      updateAccountUi(formatErrorMessage(response, body));
      return;
    }

    applyAuthState(body, "Profile saved.");
  } catch (error) {
    updateAccountUi(`Profile save failed: ${describeFetchError(error)}`);
  } finally {
    if (profileSaveButton) {
      profileSaveButton.disabled = false;
      profileSaveButton.textContent = "Save Profile";
    }
  }
}

async function logout() {
  if (!authState.authenticated || !authState.csrfToken) {
    return;
  }

  // Disable the Sign Out button for the duration of the request so a
  // slow network doesn't tempt a double-click into firing the POST
  // twice. The server's logout is idempotent (revoking an already-
  // revoked session returns the same cleared-cookie response), so the
  // duplicate is harmless server-side -- this is purely a UX nicety
  // matching the auth + profile-save buttons.
  if (authLogoutButton) {
    authLogoutButton.disabled = true;
    authLogoutButton.textContent = "Signing out...";
  }
  try {
    const response = await fetchWithTimeout("/api/auth/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: jsonHeaders(true),
      body: "{}"
    });
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));
    if (!response.ok) {
      // Same auth-state-divergence handling as the other state-changing
      // routes: a 401 means the session was already gone (e.g. revoked
      // in a sibling tab) and the UI should re-render as signed-out;
      // a 403 means a stale CSRF token (post-reauth in a sibling tab)
      // and refreshing /api/auth/me pulls a fresh one so a retry works.
      await maybeReauthOn401(response);
      updateAccountUi(body.error || `Sign out failed with status ${response.status}.`);
      return;
    }

    if (authEmail) {
      authEmail.value = "";
    }
    if (authDisplayName) {
      authDisplayName.value = "";
    }
    // Clear upload-form state too so the previous user's prefilled
    // preferences (hero name, preferred site) and any selected file
    // don't carry over to the next person signing in on the same
    // browser. The password field is already cleared in submitAuth on
    // a successful login; on logout it's typically empty already, but
    // wipe it defensively in case the user typed and then signed out
    // without submitting.
    if (authPassword) authPassword.value = "";
    if (heroInput) heroInput.value = "";
    if (siteSelect) siteSelect.value = "auto";
    if (fileInput) fileInput.value = "";
    // localStorage 'sicfun.hall.recentRuns' is browser-scoped, not
    // user-scoped: without this wipe, user B signing into the same
    // browser would see (and could Load) user A's hall configurations
    // alongside their own. Clear on explicit logout only -- a page
    // refresh leaves the data intact for the same signed-in user.
    clearRecentRuns();
    renderRecentRuns();
    applyAuthState(body, "Signed out.");
  } catch (error) {
    updateAccountUi(`Sign out failed: ${describeFetchError(error)}`);
  } finally {
    // Re-enable only on failure: a successful logout hides profileCard
    // (which contains this button) via updateAccountUi switching to the
    // sign-in panel, so its disabled state doesn't matter. On failure
    // the user stays on the profile panel and needs the button back.
    if (authLogoutButton) {
      authLogoutButton.disabled = false;
      authLogoutButton.textContent = "Sign Out";
    }
  }
}

// Map well-known server-side OIDC error codes to readable messages. Unknown
// codes fall through to the raw value (e.g. provider-side codes Google
// passes through). The server caps the value at 256 chars + truncation
// marker, so this mapping does not need to handle adversarial input -- just
// translate the small set of identifiers our own callback handler emits.
const OIDC_ERROR_MESSAGES = {
  "missing_state_cookie": "Sign-in session expired. Please try signing in again.",
  "state_cookie_mismatch": "Sign-in security check failed. Please try signing in again.",
  "missing_code_or_state": "Sign-in did not complete. Please try signing in again.",
  "oversize_callback_param": "The sign-in response was malformed. Please try again.",
  "access_denied": "Sign-in was declined. Please try a different account or method.",
  // RFC 6749 sec 5.2 standard error codes the provider might pass through
  // verbatim. They're rare in practice but worth translating so the user
  // doesn't see 'OIDC sign-in failed: temporarily_unavailable' (which
  // reads like a technical fault); the friendly versions tell them
  // whether to retry vs. contact someone.
  "temporarily_unavailable": "The sign-in provider is temporarily unavailable. Please try again in a few minutes.",
  "server_error": "The sign-in provider reported an error. Please try again.",
  "invalid_request": "The sign-in request was malformed. Please try again.",
  "unauthorized_client": "This deployment is not authorized with the sign-in provider. Please contact the operator.",
  "unsupported_response_type": "The sign-in provider returned an unsupported response. Please contact the operator.",
  "invalid_scope": "The sign-in scope is misconfigured. Please contact the operator.",
  // PlatformUserAuth.finishOidc emits this when the state token cannot be
  // resolved -- typically because the user took longer than the
  // OidcStateStore TTL (~10 minutes) between /start and /callback, or
  // because they reloaded /callback and the state was already consumed
  // by a previous attempt. Tell them what to do without leaking the
  // internal token-store mechanics.
  "OIDC login state expired or is invalid": "Your sign-in took too long or was already completed in another tab. Please try again.",
  // upsertOidcIdentity rejects when the email is already linked to a
  // different identity (e.g. local password) so two flows don't collide
  // on the same email. The user needs to use their original method.
  "an account with that email already exists; sign in with its existing method":
    "An account with that email already exists. Sign in with your original method (e.g. password) instead.",
  // The user-store cap (USER_AUTH_MAX_USERS) can trip during BOTH local
  // registration AND OIDC sign-up when the new identity does not match
  // an existing user. The server emits the same human-readable string in
  // both cases ("registration is temporarily unavailable"); the friendly
  // version below tells the OIDC user this is a deployment-side capacity
  // issue, not something they did wrong.
  "registration is temporarily unavailable": "This deployment is at capacity and not accepting new sign-ins. Please try again later or contact the operator."
};

function renderAuthFlash() {
  const params = new URLSearchParams(window.location.search);
  const authResult = params.get("auth");
  const authError = params.get("auth_error");

  if (authResult === "success") {
    updateAccountUi("OIDC sign-in completed.");
  } else if (authError) {
    const friendly = OIDC_ERROR_MESSAGES[authError];
    const displayed = friendly || `OIDC sign-in failed: ${authError.replaceAll("+", " ")}`;
    updateAccountUi(displayed);
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

function setHallSubmitting(isSubmitting) {
  if (!hallSubmitButton) {
    return;
  }
  if (requiresPlatformSignIn() && !authState.authenticated) {
    hallSubmitButton.disabled = true;
    hallSubmitButton.textContent = "Sign In Required";
    return;
  }
  hallSubmitButton.disabled = isSubmitting;
  hallSubmitButton.textContent = isSubmitting ? "Running Hall..." : "Run Playing Hall";
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

function renderHallStatus(message) {
  if (!hallStatus) {
    return;
  }
  hallStatus.innerHTML = `
    <p class="card-kicker">Status</p>
    <h3>${escapeHtml(message)}</h3>
    <p class="section-note">
      SICFUN runs the configured hall batch in the background, then returns the run summary, action mix,
      per-villain chip flow, and the generated output files here.
    </p>
  `;
}

async function pollAnalysisJob(fileName, statusUrl, initialPollAfterMs) {
  let pollAfterMs = normalizePollAfterMs(initialPollAfterMs);
  const deadline = Date.now() + MAX_POLL_WAIT_MS;

  for (;;) {
    if (Date.now() >= deadline) {
      // Frontend-side polling deadline, NOT a server-side job failure --
      // the job may still be running. Tell the user accurately so they
      // don't assume the analysis crashed; the frontend budget should
      // rarely fire because the server's analyze timeout is 2 minutes
      // (the job would have terminated long before this). Derive the
      // minute count from MAX_POLL_WAIT_MS so a future bump stays in
      // sync with the message.
      const minutes = Math.round(MAX_POLL_WAIT_MS / 60000);
      throw new Error(`Stopped polling after ${minutes} minutes. The job may still be running on the server -- check back later by reloading the page.`);
    }

    await sleep(pollAfterMs);

    const response = await fetchWithTimeout(statusUrl, {
      credentials: "same-origin",
      headers: {
        "Accept": "application/json"
      }
    }, POLL_FETCH_TIMEOUT_MS);
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error("Review job expired, was purged, or is not visible to this user session.");
      }
      if (response.status === 401) {
        // Session expired mid-poll. maybeReauthOn401 will refresh the auth
        // state and re-render the sign-in form; throw a message that tells
        // the user the JOB itself is still running server-side and that
        // they can sign back in to retry the status check, instead of the
        // generic 'session authentication required' which reads as a
        // total failure.
        await maybeReauthOn401(response);
        throw new Error("Session expired during the review. The job is still running on the server -- sign in again to keep polling, or check back later.");
      }
      throw new Error(formatErrorMessage(response, body));
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

async function pollPlayingHallJob(statusUrl, initialPollAfterMs) {
  let pollAfterMs = normalizePollAfterMs(initialPollAfterMs);
  const deadline = Date.now() + MAX_POLL_WAIT_MS;

  for (;;) {
    if (Date.now() >= deadline) {
      // Same shape as the analysis poller: this is a client-side
      // polling budget exhaustion, not a server-side job failure. The
      // hall server timeout is 15 minutes (default PLAYING_HALL_TIMEOUT_MS),
      // so reaching the frontend budget means either the server is
      // silently slow OR the queue wait was long enough to push the
      // worker's own deadline past ours. The job may still run to
      // completion. Derive the minute count from MAX_POLL_WAIT_MS so
      // a future bump stays in sync with the message.
      const minutes = Math.round(MAX_POLL_WAIT_MS / 60000);
      throw new Error(`Stopped polling after ${minutes} minutes. The hall run may still be finishing on the server -- check back later by reloading the page.`);
    }

    await sleep(pollAfterMs);

    const response = await fetchWithTimeout(statusUrl, {
      credentials: "same-origin",
      headers: {
        "Accept": "application/json"
      }
    }, POLL_FETCH_TIMEOUT_MS);
    const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error("Playing hall job expired, was purged, or is not visible to this user session.");
      }
      if (response.status === 401) {
        // Session expired mid-poll. Same shape as the analysis poller --
        // the run is still going server-side; the user can sign back in
        // and the job remains queryable until it finishes or is purged.
        await maybeReauthOn401(response);
        throw new Error("Session expired during the hall run. The job is still running on the server -- sign in again to keep polling, or check back later.");
      }
      throw new Error(formatErrorMessage(response, body));
    }

    renderHallStatus(playingHallJobStatusMessage(body.status));

    if (body.status === "completed") {
      return body.result || {};
    }

    if (body.status === "cancelled") {
      const result = body.result || {};
      result.cancelled = true;
      return result;
    }

    if (body.status === "failed") {
      throw new Error(body.error || "Playing hall failed.");
    }

    if (body.status !== "queued" && body.status !== "running") {
      throw new Error(`Unexpected playing hall job status: ${body.status || "unknown"}`);
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

function playingHallJobStatusMessage(status) {
  switch (status) {
    case "queued":
      return "Queued the playing hall run...";
    case "running":
      return "Running the playing hall in the background...";
    case "completed":
      return "Playing hall run complete.";
    case "cancelled":
      return "Playing hall run cancelled.";
    case "failed":
      return "Playing hall run failed.";
    default:
      return "Processing the playing hall run...";
  }
}

function renderResults(fileName, data) {
  renderStatus(
    `Review ready: imported ${formatInteger(data.handsImported)} hand${data.handsImported === 1 ? "" : "s"} from ${fileName}.`
  );

  summaryGrid.innerHTML = [
    summaryCard("Site", data.site, data.heroName ? `Hero: ${data.heroName}` : "Hero shown only when one clear name is resolved"),
    summaryCard("Hands", formatInteger(data.handsAnalyzed), `${formatInteger(data.handsSkipped)} skipped`),
    summaryCard("Decisions", formatInteger(data.decisionsAnalyzed), `${formatInteger(data.mistakes)} mistakes flagged`),
    summaryCard("EV Lost", formatSigned(-Math.abs(Number(data.totalEvLost || 0))), "Aggregate avoidable EV gap"),
    summaryCard("Biggest Gap", formatNumber(data.biggestMistakeEv), "Worst single decision"),
    summaryCard("Model", data.modelSource || "-", "Loaded for this review")
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

function renderHallResults(data) {
  const request = data && typeof data.request === "object" && data.request ? data.request : {};
  const summary = data && typeof data.summary === "object" && data.summary ? data.summary : {};
  const cancelled = !!(data && data.cancelled);
  const cancelledNote = cancelled ? " · CANCELLED (partial data)" : "";

  renderHallStatus(
    `Hall ready: ${formatInteger(summary.handsPlayed)} hands, ${formatSigned(summary.heroNetChips)} chips, ${formatSigned(summary.heroBbPer100)} bb/100${cancelledNote}.`
  );

  if (hallKpiGrid) {
    hallKpiGrid.innerHTML = "";
    const perHand = Array.isArray(summary.perHandHeroNet) ? summary.perHandHeroNet.map(Number) : [];
    const tail = perHand.slice(-20);
    const cumTail = [];
    let acc = 0;
    tail.forEach(v => { acc += v; cumTail.push(acc); });
    const kpiNet = document.createElement("div"); kpiNet.className = "kpi-card"; hallKpiGrid.appendChild(kpiNet);
    const kpiBb  = document.createElement("div"); kpiBb.className  = "kpi-card"; hallKpiGrid.appendChild(kpiBb);
    const kpiWlt = document.createElement("div"); kpiWlt.className = "kpi-card"; hallKpiGrid.appendChild(kpiWlt);
    SicfunCharts.renderKpiCard(kpiNet, {
      label: "Net Chips", value: formatSigned(summary.heroNetChips),
      note: `${formatInteger(summary.handsPlayed)} hands`,
      sparklineValues: cumTail.length > 1 ? cumTail : null
    });
    SicfunCharts.renderKpiCard(kpiBb, {
      label: "bb/100", value: formatSigned(summary.heroBbPer100),
      note: `${formatInteger(summary.retrains)} retrains`
    });
    SicfunCharts.renderKpiCard(kpiWlt, {
      label: "W / L / T",
      value: `${formatInteger(summary.heroWins)} / ${formatInteger(summary.heroLosses)} / ${formatInteger(summary.heroTies)}`,
      note: summary.modelId ? `Model: ${summary.modelId}` : "Uniform fallback"
    });
  }

  if (hallChartChips) {
    const perHand = Array.isArray(summary.perHandHeroNet) ? summary.perHandHeroNet.map(Number) : [];
    if (perHand.length >= 2) {
      const xs = perHand.map((_, i) => i + 1);
      const ys = [];
      let acc = 0;
      perHand.forEach(v => { acc += v; ys.push(acc); });
      SicfunCharts.renderLine(hallChartChips, {xs, ys}, {yLabel: "cumulative chips", height: 220});
    } else {
      hallChartChips.innerHTML = `<p class="section-note">No per-hand data returned for this run.</p>`;
    }
  }

  const actionEntries = objectEntries(summary.actionCounts);
  if (hallChartActionDonut) {
    if (actionEntries.length > 0) {
      SicfunCharts.renderDonut(hallChartActionDonut, {
        labels: actionEntries.map(([k]) => k),
        values: actionEntries.map(([, v]) => Number(v))
      }, {title: "Action distribution"});
    } else {
      hallChartActionDonut.innerHTML = `<p class="section-note">No hero actions recorded.</p>`;
    }
  }
  if (hallChartWlt) {
    SicfunCharts.renderStackedBarH(hallChartWlt, {
      labels: ["W", "L", "T"],
      values: [summary.heroWins || 0, summary.heroLosses || 0, summary.heroTies || 0]
    }, {title: "Outcome split"});
  }

  if (hallChartVillainBar) {
    const villainEntries = objectEntries(summary.perVillainNetChips);
    if (villainEntries.length > 0) {
      SicfunCharts.renderBarH(hallChartVillainBar, {
        labels: villainEntries.map(([k]) => k),
        values: villainEntries.map(([, v]) => Number(v)),
        signed: true
      }, {title: "Per-villain chip flow"});
    } else {
      hallChartVillainBar.innerHTML = `<p class="section-note">No per-villain breakdown returned.</p>`;
    }
  }

  if (hallChartEquityHist) {
    const equities = Array.isArray(summary.heroDecisionEquities) ? summary.heroDecisionEquities.map(Number) : [];
    if (equities.length > 0) {
      SicfunCharts.renderHistogram(hallChartEquityHist, {values: equities}, {bucketCount: 10, min: 0, max: 1, height: 200});
    } else {
      hallChartEquityHist.innerHTML = `<p class="section-note">No hero decision equities recorded.</p>`;
    }
  }

  if (hallChartMatrix) {
    const villains = objectEntries(summary.perVillainNetChips);
    if (villains.length > 0) {
      const cells = {};
      cells["Net chips"] = {};
      villains.forEach(([name, chips]) => { cells["Net chips"][name] = Number(chips); });
      SicfunCharts.renderMatrix(hallChartMatrix, {
        rows: ["Net chips"],
        cols: villains.map(([k]) => k),
        cells
      }, {title: "Villain × result matrix"});
    } else {
      hallChartMatrix.innerHTML = `<p class="section-note">No data for matrix.</p>`;
    }
  }

  if (hallOutputList) {
    const files = Array.isArray(summary.outputFiles) ? summary.outputFiles : [];
    hallOutputList.innerHTML = files.length > 0
      ? files.map(file => `
          <article class="opponent-card">
            <div class="decision-head">
              <h3 class="decision-title">Output File</h3>
              <p class="card-meta">written</p>
            </div>
            <p class="opponent-meta">${escapeHtml(file)}</p>
          </article>
        `).join("")
      : `<p class="section-note">${summary.outDir ? `Run directory: ${escapeHtml(summary.outDir)}` : "No output files were reported."}</p>`;
  }

  if (hallResults) hallResults.classList.remove("hidden");
}

function summaryCard(label, value, note) {
  // All three fields escape uniformly. Previously `note` was interpolated
  // raw, forcing every caller to pre-escape if it embedded user data --
  // easy to forget when adding a new card, which would silently turn a
  // future `summaryCard("Site", data.site, data.note)` into an XSS sink.
  // Uniform escaping closes the footgun.
  return `
    <article class="summary-card">
      <p class="summary-label">${escapeHtml(label)}</p>
      <p class="summary-value">${escapeHtml(value)}</p>
      <p class="summary-note">${escapeHtml(note)}</p>
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
  // Both fields escape uniformly. Callers today pass numeric strings from
  // formatSigned/formatPercent (no HTML chars to escape), so this is a
  // no-op on current data -- but it closes the same footgun as summaryCard:
  // a future decisionRow("Note", decision.someTextField) won't silently
  // become an XSS sink just because the helper escaped one field and not
  // the other.
  return `
    <div class="decision-row">
      <strong>${escapeHtml(label)}</strong>
      <span>${escapeHtml(value)}</span>
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

function selectedVillainPool() {
  return Array.from(document.querySelectorAll('input[name="villain-pool"]:checked'))
    .map(input => input.value)
    .filter(Boolean);
}

function numericValue(element, fallback) {
  const parsed = Number(element && "value" in element ? element.value : fallback);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function objectEntries(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? Object.entries(value)
    : [];
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

function formatFileSize(bytes) {
  const size = Number(bytes) || 0;
  if (size >= 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  if (size >= 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${size} B`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

// ========== Playing Hall v2 — validation, presets, recent-runs, progress ==========

const hallNumericInputs = [
  hallHandsInput, hallTableCountInput, hallExplorationRateInput, hallRaiseSizeInput,
  hallLearnEveryInput, hallLearningWindowInput, hallBunchingTrialsInput, hallEquityTrialsInput,
  hallSeedInput
];

function validateField(input) {
  if (!input) return true;
  // Skip validation of disabled inputs: their value isn't submitted in
  // the same form-data shape (form submits skip disabled fields), and
  // for the hall-seed case specifically the value is explicitly
  // bypassed when hallRandomSeedInput is checked (a random Long is
  // generated at submit time). A stale step-mismatching seed value
  // sitting in the disabled input would otherwise block the Run
  // button even though the value gets discarded.
  if (input.disabled) {
    input.classList.remove("invalid");
    const err = input.parentElement && input.parentElement.querySelector(".field-error");
    if (err) err.remove();
    return true;
  }
  const value = Number(input.value);
  const min = input.min !== "" ? Number(input.min) : -Infinity;
  const max = input.max !== "" ? Number(input.max) : Infinity;
  // Also reject step mismatches (e.g. 1.5 in a step=1 input). The
  // browser surfaces these via validity.stepMismatch, but only when
  // checkValidity() is called; reading the value directly doesn't
  // gate them. Without this guard a user pasting '1.5' into 'hands'
  // (step=1) would pass the frontend but get a 400 'must be an integer'
  // from optionalInt -- worse, the field would render as valid until
  // the server bounced it.
  const stepOk = !input.validity || !input.validity.stepMismatch;
  const ok = Number.isFinite(value) && value >= min && value <= max && stepOk;
  input.classList.toggle("invalid", !ok);
  let err = input.parentElement.querySelector(".field-error");
  if (!ok) {
    if (!err) {
      err = document.createElement("span");
      err.className = "field-error";
      input.parentElement.appendChild(err);
    }
    // Use the input's step to surface a specific message: step=1 reads as
    // 'Whole number required'; everything else surfaces the actual step so
    // the user sees what precision is expected (e.g. step=0.05 -> 'Use
    // multiples of 0.05'). Default fallback names the constraint without
    // a value when input.step is empty.
    const stepValue = input.step;
    const stepHint =
      stepValue === "1" ? "Whole number required" :
      stepValue ? `Use multiples of ${stepValue}` :
      "Value does not match the step";
    err.textContent =
      !Number.isFinite(value) ? "Number required" :
      value < min ? `Min ${min}` :
      value > max ? `Max ${max}` :
      stepHint;
    // Force the parent <details> open if the user collapsed the section
    // that contains an invalid field. Otherwise the submit button is
    // disabled (because validateHallForm returns false) but the red
    // highlight is hidden, so the user sees a stuck Run button with no
    // visible cause -- pure mystery. .closest finds the nearest
    // ancestor <details> regardless of nesting depth.
    const parentDetails = input.closest("details");
    if (parentDetails) parentDetails.open = true;
  } else if (err) {
    err.remove();
  }
  return ok;
}

function validateVillainPool() {
  const ok = selectedVillainPool().length > 0;
  document.querySelectorAll(".hall-pool .chip-check span").forEach(el => el.classList.toggle("invalid", !ok));
  // Same auto-open as validateField: if the villain section is collapsed
  // and the pool has no entries, the user sees a stuck Run button with
  // no visible 'why'. Force the villain <details> open so the red chips
  // are visible. Look up via the first chip's nearest ancestor so the
  // selector stays cheap and survives DOM restructuring.
  if (!ok) {
    const firstChip = document.querySelector(".hall-pool");
    const parentDetails = firstChip && firstChip.closest("details");
    if (parentDetails) parentDetails.open = true;
  }
  return ok;
}

function validateHallForm() {
  const fieldsOk = hallNumericInputs.every(validateField);
  const poolOk = validateVillainPool();
  const allOk = fieldsOk && poolOk;
  if (hallSubmitButton) {
    const locked = requiresPlatformSignIn() && !authState.authenticated;
    hallSubmitButton.disabled = !allOk || locked;
  }
  return allOk;
}

function wireHallValidation() {
  if (!hallForm) return;
  hallNumericInputs.forEach(input => {
    if (input) input.addEventListener("input", () => validateHallForm());
  });
  document.querySelectorAll('input[name="villain-pool"]').forEach(el => {
    el.addEventListener("change", () => validateHallForm());
  });
  validateHallForm();
}

// ----- Presets -----

const HALL_PRESETS = [
  {id: "smoke",          label: "Smoke 50",       config: {hands: 50,   tableCount: 1, playerCount: 2, heroStyle: "adaptive", heroPosition: "Button", gtoMode: "exact", villainPool: ["gto"], heroExplorationRate: 0, raiseSize: 2.5, bunchingTrials: 20, equityTrials: 120, learnEveryHands: 0, learningWindowSamples: 0, seed: 42, saveReviewHandHistory: false, fullRing: false}},
  {id: "standard",       label: "Standard 240",   config: {hands: 240,  tableCount: 2, playerCount: 6, heroStyle: "adaptive", heroPosition: "Button", gtoMode: "exact", villainPool: ["tag","lag","gto"], heroExplorationRate: 0, raiseSize: 2.5, bunchingTrials: 40, equityTrials: 240, learnEveryHands: 0, learningWindowSamples: 200, seed: 42, saveReviewHandHistory: false, fullRing: false}},
  {id: "long",           label: "Long 1000",      config: {hands: 1000, tableCount: 2, playerCount: 6, heroStyle: "adaptive", heroPosition: "Button", gtoMode: "exact", villainPool: ["tag","lag","gto"], heroExplorationRate: 0, raiseSize: 2.5, bunchingTrials: 40, equityTrials: 240, learnEveryHands: 100, learningWindowSamples: 200, seed: 42, saveReviewHandHistory: false, fullRing: false}},
  {id: "gto-vs-tag",     label: "GTO vs TAGs",    config: {hands: 500,  tableCount: 2, playerCount: 6, heroStyle: "gto",      heroPosition: "Button", gtoMode: "exact", villainPool: ["tag"], heroExplorationRate: 0, raiseSize: 2.5, bunchingTrials: 40, equityTrials: 240, learnEveryHands: 0, learningWindowSamples: 0, seed: 42, saveReviewHandHistory: false, fullRing: false}},
  {id: "adaptive-mixed", label: "Adaptive mixed", config: {hands: 500,  tableCount: 2, playerCount: 6, heroStyle: "adaptive", heroPosition: "Button", gtoMode: "exact", villainPool: ["tag","lag","nit","station"], heroExplorationRate: 0.05, raiseSize: 2.5, bunchingTrials: 40, equityTrials: 240, learnEveryHands: 100, learningWindowSamples: 200, seed: 42, saveReviewHandHistory: false, fullRing: false}},
  {id: "strategic",      label: "Strategic",      config: {hands: 500,  tableCount: 2, playerCount: 6, heroStyle: "strategic", heroPosition: "Button", gtoMode: "exact", villainPool: ["maniac","station"], heroExplorationRate: 0, raiseSize: 2.5, bunchingTrials: 40, equityTrials: 240, learnEveryHands: 0, learningWindowSamples: 0, seed: 42, saveReviewHandHistory: false, fullRing: false}},
  {id: "heads-up",       label: "Heads-up GTO",   config: {hands: 500,  tableCount: 1, playerCount: 2, heroStyle: "gto",      heroPosition: "Button", gtoMode: "exact", villainPool: ["gto"], heroExplorationRate: 0, raiseSize: 2.5, bunchingTrials: 40, equityTrials: 240, learnEveryHands: 0, learningWindowSamples: 0, seed: 42, saveReviewHandHistory: false, fullRing: false}}
];

function applyHallConfig(config) {
  if (!config) return;
  if (hallHandsInput && config.hands != null) hallHandsInput.value = config.hands;
  if (hallTableCountInput && config.tableCount != null) hallTableCountInput.value = config.tableCount;
  if (hallPlayerCountSelect && config.playerCount != null) hallPlayerCountSelect.value = String(config.playerCount);
  if (hallHeroStyleSelect && config.heroStyle) hallHeroStyleSelect.value = config.heroStyle;
  if (hallHeroPositionSelect && config.heroPosition) hallHeroPositionSelect.value = config.heroPosition;
  if (hallGtoModeSelect && config.gtoMode) hallGtoModeSelect.value = config.gtoMode;
  if (hallExplorationRateInput && config.heroExplorationRate != null) hallExplorationRateInput.value = config.heroExplorationRate;
  if (hallRaiseSizeInput && config.raiseSize != null) hallRaiseSizeInput.value = config.raiseSize;
  if (hallLearnEveryInput && config.learnEveryHands != null) hallLearnEveryInput.value = config.learnEveryHands;
  if (hallLearningWindowInput && config.learningWindowSamples != null) hallLearningWindowInput.value = config.learningWindowSamples;
  if (hallBunchingTrialsInput && config.bunchingTrials != null) hallBunchingTrialsInput.value = config.bunchingTrials;
  if (hallEquityTrialsInput && config.equityTrials != null) hallEquityTrialsInput.value = config.equityTrials;
  if (hallSeedInput && config.seed != null) hallSeedInput.value = config.seed;
  if (hallSaveReviewInput) hallSaveReviewInput.checked = Boolean(config.saveReviewHandHistory);
  if (hallFullRingInput) hallFullRingInput.checked = Boolean(config.fullRing);
  if (Array.isArray(config.villainPool)) {
    document.querySelectorAll('input[name="villain-pool"]').forEach(input => {
      input.checked = config.villainPool.includes(input.value);
    });
  }
  validateHallForm();
}

function renderPresetBar() {
  if (!hallPresetBar) return;
  hallPresetBar.innerHTML = HALL_PRESETS.map(p =>
    `<button type="button" class="preset-button" data-preset="${p.id}">${escapeHtml(p.label)}</button>`
  ).join("");
  hallPresetBar.querySelectorAll(".preset-button").forEach(btn => {
    btn.addEventListener("click", () => {
      const preset = HALL_PRESETS.find(p => p.id === btn.dataset.preset);
      if (preset) applyHallConfig(preset.config);
    });
  });
}

// ----- Recent runs -----

const RECENT_RUNS_KEY = "sicfun.hall.recentRuns";
const RECENT_RUNS_MAX = 20;

function readRecentRuns() {
  try {
    const raw = localStorage.getItem(RECENT_RUNS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) { return []; }
}

function writeRecentRuns(entries) {
  try { localStorage.setItem(RECENT_RUNS_KEY, JSON.stringify(entries.slice(0, RECENT_RUNS_MAX))); }
  catch (_) { /* quota or private mode - ignore */ }
}

function clearRecentRuns() {
  try { localStorage.removeItem(RECENT_RUNS_KEY); }
  catch (_) { /* private mode - ignore */ }
}

function pushRecentRun(request, summary) {
  const entry = {
    timestamp: Date.now(),
    request,
    summary: {
      handsPlayed: summary.handsPlayed,
      heroNetChips: summary.heroNetChips,
      heroBbPer100: summary.heroBbPer100,
      heroWins: summary.heroWins,
      heroLosses: summary.heroLosses,
      heroTies: summary.heroTies
    }
  };
  const existing = readRecentRuns();
  writeRecentRuns([entry, ...existing]);
  renderRecentRuns();
}

function renderRecentRuns() {
  if (!hallRecentList || !hallRecentCount) return;
  const entries = readRecentRuns();
  hallRecentCount.textContent = entries.length;
  if (entries.length === 0) {
    hallRecentList.innerHTML = `<p class="section-note">No runs yet. Launch one from the form.</p>`;
    return;
  }
  // Header bar above the list: "N entries" + Clear all. The per-entry ×
  // button handles one-off removal; this gives a bulk path for the no-
  // auth deployments that don't have a logout-clear, and lets a user
  // wipe the list without 20 individual clicks. requestConfirm prompt
  // is a native dialog to keep the codebase free of a modal dependency.
  const headerBar = `
    <div class="recent-runs-header">
      <span class="section-note">${entries.length} ${entries.length === 1 ? "run" : "runs"} stored locally</span>
      <button type="button" class="button button-secondary" id="hall-recent-clear">Clear all</button>
    </div>
  `;
  hallRecentList.innerHTML = headerBar + entries.map((entry, idx) => {
    const ts = new Date(entry.timestamp).toLocaleString();
    const pool = Array.isArray(entry.request.villainPool) ? entry.request.villainPool.join(", ") : "-";
    return `
      <article class="recent-run">
        <div class="recent-run-head">
          <span class="recent-run-ts">${escapeHtml(ts)}</span>
          <span class="recent-run-actions">
            <button type="button" class="button button-secondary" data-recent-index="${idx}">Load</button>
            <button type="button" class="button button-secondary recent-run-remove" data-recent-remove="${idx}" aria-label="Remove this run from history" title="Remove">×</button>
          </span>
        </div>
        <p class="recent-run-meta">
          ${escapeHtml(entry.request.heroStyle || "-")} &middot; ${formatInteger(entry.request.hands)} hands &middot;
          ${formatSigned(entry.summary.heroNetChips)} chips &middot; [${escapeHtml(pool)}] &middot; seed ${escapeHtml(entry.request.seed)}
        </p>
      </article>
    `;
  }).join("");
  hallRecentList.querySelectorAll("button[data-recent-index]").forEach(btn => {
    btn.addEventListener("click", () => {
      const entry = entries[Number(btn.dataset.recentIndex)];
      if (entry) applyHallConfig(entry.request);
    });
  });
  // Per-entry remove: localStorage is bounded at RECENT_RUNS_MAX so old
  // entries age out automatically, but a user who ran a one-off
  // exploratory config they don't want to remember had no way to drop
  // it before it cycled off. Remove + re-render rather than just hide
  // so the entry doesn't reappear on the next renderRecentRuns.
  hallRecentList.querySelectorAll("button[data-recent-remove]").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.recentRemove);
      const next = entries.filter((_, i) => i !== idx);
      writeRecentRuns(next);
      renderRecentRuns();
    });
  });
  // Clear all: complement to per-entry remove for users who want to
  // wipe the whole history (e.g., no-auth deployments where logout-
  // clear isn't an option). Use the native window.confirm rather than
  // building a modal so the codebase stays dependency-free; the prompt
  // text names the count + that the data is local so the user knows
  // what they're discarding.
  const clearBtn = document.getElementById("hall-recent-clear");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      const noun = entries.length === 1 ? "1 stored hall run" : `all ${entries.length} stored hall runs`;
      const proceed = window.confirm(`Discard ${noun} from this browser?`);
      if (proceed) {
        clearRecentRuns();
        renderRecentRuns();
      }
    });
  }
}

// ----- Progress / elapsed timer / cancel -----

let hallElapsedTimer = null;
let hallActiveJobId = null;
let hallActiveStartedAt = 0;

function startHallElapsed(jobId) {
  hallActiveJobId = jobId;
  hallActiveStartedAt = Date.now();
  if (hallProgress) hallProgress.classList.remove("hidden");
  if (hallCancelButton) {
    hallCancelButton.disabled = false;
    hallCancelButton.textContent = "Cancel Run";
  }
  stopHallElapsed();
  hallElapsedTimer = window.setInterval(tickHallElapsed, 1000);
  tickHallElapsed();
}

function stopHallElapsed() {
  if (hallElapsedTimer) { window.clearInterval(hallElapsedTimer); hallElapsedTimer = null; }
}

function tickHallElapsed() {
  if (!hallElapsed) return;
  const sec = Math.floor((Date.now() - hallActiveStartedAt) / 1000);
  const mm = String(Math.floor(sec / 60)).padStart(2, "0");
  const ss = String(sec % 60).padStart(2, "0");
  hallElapsed.textContent = `Elapsed ${mm}:${ss}`;
}

function finishHallProgress() {
  stopHallElapsed();
  hallActiveJobId = null;
  if (hallProgress) hallProgress.classList.add("hidden");
}

if (hallCancelButton) {
  hallCancelButton.addEventListener("click", async () => {
    if (!hallActiveJobId) return;
    hallCancelButton.disabled = true;
    hallCancelButton.textContent = "Cancelling...";
    try {
      const response = await fetchWithTimeout(`/api/playing-hall/jobs/${encodeURIComponent(hallActiveJobId)}`, {
        method: "DELETE",
        credentials: "same-origin",
        headers: jsonHeaders(true)
      });
      if (response.status === 404) {
        renderHallStatus("Job no longer available.");
        finishHallProgress();
      } else if (response.status === 409) {
        // Already terminal -- the polling loop is about to (or already did)
        // resolve with whatever the server's final status was, so let it own
        // the user-facing message. Don't reset the button here: it stays
        // disabled until finishHallProgress hides the whole progress card.
      } else if (response.ok) {
        // 200/202: server accepted the cancel. The job is now winding down
        // server-side; the polling loop will see status="cancelled" within
        // ~pollAfterMs (capped at 5s) and trigger finishHallProgress. Give
        // the user IMMEDIATE feedback so they know their click landed and
        // the UI didn't freeze on "Cancelling..." for the poll window.
        renderHallStatus("Cancel accepted. Finishing in-flight hands...");
      } else {
        await maybeReauthOn401(response);
        // Run the response body through formatErrorMessage so 429 / 503
        // responses surface their Retry-After info ('... Try again in N
        // seconds.') the same way the analyze + hall submit error paths
        // do. Cancel goes through the same job-status rate-limit bucket
        // as polling so a 429 is plausible if a script is hammering.
        const body = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));
        renderHallStatus(`Cancel failed: ${formatErrorMessage(response, body)}`);
        // Restore the button so the user can retry. Without this the user
        // is stuck staring at a disabled "Cancelling..." button after a
        // transient 500 / 503 / network-layer failure.
        hallCancelButton.disabled = false;
        hallCancelButton.textContent = "Cancel Run";
      }
    } catch (error) {
      renderHallStatus(`Cancel failed: ${describeFetchError(error)}`);
      hallCancelButton.disabled = false;
      hallCancelButton.textContent = "Cancel Run";
    }
  });
}
