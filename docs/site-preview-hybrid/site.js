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

// Polling deadline for both the analyze and hall job pollers. Initial
// value is 16 minutes -- 1 minute of slack over the default
// PLAYING_HALL_TIMEOUT_MS (15 min) so the frontend deadline check at
// the top of the poll loop doesn't throw "timed out" the moment the
// server's own timeout fires. Without slack the two deadlines line up
// exactly, and the queue wait between submit and the worker start
// (which the server measures from) means the server can return a
// terminal status (completed OR timeout-failed) microseconds after
// the frontend's pre-poll deadline check has already given up. The
// user then sees a misleading client-side "timed out" instead of the
// server's actual outcome. ANALYZE jobs use a 2-min server timeout by
// default so they have 14 min of slack and aren't affected.
//
// probeServerLimits() at boot reads /api/health.analysisTimeoutMs and
// /api/health.playingHallTimeoutMs and EXTENDS this value (never
// shortens) so an operator who raises PLAYING_HALL_TIMEOUT_MS beyond
// 15 min also gets a matching frontend deadline without a frontend
// rebuild. Pure extension: a network failure on the probe leaves the
// 16-min default in place, which is still correct for the shipped
// server defaults.
let maxPollWaitMs = 16 * 60 * 1000;

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
// fetch fails well before maxPollWaitMs (16 min default) is exhausted.
// Without this, an OS-level TCP idle timeout (minutes) could leave one
// poll waiting while the deadline check at the top of the loop never
// gets a chance to re-evaluate.
const POLL_FETCH_TIMEOUT_MS = 30_000;

function fetchWithTimeout(url, options, timeoutMs = SINGLE_SHOT_FETCH_TIMEOUT_MS) {
  // Three-tier feature detection so the timeout actually fires on
  // every browser that supports the underlying AbortController
  // primitive, not just the ones with the modern sugar.
  //
  // Tier 1: AbortSignal.timeout (Chrome 103+, Firefox 100+, Safari
  // 16+). One-liner; throws DOMException name='TimeoutError' that
  // describeFetchError translates to actionable text. Use it when
  // available.
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
    return fetch(url, { ...options, signal: AbortSignal.timeout(timeoutMs) });
  }
  // Tier 2: AbortController + setTimeout (Chrome 66+, Firefox 57+,
  // Safari 11.1+ -- 2018+ browsers). The aborted fetch rejects with
  // DOMException name='AbortError' which describeFetchError also
  // recognizes. Wider support than AbortSignal.timeout by ~4 years.
  // Without this tier, every fetch on Chrome 66..102 / Firefox
  // 57..99 / Safari 11.1..15 ran without a timeout (could hang on a
  // dropped TCP socket for the OS-level idle timeout, often minutes).
  if (typeof AbortController !== "undefined") {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
    return fetch(url, { ...options, signal: controller.signal })
      .finally(() => window.clearTimeout(timeoutId));
  }
  // Tier 3: truly ancient browser without AbortController. Best
  // effort: no timeout. Below this floor the page also doesn't
  // support fetch credentials, ESM, async/await, etc. -- so the
  // bundled UI wouldn't fully boot anyway. Documented in the
  // deployment doc's browser-support floor.
  return fetch(url, options);
}

// Translate a thrown async error into a user-facing message. The cases
// the user benefits from naming explicitly:
//   - AbortSignal.timeout throws DOMException with name='TimeoutError' and
//     a terse "signal timed out" message that reads like a bug to a non-
//     developer. Translate to actionable language.
//   - A network-layer failure (server unreachable, DNS down, offline)
//     throws TypeError "Failed to fetch" in Chrome / "NetworkError when
//     attempting to fetch resource" in Firefox -- match either substring.
//   - file.text() throws DOMException with name='NotReadableError' when
//     the picked file has gone away (renamed, moved, deleted, removable
//     drive unplugged) since the user clicked Choose File. The default
//     Chrome message reads as a developer log entry; translate to
//     actionable "pick the file again" language.
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
  if (error.name === "NotReadableError") {
    return "the file could not be read -- it may have been moved or deleted since you picked it; pick the file again and retry";
  }
  return error instanceof Error ? error.message : "unknown error";
}

let authState = normalizeAuthState({});

// boot() is async and runs network probes AFTER a batch of synchronous
// DOM setup (mirrorHelpDataToAriaLabel, renderPresetBar,
// renderRecentRuns, wireHallValidation, syncRandomSeedState). Any
// throw -- sync from the setup batch or async from the network probes
// -- without this catch becomes an unhandled rejection (browser
// console only, invisible to the user) and the page silently loses
// the half of its UI that boot() was supposed to wire up.
// This is primarily defense-in-depth -- every realistic failure path
// today is ALREADY guarded internally: readRecentRuns /
// writeRecentRuns / clearRecentRuns each wrap localStorage in
// try/catch so Safari-private-mode SecurityError and Firefox-quota
// QuotaExceededError get swallowed there before they can reach
// boot(); refreshAuthState / probeServerLimits each have their own
// try/catch around fetchWithTimeout so a network/timeout/CSP failure
// surfaces as a no-op rather than a thrown rejection at the
// Promise.all boundary; the DOM-touching functions all early-return
// on null element references rather than dereferencing. So the
// realistic "this catch fires today" scenarios are narrow:
//   - A future refactor inside any of the above functions strips
//     one of those internal try/catches or removes a null-guard,
//     and the previously-swallowed error escapes for the first time.
//   - A browser bug or extension injects a synchronous throw into
//     a DOM API (querySelectorAll, addEventListener) the page
//     trusts -- rare but possible, particularly with aggressive
//     content-blockers that monkey-patch fetch / DOM APIs.
//   - A pathological mismatch between site.js and index.html where
//     an element id the JS dereferences without a null-guard moved
//     in the HTML (this codebase pretty consistently null-guards,
//     but a future addition could miss one).
// Without the catch, the user-visible symptom is "loaded the page,
// the sign-in panel never appeared, no error in sight". With the
// catch the user sees the #page-init-error banner in index.html
// directing them to reload, and the console still has the full
// stack for operator/dev triage. The banner mirrors the noscript-
// notice's visual treatment so the two failure modes ("JS disabled"
// vs "JS ran but threw") read identically as page-level warnings.
boot().catch(showBootError);

function showBootError(error) {
  // Surface the technical detail to the developer/operator via the
  // browser console; the user sees only the generic banner.
  // Intentionally no /api/client-error report: this codebase doesn't
  // ship a client-error sink endpoint (by design), and a CSP /
  // network error in boot() itself could be exactly the thing
  // preventing the report from getting through.
  if (typeof console !== "undefined" && typeof console.error === "function") {
    console.error("SICFUN: page initialization failed", error);
  }
  const banner = document.getElementById("page-init-error");
  // No-op silently if the banner element is absent (HTML/JS partial-
  // deploy mismatch). The fallback under that scenario is the
  // existing "silently half-broken page" symptom -- no worse than
  // pre-catch behavior, just lacking the visible banner this commit
  // adds. The console.error above still fires regardless.
  if (banner) banner.classList.remove("hidden");
}

if (form && fileInput && siteSelect && heroInput) {
  form.addEventListener("submit", async event => {
    // preventDefault skips the form's HTML POST fallback (method="post"
    // action="/api/analyze-hand-history" on the #hand-upload-form element
    // in index.html -- grep for `id="hand-upload-form"`). The
    // HTML attributes are the JS-failed-to-load belt-and-braces -- if
    // this handler never runs (bundle 404, CSP block), the browser
    // default-submits to a real endpoint and the analyze service 415s
    // the form-urlencoded body, keeping the upload from leaking via a
    // GET-with-query-params fallback. When the JS path IS live we want
    // the proper application/json XHR via fetchWithTimeout below, not
    // the form-default POST also firing for the same submit event.
    // See the comment block above `authForm.addEventListener("submit",
    // ...)` later in this file for the full two-layer defense
    // rationale; same shape applies here.
    event.preventDefault();

    if (requiresPlatformSignIn() && !authState.authenticated) {
      renderStatus("Sign in to queue a review job for this deployment.");
      reviewResults.classList.add("hidden");
      return;
    }

    // Defensive guard against re-entry. setSubmitting(true) disables the
    // submit BUTTON before the first await, but a rapid Enter key in
    // any focused form field (file input, site select, hero name input)
    // ALSO fires the submit event -- the button-disabled state only
    // blocks button-click submits, not keyboard submits. Without this
    // guard a user mashing Enter would fire two parallel submit
    // handlers, each posting its own /api/analyze-hand-history and
    // each polling its own jobId; the frontend would track only the
    // last one and the first job becomes a ghost server-side. Check
    // submitButton.disabled (which setSubmitting flips synchronously
    // before the first await) and bail out if a submit is already in
    // flight -- the user gets visual feedback from the still-disabled
    // button and the existing 'Submitting <file>' status message,
    // rather than a confusing duplicate-job condition.
    if (submitButton && submitButton.disabled) {
      return;
    }

    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      renderStatus("Choose a `.txt` hand-history export to start the review.");
      reviewResults.classList.add("hidden");
      // Move focus to the file input so a keyboard / screen-reader user
      // can act on the announced status immediately. Without the focus
      // move, focus stays on the submit button after the renderStatus
      // call and the user has to Shift-Tab back through the hero-name
      // and site fields to reach the input the announcement is asking
      // them to fix. The aria-live region announces WHAT to fix; this
      // focus move puts the cursor on WHERE to fix it, collapsing the
      // announce-then-act loop into one step. preventScroll omitted
      // because the file input sits above the submit button on the same
      // card -- the natural scroll lands the user on it. Same focus-
      // restoration pattern as the post-logout `authEmail.focus()` in
      // logout() and the post-row-remove `hallSubmitButton.focus()` in
      // the renderRecentRuns load-handler; mirrors the codebase's
      // existing "move focus to the next action target on every state
      // transition" convention. (Line numbers intentionally omitted --
      // an earlier version of this comment cited "line 1378" and "line
      // 2936" anchors that drifted 82 and 102 lines respectively as
      // the file grew; symbol-name references survive that rot.)
      fileInput.focus();
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
      // Same focus-to-file-input restoration as the !file case above --
      // the announced action ("Trim or split the hand history and try
      // again") presupposes the user re-picks a different file via the
      // file input, so keyboard/SR users land on that widget directly.
      fileInput.focus();
      return;
    }

    if (file.size === 0) {
      // The server also rejects empty handHistoryText (post-trim), but a
      // synchronous local check beats round-tripping a 400. Common cause:
      // the user picked the wrong file from a "Save As" template that
      // left only an empty placeholder.
      renderStatus(`${file.name} is empty. Pick a hand-history export with at least one hand.`);
      reviewResults.classList.add("hidden");
      // Same focus-to-file-input restoration as the two early-return
      // paths above -- the "Pick a hand-history export with at least
      // one hand" message presupposes the user re-picks via the file
      // input, so keyboard/SR users land on that widget directly.
      fileInput.focus();
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
    setTitleStatus(`Submitting ${file.name}`);
    reviewResults.classList.add("hidden");
    // Capture the submit-click timestamp so renderResults can report
    // the review duration in the completion message. Same rationale as
    // the hall flow's hallActiveStartedAt: the analyze form has no
    // visible elapsed timer, so without this the user has no signal
    // of how long the review took -- a question that matters when
    // operators are tuning ANALYSIS_TIMEOUT_MS or comparing review
    // latency across deployments. Local to the handler closure so it
    // doesn't leak across concurrent submits (which the re-entry
    // guard prevents anyway, but local scope is the natural fit).
    const analyzeStartedAt = Date.now();

    let resultReady = false;
    let resultFailed = false;
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
        resultFailed = true;
        return;
      }

      if (body.jobId) {
        const statusUrl = body.statusUrl || response.headers.get("Location");
        if (!statusUrl) {
          renderStatus("Server accepted the upload but did not return a job status URL.");
          resultFailed = true;
          return;
        }

        renderStatus(jobStatusMessage(file.name, body.status));
        const result = await pollAnalysisJob(file.name, statusUrl, body.pollAfterMs);
        renderResults(file.name, result, analyzeStartedAt);
        resultReady = true;
        return;
      }

      renderResults(file.name, body, analyzeStartedAt);
      resultReady = true;
    } catch (error) {
      // Same duration-on-failure treatment as the hall submit catch.
      // The success path appends "(took Xm Ys)" via renderResults;
      // failures didn't have that signal -- a 60-second polling
      // timeout looked identical to an instant validation error in
      // the rendered message. Include the elapsed-since-submit time
      // so the user can distinguish those modes. analyzeStartedAt is
      // captured immediately after setSubmitting(true) so it covers
      // file.text() + POST + poll-loop, matching what the user
      // perceives as "the request" duration.
      const failDurationMs = analyzeStartedAt > 0 ? Date.now() - analyzeStartedAt : 0;
      const failDurationStr = failDurationMs > 0 ? ` (after ${formatDuration(failDurationMs)})` : "";
      renderStatus(`Request failed: ${describeFetchError(error)}${failDurationStr}`);
      resultFailed = true;
    } finally {
      setSubmitting(false);
      // If the review reached a terminal state while the user had the
      // tab backgrounded, leave a completion cue in the title so they
      // spot the outcome in their browser tab list without refocusing
      // the tab: 'Review ready' on success, 'Review failed' when the
      // upload errored, the server rejected the request, or polling
      // threw. Without the failure cue the title would silently reset
      // to default and a user who walked away would have no signal at
      // all that the run ended. The visibilitychange listener clears
      // either title the moment they return -- the renderStatus panel
      // is the source of truth once the user is looking again.
      if (document.hidden) {
        if (resultReady) setTitleStatus("Review ready");
        else if (resultFailed) setTitleStatus("Review failed");
        else setTitleStatus(null);
      } else {
        setTitleStatus(null);
      }
    }
  });
}

if (hallForm) {
  hallForm.addEventListener("submit", async event => {
    // preventDefault skips the form's HTML POST fallback (method="post"
    // action="/api/playing-hall" on the #playing-hall-form element in
    // index.html -- grep for `id="playing-hall-form"`). The hall form
    // has 9+ keyboard-reachable inputs (hands / tables / seed / equity
    // trials / etc.) so an Enter press in ANY of them fires this
    // handler -- without preventDefault, the JS path's JSON XHR would
    // race the form's default POST against the same /api/playing-hall
    // endpoint, doubling the rate-limit + admission cost and racing
    // two CSRF-bearing requests against the same token. The HTML
    // method+action exist as the JS-failed-to-load belt-and-braces
    // (browser default-POSTs to a real endpoint that 415s the form-
    // urlencoded body, keeping the configured hall args off the URL
    // bar / history / Referer where a method-less default-GET would
    // persist them). See the comment block above
    // `authForm.addEventListener("submit", ...)` later in this file
    // for the full two-layer defense rationale.
    event.preventDefault();

    if (requiresPlatformSignIn() && !authState.authenticated) {
      renderHallStatus("Sign in to launch a playing hall run on this deployment.");
      hallResults.classList.add("hidden");
      return;
    }

    // Same re-entry guard as the analyze submit. Hall submits can fire
    // multiple times in rapid succession via the Enter key in any
    // focused form field (the form has 9+ keyboard-reachable inputs
    // including hands/tables/seed numbers); button-disabled blocks
    // button-click submits but not keyboard submits. The hall case is
    // worse than analyze because the random-seed checkbox runs a
    // fresh Math.random() per submit-handler invocation, so two
    // rapid Enter presses with random-seed checked queue two hall
    // runs with DIFFERENT seeds -- not reproducible, double the
    // worker-pool slot consumption, and the frontend tracks only
    // the second.
    if (hallSubmitButton && hallSubmitButton.disabled) {
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
    setTitleStatus("Hall run queued");
    hallResults.classList.add("hidden");

    let runReady = false;
    let runFailed = false;
    // Distinguish a user-initiated cancel from a clean completion so the
    // backgrounded-tab title cue at the bottom of the finally block can
    // report "Hall cancelled" instead of "Hall done". Without this, a
    // cancelled run that lands while the user is on another tab gets a
    // "Hall done" title cue -- semantically wrong: the user told the
    // server to stop, calling that "done" misrepresents what happened.
    let runCancelled = false;
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
        runFailed = true;
        return;
      }

      if (body.jobId) {
        const statusUrl = body.statusUrl || response.headers.get("Location");
        if (!statusUrl) {
          renderHallStatus("Server accepted the hall run but did not return a job status URL.");
          runFailed = true;
          return;
        }

        startHallElapsed(body.jobId);
        renderHallStatus(playingHallJobStatusMessage(body.status));
        const result = await pollPlayingHallJob(statusUrl, body.pollAfterMs);
        // Capture duration once before finishHallProgress (in the
        // finally block) nulls hallActiveStartedAt -- shared between
        // renderHallResults (for the "(took Xm Ys)" status tail) and
        // pushRecentRun (so the Recent runs panel can show duration
        // on each entry). renderHallResults still falls back to its
        // own compute if data.durationMs is missing, so old call
        // patterns continue to work.
        const hallDurationMs = hallActiveStartedAt > 0 ? Date.now() - hallActiveStartedAt : 0;
        renderHallResults(result);
        pushRecentRun(payload, (result && result.summary) || {}, {cancelled: !!(result && result.cancelled), durationMs: hallDurationMs});
        // Branch the terminal state: a result.cancelled return is a
        // user-initiated stop, not a successful completion, and the
        // backgrounded-tab title cue distinguishes them ("Hall
        // cancelled" vs "Hall done"). The Recent runs panel already
        // distinguishes them via the cancelled flag passed above; this
        // is the tab-title-only parity. The poll loop above only
        // returns with result.cancelled when body.status === "cancelled"
        // (grep `pollPlayingHallJob` for the `body.status === "cancelled"`
        // branch -- it's the only path that returns a non-throw result
        // with the cancelled flag set), so this branch is reachable iff
        // the server actually flipped the job to cancelled state.
        // (Line numbers intentionally omitted -- an earlier version of
        // this comment cited "line 1622-1626" which by the time of this
        // fix was the URLSearchParams +-vs-%20 handling comment in the
        // OIDC error path, ~1560 lines stale AND in the wrong section.)
        if (result && result.cancelled) runCancelled = true;
        else runReady = true;
        return;
      }

      renderHallResults(body);
      pushRecentRun(payload, (body && body.summary) || {});
      runReady = true;
    } catch (error) {
      // Include the run duration in the failure message too, parallel
      // to the success path's "(took Xm Ys)" tail. Reading "Playing
      // hall request failed: ... (after 5m 30s)" tells the user
      // immediately whether they hit an instant validation error vs
      // a timeout-after-long-run vs a network blip mid-poll -- three
      // failure modes that look identical without the duration
      // context. hallActiveStartedAt is still set here (finishHallProgress
      // runs in the finally below); guard against 0 for the early-
      // failure paths (sign-in not authenticated, validation rejection,
      // double-submit) that throw before startHallElapsed was called.
      const failDurationMs = hallActiveStartedAt > 0 ? Date.now() - hallActiveStartedAt : 0;
      const failDurationStr = failDurationMs > 0 ? ` (after ${formatDuration(failDurationMs)})` : "";
      renderHallStatus(`Playing hall request failed: ${describeFetchError(error)}${failDurationStr}`);
      runFailed = true;
    } finally {
      setHallSubmitting(false);
      finishHallProgress();
      // Same backgrounded-tab signaling as the analyze submit -- if the
      // hall run reaches a terminal state while the user is looking at
      // another tab, leave a completion cue in the title: 'Hall done'
      // on success, 'Hall failed' when the queue rejected the request
      // or the poll loop threw. Without the failure cue the title
      // would reset to default and a user who walked away would have
      // no signal at all that the run ended. visibilitychange clears
      // either title on return.
      if (document.hidden) {
        if (runReady) setTitleStatus("Hall done");
        else if (runCancelled) setTitleStatus("Hall cancelled");
        else if (runFailed) setTitleStatus("Hall failed");
        else setTitleStatus(null);
      } else {
        setTitleStatus(null);
      }
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

// Browsers use the password input's autocomplete attribute as a strong
// signal to decide whether to OFFER to save the entered password:
//   - "current-password" -> sign-in form, do NOT prompt to save (user is
//     entering an existing credential the browser already knows or will
//     learn from this submit)
//   - "new-password"     -> registration form, DO prompt to save (this is
//     a freshly chosen credential)
// The HTML defaults the field to "current-password" because login is the
// more common action and Enter-to-submit routes to login. But when the
// user clicks Register, the password is a fresh credential and the save
// prompt is the whole point -- flip the attribute just before the fetch
// so Chrome/Edge/Firefox surface their save UI. We flip it back on the
// login path so a register-then-cancel-then-login sequence doesn't leave
// the wrong hint sticky on the input.
function setAuthPasswordIntent(intent) {
  if (authPassword) {
    authPassword.autocomplete = intent === "register" ? "new-password" : "current-password";
  }
}

if (authLoginButton) {
  authLoginButton.addEventListener("click", () => {
    setAuthPasswordIntent("login");
    void submitAuth("/api/auth/login", false);
  });
}

if (authRegisterButton) {
  authRegisterButton.addEventListener("click", () => {
    setAuthPasswordIntent("register");
    void submitAuth("/api/auth/register", true);
  });
}

if (authForm) {
  // Catch the form's implicit submit (Enter in email / password / display-
  // name) and route to submitAuth -- the JSON XHR path the server actually
  // accepts. POST `/api/auth/login` requires `application/json` per the
  // readRequestBody Content-Type check, so the form's default-submit POST
  // would 415 anyway; the JS handler converts it into the proper JSON
  // request that the server will process.
  //
  // The HTML form ALSO carries `method="post" action="/api/auth/login"`
  // (see the long inline rationale comment inside the #auth-form element
  // in index.html -- grep for `id="auth-form"`) as a defense-in-
  // depth fallback for the case where THIS handler never runs because the
  // JS bundle 404s mid-deploy, hits a CSP block, or otherwise fails to
  // load. In that scenario the browser's default Enter-key submit still
  // POSTs to a real endpoint (server then 415s the form-urlencoded body
  // since it isn't application/json), and the password lands in the
  // request body rather than the URL bar / history / Referer header
  // (which a method-less form would expose via the default GET). Earlier
  // versions of this comment claimed the form had no action/method --
  // stale; the HTML attributes have been there for a while and the
  // index.html comment explains the belt-and-braces rationale.
  //
  // preventDefault here skips the form's POST fallback when the JS path
  // IS live, so the same Enter press doesn't fire BOTH the JSON XHR AND
  // the form-default POST against the same endpoint. Default action is
  // login (the most common Enter-press intent); the user can still
  // explicitly click Register if that was their goal.
  authForm.addEventListener("submit", event => {
    event.preventDefault();
    if (authState.authenticationMode !== "users") return;
    if (authState.authenticated) return;
    setAuthPasswordIntent("login");
    void submitAuth("/api/auth/login", false);
  });
}

if (profileForm) {
  // Same two-layer defense as the auth-form submit handler above: the
  // #profile-form element in index.html (grep for `id="profile-form"`)
  // carries `method="post" action="/api/auth/profile"`
  // as a JS-failed-to-load fallback. If the JS bundle 404s mid-deploy,
  // hits a CSP block, or otherwise never wires this handler, the
  // browser's default Enter-key submit still POSTs to a real endpoint --
  // server then 415s the form-urlencoded body (the profile route
  // requires `application/json` per readRequestBody) but the
  // displayName / heroName / preferredSite / timeZone values stay in
  // the request body rather than landing in the URL bar, browser
  // history, or Referer header (which a method-less form would expose
  // via the default GET, persisting identifying info -- display name,
  // poker handle, IANA tz -- into shareable URL artifacts). When the
  // JS path IS live, preventDefault here skips the form's POST fallback
  // so we don't double-submit (JSON XHR via saveProfile + browser-
  // default POST firing for the same Enter press). See the inline
  // rationale comment immediately above the #profile-form element in
  // index.html for the full HTML-side reasoning.
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
  // Run the synchronous, network-independent setup BEFORE awaiting any
  // network probes -- a screen-reader user who Tabs through the page
  // while boot is mid-await would otherwise hit help icons / hall
  // preset buttons / Recent runs entries with their aria-labels still
  // unset. mirrorHelpDataToAriaLabel in particular is pure DOM mutation
  // against the static HTML in index.html and has no reason to wait
  // for /api/health or /api/auth/me to return. renderPresetBar and
  // wireHallValidation are also DOM-only. renderRecentRuns reads
  // localStorage (synchronous) so it too is safe to run pre-await.
  // syncRandomSeedState is just one input.disabled assignment.
  mirrorHelpDataToAriaLabel();
  renderPresetBar();
  renderRecentRuns();
  wireHallValidation();
  syncRandomSeedState();
  // /api/health is unauthenticated and cheap; do it in parallel with the
  // auth probe so a slow auth-state response doesn't delay the upload cap
  // sync. Both Promises are fire-and-forget on error.
  await Promise.all([refreshAuthState(), probeServerLimits()]);
  // renderAuthFlash depends on authState being populated (it checks the
  // current mode + queries authState.providers), so it stays AFTER the
  // refreshAuthState await -- moving it earlier would render the flash
  // against a stale empty authState.
  renderAuthFlash();
}

// Read server limits from /api/health and adopt them on the client side
// so an operator who raises MAX_UPLOAD_BYTES or the job timeouts doesn't
// need a frontend rebuild for the page to honor the new values.
//   - maxUploadBytes -> client-side file-size check (refuses oversize
//     files before reading + uploading them).
//   - analysisTimeoutMs / playingHallTimeoutMs -> extend (never shorten)
//     maxPollWaitMs so the frontend's poll deadline outlasts whichever
//     server timeout would fire later. Pure extension: if the probe
//     fails or the server uses default timeouts, the 16-min default
//     stays in place, which is already correct for the shipped server
//     defaults.
// Falls back silently to the initial 2 MiB upload cap + 16-min poll
// budget if the probe fails -- worst case the frontend is stricter
// than the server, which is strictly safer than the opposite. Uses
// the same AbortSignal timeout as other single-shot fetches.
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
    // Extend the poll deadline so it outlasts the larger of the two
    // server timeouts (analyze vs hall) plus a 1-minute slack. Math.max
    // ensures we never SHORTEN the default 16-min cap -- if both server
    // timeouts are small (defaults), the default stays. A 0 (disabled)
    // server timeout contributes nothing here so the default also
    // stays; we don't want to wait indefinitely on a misconfigured
    // bottomless job.
    const analyzeTimeoutMs = Number(body && body.analysisTimeoutMs);
    const hallTimeoutMs = Number(body && body.playingHallTimeoutMs);
    const slackMs = 60 * 1000;
    const candidates = [maxPollWaitMs];
    if (Number.isFinite(analyzeTimeoutMs) && analyzeTimeoutMs > 0) {
      candidates.push(analyzeTimeoutMs + slackMs);
    }
    if (Number.isFinite(hallTimeoutMs) && hallTimeoutMs > 0) {
      candidates.push(hallTimeoutMs + slackMs);
    }
    maxPollWaitMs = Math.max(...candidates);
  } catch (_) {
    // Probe is best-effort. A network blip leaves the 2 MiB default in
    // place, which still allows legitimate hand-history uploads, AND
    // leaves the 16-min poll deadline in place, which still works for
    // any deployment that hasn't raised PLAYING_HALL_TIMEOUT_MS past
    // its shipped 15-min default.
  } finally {
    // Always refresh the visible size hint -- on probe failure it'll
    // show 'Max 2 MB' (the default), on success it reflects the actual
    // server cap. Either way the user sees the cap that the validation
    // will enforce.
    const hint = document.getElementById("hand-history-file-hint");
    if (hint) {
      hint.textContent = `Max ${formatFileSize(maxUploadFileBytes)} · .txt export from PokerStars, Winamax, or GGPoker`;
    }
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

// Extract retry-after seconds from a server response. The 429 body carries
// retryAfterSeconds explicitly (the precise server-computed wait), and both
// 429 and 503 also set the Retry-After header (per RFC 7231 sec 7.1.3 +
// 6.6.4). Prefer the body field when present because it survives proxies
// that strip headers; fall back to the header for 503 (which has no body
// field). Returns null when neither is present or parseable so callers can
// pick their own default.
function parseRetryAfterSeconds(response, body) {
  if (body && Number.isFinite(Number(body.retryAfterSeconds))) {
    const fromBody = Number(body.retryAfterSeconds);
    if (fromBody > 0) return fromBody;
  }
  const headerValue = response.headers.get("Retry-After");
  if (!headerValue) return null;
  // RFC 7231 sec 7.1.3 allows Retry-After to be EITHER a delta-seconds
  // integer ("Retry-After: 5") OR an HTTP-date ("Retry-After: Fri, 31 Dec
  // 1999 23:59:59 GMT"). The origin server always emits delta-seconds
  // (see HandHistoryReviewServerApi.scala's `retryAfterSeconds`, which
  // formats `pollAfterMs` as an integer second count), so for direct
  // origin traffic the integer branch below covers every case. The
  // HTTP-date fallback after it exists for deployments behind a reverse
  // proxy or CDN that rewrites Retry-After into HTTP-date form for its
  // own back-pressure model (some CDNs do this when they layer their
  // own 429/503 behavior on top of the origin's). Without the fallback
  // the frontend would silently drop the proxy's retry hint and surface
  // only the base error message -- the user gets "rate limited" with no
  // wait-time guidance, even though the proxy is broadcasting one.
  const parsedSeconds = parseInt(headerValue, 10);
  if (Number.isFinite(parsedSeconds) && parsedSeconds > 0) return parsedSeconds;
  // Date.parse accepts the three HTTP-date formats RFC 7231 names (RFC
  // 1123 / RFC 850 / asctime) and returns NaN on parse failure. Compute
  // the delta against the client's clock and round UP to the nearest
  // second so a "1500 ms from now" deadline surfaces as "2 seconds"
  // rather than "1" -- under-reporting would have the client retry
  // slightly before the proxy's intended backoff window expires and
  // bounce again at the still-active rate limit. Caveat worth knowing:
  // this branch uses the CLIENT's wall clock vs the server-emitted
  // HTTP-date, so a clock-skewed client (laptop with stale time on a
  // long-suspended session, embedded device with no NTP) computes a
  // wrong delta. Clock skew of a few seconds is harmless given the
  // typical 5-60 second back-pressure horizon; skew of minutes would
  // surface as a wildly-wrong "Try again in N seconds" message but the
  // user can still retry manually. The origin-direct path (parseInt
  // above) avoids this entirely -- the value IS the delta, no clock
  // arithmetic involved -- which is why the server prefers integer
  // emission rather than HTTP-date even though the RFC allows both.
  const dateMs = Date.parse(headerValue);
  if (Number.isFinite(dateMs)) {
    const deltaSec = Math.ceil((dateMs - Date.now()) / 1000);
    if (deltaSec > 0) return deltaSec;
  }
  return null;
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
  const retrySeconds = parseRetryAfterSeconds(response, body);
  if (retrySeconds == null) return base;
  const unit = retrySeconds === 1 ? "second" : "seconds";
  return `${base} Try again in ${retrySeconds} ${unit}.`;
}

function applyAuthState(data, flashMessage = "") {
  // Detect an authenticated -> unauthenticated transition before
  // overwriting the cached authState. Three call sites flow through
  // this function on a logout-equivalent:
  //   - logout() success path with the post-logout /api/auth/logout body
  //   - maybeReauthOn401 -> refreshAuthState -> applyAuthState when a
  //     mid-poll request 401s because the session ended (either the
  //     cookie's 12h Max-Age FIXED from login time expired client-
  //     side, OR the server-side record was idle past its ttlMs and
  //     got purged -- the two-layer mechanic is documented in the
  //     deploy doc's USER_AUTH_SESSION_TTL_MS bullet) or was revoked
  //     in a sibling tab
  //   - refreshAuthState called from any later trigger that observes a
  //     server-side session change (cross-tab logout, OIDC re-auth
  //     elsewhere, etc.)
  // ALL of them previously left the analyze + hall result panels
  // visible underneath the freshly-restored sign-in form -- a privacy
  // leak on shared-computer setups where the next person sitting down
  // sees the previous user's hero name, decision EVs, per-opponent
  // exploit hints (analyze) or per-villain chip flow + model id (hall).
  // An earlier version of the codebase closed only the explicit-logout
  // path by adding the hide code inline to logout(); centralising it
  // here covers the session-expiry + sibling-tab paths too. Same
  // render functions and initial copy used historically so the
  // transition lands on a clean slate identical to a freshly-loaded
  // page.
  //
  // The explicit logout() ALSO does a fuller wipe (heroInput +
  // siteSelect + hall form via .reset() + Recent runs from
  // localStorage + auth fields) before reaching here, because "user
  // signed out" implies "the next person might sit down" -- preserve
  // nothing. The session-expiry / sibling-tab paths through this
  // branch deliberately preserve form state because the user typically
  // wants to re-auth and resume their working configuration -- wiping
  // their in-progress hall preset and Recent runs on a stale-session
  // 401 would be hostile UX, the user didn't ASK to sign out.
  const nextAuthState = normalizeAuthState(data);
  if (authState.authenticated && !nextAuthState.authenticated) {
    if (reviewResults) reviewResults.classList.add("hidden");
    if (hallResults) hallResults.classList.add("hidden");
    renderStatus("Upload a hand-history file to start a local review job.");
    renderHallStatus("Configure a hall run and launch it from the browser.");
    setTitleStatus(null);
  }
  authState = nextAuthState;
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
  // hydrateUploadDefaults already prefills heroInput from
  // authState.user.heroName when the user signs in, so whatever the
  // input shows is the user's effective choice. Returning the saved
  // heroName here too used to OVERRIDE an explicit clear: a signed-in
  // user with heroName='Mig_Pro22' who deleted the input value
  // intending to fall back to auto-detect for THIS upload silently got
  // 'Mig_Pro22' sent anyway, because heroInput.value.trim() === ''
  // fell through to the saved-name branch. Trust the input; it
  // already reflects the prefill, and an explicit clear means 'auto-
  // detect this run'. Symmetric with the resolvedUploadSite fix above.
  return heroInput.value.trim() || null;
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
    // Same in-flight-pinning preservation as the hallSubmitButton block
    // below (the pair of earlier fixes that closed the concurrent-
    // submit race + the validateHallForm-vs-poll-phase clobber).
    // analyzeInFlight is set true by
    // setSubmitting(true) at submit-handler entry and stays true until
    // the handler's finally block runs setSubmitting(false). If
    // updateUploadAvailability fires in between (auth probe completion,
    // post-401 refreshAuthState, post-CSRF-refresh /api/auth/me re-pull,
    // sibling-tab storage-event sign-in/-out), the pre-fix unconditional
    // assignments would clobber the "Queueing Review..." pinning: the
    // button would flip back to "Queue Review" + enabled mid-flight,
    // telling the user the run is over when it isn't and opening the
    // same re-entry race the hall side had. Skip when in flight so
    // setSubmitting's pinning survives auth-state churn; the handler's
    // finally will overwrite cleanly once the analyze poll loop exits.
    if (!analyzeInFlight) {
      submitButton.disabled = locked;
      submitButton.textContent = locked ? "Sign In Required" : "Queue Review";
    }
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
    // hallActiveJobId is non-null while a hall run is being polled,
    // hallSubmittingPending is true during the POST-round-trip
    // window between setHallSubmitting(true) and startHallElapsed,
    // and setHallSubmitting(true) has already pinned the button to
    // disabled + "Running Hall..." text. updateUploadAvailability
    // fires on every applyAuthState (auth probe completion, sibling-
    // tab sign-in/-out via the storage event, post-CSRF-refresh
    // /api/auth/me re-pull) -- rare during a 5-15 min hall run but
    // not impossible. The pre-fix unconditional assignments would
    // clobber the in-flight state: disabled flips to !locked (false
    // when the user is authenticated, which is always the case mid-
    // run), and the text resets from "Running Hall..." to "Run
    // Playing Hall" -- both signals telling the user the run is over
    // when it isn't, AND opening the same concurrent-submit race
    // an earlier fix to validateHallForm closed. Skip the in-flight case so
    // setHallSubmitting's pinning survives auth-state churn at every
    // stage (POST window + poll window). The submit-handler's finally
    // block will overwrite this freshly once both flags clear.
    const hallInFlight = hallActiveJobId !== null || hallSubmittingPending;
    if (!hallInFlight) {
      hallSubmitButton.disabled = locked;
      hallSubmitButton.textContent = locked ? "Sign In Required" : "Run Playing Hall";
    }
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

  // Same re-entry guard as the analyze + hall submit handlers: the auth
  // form's <form> wrapper means an Enter press in email / password /
  // display-name fires the form's submit handler, which calls submitAuth.
  // setAuthButtonsBusy(true) disables both auth buttons synchronously
  // before the first await, but only blocks subsequent button clicks --
  // a rapid Enter mash still fires the form-submit handler again and
  // would race a duplicate POST against the same login (wasting one
  // PBKDF2 verify per duplicate, plus a CSRF-token-and-cookie roundtrip
  // the rate-limiter charges to the auth bucket). Bail out when the
  // login button is already disabled (the canonical busy signal --
  // setAuthButtonsBusy sets it regardless of which path is in flight).
  if (authLoginButton && authLoginButton.disabled) {
    return;
  }

  const email = authEmail ? authEmail.value.trim() : "";
  const password = authPassword ? authPassword.value : "";
  const displayName = authDisplayName ? authDisplayName.value.trim() : "";

  if (!email || !password) {
    updateAccountUi("Email and password are required.");
    // Move focus to the first empty field so a keyboard / screen-reader
    // user can act on the announced message immediately rather than
    // Shift-Tabbing back from the (now-disabled) Sign In / Register
    // button. Worth noting: the auth login + register buttons are
    // `type="button"` on the #auth-login and #auth-register elements in
    // index.html (grep for `id="auth-login"`) -- NOT `type="submit"` --
    // so HTML5 form validation -- which would normally catch the empty
    // case via the inputs' `required minlength` attributes before this
    // handler ran -- is bypassed on button click. That makes this JS
    // gate the only validation fence in the button-click path, not a
    // defense-in-depth fallback, so the focus restoration here matters
    // in modern browsers that otherwise honor `required`. Email goes
    // first because the user reads top-to-bottom on a vertical form and
    // an empty email is the more common starting state (typically a
    // password manager filled the password but not the email after a
    // domain change). Same focus-restoration pattern as the analyze
    // form's three `fileInput.focus()` early-return paths inside the
    // upload form's `form.addEventListener("submit", ...)` handler
    // above (grep this file for `fileInput.focus()` to land on all
    // three). Line / SHA references intentionally omitted: the
    // anchor previously cited "near line 226 (commit de7d982)" but
    // line 226 has since drifted to a completely unrelated function
    // (showBootError) -- symbol-name references survive that rot.
    if (!email && authEmail) {
      authEmail.focus();
    } else if (authPassword) {
      authPassword.focus();
    }
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

  // Same re-entry guard as the analyze + hall + submitAuth handlers:
  // the profile form's <button type="submit"> means Enter in any of the
  // four profile fields (display name, hero name, preferred site,
  // timezone) fires the form's submit handler, which calls saveProfile.
  // The button-disabled toggle below happens synchronously before the
  // first await, but only blocks button-click submits -- a rapid Enter
  // would otherwise race a duplicate POST with the same CSRF token
  // against the same profile. Idempotent server-side, but the duplicate
  // burns rate-limit budget and the second response's applyAuthState
  // arrives on a slightly later tick where the user might already be
  // looking at a different panel.
  if (profileSaveButton && profileSaveButton.disabled) {
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
    // Same browser-scoped argument applies to the live hall form:
    // user A's last-edited values (hands, table count, hero style,
    // villain pool, seed, etc.) sit in the DOM until the page reloads.
    // Without this reset, user B signing in on the same browser sees
    // A's working configuration prefilled in the form -- less load-
    // bearing than the persisted Recent runs panel (no "Load" link to
    // dispatch A's choices in one click) but still a leak of a logged-
    // out user's session state to whoever sits down next, and one of
    // the values is a captured RNG seed that's plausibly worth
    // hiding. Native form.reset() snaps every control back to its
    // HTML default value=/checked=/selected= attribute; re-run
    // validateHallForm and syncRandomSeedState afterwards so the
    // rendered form matches the reset values (no stale aria-invalid
    // markers, no editable seed input next to an unchecked random-
    // seed box).
    if (hallForm) {
      hallForm.reset();
      validateHallForm();
      syncRandomSeedState();
    }
    // applyAuthState below detects the authenticated -> unauthenticated
    // transition and hides reviewResults + hallResults, resets the two
    // status cards to their initial pre-run copy, and clears any
    // terminal document.title -- closes the privacy leak on shared
    // browsers (the previous user's hero name, decision EVs, opponent
    // reads, per-villain chip flow). Moved into applyAuthState (from
    // an earlier inline-in-logout() location) so the same hide also
    // fires on session-expiry-mid-poll (maybeReauthOn401 ->
    // refreshAuthState -> apply) and cross-tab logout, not just on
    // this explicit-logout path.
    applyAuthState(body, "Signed out.");
    // applyAuthState -> updateAccountUi hides the profile card (which
    // contained the Sign Out button the user just clicked), so the
    // browser moves focus to document.body. A keyboard / screen-reader
    // user would then have to Tab through skip-link + brand + nav
    // links before reaching the now-visible sign-in form -- a real
    // "where did focus go?" disruption matching the pattern Remove-on-
    // recent-runs fixed in 6ba388b. Move focus to authEmail, the first
    // interactive element on the now-visible auth form, so they can
    // continue working without the Tab re-entry penalty. authEmail is
    // always rendered post-logout in platform-user mode (basic-auth
    // mode doesn't surface the Sign Out button in the first place,
    // and no-auth mode never had an auth panel). preventScroll is
    // intentionally omitted -- the auth form sits at the top of the
    // page, so the implicit scroll lands the user on exactly the
    // panel they need to act on next.
    if (authEmail) {
      authEmail.focus();
    }
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
  // OpenID Connect Core 1.0 sec 3.1.2.6 codes. These mostly fire when
  // the relying party requested prompt=none and the provider couldn't
  // satisfy it silently; our flow doesn't set prompt=none, so they're
  // edge cases, but the provider can still emit them.
  "login_required": "Sign-in requires a fresh login. Please try again and log in when prompted.",
  "consent_required": "The sign-in provider needs you to grant consent. Please try again and accept the requested permissions.",
  "interaction_required": "Sign-in requires extra interaction with the provider. Please try again.",
  "account_selection_required": "Pick which account you want to use, then try signing in again.",
  // PlatformUserAuth.finishOidc emits this when the state token cannot be
  // resolved -- typically because the user took longer than the
  // OidcStateStore TTL (~10 minutes) between /start and /callback, or
  // because they reloaded /callback and the state was already consumed
  // by a previous attempt. Tell them what to do without leaking the
  // internal token-store mechanics.
  "OIDC login state expired or is invalid": "Your sign-in took too long or was already completed in another tab. Please try again.",
  // upsertOidcIdentity rejects when the provider returns a `sub` claim
  // longer than 256 chars. Google subjects are short (~21 digits) so
  // this almost never trips in practice, but a non-Google IdP added in
  // the future could legitimately emit longer subjects -- and a hostile
  // provider could deliberately send a giant one to probe the failure
  // path. Either way the user can't fix it; surface a friendly message
  // that names the actionable response without exposing the 256-char
  // cap to a probing attacker.
  "OIDC subject is too long": "The sign-in provider returned an unexpectedly long account identifier. Please try again, or contact the operator if this keeps happening.",
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
  "registration is temporarily unavailable": "This deployment is at capacity and not accepting new sign-ins. Please try again later or contact the operator.",
  // Google-specific failures from parseGoogleUserInfo / token exchange.
  // The server emits these verbatim from the OIDC exchange flow; the
  // raw strings are accurate but read as a developer log entry, so
  // translate to actionable language that names what's wrong and how
  // the user can fix it (verified-email failure) or who to ask
  // (operator misconfig). Rare in practice -- most Google accounts
  // have email_verified=true and the userinfo shape is stable -- but
  // worth covering so the user never sees "OIDC sign-in failed:
  // Google userinfo response was missing required identity fields".
  "Google did not return a verified email address for this account":
    "Google reports your account email is not verified. Verify your email in your Google account settings, then try signing in again.",
  "Google userinfo response was missing required identity fields":
    "Google did not return the required account info. This is usually a deployment-side OIDC scope misconfiguration -- please contact the operator.",
  "Google token exchange did not return an access token":
    "Sign in with Google did not complete -- the provider did not return an access token. Please try again, or contact the operator if this keeps happening.",
  // The OIDC callback re-runs `validateEmail` on whatever the provider
  // returned as the userinfo `email` field (PlatformUserAuth.scala calls
  // it inside upsertOidcIdentity for defense in depth against a
  // malformed-provider response). Google's emails always pass in
  // practice, but a future provider added to the supported set OR a
  // tampered userinfo response from a compromised upstream proxy could
  // deliver a value that trips one of these three messages. The user
  // cannot fix any of them (the email is provider-controlled), so all
  // three friendly mappings name it as a provider-side issue and route
  // the user to the operator rather than asking them to retry. Closes
  // the 13-string `finishOidc` Left coverage previously missing here --
  // without these three, the user saw a raw "OIDC sign-in failed: email
  // must be at most 254 characters" surface, which reads as a developer
  // log entry rather than actionable advice.
  "email must be at most 254 characters":
    "The sign-in provider returned an email address longer than allowed. Please contact the operator -- this is a provider-side issue.",
  "email must not contain whitespace or control characters":
    "The sign-in provider returned a malformed email address. Please contact the operator -- this is a provider-side issue.",
  "email must be a valid address":
    "The sign-in provider returned an unparseable email address. Please contact the operator -- this is a provider-side issue."
};

// Translate a server-emitted OIDC error code to a user-readable message.
// Tries an exact-match lookup in OIDC_ERROR_MESSAGES first, then falls
// back to prefix patterns for the three Google-exchange error strings
// that embed a variable suffix (HTTP status code or upstream exception
// message): finishOidc emits `Google token exchange failed with status
// <N>`, `Google userinfo request failed with status <N>`, and `Google
// OIDC exchange failed: <NonFatal.getMessage>`. Without the prefix
// pass these fell through to the raw `OIDC sign-in failed: Google
// token exchange failed with status 401` fallback -- readable but
// with awkward double-failed wording and no actionable advice. The
// prefix translations name the actionable response (retry, then
// contact the operator) and stay consistent with the exact-match
// mappings on the success surface. Returns null when nothing matches
// so the caller can decide its own fallback.
function lookupOidcErrorMessage(authError) {
  const exact = OIDC_ERROR_MESSAGES[authError];
  if (exact) return exact;
  if (authError.startsWith("Google token exchange failed with status ")) {
    return "Sign in with Google could not complete -- the provider rejected our token exchange. Please try again, or contact the operator if this keeps happening.";
  }
  if (authError.startsWith("Google userinfo request failed with status ")) {
    return "Sign in with Google could not complete -- the provider rejected our account-info request. Please try again, or contact the operator if this keeps happening.";
  }
  if (authError.startsWith("Google OIDC exchange failed: ")) {
    return "Sign in with Google did not complete due to a provider or network error. Please try again, or contact the operator if this keeps happening.";
  }
  return null;
}

function renderAuthFlash() {
  const params = new URLSearchParams(window.location.search);
  const authResult = params.get("auth");
  const authError = params.get("auth_error");

  if (authResult === "success") {
    updateAccountUi("OIDC sign-in completed.");
  } else if (authError) {
    const friendly = lookupOidcErrorMessage(authError);
    // No `.replaceAll("+", " ")` here: URLSearchParams.get() already
    // applies the application/x-www-form-urlencoded `+`-to-space
    // conversion during parsing per the WHATWG URL spec, AND the
    // server's PlatformUserAuth.urlEncode emits `%20` for spaces (not
    // `+`) precisely so the client doesn't have to guess. Any literal
    // `+` in `authError` is a percent-decoded `%2B` representing a
    // real `+` in the provider's error string (e.g. an upstream
    // response containing a literal `+`), and turning it back into a
    // space would silently lose data.
    const displayed = friendly || `OIDC sign-in failed: ${authError}`;
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
  // Content-Type names what we're SENDING; Accept names what we want in
  // the response. The four GET fetches that go directly to /api/health,
  // /api/auth/me, and the two poll endpoints already set Accept:
  // application/json inline; this brings the six POST/DELETE call sites
  // (analyze submit, hall submit, register, login, profile save,
  // logout, hall cancel) onto the same shape -- explicit content-
  // negotiation hint to any intermediate proxy that might honor it
  // (rare in our reverse-proxy stack but a documented HTTP idiom), and
  // future-proofs against a server endpoint that someday negotiates
  // text/html vs application/json off Accept. fetch() defaults Accept
  // to */* when unspecified, so this is a strict-narrowing change with
  // no on-the-wire regression for the current server which ignores the
  // header.
  const headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
  };
  if (includeCsrf && authState.csrfToken) {
    headers["X-CSRF-Token"] = authState.csrfToken;
  }
  return headers;
}

// In-flight flag for the analyze submit handler. Parallels the hall
// flow's hallActiveJobId, but the analyze flow has no globally-tracked
// jobId to read (each submit's body.jobId is local to the handler
// closure, no DELETE-cancel path that would need a shared identifier).
// Instead, setSubmitting writes this flag so updateUploadAvailability
// can decide whether to clobber the button's mid-flight pinning -- the
// same fix shape as the hall path's parallel in-flight-preservation
// guard above (which keys on `hallActiveJobId !== null ||
// hallSubmittingPending`). False at module
// init, flipped true by setSubmitting(true) at submit-handler entry,
// flipped back to false by setSubmitting(false) in the handler's
// finally block. Module-level scope (not nested in the submit handler
// closure) so updateUploadAvailability can read it across calls.
let analyzeInFlight = false;

function setSubmitting(isSubmitting) {
  if (!submitButton) {
    return;
  }
  analyzeInFlight = isSubmitting;
  if (requiresPlatformSignIn() && !authState.authenticated) {
    submitButton.disabled = true;
    submitButton.textContent = "Sign In Required";
    return;
  }
  submitButton.disabled = isSubmitting;
  submitButton.textContent = isSubmitting ? "Queueing Review..." : "Queue Review";
}

// In-flight flag for the hall submit POST window. hallActiveJobId
// (set by startHallElapsed once the server returns body.jobId, read by
// validateHallForm + updateUploadAvailability as the in-flight signal)
// is the in-flight signal during the POLL phase, but it's NULL during
// the 100ms-2s POST round-trip that sits BETWEEN setHallSubmitting(true)
// and startHallElapsed. A user typing into a hall form field during
// that window triggers validateHallForm, which would see inFlight=false
// and re-enable the submit button -- reopening the concurrent-submit
// race those two fixes closed for the poll phase. Parallels analyzeInFlight
// (which closes the equivalent race for the analyze side via setSubmitting).
// False at module init, flipped true by setHallSubmitting(true) BEFORE
// the POST kicks off, flipped back to false by setHallSubmitting(false)
// in the parent finally block AFTER finishHallProgress has already
// nulled hallActiveJobId. So the OR (hallActiveJobId !== null ||
// hallSubmittingPending) covers the union of POST-window + poll-window.
let hallSubmittingPending = false;

function setHallSubmitting(isSubmitting) {
  if (!hallSubmitButton) {
    return;
  }
  hallSubmittingPending = isSubmitting;
  if (requiresPlatformSignIn() && !authState.authenticated) {
    hallSubmitButton.disabled = true;
    hallSubmitButton.textContent = "Sign In Required";
    return;
  }
  hallSubmitButton.disabled = isSubmitting;
  hallSubmitButton.textContent = isSubmitting ? "Running Hall..." : "Run Playing Hall";
}

// Cache the last-rendered status string so a poll loop that calls
// renderStatus with the SAME message every ~750ms (e.g. status='running'
// throughout a 2-min analyze run) doesn't keep mutating the DOM. The
// #review-status div carries aria-live="polite" (moved down from the
// parent #review-panel article to narrow the announcement
// surface), and while most
// modern screen readers de-dupe identical announcements per the aria-
// live spec's "if the announcement text is identical to the previous
// one, skip it" guidance, the spec doesn't MANDATE de-dup and some
// readers (older NVDA, certain JAWS configs) re-announce on every DOM
// mutation regardless of text equality. A cached string + early-return
// closes that variability AND saves the innerHTML reparse cost on the
// hot polling path. Reset to null on any path that intentionally clears
// the panel so the next non-empty render isn't suppressed.
let lastRenderedStatus = null;
function renderStatus(message) {
  if (message === lastRenderedStatus) {
    return;
  }
  lastRenderedStatus = message;
  reviewStatus.innerHTML = `
    <p class="card-kicker">Status</p>
    <h3>${escapeHtml(message)}</h3>
    <p class="section-note">
      SICFUN accepts the upload quickly, analyzes it in a background job, and fills this board with hand
      counts, EV gaps, warnings, and opponent notes when the review is ready.
    </p>
  `;
}

// Mirror status into the document title so users with many tabs can see
// at a glance which one has work in flight. Pass null/empty to reset to
// the default title (kept here as the source of truth so the meta and
// the page title agree). Called from the analyze + hall submit flows
// and finally blocks; idempotent so a final reset doesn't overwrite a
// later flow's in-progress title.
const DEFAULT_TITLE = "SICFUN | Hand-History Review & Playing Hall";
// Terminal status messages whose backgrounded-tab title cue should reset
// to default the moment the user returns. These are the EXACT `message`
// arguments passed to setTitleStatus on the corresponding completion
// paths (lines 345-346 for analyze, 516-518 for hall) -- defined here
// as plain message strings so the visibilitychange handler below can
// reconstruct the full title via the same `${message} · SICFUN`
// template setTitleStatus uses. Previous version hardcoded the full
// composed strings ("Review ready · SICFUN" etc.) in the handler,
// which silently desyncs from setTitleStatus if the title-format
// template ever changes (` · SICFUN` -> something else) -- the
// terminal-title reset would just stop firing with no failure signal.
// Centralising the message list here ties the two sides to a single
// source-of-truth string set.
const TERMINAL_TITLE_STATUSES = [
  "Review ready",
  "Review failed",
  "Hall done",
  "Hall cancelled",
  "Hall failed"
];
function setTitleStatus(message) {
  document.title = message ? `${message} · SICFUN` : DEFAULT_TITLE;
}

// Job-terminal titles linger when the user has the tab backgrounded so
// they get a completion cue in their browser tab list. Reset to default
// the moment they come back -- the title has served its purpose and the
// renderStatus panel is now visible as the source of truth; leaving a
// stale terminal title would be misleading on the next visit. Match the
// exact terminal strings (via TERMINAL_TITLE_STATUSES above) so a mid-
// poll switch back ('Reviewing X', 'Hall running') doesn't get
// clobbered -- those transient titles aren't in the terminal list so
// they survive the visibility change.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  const current = document.title;
  if (TERMINAL_TITLE_STATUSES.some(status => current === `${status} · SICFUN`)) {
    setTitleStatus(null);
  }
});

// Cache last-rendered (message, badge) so the #hall-status div's polite
// aria-live region (moved down from the parent .hall-panel article
// to narrow the announcement surface to status text only) doesn't
// keep re-announcing the SAME status text on every ~750ms poll tick
// during a 5-15 min hall run -- same screen-reader-spam fix renderStatus
// got above, with a bigger payoff here because hall polls run far longer
// (the analyze flow times out at 2 min default; hall runs default to 15
// min and can be tuned higher via PLAYING_HALL_TIMEOUT_MS). Cache both
// args because a status transition like queued->running keeps the
// message stable while only the badge would change (e.g. a future per-
// status badge), and vice versa for cancelled vs completed at the same
// message.
let lastRenderedHallMessage = null;
let lastRenderedHallBadge = null;
function renderHallStatus(message, badge) {
  if (!hallStatus) {
    return;
  }
  // Normalise undefined to null for the cache comparison -- the function
  // is called with badge omitted (e.g. renderHallStatus("Queueing...")),
  // making `badge` === undefined, vs explicit null elsewhere. Treat
  // both as the same "no badge" state so the equality check doesn't
  // fire spuriously on the first explicit-null call after an
  // undefined-arg call.
  const normalizedBadge = badge == null ? null : badge;
  if (message === lastRenderedHallMessage && normalizedBadge === lastRenderedHallBadge) {
    return;
  }
  lastRenderedHallMessage = message;
  lastRenderedHallBadge = normalizedBadge;
  // Optional `badge` renders a styled <span class="cancelled-badge">
  // (defined in site.css, originally added for this purpose but
  // never wired up -- the cancelled indicator was previously inlined
  // as " · CANCELLED (partial data)" plain text into the H3, dropping
  // the styled box the CSS was designed for). The badge text gets
  // escapeHtml just like the message; the surrounding span is the
  // only fixed HTML.
  const badgeHtml = normalizedBadge ? ` <span class="cancelled-badge">${escapeHtml(normalizedBadge)}</span>` : "";
  hallStatus.innerHTML = `
    <p class="card-kicker">Status</p>
    <h3>${escapeHtml(message)}${badgeHtml}</h3>
    <p class="section-note">
      SICFUN runs the configured hall batch in the background, then returns the run summary, action mix,
      per-villain chip flow, and the generated output files here.
    </p>
  `;
}

async function pollAnalysisJob(fileName, statusUrl, initialPollAfterMs) {
  let pollAfterMs = normalizePollAfterMs(initialPollAfterMs);
  const deadline = Date.now() + maxPollWaitMs;

  for (;;) {
    if (Date.now() >= deadline) {
      // Frontend-side polling deadline, NOT a server-side job failure --
      // the job may still be running. Tell the user accurately so they
      // don't assume the analysis crashed; the frontend budget should
      // rarely fire because the server's analyze timeout is 2 minutes
      // (the job would have terminated long before this). Derive the
      // minute count from maxPollWaitMs so the message reflects the
      // value actually in effect, including any boot-time extension
      // probeServerLimits applied.
      const minutes = Math.round(maxPollWaitMs / 60000);
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
      if (response.status === 429) {
        // Status-poll rate limit hit -- most plausibly from a handful of
        // sibling tabs polling the same user/session concurrently (the
        // JobStatus bucket caps at 240/min/principal by default, and each
        // tab polls at ~750ms = 80/min, so 3+ tabs can push past the
        // ceiling). The job itself is still running server-side; throwing
        // here would blow up the polling loop on a transient retryable
        // condition and leave the user thinking the review failed. Read
        // Retry-After (capped at 30s so a misconfigured server can't pin
        // us indefinitely), surface what's happening so the user doesn't
        // read the long sleep as a frozen page, then continue. The
        // existing deadline check at the top of the loop guards against
        // an infinite retry loop -- maxPollWaitMs eventually expires and
        // throws the deadline error if the rate limit never clears.
        const retrySeconds = parseRetryAfterSeconds(response, body) || 5;
        const waitMs = Math.min(retrySeconds, 30) * 1000;
        renderStatus(`Polling rate-limited; resuming in ${Math.round(waitMs / 1000)}s.`);
        await sleep(waitMs);
        continue;
      }
      throw new Error(formatErrorMessage(response, body));
    }

    renderStatus(jobStatusMessage(fileName, body.status));
    // Mirror the polling state into the tab title too -- otherwise the
    // 'Submitting <file>' title set at submit time would persist through
    // the entire reviewing phase, undercutting the multi-tab visibility
    // win the dynamic title was meant to provide.
    if (body.status === "running") setTitleStatus(`Reviewing ${fileName}`);
    else if (body.status === "queued") setTitleStatus(`Queued ${fileName}`);

    if (body.status === "completed") {
      return body.result || {};
    }

    if (body.status === "failed") {
      // Translate the server's "analysis timed out after 120000ms" raw
      // message into a friendlier duration. The server emits raw
      // milliseconds (JobQueue.scala's `timeoutFailure` formats the
      // configured ANALYSIS_TIMEOUT_MS verbatim into the error string)
      // because that's the only form it has -- the env var IS specified
      // in ms -- but non-developer users read durations in minutes/
      // seconds, and "120000ms" reads as a developer log entry rather
      // than actionable feedback. Keyed on body.errorStatus === 504
      // (the timeout-specific HTTP equivalent set by timeoutFailure)
      // AND a regex match on the message shape, so a server change to
      // the message format only loses the prettification (falls back
      // to the verbatim string), never breaks the failure path. Same
      // formatDuration helper the success path uses for "(took Xm Ys)"
      // so the time format is consistent across success and timeout.
      if (body.errorStatus === 504 && typeof body.error === "string") {
        const match = body.error.match(/^analysis timed out after (\d+)ms$/);
        if (match) {
          throw new Error(`Analysis timed out after ${formatDuration(Number(match[1]))}.`);
        }
      }
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
  const deadline = Date.now() + maxPollWaitMs;

  for (;;) {
    if (Date.now() >= deadline) {
      // Same shape as the analysis poller: this is a client-side
      // polling budget exhaustion, not a server-side job failure. The
      // hall server timeout defaults to 15 minutes (PLAYING_HALL_TIMEOUT_MS),
      // and probeServerLimits at boot extends maxPollWaitMs to match
      // any raised server timeout, so reaching the frontend budget
      // means either the server is silently slow OR the queue wait
      // was long enough to push the worker's own deadline past ours.
      // The job may still run to completion. Derive the minute count
      // from maxPollWaitMs so the message reflects the active value
      // including the boot-time extension.
      const minutes = Math.round(maxPollWaitMs / 60000);
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
      if (response.status === 429) {
        // Same retry-after-respecting backoff the analysis poller does.
        // Hall runs poll for much longer (15-min server-side default vs
        // 2-min analyze), so the multi-tab rate-limit collision is even
        // more likely to bite -- a single 429 mid-run should pause and
        // resume polling, not kill the whole poll loop and report the
        // hall run as failed when it's actually still executing.
        const retrySeconds = parseRetryAfterSeconds(response, body) || 5;
        const waitMs = Math.min(retrySeconds, 30) * 1000;
        renderHallStatus(`Polling rate-limited; resuming in ${Math.round(waitMs / 1000)}s.`);
        await sleep(waitMs);
        continue;
      }
      throw new Error(formatErrorMessage(response, body));
    }

    // When the user has clicked Cancel and the server has accepted but
    // not yet flipped the job to cancelled, suppress the normal
    // "Running the playing hall in the background..." / "Queued the
    // playing hall run..." messages -- they'd overwrite the "Cancel
    // accepted. Finishing in-flight hands..." message and make the
    // user think their cancel was dropped. Render a cancelling-aware
    // message instead until the next poll sees status=cancelled and
    // exits via finishHallProgress (which clears the flag). Title
    // mirroring also pivots to "Hall cancelling" for the same reason.
    if (hallCancelRequested && (body.status === "queued" || body.status === "running")) {
      // Differentiate the in-flight-hands message based on whether the
      // worker has actually started. status="queued" means the job is
      // still in the FIFO queue waiting for a free worker slot
      // (typically because MAX_CONCURRENT_JOBS is saturated) -- the
      // worker hasn't started, so there are NO in-flight hands and
      // the prior "Finishing in-flight hands..." message was
      // misleading. status="running" means the worker is mid-loop
      // and may have partial-data hands captured before the cancel
      // flag is observed at the next per-hand check. Both cases
      // resolve via the cancel flag (jobStore.cancel sets the
      // cancelFlags atomic; a queued job's worker checks the flag
      // immediately on pickup and exits before playing any hands;
      // a running job's worker checks at the next per-hand boundary
      // and exits then). The hallCancelRequested flag stays set
      // through both cases until the next poll observes status=
      // "cancelled" and finishHallProgress clears it.
      const cancelMessage = body.status === "queued"
        ? "Cancel accepted. Removing job from queue..."
        : "Cancel accepted. Finishing in-flight hands...";
      renderHallStatus(cancelMessage);
      setTitleStatus("Hall cancelling");
    } else {
      renderHallStatus(playingHallJobStatusMessage(body.status));
      // Same title-mirroring as the analyze poller -- the 'Hall run queued'
      // title set at submit time should advance to 'Hall running' once the
      // worker picks it up.
      if (body.status === "running") setTitleStatus("Hall running");
      else if (body.status === "queued") setTitleStatus("Hall queued");
    }

    if (body.status === "completed") {
      return body.result || {};
    }

    if (body.status === "cancelled") {
      const result = body.result || {};
      result.cancelled = true;
      return result;
    }

    if (body.status === "failed") {
      // Same timeout-message prettification as the analyze poller above:
      // the server emits "playing hall timed out after 900000ms" verbatim
      // (15-min default formatted as raw ms) which reads as a developer
      // log entry; translate to "Playing hall timed out after 15m 0s."
      // when the errorStatus + message shape both match, fall through
      // to the verbatim string on any mismatch so a future server-side
      // message change doesn't break the failure path. PLAYING_HALL_TIMEOUT_MS
      // defaults to 15 min but can be raised to hours via env var, so the
      // formatDuration h:mm:ss promotion in formatDuration matters here
      // more than it does for the analyze flow (2-min default).
      if (body.errorStatus === 504 && typeof body.error === "string") {
        const match = body.error.match(/^playing hall timed out after (\d+)ms$/);
        if (match) {
          throw new Error(`Playing hall timed out after ${formatDuration(Number(match[1]))}.`);
        }
      }
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

function renderResults(fileName, data, startedAt) {
  // Same duration-reporting treatment as renderHallResults: include
  // how long the review took in the completion message. The analyze
  // form has no visible elapsed timer (the hall flow's progress card
  // pattern doesn't apply because analyze typically completes in
  // seconds-to-a-minute vs hall's 5-15 min), so this is the only
  // place the user sees the latency. Useful for operators tuning
  // ANALYSIS_TIMEOUT_MS or comparing review latency across
  // deployments. Caller passes the submit-click timestamp; if
  // unavailable (defensive fallback) the duration tail is skipped.
  const durationMs = Number.isFinite(startedAt) && startedAt > 0 ? Date.now() - startedAt : 0;
  const durationStr = durationMs > 0 ? ` (took ${formatDuration(durationMs)})` : "";
  renderStatus(
    `Review ready: imported ${formatInteger(data.handsImported)} hand${data.handsImported === 1 ? "" : "s"} from ${fileName}${durationStr}.`
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

  // Pass the cancelled state as a renderHallStatus badge rather than
  // inline " · CANCELLED" text. The badge picks up the warning-color
  // border + monospace styling from `.cancelled-badge` in site.css
  // (originally added for exactly this purpose but never wired up
  // before this commit), and reads as a visually distinct marker
  // rather than buried punctuation in a long status sentence.
  //
  // Include the run duration in the status message: hallActiveStartedAt
  // is still set when renderHallResults fires (the submit handler's
  // finally block calls finishHallProgress, which clears it, AFTER
  // renderHallResults returns). Without this, the live "Elapsed mm:ss"
  // counter on the progress card disappears the moment the card hides,
  // and the result panel never says how long the run took -- a
  // common-enough operator + power-user question that the timer is
  // worth preserving in the completion message. Client-side duration
  // (rather than the server's response.durationMs, which would require
  // restructuring pollPlayingHallJob's return shape) is close enough:
  // the polling cadence is ~750 ms so the gap between server completion
  // and frontend observing it is bounded.
  const durationMs = hallActiveStartedAt > 0 ? Date.now() - hallActiveStartedAt : 0;
  const durationStr = durationMs > 0 ? ` (took ${formatDuration(durationMs)})` : "";
  // Switch the leading verb on cancellation: "Hall ready" is wrong when
  // the user explicitly clicked Cancel -- the run did NOT complete in
  // any meaningful sense, only a worker that was already mid-run was
  // interrupted and surfaced partial data. The CANCELLED badge below
  // already marks the partial-data state, but the leading "Hall ready"
  // semantically claims completion that didn't happen and reads as a
  // contradiction next to its own CANCELLED badge ("Hall ready: 200
  // hands [CANCELLED]" -- which is it?). Using "Hall cancelled" as the
  // verb on the cancelled path makes the title-cue ladder (set in the
  // hall submit handler's finally block where setTitleStatus dispatches
  // on runReady / runCancelled / runFailed -- grep for `setTitleStatus("Hall
  // done")` to find the cluster) consistent with the hall-status panel
  // verb here. (Line numbers intentionally omitted -- an earlier version
  // of this comment cited "line 491-493" which referenced the wrong block
  // by ~55 lines because the comment didn't update when the submit handler
  // grew; symbol-name references stay correct as the file evolves.)
  const verb = cancelled ? "Hall cancelled" : "Hall ready";
  // Distinguish two cancel sub-cases for the trailing badge:
  // - Cancel landed AFTER the worker had captured at least one hand
  //   (handsPlayed > 0): the result has partial data, badge reads
  //   "CANCELLED (partial data)" and the leading verb's "0 hands,
  //   +0.00 chips, ..." numbers DO reflect a real sample.
  // - Cancel landed BEFORE the worker captured anything (handsPlayed
  //   <= 0): the result has no useful summary; the displayed
  //   "0 hands, +0.00 chips, +0.00 bb/100" numbers are placeholders,
  //   not a real partial sample. Badge "CANCELLED (no data captured)"
  //   is honest about that -- "partial data" claimed data that
  //   wasn't there. Either summary.handsPlayed being missing or zero
  //   triggers the no-data branch.
  let badge = null;
  if (cancelled) {
    const handsCaptured = Number(summary.handsPlayed);
    badge = Number.isFinite(handsCaptured) && handsCaptured > 0
      ? "CANCELLED (partial data)"
      : "CANCELLED (no data captured)";
  }
  renderHallStatus(
    `${verb}: ${formatInteger(summary.handsPlayed)} hands, ${formatSigned(summary.heroNetChips)} chips, ${formatSigned(summary.heroBbPer100)} bb/100${durationStr}.`,
    badge
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
      SicfunCharts.renderLine(hallChartChips, {xs, ys}, {yLabel: "cumulative chips", height: 220, title: "Hero cumulative chips over time"});
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
      SicfunCharts.renderHistogram(hallChartEquityHist, {values: equities}, {bucketCount: 10, min: 0, max: 1, height: 200, title: "Hero decision-equity distribution"});
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
      ? files.map(file => {
          // Use the file's basename as the card title rather than a
          // generic "Output File". A hall run can emit up to five
          // distinct files (hands.tsv, learning.tsv,
          // training-selfplay.tsv, ddre-training-selfplay.tsv,
          // review-upload-pokerstars.txt -- enumerated by
          // existingOutputFiles in HandHistoryReviewServerApi.scala),
          // and rendering each as "Output File" makes the list read as
          // five duplicate cards with the only distinguishing detail
          // buried in the trailing path meta. Basename in the title +
          // full path in the meta gives both signals at the right
          // visual weight: glance the list to see WHICH files got
          // written, dig into the path for the absolute location.
          // Split on both `/` and `\` so absolute paths on Windows
          // (e.g. `C:\...\hands.tsv`) and Unix-style paths from
          // outDir.resolve(...) both produce the right basename.
          // The `|| file` fallback handles the unlikely degenerate
          // case of an empty-string path component.
          const basename = file.split(/[/\\]/).pop() || file;
          return `
            <article class="opponent-card">
              <div class="decision-head">
                <h3 class="decision-title">${escapeHtml(basename)}</h3>
                <p class="card-meta">written</p>
              </div>
              <p class="opponent-meta">${escapeHtml(file)}</p>
            </article>
          `;
        }).join("")
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

// All five format* helpers below use `coerceNumber` to defend against a
// server response that ever ships a non-numeric value where the renderer
// expects a number. The earlier pattern `Number(value || 0)` only handled
// null / undefined / empty-string / 0 -- it left `Number("not-a-number")`
// as NaN, which then flows through `.toFixed(2)` and `.toLocaleString()`
// as the literal string "NaN" in the rendered summary card. That's
// operator-visible as a defect rather than the intended graceful "0"
// fallback. coerceNumber additionally guards against ±Infinity for the
// same reason -- `(Infinity).toFixed(2)` returns "Infinity" rather than
// a numeric-looking string. Both NaN and Infinity are improbable in
// practice (server fields are well-typed Long / Double), but cheap to
// defend.
function coerceNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function formatInteger(value) {
  return coerceNumber(value).toLocaleString("en-US");
}

function formatNumber(value) {
  return coerceNumber(value).toFixed(2);
}

function formatSigned(value) {
  const number = coerceNumber(value);
  return `${number > 0 ? "+" : ""}${number.toFixed(2)}`;
}

function formatMetric(value) {
  return coerceNumber(value).toFixed(3);
}

function formatPercent(value) {
  return `${(coerceNumber(value) * 100).toFixed(1)}%`;
}

function formatFileSize(bytes) {
  // Routes through the same coerceNumber helper as the format* group
  // above (added af13670). The earlier `Number(bytes) || 0` short-circuit
  // caught NaN (falsy → falls back to 0), but Infinity is truthy + non-
  // zero and would have leaked through as `"Infinity MB"`. In practice
  // bytes comes from the File API's `file.size` which is always a
  // non-negative integer, but defending here keeps the helper uniform
  // with the rest of the format-family + survives any future call site
  // that passes a less-disciplined source (e.g. a network-reported
  // size, an estimated value, a computed-but-unbounded total). Also
  // catches negative inputs (`size < 0` is finite but nonsensical for
  // a byte count) -- the helper now coerces them to 0 via the same
  // `Math.max(0, ...)` guard so `formatFileSize(-100)` returns `"0 B"`
  // rather than the misleading `"-100 B"` the prior implementation
  // would have produced. Same defense-in-depth rationale: improbable
  // today, cheap to defend.
  const size = Math.max(0, coerceNumber(bytes));
  if (size >= 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  if (size >= 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${size} B`;
}

// Human-readable elapsed time in ms. Used in the analyze + hall
// completion status messages, the failure messages, and the Recent
// runs panel so the user sees the actual run duration after
// finishHallProgress hides the live elapsed timer. Scales the unit
// to the magnitude: sub-second renders as `<1s` (not `0s` -- a
// 200ms validation failure showing "(after 0s)" reads as a defect
// rather than "instant"), sub-minute as plain seconds, sub-hour as
// "Xm Ys", longer as "Xh Ym Zs". The mm:ss-pad-format used by the
// live tickHallElapsed isn't appropriate here -- this is one-shot
// reporting, not a moving counter, so prose-form scales better
// across PLAYING_HALL_TIMEOUT_MS values raised past the 15-min
// default.
function formatDuration(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 1000) return "<1s";
  const sec = Math.round(n / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  const remSec = sec % 60;
  if (min < 60) return `${min}m ${remSec}s`;
  const hr = Math.floor(min / 60);
  const remMin = min % 60;
  return `${hr}h ${remMin}m ${remSec}s`;
}

function escapeHtml(value) {
  // null / undefined render as empty string rather than the literal
  // text "null" / "undefined". The default String() coercion turns
  // them into those four / nine character strings respectively, which
  // then sail through the replaceAll chain unchanged and surface in
  // the UI verbatim -- a universally-recognized "the site is broken"
  // signal whenever a server response is missing an optional field,
  // a normalize* helper hasn't run yet, or a recent-runs entry was
  // persisted before a new field existed on the entry shape. Falsy-
  // but-meaningful values (0, false, the empty string itself) still
  // round-trip through String() to their canonical text form, which
  // is what the existing call sites expect (e.g. "(took 0s)" reads
  // fine, "after false" doesn't but no call site builds that).
  if (value == null) return "";
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
    clearFieldErrorAria(input);
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
      // Wire the error span to the input via aria-describedby + aria-
      // invalid (set below) so screen-reader users get an audible "this
      // field is invalid: Max 5000" announcement instead of a silent
      // disabled Run button. aria-live="polite" queues the message
      // until the user pauses typing rather than interrupting each
      // keystroke. Stable id ($input.id-error) so re-validation reuses
      // the same node without orphaning aria-describedby references.
      err.id = `${input.id}-error`;
      err.setAttribute("aria-live", "polite");
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
    input.setAttribute("aria-invalid", "true");
    // aria-describedby is a space-separated list of IDs per WAI-ARIA, and
    // most of the validated inputs already point at a `<field-id>-hint`
    // span from index.html (e.g. hero-name -> hero-name-hint, profile-
    // time-zone -> profile-time-zone-hint). Pre-fix this line wrote
    // `err.id` as a single value, CLOBBERING the hint reference -- so a
    // field that ever failed validation lost its hint announcement for
    // the rest of the session, and clearFieldErrorAria below couldn't
    // restore it because the original value was gone. Append the error
    // id to the existing list (de-duped) so screen readers announce
    // BOTH the hint and the error; clearFieldErrorAria mirrors this
    // shape on the other side, removing only the error id and leaving
    // the hint reference intact for the next focus.
    const existingDescribedBy = input.getAttribute("aria-describedby") || "";
    const describedByIds = existingDescribedBy.split(/\s+/).filter(Boolean);
    if (!describedByIds.includes(err.id)) describedByIds.push(err.id);
    input.setAttribute("aria-describedby", describedByIds.join(" "));
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
    clearFieldErrorAria(input);
  }
  return ok;
}

// Removes the aria-invalid attribute and the aria-describedby link the
// field-error span owned. Only clears aria-describedby if it currently
// points at the field-error id (so a future input with multiple
// describing relationships -- e.g., a separate hint span -- won't be
// stripped). Idempotent if the attributes were never set.
function clearFieldErrorAria(input) {
  input.removeAttribute("aria-invalid");
  const errorId = `${input.id}-error`;
  // Mirror the append-not-overwrite shape validateField uses on the
  // failing side: remove only the error id from the space-separated
  // aria-describedby list, leaving any other ids (typically the
  // `<field-id>-hint` reference from index.html) attached so the
  // hint announcement returns to screen readers once the field
  // re-validates. Pre-fix this branch matched the FULL attribute
  // value against the errorId, then removed the whole attribute --
  // which (a) failed to match if validateField had appended to an
  // existing hint (which it now does, post the same-fire fix above)
  // and (b) erased the hint reference entirely when it did match,
  // breaking the field's documented aria-describedby contract for
  // the rest of the session.
  const existing = input.getAttribute("aria-describedby") || "";
  const remaining = existing.split(/\s+/).filter(id => id && id !== errorId);
  if (remaining.length > 0) {
    input.setAttribute("aria-describedby", remaining.join(" "));
  } else {
    input.removeAttribute("aria-describedby");
  }
}

function validateVillainPool() {
  const ok = selectedVillainPool().length > 0;
  document.querySelectorAll(".hall-pool .chip-check span").forEach(el => el.classList.toggle("invalid", !ok));
  const pool = document.querySelector(".hall-pool");
  const group = pool && pool.closest('[role="group"]');
  // Manage aria-invalid + aria-describedby on the role="group" wrapper
  // so screen-reader users get the same audible "invalid: select at
  // least one" announcement as the numeric fields do via validateField.
  // The .invalid CSS class on each chip span only signals visually --
  // without an ARIA pairing a user with assistive tech sees a stuck
  // Run button with no spoken cause. Append a polite aria-live span
  // mirroring the .field-error pattern validateField uses, then
  // remove it when the pool re-validates so a recovered form
  // doesn't leak a stale "select at least one" announcement on the
  // next focus.
  if (group) {
    const errorId = "villain-pool-error";
    let err = group.querySelector("#" + errorId);
    if (!ok) {
      if (!err) {
        err = document.createElement("span");
        err.id = errorId;
        err.className = "field-error";
        err.setAttribute("aria-live", "polite");
        err.textContent = "Select at least one villain.";
        group.appendChild(err);
      }
      group.setAttribute("aria-invalid", "true");
      // Append-not-overwrite, same shape as validateField's
      // aria-describedby handling above (grep for `validateField` in
      // this file). The villain-pool role="group" element doesn't currently carry an
      // aria-describedby in index.html, so the previous single-attribute
      // overwrite was benign on the shipped markup -- but if a future
      // commit adds a hint reference (e.g., a "pick from nit / tag /
      // lag / ... / gto" explainer) the overwrite would silently clobber
      // it. Use the same space-separated append/remove dance the
      // numeric fields use so the two code paths stay in lockstep.
      const existingDescribedBy = group.getAttribute("aria-describedby") || "";
      const describedByIds = existingDescribedBy.split(/\s+/).filter(Boolean);
      if (!describedByIds.includes(errorId)) describedByIds.push(errorId);
      group.setAttribute("aria-describedby", describedByIds.join(" "));
    } else {
      if (err) err.remove();
      group.removeAttribute("aria-invalid");
      // Mirror the filter-not-clear shape clearFieldErrorAria uses:
      // remove only the errorId from the space-separated list, leaving
      // any future hint reference attached. The pre-fix exact-match
      // check would have failed if validateVillainPool had appended to
      // an existing hint (which it now does, per the append branch
      // above) and erased the whole attribute when it did match --
      // same two-bug shape (append-not-overwrite + filter-not-clear)
      // that clearFieldErrorAria handles for the numeric-fields path.
      const existing = group.getAttribute("aria-describedby") || "";
      const remaining = existing.split(/\s+/).filter(id => id && id !== errorId);
      if (remaining.length > 0) {
        group.setAttribute("aria-describedby", remaining.join(" "));
      } else {
        group.removeAttribute("aria-describedby");
      }
    }
  }
  // Same auto-open as validateField: if the villain section is collapsed
  // and the pool has no entries, the user sees a stuck Run button with
  // no visible 'why'. Force the villain <details> open so the red chips
  // are visible. Look up via the first chip's nearest ancestor so the
  // selector stays cheap and survives DOM restructuring.
  if (!ok) {
    const parentDetails = pool && pool.closest("details");
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
    // hallActiveJobId is set non-null by startHallElapsed once the
    // server accepts a hall job and stays non-null until
    // finishHallProgress runs in the submit handler's finally block.
    // hallSubmittingPending covers the additional POST-round-trip
    // window between setHallSubmitting(true) and startHallElapsed,
    // where hallActiveJobId is still null but a submit IS in flight.
    // validateHallForm fires on every form input event, so a user
    // typing into hall fields to prep a follow-up run (a common
    // pattern during the 5-15 min wait for a long simulation) -- or
    // typing during the 100ms-2s POST round-trip -- would otherwise
    // re-enable the submit button while setHallSubmitting (true) has
    // it disabled, letting them fire a SECOND submit before the first
    // poll loop has finished, ending up with two concurrent poll
    // loops fighting over the hall-status panel, the elapsed timer,
    // and the title-cue ladder. The OR covers the union of POST-
    // window + poll-window so the form-validity recompute never
    // undoes setHallSubmitting's disabled gate at any in-flight stage.
    const inFlight = hallActiveJobId !== null || hallSubmittingPending;
    hallSubmitButton.disabled = !allOk || locked || inFlight;
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
      if (preset) {
        applyHallConfig(preset.config);
        // Announce the preset load so screen-reader users (and sighted
        // keyboard users at a distant focus point) know the form
        // values just changed. Programmatic input.value assignments
        // don't fire aria-live events even inside live regions, so
        // applyHallConfig's silent mutation of 15+ fields would
        // otherwise be invisible -- the user clicks a preset, hears
        // nothing, and has to Tab through the form to verify. The
        // #hall-status div is aria-live=polite (moved down from the
        // parent .hall-panel article) so this
        // renderHallStatus call queues a polite announcement; the
        // message is also visible context for sighted users who
        // already saw the values change. Transient: the next
        // renderHallStatus on submit / validation / poll overwrites it.
        renderHallStatus(`Loaded preset: ${preset.label}.`);
      }
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
    if (!Array.isArray(parsed)) return [];
    // Filter to entries whose shape matches what renderRecentRuns
    // dereferences without null-guards: entry.request must be an
    // object (the render path reads .villainPool / .heroStyle /
    // .hands / .seed directly), entry.summary must be an object
    // (it reads .heroNetChips), AND entry.timestamp must be a finite
    // POSITIVE number (the render path passes it to
    // `new Date(entry.timestamp).toLocaleString()`). The Number.isFinite
    // check alone catches the NaN / non-number / ±Infinity cases that
    // would have rendered as the literal string "Invalid Date"; the
    // additional `> 0` check catches negative-finite values too, which
    // are syntactically valid (e.g. `new Date(-1).toLocaleString()`
    // returns a 1969 date string, NOT "Invalid Date") but semantically
    // nonsensical for a "Recent run" entry -- a run can't have
    // happened before Unix epoch. pushRecentRun always writes
    // `Date.now()` so well-formed entries always pass; this check
    // guards against the same corruption vectors the request/summary
    // checks already address (future schema change drops or renames
    // the field, half-written cross-tab race leaves a partial object,
    // hand-edited localStorage). Filtering here makes the panel render
    // the surviving entries and the bad ones age out naturally on the
    // next pushRecentRun (which writes back the filtered list capped
    // at RECENT_RUNS_MAX).
    return parsed.filter(entry =>
      entry !== null && typeof entry === "object"
        && entry.request !== null && typeof entry.request === "object"
        && entry.summary !== null && typeof entry.summary === "object"
        && Number.isFinite(entry.timestamp) && entry.timestamp > 0
    );
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

function pushRecentRun(request, summary, options) {
  // `options.cancelled` flags entries that came from a cancelled hall
  // run -- summary is typically empty (the worker didn't capture
  // partial data before the interrupt) so renderRecentRuns would
  // otherwise display "0 hands, +0.00 chips" and make the entry
  // look like a broken run. The flag lets the render path show a
  // "(cancelled)" marker instead, preserving the config for re-
  // launch while honestly representing the outcome.
  // `options.durationMs` records the client-observed run latency
  // (submit click -> result rendered) so the Recent runs panel can
  // show "took Xm Ys" on each entry -- useful for comparing config-
  // to-latency tradeoffs across the captured history. Optional;
  // missing means "no duration data was captured at the time," and
  // renderRecentRuns just omits the duration field rather than
  // displaying "0s".
  const durationMs = options && Number.isFinite(options.durationMs) && options.durationMs > 0
    ? options.durationMs
    : null;
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
    },
    cancelled: !!(options && options.cancelled),
    durationMs
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
    // Per-button aria-labels include the timestamp so screen-reader users
    // navigating by button list (Tab key, rotor, etc.) don't hear "Load
    // button, Remove button" repeated N times with no way to tell which
    // run is which. The visible "Load" / "x" stays terse for sighted
    // users; the label only kicks in for assistive tech.
    const tsForLabel = escapeHtml(ts);
    // Mark cancelled entries explicitly. Without this, a cancelled
    // run with no captured partial data renders as "0 hands · +0.00
    // chips" -- looks like a broken or zero-result successful run.
    // The "(cancelled)" suffix on the timestamp + the muted meta
    // line tells the user the entry came from an intentional
    // cancellation, and the Load button still works for re-launch
    // with the captured config. `entry.cancelled` is falsy for old
    // entries (added before this field existed) so legacy data
    // renders unchanged.
    const cancelled = !!entry.cancelled;
    const cancelledSuffix = cancelled ? ` <span class="recent-run-cancelled">(cancelled)</span>` : "";
    // Append duration when present. `entry.durationMs` is captured at
    // push time (commit history); old entries written before this
    // field existed leave it null, in which case the duration is
    // omitted entirely rather than rendering an awkward "0s" or
    // "unknown" placeholder. Cancelled entries also get the duration
    // since "cancelled in 2m 15s" gives the user a sense of how
    // long it ran before the interrupt landed.
    const durationStr = Number.isFinite(entry.durationMs) && entry.durationMs > 0
      ? ` &middot; ${escapeHtml(formatDuration(entry.durationMs))}`
      : "";
    // Cancelled-run meta line surfaces captured-vs-target so a reader of
    // the panel can tell a 250-of-1000 partial run from a 0-of-1000
    // zero-data abort without expanding the entry. summary.handsPlayed
    // is what the server actually emitted; Number coerces a missing /
    // null value to NaN which Number.isFinite then catches, falling
    // back to 0. Same partial-vs-no-data distinction the hall-status
    // panel badge got in 594e7de, applied to the historical record.
    const cancelCaptured = Number(entry.summary && entry.summary.handsPlayed);
    const cancelCapturedHands = Number.isFinite(cancelCaptured) && cancelCaptured > 0 ? cancelCaptured : 0;
    const metaLine = cancelled
      ? `${escapeHtml(entry.request.heroStyle || "-")} &middot; ${formatInteger(cancelCapturedHands)} of ${formatInteger(entry.request.hands)} hands &middot; cancelled${durationStr} &middot; [${escapeHtml(pool)}] &middot; seed ${escapeHtml(entry.request.seed)}`
      : `${escapeHtml(entry.request.heroStyle || "-")} &middot; ${formatInteger(entry.request.hands)} hands &middot; ${formatSigned(entry.summary.heroNetChips)} chips${durationStr} &middot; [${escapeHtml(pool)}] &middot; seed ${escapeHtml(entry.request.seed)}`;
    return `
      <article class="recent-run">
        <div class="recent-run-head">
          <span class="recent-run-ts">${escapeHtml(ts)}${cancelledSuffix}</span>
          <span class="recent-run-actions">
            <button type="button" class="button button-secondary" data-recent-index="${idx}" aria-label="Load run from ${tsForLabel}${cancelled ? ", cancelled" : ""}">Load</button>
            <button type="button" class="button button-secondary recent-run-remove" data-recent-remove="${idx}" aria-label="Remove run from ${tsForLabel}${cancelled ? ", cancelled" : ""}" title="Remove">×</button>
          </span>
        </div>
        <p class="recent-run-meta">${metaLine}</p>
      </article>
    `;
  }).join("");
  hallRecentList.querySelectorAll("button[data-recent-index]").forEach(btn => {
    btn.addEventListener("click", () => {
      const entry = entries[Number(btn.dataset.recentIndex)];
      if (!entry) return;
      applyHallConfig(entry.request);
      // Recent runs capture the SEED THAT ACTUALLY RAN, even when the
      // submit toggled random-seed at the time. Loading a recent run is
      // the user explicitly asking to reproduce that captured state, so
      // also turn OFF the random-seed checkbox + re-enable the seed
      // input -- otherwise a user with random-seed currently checked
      // would see the captured seed populated but silently overridden
      // by a fresh Math.random() at submit time. Presets DO NOT do
      // this (they're configurable starting points, and a user toggling
      // random-seed deliberately would expect that preference to
      // survive a preset click), so the override lives at the recent-
      // run handler rather than inside applyHallConfig.
      if (hallRandomSeedInput && hallSeedInput && entry.request && entry.request.seed != null) {
        hallRandomSeedInput.checked = false;
        hallSeedInput.disabled = false;
        validateHallForm();
      }
      // Same a11y announcement the preset bar does. Loading a recent
      // run mutates 15+ form values plus the random-seed checkbox;
      // without an announcement screen readers and distant-focus
      // keyboard users have no signal that the click did anything.
      // The #hall-status div is aria-live=polite (moved down from
      // the parent .hall-panel article) so renderHallStatus
      // queues the announcement.
      const ts = new Date(entry.timestamp).toLocaleString();
      renderHallStatus(`Loaded run from ${ts}.`);
      // The Recent runs <details> sits ABOVE the hall form in DOM
      // order. When the user expands the panel to pick a saved run,
      // their viewport is on the entries -- and clicking Load leaves
      // them there, with the now-populated form scrolled off-screen
      // below. They have to manually scroll down to see what got
      // loaded, defeating the "I want to re-run this captured state"
      // intent that drove the Load click. Move focus to the hall
      // submit button: the browser's default focus-scroll lands the
      // button at the bottom of the viewport, which puts the
      // populated form fields visible above it. The user sees the
      // values they just loaded AND lands on the action button they
      // most likely want to click next; a Tab back is the cheap path
      // for the minority case of "load this then tweak one value
      // before running". Presets don't need this treatment -- the
      // preset bar lives INSIDE the hall form, so a preset click
      // never scrolls the form out of view to begin with.
      if (hallSubmitButton) {
        hallSubmitButton.focus();
      }
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
      // Restore keyboard focus after the renderRecentRuns rebuild
      // wipes the original Remove button. Without this, a keyboard
      // user removing entry N has their focus drop to document-start
      // (the Remove button that was focused no longer exists, so the
      // browser's default focus-restore lands on <body>), and the
      // next Tab moves to the Skip-to-content link -- forcing them
      // to re-Tab through the page to get back to the Recent runs
      // panel just to remove the next entry. Set focus to the entry
      // that now sits at the same index (i.e., the entry that was
      // just below the removed one), clamped to the last entry if
      // we removed the tail. If no entries remain after the removal,
      // focus the panel's own <summary> -- the disclosure toggle
      // remains visible and focusable, and lands the user on the
      // Recent runs region itself rather than dropping focus to
      // <body> (where Tab-from-body behavior varies across browsers:
      // some resume from where focus was, some reset to document
      // start). Empty-state focusing the summary also keeps screen-
      // reader users oriented -- they hear "Recent runs, 0, summary,
      // expanded" rather than the page-language announcement that
      // follows a focus-to-body drop.
      const remaining = hallRecentList.querySelectorAll("button[data-recent-remove]");
      if (remaining.length > 0) {
        const focusIdx = Math.min(idx, remaining.length - 1);
        remaining[focusIdx].focus();
      } else {
        const recentRunsDetails = document.getElementById("hall-recent-runs");
        const summary = recentRunsDetails ? recentRunsDetails.querySelector("summary") : null;
        if (summary) summary.focus({preventScroll: true});
      }
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
        // Mirror the per-entry Remove handler's empty-state focus
        // restore (see the comment above). After Clear all wipes
        // the list, renderRecentRuns rebuilds hallRecentList's
        // innerHTML and the Clear-all button itself is gone -- so
        // a keyboard user's focus would drop to document.body, and
        // their next Tab would resume from the Skip-to-content link
        // (or arbitrary position depending on the browser's focus-
        // from-body policy). Move focus to the Recent runs panel's
        // own <summary> instead -- the same target the per-entry
        // Remove handler uses when the last entry is removed.
        // preventScroll keeps the page position stable; the panel
        // sits above the hall form so the user's viewport doesn't
        // need to move. Screen-reader users also get a coherent
        // announcement ("Recent runs, 0, summary, expanded")
        // instead of the page-language announcement that follows
        // a focus-to-body drop.
        const recentRunsDetails = document.getElementById("hall-recent-runs");
        const summary = recentRunsDetails ? recentRunsDetails.querySelector("summary") : null;
        if (summary) summary.focus({preventScroll: true});
      }
    });
  }
}

// Cross-tab sync for the Recent runs panel. The WHATWG HTML spec
// fires `storage` events only in tabs OTHER than the one that wrote
// localStorage, so tab A pushing a new entry (via pushRecentRun on
// hall completion) lands in tab B's listener and re-renders the
// panel without a page refresh -- a user keeping the page open in
// multiple tabs (e.g., one for analyze, one for hall) sees runs
// from the sibling tab show up live instead of catching up only on
// the next reload. The current tab's own writes are correctly
// excluded by the spec, so pushRecentRun's existing inline
// renderRecentRuns call still owns the within-tab update with no
// double-render. Match RECENT_RUNS_KEY directly, OR event.key ===
// null which the spec fires on localStorage.clear() from another
// tab (no current call site, but defensive against a future
// 'reset all browser data' feature wiping the key).
window.addEventListener("storage", event => {
  if (event.key === null || event.key === RECENT_RUNS_KEY) {
    renderRecentRuns();
  }
});

// ----- Progress / elapsed timer / cancel -----

let hallElapsedTimer = null;
let hallActiveJobId = null;
let hallActiveStartedAt = 0;
// True between the Cancel click (when the DELETE returns 200) and the
// next poll seeing status=cancelled. Lets pollPlayingHallJob suppress
// the normal "Running the playing hall in the background..." status
// message in favor of a "Cancelling..." form, so a polling tick that
// fires after Cancel but before the server flips the job to cancelled
// doesn't overwrite the "Cancel accepted. Finishing in-flight hands..."
// message with "Running..." and make the user think their cancel was
// dropped. Reset on every startHallElapsed (new job) and
// finishHallProgress (any job winding down) so a fresh run starts
// clean.
let hallCancelRequested = false;

function startHallElapsed(jobId) {
  hallActiveJobId = jobId;
  hallActiveStartedAt = Date.now();
  hallCancelRequested = false;
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
  const hh = Math.floor(sec / 3600);
  const mm = String(Math.floor((sec % 3600) / 60)).padStart(2, "0");
  const ss = String(sec % 60).padStart(2, "0");
  // Promote to h:mm:ss format past one hour so a long-running hall
  // job doesn't display "Elapsed 120:00" once minutes overflow the
  // two-digit pad. With the shipped PLAYING_HALL_TIMEOUT_MS default
  // of 15 min, sub-hour is the only case in practice -- but the
  // deployment doc explicitly supports operators raising the knob
  // (and the dynamic-poll-deadline work extends maxPollWaitMs to
  // match), so a 2-hour-timeout configuration could otherwise show
  // an ambiguous three-digit-minute count where the layout expects
  // two. Common case (under one hour) stays mm:ss for visual
  // stability with the surrounding labels.
  hallElapsed.textContent = hh > 0
    ? `Elapsed ${hh}:${mm}:${ss}`
    : `Elapsed ${mm}:${ss}`;
}

function finishHallProgress() {
  stopHallElapsed();
  hallActiveJobId = null;
  hallCancelRequested = false;
  // If focus is currently inside the hall-progress region we're about
  // to hide (typical path: the user just clicked Cancel and the Cancel
  // button lives inside this card), the browser would drop focus to
  // document.body once the region becomes display:none. A keyboard /
  // screen-reader user would then have to Tab from the very top of
  // the page to reach anything useful. Match the logout-focus-restore
  // pattern from 948fccc: check if we own the active focus before
  // hiding, and if so move it to the hall submit button -- the
  // natural next action after a cancelled or completed run. By this
  // point setHallSubmitting(false) has already run from the parent
  // finally block, so the submit button is enabled and a sensible
  // target. preventScroll keeps the page position stable; the hall
  // form and progress card sit in adjacent viewport rows so the
  // submit button is already on screen. The check is skipped if
  // focus is outside the progress card (e.g. the user was reading
  // the Recent runs panel while the run finished on its own --
  // moving their focus would be a worse disruption than leaving
  // it alone).
  if (
    hallProgress &&
    hallProgress.contains(document.activeElement) &&
    hallSubmitButton
  ) {
    hallSubmitButton.focus({preventScroll: true});
  }
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
        // Any 2xx: server accepted the cancel. The current handler returns
        // 200 with a minimal `{jobId, status: "cancelled"}` body (see the
        // DELETE bullet in HAND_HISTORY_WEB_DEPLOYMENT.md -- intentionally
        // doesn't carry result/timestamps/durationMs; this is the cancel
        // ACKNOWLEDGMENT, not the final terminal state). `response.ok`
        // instead of `response.status === 200` so a future move to another
        // 2xx code (e.g. 202 Accepted to make the async nature explicit,
        // or 204 No Content if the body were ever dropped) doesn't break
        // this branch. The job is now winding down server-side; the
        // polling loop will see status="cancelled" within ~pollAfterMs
        // (capped at 5s) and trigger finishHallProgress, which is what
        // actually surfaces any partial result captured before interrupt
        // (via the next poll's full job-status response). Give the user
        // IMMEDIATE feedback so they know their click landed and the UI
        // didn't freeze on "Cancelling..." for the poll window. Set
        // hallCancelRequested so the next polling tick that fires BEFORE
        // the server flips the job to cancelled doesn't overwrite this
        // message with "Running the playing hall in the background..."
        // (which would make the user think the cancel was lost). The
        // poll-status renderer reads this flag and uses a "Cancelling..."
        // form for queued/running statuses while it's set.
        hallCancelRequested = true;
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
