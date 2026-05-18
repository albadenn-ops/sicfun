# SICFUN Hand-History Web Deployment

This guide is for the packaged Windows hand-history review product under `dist/hand-history-web`.

## Two Release Variants

Two release scripts produce two artifact shapes. Both serve the same web app; the difference is what they bundle for the host.

| Script | Output | Java handling | Click-to-run |
|---|---|---|---|
| `scripts/release-hand-history-web.ps1` | `dist/hand-history-web/` directory only | Requires Java 17+ on `PATH` | No |
| `scripts/release-hand-history-web-installer.ps1` | Same directory **plus** versioned outer ZIP at `dist/hand-history-web-<version>.zip` | Embeds a jlink runtime under `runtime/bin/java.exe`; falls back to PATH if missing | Yes: `Setup.cmd` verifies the manifest then launches the service |

Use the installer variant when the target host may not have Java pre-installed.

## What You Ship

The release bundle contains:

- `bin/run-hand-history-web.ps1`: start the app directly
- `bin/install-hand-history-web-service.ps1`: install an NSSM-backed Windows service
- `bin/start-hand-history-web-service.ps1`: start the service and wait for readiness
- `bin/drain-stop-hand-history-web-service.ps1`: drain and stop the service cleanly
- `bin/uninstall-hand-history-web-service.ps1`: remove the service
- `bin/verify-release-manifest.ps1`: verify bundle integrity against `manifest.sha256`
- `conf/hand-history-web.env`: runtime configuration template
- `static/`: packaged upload UI
- `lib/`: application jars
- `model/`: optional packaged model artifacts if they were included at build time

Installer-variant additions:

- `Setup.cmd`: click-to-run entry point that verifies the manifest then launches the service
- `runtime/`: jlink-trimmed Java runtime; the launcher and service-common shim prefer this over `PATH`
- `runtime/BUILD_INFO.txt`: source JDK version and jdeps-detected module list

## Host Prerequisites

- Windows host
- PowerShell 5.1+
- Java 17+ on `PATH` before startup (not required for the installer variant — `runtime/` is embedded)
- NSSM if you want the Windows service workflow (drop `nssm.exe` into `bin/` before zipping or pass `-NssmPath` at install time; auto-download is disabled because corporate AV often blocks `nssm.cc`)
- Client browser: a modern build (Chrome 103+, Edge 103+, Firefox 100+, Safari 16+) gets the one-line `AbortSignal.timeout` fetch-timeout path. Older browsers fall back to an `AbortController + setTimeout` shim that delivers the same fetch-timeout safety net; that primitive shipped in Chrome 66 / Firefox 57 / Safari 11.1 (2018+) which is below the page's actual rendering floor, so every browser that can run the page also gets the timeout. `String.prototype.replaceAll` (used by HTML escaping) requires Chrome 85+ / Edge 85+ / Firefox 77+ / Safari 13.1+ — that's the real floor below which the page won't render at all. Modern Edge is Chromium-based so it tracks the Chrome version.

## Quick Start

From the package root:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\verify-release-manifest.ps1
powershell -ExecutionPolicy Bypass -File .\bin\run-hand-history-web.ps1
```

Default behavior:

- Binds to `127.0.0.1:8080`
- Serves the upload UI from `static/`
- Loads settings from `conf/hand-history-web.env` (override the file location via `-ConfigFile <path>` on the launcher CLI or `CONFIG_FILE=<path>` in the environment — useful when one host runs multiple sicfun instances on different ports with sibling config files)
- Exposes `/api/health` for liveness and `/api/ready` for readiness
- Exposes async job endpoints for hand-history analysis and Playing Hall simulation (see "HTTP Endpoints" below)

## HTTP Endpoints

`/api/health` and `/api/ready` are unauthenticated and accept `GET`, `HEAD`, and `OPTIONS`. HEAD returns the same status code and security headers as GET but with no body, so monitoring tools that probe with HEAD see `200 OK` while idle and `503` when readiness fails — same as a GET probe. The hand-history analysis and Playing Hall job routes enforce whichever auth mode is configured (Basic auth or platform-user); on a no-auth deployment (loopback default, or non-loopback with `ALLOW_UNAUTHENTICATED_PUBLIC_BIND=true`) they accept anonymous requests. Either way they are rate-limited (Submit / JobStatus buckets — see "Rate limiting" below); rate limiting runs independently of auth. Auth bootstrap routes under `/api/auth/*` are open by design (login/register cannot require an existing session). Submissions return `202 Accepted` with `Location` and `Retry-After` headers plus a JSON body containing `jobId`, `status` (always the literal string `"queued"` on a fresh `202` — the worker hasn't started yet; the GET poll endpoint surfaces later state transitions), `statusUrl`, `submittedAtEpochMs`, `pollAfterMs`.

The read-only endpoints `/api/auth/me`, `/api/analyze-hand-history/jobs/{jobId}`, and `/api/playing-hall/jobs/{jobId}` also accept `HEAD` alongside `GET`. The HEAD response carries the same status code and headers (security headers, `Allow`, `Retry-After` on 503, etc.) as the corresponding GET response but with no body, so a probe that only needs to check existence or rate-limit posture can skip the JSON payload. Static asset routes likewise support HEAD with the same `ETag` / `Last-Modified` / `Content-Encoding` advertisement as a GET would emit.

All body-reading endpoints (`POST /api/auth/*`, `POST /api/analyze-hand-history`, `POST /api/playing-hall`) require `Content-Type: application/json` and return `415 Unsupported Media Type` otherwise. The `application/json; charset=utf-8` form is also accepted. The check forecloses a cross-origin form-CSRF vector against the login endpoint (a hostile site auto-submitting an HTML `<form action="/api/auth/login">` would send `application/x-www-form-urlencoded` and now bounces at the Content-Type guard before any parsing). The same endpoints also reject `Content-Encoding` values other than absent or `identity` with 415; the server does not decompress request bodies, so accepting `Content-Encoding: gzip` would either produce a confusing "invalid JSON" 400 or, worse, open a zip-bomb vector against `MAX_UPLOAD_BYTES`.

Error responses across the JSON API (400 invalid input, 401 unauthorized, 403 CSRF / forbidden, 404 unknown / not-yours, 409 already terminal, 413 oversize, 415 wrong Content-Type/Encoding, 429 rate-limited, 503 draining / queue full) all return `Content-Type: application/json; charset=utf-8` with a body of `{"error": "<message>"}`. 429 responses additionally include `"retryAfterSeconds": N` alongside `error`, and the `Retry-After` header carries the same integer. 4xx/5xx bodies are stable enough for scripted clients to display `body.error` verbatim, with the caveat that the messages are operator-readable English rather than i18n-keyed.

Hand-history analysis:

- `POST /api/analyze-hand-history` — submit a hand history for analysis. Body: JSON `{handHistoryText, heroName?, site?}`. `handHistoryText` is required (whitespace-only and BOM-only payloads are rejected with `400`); `heroName` is capped at 64 chars and may not contain C0 control characters or DEL (defense in depth against log injection if any future code path logs the value); `site` accepts one of the recognised aliases (`pokerstars`/`stars`/`winamax`/`wina`/`ggpoker`/`gg`/`ggnetwork`), or the literal string `auto` / an absent/empty value to ask the server to auto-detect from the file's first non-empty line, and is otherwise capped at 64 chars before any other validation runs. Rate-limited as Submit.
- `GET /api/analyze-hand-history/jobs/{jobId}` — poll job status. Returns `200` with `status=queued` / `running` / `completed` / `failed`, the universal fields `jobId` / `statusUrl` / `submittedAtEpochMs` / `startedAtEpochMs` / `completedAtEpochMs` (the latter two are `null` until the worker reaches each transition), and `pollAfterMs` while the job is still queued or running. Terminal states add `durationMs` (worker run time, computed from `completedAtEpochMs - startedAtEpochMs`). Completed responses additionally include `result` (the analyzer payload — summary, decisions, opponents, warnings, model source). Failed responses include `errorStatus` (HTTP-equivalent: `500` backend exception or `504` timeout) and `error` (text) inside the JSON body. `404` if the job is unknown (never existed, expired off the 15-min terminal-job retention, or — under platform-user auth — was submitted by a different user). Rate-limited as JobStatus.

Playing Hall simulation:

- `POST /api/playing-hall` — submit a hall simulation. All JSON body fields are optional; omitted values fall back to documented defaults. Numeric ranges are enforced server-side with `400` on out-of-range values:
  - `hands` (1..5000, default 240), `tableCount` (1..24, default 2), `playerCount` (2..9, default 6)
  - `heroStyle` (`adaptive`/`gto`/`strategic`, default `adaptive`)
  - `heroPosition` (`SmallBlind`/`BigBlind`/`UTG`/`UTG1`/`UTG2`/`Middle`/`Hijack`/`Cutoff`/`Button`, default `Button`)
  - `gtoMode` (`fast`/`exact`, default `exact`)
  - `villainPool` (array of 1..8 entries from `nit`/`tag`/`lag`/`station`/`callingstation`/`maniac`/`gto`, each entry capped at 32 chars; default `[tag, gto]`)
  - `heroExplorationRate` (0.0..1.0, default 0.0), `raiseSize` (0.25..20.0, default 2.5)
  - `bunchingTrials` (1..600, default 40), `equityTrials` (1..6000, default 240)
  - `learnEveryHands` (0..5000, default 0), `learningWindowSamples` (0..500000, default 200)
  - `seed` (any Long, default 42), `saveReviewHandHistory` (bool, default false), `fullRing` (bool, default false)

  Rate-limited as Submit; admission shares the same `MAX_CONCURRENT_JOBS` / `MAX_QUEUED_JOBS` budget as hand-history analysis.
- `GET /api/playing-hall/jobs/{jobId}` — poll job status. Returns `200` with `status=queued` / `running` / `completed` / `failed` / `cancelled` (cancelled is reachable only via `DELETE` below; analyze jobs share the rest of the state set but never reach cancelled), the universal fields `jobId` / `statusUrl` / `submittedAtEpochMs` / `startedAtEpochMs` / `completedAtEpochMs` (the latter two are `null` until the worker reaches each transition), and `pollAfterMs` while the job is still queued or running. Terminal states add `durationMs` (worker run time; on cancelled responses, only present when the worker had already started before the interrupt landed). Completed responses additionally include `result` (the hall payload — summary, request mirror, per-hand hero deltas, equity histogram, opponent rollups). Failed responses include `errorStatus` (`500` / `504`) and `error` text; cancelled responses include partial `result` when the worker captured it before interrupt. `404` if the job is unknown (never existed, expired off retention, or owned by a different user under platform-user auth). Rate-limited as JobStatus.
- `DELETE /api/playing-hall/jobs/{jobId}` — request cooperative cancellation of an in-flight Playing Hall job. Returns `200` with `status=cancelled` while the job is still cancellable (queued or running), `409` if it already reached a terminal state (completed / failed / cancelled — re-DELETEing an already-cancelled job returns 409, not 200), `404` if unknown / expired / cross-user. Rate-limited as JobStatus.

Platform-user auth (when `USER_STORE_PATH` is set):

- `GET /api/auth/me`, `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`, `POST /api/auth/profile`
- `POST /api/auth/register` and `POST /api/auth/login` are rate-limited as Auth (see `RATE_LIMIT_AUTH_PER_MINUTE` below; default 10/min/IP). Throttles credential-stuffing and slow-disk-fill registration abuse without burning a PBKDF2 verify per attempt. `/me`, `/logout`, and `/profile` are not rate-limited — they require an authenticated session, so the same-IP flood vector is bounded by sign-in cost.
- `GET /api/auth/me` returns the live auth-state JSON: `authenticationEnabled` (bool), `authenticationMode` (`users` / `basic` / `none`), `authenticated` (bool), `allowLocalRegistration` (bool — frontends gate the Register button on this), `providers` (array of OIDC provider summaries, each `{id, startPath}` where `startPath` is the `/api/auth/oidc/<id>/start` route the frontend Continue-with-X link targets), `user` (the signed-in account view with `userId` / `email` / `displayName`, plus optional `heroName` / `preferredSite` / `timeZone` / `avatarUrl` and the `linkedProviders` array naming any OIDC ids linked to the account; `null` when not signed in), and `csrfToken` (string the client must echo in `X-CSRF-Token` for state-changing requests under platform-user auth; `null` outside platform-user mode or when no session is on the request). The exact same shape is returned by every state-changing platform-user response: successful `POST /api/auth/register` (`201`) carries the newly-created session's state, `POST /api/auth/login` (`200`) the just-established session, `POST /api/auth/logout` (`200`) the post-signout degenerate state (`authenticated: false`, `user: null`, `csrfToken: null` — useful for the client to reset local state without an extra `/me` round trip), and `POST /api/auth/profile` (`200`) the same state with the `user` field reflecting the newly-saved profile values (also includes a fresh `csrfToken` for the next state-changing request). Under basic-auth and no-auth modes `/api/auth/me` returns a degenerate form (`authenticated: false`, `providers: []`, `user: null`, `csrfToken: null`) so a frontend can switch deployment modes from the same probe.
- Optional OIDC start/callback at `GET /api/auth/oidc/{provider}/start` and `GET /api/auth/oidc/{provider}/callback` (e.g. `GET /api/auth/oidc/google/callback`) when the matching `*_OIDC_*` settings are configured. Both are GET-only top-level redirects (the user agent receives a `302 Location: <provider>` from `/start`, the provider sends the user back to `/callback?state=...&code=...` via a navigation, and `/callback` issues a final `302` to `/?auth=success` or `/?auth_error=...`). Non-GET methods return `405 Allow: GET, OPTIONS`.
- The `register` / `login` JSON responses include a `csrfToken` field. State-changing routes under platform-user auth (`POST /api/auth/logout`, `POST /api/auth/profile`, `POST /api/analyze-hand-history`, `POST /api/playing-hall`, `DELETE /api/playing-hall/jobs/{id}`) require this token in the `X-CSRF-Token` request header in addition to the session cookie; missing or mismatched values produce `403` `"missing or invalid csrf token"`. Scripted clients can either re-call `GET /api/auth/me` to refresh the token or stash it from the original login response. The OIDC callback also issues a fresh token via the same response shape.

## Core Configuration

Edit `conf/hand-history-web.env` instead of hard-coding long-lived settings into service command lines.

Common settings:

- `HOST` / `PORT`: bind address and port (default `127.0.0.1` / `8080`)
- `ALLOW_UNAUTHENTICATED_PUBLIC_BIND` (default `false`): explicit override for non-loopback binds without auth on a trusted private network. The server otherwise refuses to bind to a non-loopback host without one of the auth modes configured.
- `STATIC_DIR`: directory served as the upload UI (default `docs/site-preview-hybrid` in source mode; `static/` relative to the bundle root after `release-hand-history-web.ps1` packages it). Override only when you've copied or modified the static assets to a non-standard location.
- `MODEL_DIR`: optional path to a packaged action-model artifact directory. When set, the analyze backend loads it for EV-gap and recommended-line scoring; when unset, the backend falls back to a uniform action model and the analyze results lose their model-specific edge (recommendations are correct in shape but undifferentiated). `/api/health` surfaces this as `modelSource = "configured artifact dir"` vs `"uniform fallback"` so dashboards can flag instances accidentally running without their intended model.
- `MAX_UPLOAD_BYTES`: upload cap in bytes (default `2097152`, i.e. 2 MiB; applies to both `/api/analyze-hand-history` and `/api/playing-hall` request bodies). The frontend probes `/api/health` at page load and adopts this value as its client-side file-size check, so raising the server cap automatically raises what the upload form accepts without a frontend rebuild.
- `ANALYSIS_TIMEOUT_MS`: per-job timeout for `/api/analyze-hand-history` jobs (default `120000`, i.e. 2 min); `0` disables it. The shipped frontend probes `/api/health.analysisTimeoutMs` at boot and rolls it into the same `max(16 min default, server timeout + 1 min slack)` poll deadline described under `PLAYING_HALL_TIMEOUT_MS` below, so raising this knob does NOT need a frontend rebuild.
- `PLAYING_HALL_TIMEOUT_MS`: per-job timeout for `/api/playing-hall` jobs (default `900000`, i.e. 15 min); `0` disables it. The shipped frontend probes `/api/health.playingHallTimeoutMs` at boot and extends its poll deadline to `max(16 min, server timeout + 1 min slack)`, so raising this knob on the server side automatically extends the frontend's patience without a frontend rebuild. `0` (disabled) on the server side falls back to the 16-min frontend default — the page won't poll indefinitely on a bottomless job.
- `MAX_CONCURRENT_JOBS` / `MAX_QUEUED_JOBS`: admission limits shared across both job stores. Defaults derive from host CPU: concurrent is `clamp(availableProcessors - 1, 1, 4)` (so a 2-core host gets 1, an 8-core host gets 4, and a 32-core host still gets 4 — the cap stops the worker pool from contending with the JVM's own threads), and queued is `max(8, concurrent * 8)`. A small dual-core deployment lands at 1 concurrent + 8 queued; a typical 8-core lands at 4 + 32.
- `SHUTDOWN_GRACE_MS`: graceful shutdown budget (default `5000`, i.e. 5 sec)
- `DRAIN_SIGNAL_FILE`: path used to mark the instance unready before shutdown; while present, new submissions for both analysis and Playing Hall are rejected with `503`

Auth modes:

- Basic auth: set `BASIC_AUTH_USER` and `BASIC_AUTH_PASSWORD` (both required together — setting one without the other fails startup with a clear error). The username may not contain `:` (HTTP Basic uses it as the user:password separator per RFC 7617), and both values are trimmed; empty / whitespace-only values are treated as unset.
- Platform-user auth: set `USER_STORE_PATH`
- `ALLOW_INSECURE_USER_AUTH` (default `false`): explicit override for non-loopback platform-user auth without secure cookies or an HTTPS OIDC callback on a trusted private-network test deployment. The server otherwise refuses to start when `USER_STORE_PATH` is set on a non-loopback host without `USER_AUTH_COOKIE_SECURE=true`.
- Do not enable both at the same time
- For safety, non-loopback binds now require one of those auth modes unless you explicitly set `ALLOW_UNAUTHENTICATED_PUBLIC_BIND=true` for a trusted private network
- For safety, non-loopback platform-user auth also requires `USER_AUTH_COOKIE_SECURE=true` unless you explicitly set `ALLOW_INSECURE_USER_AUTH=true` for trusted private-network testing
- When `USER_AUTH_COOKIE_SECURE=true` the session cookie is emitted with the RFC 6265 `__Host-` prefix (`__Host-sicfun_session`) so the browser also blocks a sibling subdomain from overwriting or planting the session cookie. Operator-side log greps and proxy ACLs should account for both `sicfun_session` (insecure mode) and `__Host-sicfun_session` (secure mode). The full session-cookie attribute set is `Path=/; Max-Age=<USER_AUTH_SESSION_TTL_MS/1000>; HttpOnly; SameSite=Lax` plus `Secure` when cookie-secure is on — the same defense-in-depth shape as the OIDC state cookie, minus the 10-min Max-Age cap (the session uses the configured sliding TTL instead).
- `USER_AUTH_COOKIE_SECURE` defaults to `false` so loopback testing over plain HTTP works (a `Secure` cookie would be discarded by the browser on `http://127.0.0.1`). The non-loopback bind check above flips this knob to required-true at startup so internet-facing deployments don't accidentally ship plain-HTTP cookies.
- `USER_AUTH_MAX_USERS` (default `100000`) caps the total number of registered users. Once reached, further registration attempts return `400` with `"registration is temporarily unavailable"`. Defends against slow disk-fill via public-registration abuse (the auth rate-limit alone caps in-flight attempts at 10/min/IP, but without a per-store cap a persistent bot can still grow the JSON file unbounded over days/weeks). Surfaced in `/api/health` as `userAuthMaxUsers` so dashboards can show "users used / max" and alert when capacity is being approached.
- `USER_AUTH_ALLOW_REGISTRATION` (default `true`): when `false`, `POST /api/auth/register` returns `400` with `"local registration is disabled"` regardless of the cap. Use this for closed deployments where accounts are minted out-of-band (e.g. via an operator-only flow that calls `registerLocal` directly) and the public registration form should be locked. OIDC sign-up is still allowed since it doesn't go through `/api/auth/register`. The `/api/auth/me` response surfaces this as `allowLocalRegistration: false` so the frontend disables the Register button.
- `USER_AUTH_SESSION_TTL_MS` (default `43200000`, i.e. 12 hours): sliding TTL on the session cookie. Every authenticated request resets the expiry, so a continuously-active user stays signed in indefinitely; an idle user is signed out after this many milliseconds without a successful resolveSession. Shorter TTLs trade convenience for tighter post-leak windows; longer TTLs trade exposure for less re-auth friction.

Optional OIDC:

- Set `GOOGLE_OIDC_CLIENT_ID`, `GOOGLE_OIDC_CLIENT_SECRET`, and `GOOGLE_OIDC_REDIRECT_URI`. OIDC layers on top of platform-user auth so `USER_STORE_PATH` must also be set; otherwise startup fails with `user auth with OIDC requires --userStorePath/USER_STORE_PATH`. All three Google knobs are required together (configuring just one or two also fails at startup). The redirect URI's PATH component must be exactly `/api/auth/oidc/google/callback` (the path the server registers its callback handler on); any other path -- e.g. `https://example.com/oidc-return` -- fails startup with `--googleOidcRedirectUri/GOOGLE_OIDC_REDIRECT_URI path must be '/api/auth/oidc/google/callback'`. Configure the same path in the Google Cloud Console's Authorized Redirect URIs.
- On non-loopback binds, set `USER_AUTH_COOKIE_SECURE=true` and use an `https://...` redirect URI unless you explicitly set `ALLOW_INSECURE_USER_AUTH=true` for trusted private-network testing
- OIDC vs local-password collision: if a user already has a local-password account with email X, an OIDC sign-in returning the same email X is rejected with `?auth_error=an account with that email already exists; sign in with its existing method` (translated to a friendlier message by the bundled UI). The collision is deliberate -- linking the two identities silently would let any party who controls the Google account claim the password user's data, and silently overwriting either way would surprise the user. Operators triaging "OIDC sign-in suddenly stopped working" support tickets should ask whether the user has a prior local-password account on the same email; if so, the user should sign in with their password instead, OR the operator can manually clear the local-password account from `USER_STORE_PATH` (the OIDC flow will create a fresh user record on the next sign-in).

Rate limiting:

- `RATE_LIMIT_SUBMITS_PER_MINUTE` — cap on analysis/Playing Hall submissions per IP per minute (defaults to `6`; bounds the worker pool against burst-submit abuse without throttling a human operator queuing a handful of files). Under platform-user auth the bucket keys on the signed-in user (`user:<userId>`) instead of the IP, so a noisy authenticated user can't throttle unrelated users on the same NAT -- see the rate-limit `client=` log-line documentation below for the full keying rules.
- `RATE_LIMIT_STATUS_PER_MINUTE` — cap on job-status polls per IP per minute (defaults to `240`; the server-suggested poll cadence is `750 ms` so one tab steady-state polls ~80 times/min, and the 240 cap leaves headroom for ~3 concurrent tabs from the same IP or one tab plus a sidecar monitor). Same per-user keying override as SUBMITS under platform-user auth, so the headroom math is per-signed-in-user there rather than per-IP.
- `RATE_LIMIT_AUTH_PER_MINUTE` — cap on `/api/auth/register` + `/api/auth/login` attempts per IP per minute (defaults to `10`; throttles PBKDF2-cost credential stuffing). Stays IP-keyed across all auth modes because there's no authenticated user yet at the register/login step.
- All three buckets use a **fixed 60-second window** anchored at each client's first observed request per bucket (not a sliding window). Operationally this means a client can burst up to ~2× the per-minute limit right at a window boundary -- six submits between `T+55s` and `T+60s` plus six more in the new window starting around `T+61s` both go through and the client only sees `429` from `T+62s` onward. The fixed-window choice keeps the in-process limiter to a single `ConcurrentHashMap.compute` per request; if you need true sliding-window smoothing for fleet-wide burst control, do it at the upstream proxy (nginx `limit_req`, HAProxy `stick-table`, etc.) and treat the in-process caps as a defense-in-depth backstop.
- `RATE_LIMIT_CLIENT_IP_HEADER` — name of a single-value header the reverse proxy injects with the real client IP (e.g. `X-Real-IP`). When set and the request's TCP peer is loopback or in `RATE_LIMIT_TRUSTED_PROXY_IPS`, the rate limiter and audit-log `remote=` field both key on the trusted-header value instead of the proxy's peer address, so a single grep for one client IP correlates rate-limit rejections AND the auth events that triggered them. The header value must be ONE IP literal -- comma-separated chains (the natural form of `X-Forwarded-For` for multi-hop proxies, e.g. `1.2.3.4, 5.6.7.8`) are deliberately ignored, and the server falls back to the TCP peer; configure the proxy to extract the left-most untrusted hop and re-emit it as a single-value header (most proxies expose `X-Real-IP` or equivalent for exactly this), or place the rate limiter knob behind a proxy that already does the unwrap. Do NOT set this if the app is directly internet-facing -- clients can spoof the header.
- `RATE_LIMIT_TRUSTED_PROXY_IPS` — comma-separated allowlist of proxy peer IP literals authorized to inject `RATE_LIMIT_CLIENT_IP_HEADER`. Same-host loopback proxies are trusted automatically; this knob is for proxies on a different host. The IP comparison is exact (no CIDR support); list each proxy explicitly.

Set any of the three caps to `0` to disable that bucket. Only trust a client IP header when the app is behind a proxy you control.

## Service Install

Install:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\install-hand-history-web-service.ps1 `
  -NssmPath C:\tools\nssm\nssm.exe
```

Start and wait for readiness:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\start-hand-history-web-service.ps1
```

Drain and stop:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\drain-stop-hand-history-web-service.ps1
```

Uninstall:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\uninstall-hand-history-web-service.ps1
```

The service helper scripts write stdout/stderr logs under `logs/`.
If startup fails or readiness times out, `bin/start-hand-history-web-service.ps1` now includes the current service state plus recent stdout/stderr tail in its error output to shorten first-response debugging.
If drain/stop times out, `bin/drain-stop-hand-history-web-service.ps1` now reports the last readiness/health probe summary before forcing the stop.

## Reverse Proxy And Exposure

- Keep the raw app on loopback unless you intentionally want LAN exposure
- If you intentionally bind to `0.0.0.0`, a LAN IP, or another non-loopback host without auth, set `ALLOW_UNAUTHENTICATED_PUBLIC_BIND=true` in `conf/hand-history-web.env` so that choice is explicit
- Put HTTPS and stronger edge rate limiting in front of the app for internet-facing deployments
- Use `/api/ready` for load balancers and service managers
- Use `/api/health` for liveness and coarse metrics
- Configure the proxy to **preserve, not strip**, the response headers the origin emits:
  - `Content-Encoding: gzip` + `Vary: Accept-Encoding` — the origin negotiates gzip itself; stripping `Vary` would cause the proxy to serve gzipped bodies to clients that don't accept gzip
  - `ETag` + `Last-Modified` — preserve so the proxy and client can revalidate; the origin honors `If-None-Match` and `If-Modified-Since` and replies `304` to save bandwidth
  - `Cache-Control` — `no-store` on API responses (do not cache), `public, max-age=...` on static assets (safe to cache and revalidate)
  - `Content-Security-Policy`, `Permissions-Policy`, `X-Frame-Options`, `Referrer-Policy`, `X-Content-Type-Options`, `Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy`, `X-Robots-Tag` — defense-in-depth headers the origin sets on every response. A proxy that drops them weakens the browser-side protections; in particular `X-Robots-Tag: noindex, nofollow` prevents search engines from indexing the app if the deployment is accidentally reachable from the public internet.
- The origin does NOT emit `Strict-Transport-Security` because it does not terminate TLS. For internet-facing deployments behind an HTTPS-terminating proxy, configure the proxy to add an HSTS header (e.g. `Strict-Transport-Security: max-age=31536000; includeSubDomains`). Recommended only AFTER the deployment is verified to work over HTTPS — once `max-age` is in flight a browser refuses to fall back to HTTP for that host, so a misconfigured HSTS during testing can lock you out of plain-HTTP access until the header's TTL expires. For private-network/loopback deployments, HSTS is unnecessary.

## Logs And Observability

- The service helper scripts route stdout/stderr to files under `logs/`. Tail those or pipe to a log aggregator.
- `/api/health` reports configuration + queue state (auth mode, rate-limit caps, max concurrent/queued jobs, `analysisTimeoutMs`, `playingHallTimeoutMs`, `maxUploadBytes`, active HTTP requests, queued/running jobs, timed-out workers still unwinding, retained terminal jobs, the machine-readable readiness reason as `readyReason` -- same enum as `/api/ready`'s `reason` field (`accepting-traffic` / `draining` / `timed-out-worker` / `queue-full`) but surfaced under a different field name on each endpoint, so a dashboard parsing both probes reads `readyReason` here and `reason` there, plus `service`/`host`/`port`/`startedAtEpochMs`/`uptimeMs` for fleet correlation). Under platform-user auth it also reports user-store metrics: `userAuthMaxUsers` (the cap), `userAuthStoredUsers` (current count -- dashboard at e.g. 80%/90% to spot capacity pressure before legitimate registrations fail), `userAuthActiveSessions` (live in-memory session count -- an off-hours spike is a credential-stuffing-success / leaked-automation heads-up), and `userAuthPendingOidcFlows` (started but not callback-completed OIDC flows -- a /start storm with no completions usually means a misconfigured provider redirect URI). All four are `null` in basic-auth and no-auth modes. Use it for dashboards.
- `/api/ready` is 200 only when the instance is accepting traffic; 503 otherwise (drain mode, queue full, or stuck worker). Use it for load-balancer health checks.
- Auth events emit structured log lines: `auth.login.success`, `auth.login.failure`, `auth.register.success`, `auth.register.failure`, `auth.logout`, `auth.oidc.start`, `auth.oidc.start.failure`, `auth.oidc.success`, `auth.oidc.failure`. INFO level for success/expected events, WARN for failures. Each line carries `email=` (where applicable) and `remote=` so a high WARN rate from a single `remote=` reveals brute-force or credential-stuffing attempts. Success lines log the **canonical** (normalized: trim + lowercase) email so all events for one user grep identically; failure lines log the **submitted** email so brute-force probes are visible in the form the attacker typed.
- `remote=` is the **resolved client identity**, not necessarily the TCP peer. When `RATE_LIMIT_CLIENT_IP_HEADER` is set and the peer is loopback or in `RATE_LIMIT_TRUSTED_PROXY_IPS`, `remote=` is the IP the trusted header resolves to (no port — `X-Forwarded-For`/`X-Real-IP` carry none). Otherwise it's the direct TCP peer as `host:port` with IPv6 bracketed per RFC 3986 §3.2.2. This matches the rate-limit `clientKey=` field so an operator grepping for one IP sees both rate-limit rejections AND the auth events that triggered them.
- The OIDC `/start` endpoint Set-Cookies a short-lived (10-minute, `HttpOnly`, `SameSite=Lax`) `sicfun_oidc_state` cookie bound to the random state value embedded in the provider redirect URL. The `/callback` endpoint requires the cookie to match the URL `state` parameter before exchanging the authorization code — an OAuth 2.0 BCP "covert-redirect" / login-CSRF mitigation. If the cookie is missing or mismatched, the callback redirects to the failure landing page with `?auth_error=missing_state_cookie` or `?auth_error=state_cookie_mismatch` rather than calling the provider's token endpoint. The cookie also carries the `__Host-` prefix when `USER_AUTH_COOKIE_SECURE=true`, the same hardening applied to the session cookie. The redirect URI also sets `prompt=select_account` on Google's authorization endpoint, so every `/start` forces Google's account picker even when the user has exactly one Google account already signed in — useful for multi-account users (Work vs personal Google), and an explainer for the "why does it ask me every time?" support question on single-account users.
- Job lifecycle events log to stderr at INFO/WARN with specific prefixes a log-monitoring rule can match on: `job accepted`, `job started`, `job completed` (all INFO), `job rejected unavailable`, `job rejected queue full`, `job failed`, `job timed out` (all WARN). Playing Hall jobs emit a parallel set under the prefix `playing hall job ...`, plus `playing hall job cancelled` (INFO) when the DELETE-cancel path interrupts a running worker. Each line carries `jobId=` plus a context-appropriate subset of `queuedJobs=` / `runningJobs=` / `durationMs=` / `errorStatus=` / `timeoutMs=` / `error=` / `reason=` so an operator can chart submit→start latency, completion-rate, timeout-rate, and queue depth from the audit log alone.
- Rate-limit rejections log as `request rate limited path=... client=... bucket=... limitPerMinute=... retryAfterMs=...` at WARN. `client=` is `user:<userId>` when a platform-user session is on the request, `header:<ip>` when `RATE_LIMIT_CLIENT_IP_HEADER` is configured and the TCP peer is in the trusted set, or `remote:<addr>` (TCP peer) otherwise — the user-keyed form means a noisy authenticated user does not throttle unrelated users behind the same NAT. `bucket=` is one of `submit` / `job-status` / `auth`.
- Unauthenticated requests against protected paths log as `request unauthorized path=... remote=... reason=...` at WARN.
- CSRF-failed state-changing requests (POST/DELETE on auth/logout, auth/profile, analyze-hand-history, playing-hall) log as `request forbidden path=... remote=... email=... reason=csrf-missing-or-invalid` at WARN. A burst from one `remote=` against many endpoints is a likely automated probe; the same from a known user usually means their JS frontend lost the cookie mid-session.
- NSSM rotates the service stdout/stderr logs at 10 MB (`AppRotateBytes=10485760`, `AppRotateOnline=1`). Rotated files keep their timestamp suffix in `logs/`; nothing deletes them automatically, so prune or ship them off the host periodically if the audit history matters and the partition is small.
- Structured-log values that could legitimately contain a space — submitted email at register/login, analyze `heroName=`, startup banner `modelSource=` / `drainSignalFile=` / `rateLimitClientIpSource=`, plus the `reason=` field in admission-rejected (`job rejected unavailable`) and auth-failure (`auth.register.failure`, `auth.login.failure`, `auth.oidc.start.failure`, `auth.oidc.failure`) log lines, and the `error=` field in `job failed` lines — are `%20`-escaped before they reach the log line. Without this, a Windows path like `C:\Program Files\model`, a hero name like `"Alice Smith"`, or a backend message like `analysis timed out after 120000ms` would split the surrounding key=value pairs and confuse log aggregators that tokenize on whitespace. Operator-facing: when you grep for an email, path, or other space-bearing value in the logs, type the value with `%20` in place of any space.

## State And Backups

- In-flight and completed review jobs are in-memory only
- A restart drops queued/running jobs
- Platform-user auth persists account/profile data under `USER_STORE_PATH`. The file contains email addresses, PBKDF2-HMAC-SHA256 password hashes (salted, 210k iterations -- not plaintext), profile fields (displayName, heroName, preferredSite, timeZone), OIDC subject identifiers, per-account UUIDs (internal `userId`), and event timestamps (`createdAtEpochMs` / `updatedAtEpochMs` / `lastLoginAtEpochMs`). Treat it as PII: restrict filesystem permissions to the service account (e.g. `icacls` on Windows or `chmod 600` + correct owner on Linux), encrypt backups, and keep it off any tier where unprivileged readers could grep it. A leaked store is not a credential leak (the hashes resist offline cracking), but the linked emails + OIDC subjects + profile fields are still PII a compliance audit would flag.
- Back up `conf/hand-history-web.env` and `USER_STORE_PATH` if they matter operationally. `hand-history-web.env` contains the Basic-auth credentials and/or `GOOGLE_OIDC_CLIENT_SECRET` in plaintext if either is configured; back up with the same protections you give other secrets.

## Bundle Integrity

Verify the shipped bundle after copy/deploy:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\verify-release-manifest.ps1
```

That script checks every shipped file against `manifest.sha256` and fails if a file is missing, added, or modified. Runtime marker files at the bundle root (filenames starting with `.`, e.g. `.manifest-verified` written by the installer-variant's `Setup.cmd`) are intentionally ignored.

The installer variant's `Setup.cmd` calls `bin\verify-if-needed.ps1` instead of running the full verifier on every launch. That helper writes a `.manifest-verified` marker after a successful check and short-circuits subsequent launches as long as the marker's timestamp is at least as new as `manifest.sha256`. Re-extracting the bundle advances `manifest.sha256`'s mtime and triggers a fresh verification. To force a re-verify manually, delete `.manifest-verified` at the bundle root.
