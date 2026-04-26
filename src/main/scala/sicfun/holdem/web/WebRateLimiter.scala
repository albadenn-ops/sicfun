package sicfun.holdem.web

import com.sun.net.httpserver.HttpExchange

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/** Sliding-window per-client rate limiter for the embedded HTTP surface.
  *
  * Carved out of [[HandHistoryReviewServer]] (F3 / B3 split slice). The previous
  * inner `RequestRateLimiter` class reached into the server's IP-resolution
  * helpers via a bareword call to `rateLimitClientKey`. This module accepts the
  * key resolver as a constructor callback so the windowing logic has no
  * coupling to the server's auth or proxy stack.
  *
  *   - One window per `(bucket, clientKey)` tuple, sized at `windowMs`.
  *   - Counts reset when the window expires.
  *   - The first request that crosses the per-minute limit gets a
  *     [[RateLimitRejection]] with the retry-after delta.
  *   - Stale windows are reaped on each `check` call (cheap CAS-guarded sweep).
  *
  * Limits of `0` disable that bucket (every request returns `None`).
  */
object WebRateLimiter:

  /** Default sliding window: a rolling minute. */
  val DefaultRateLimitWindowMs: Long = 60L * 1000L

  /** Rate-limit bucket -- distinct counter per bucket per client. */
  enum RateLimitBucket:
    case Submit, JobStatus

    def id: String = this match
      case Submit => "submit"
      case JobStatus => "job-status"

    def description: String = this match
      case Submit => "submit"
      case JobStatus => "job status"

  /** Per-bucket-per-client window: how many requests since when. */
  final case class RateLimitState(
      windowStartedAtMs: Long,
      requestCount: Int
  )

  /** Returned when a request would exceed the limit; carries enough info for
    * the caller to emit a 429 with `Retry-After`.
    */
  final case class RateLimitRejection(
      bucket: RateLimitBucket,
      limitPerMinute: Int,
      retryAfterMs: Long,
      clientKey: String
  )

/** Rate-limiter instance. One per server.
  *
  * @param submitsPerMinute   per-client cap for [[WebRateLimiter.RateLimitBucket.Submit]]; 0 disables.
  * @param statusPerMinute    per-client cap for [[WebRateLimiter.RateLimitBucket.JobStatus]]; 0 disables.
  * @param clientKeyFor       resolves an `HttpExchange` to an opaque client-key string.
  *                           The server owns IP / X-Forwarded-For / trusted-proxy logic.
  * @param nowMillis          clock; defaulted for production, overrideable for tests.
  * @param windowMs           sliding window length; defaulted to one minute.
  */
final class WebRateLimiter(
    submitsPerMinute: Int,
    statusPerMinute: Int,
    clientKeyFor: HttpExchange => String,
    nowMillis: () => Long = () => System.currentTimeMillis(),
    windowMs: Long = WebRateLimiter.DefaultRateLimitWindowMs
):
  import WebRateLimiter.{RateLimitBucket, RateLimitRejection, RateLimitState}

  private val windows = new ConcurrentHashMap[String, RateLimitState]()
  private val lastCleanupAtMs = new AtomicLong(0L)

  /** Check whether this request fits inside the bucket's window for the
    * resolved client key. Returns `None` on accept, `Some(rejection)` on
    * deny. `principalKey` lets authenticated callers be tracked by user id
    * instead of IP (so multiple users behind the same NAT don't collide).
    */
  def check(
      exchange: HttpExchange,
      bucket: RateLimitBucket,
      principalKey: Option[String] = None
  ): Option[RateLimitRejection] =
    val limitPerMinute = bucket match
      case RateLimitBucket.Submit => submitsPerMinute
      case RateLimitBucket.JobStatus => statusPerMinute
    if limitPerMinute <= 0 then None
    else
      val now = nowMillis()
      cleanupIfDue(now)
      val clientKey = principalKey.getOrElse(clientKeyFor(exchange))
      val key = s"${bucket.id}|$clientKey"
      var rejection = Option.empty[RateLimitRejection]
      windows.compute(
        key,
        (_, existing) =>
          if existing == null || now - existing.windowStartedAtMs >= windowMs then
            RateLimitState(windowStartedAtMs = now, requestCount = 1)
          else if existing.requestCount < limitPerMinute then
            existing.copy(requestCount = existing.requestCount + 1)
          else
            rejection = Some(
              RateLimitRejection(
                bucket = bucket,
                limitPerMinute = limitPerMinute,
                retryAfterMs = math.max(1L, windowMs - (now - existing.windowStartedAtMs)),
                clientKey = clientKey
              )
            )
            existing
      )
      rejection

  /** Sweep stale windows on a CAS-guarded cadence: at most once per windowMs. */
  private def cleanupIfDue(now: Long): Unit =
    val lastCleanup = lastCleanupAtMs.get()
    if now - lastCleanup >= windowMs && lastCleanupAtMs.compareAndSet(lastCleanup, now) then
      val cutoff = now - (windowMs * 2L)
      val iterator = windows.entrySet().iterator()
      while iterator.hasNext do
        val entry = iterator.next()
        if entry.getValue.windowStartedAtMs < cutoff then
          iterator.remove()
