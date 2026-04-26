package sicfun.holdem.web

import com.sun.net.httpserver.HttpExchange
import munit.FunSuite

import java.util.concurrent.atomic.AtomicLong

/** Unit tests for [[WebRateLimiter]].
  *
  * Pre-extraction the rate-limiter was an inner class of `HandHistoryReviewServer`
  * that called bareword `rateLimitClientKey(exchange, ...)`. That coupling made
  * unit-testing the windowing logic hard -- the only way to exercise it was via
  * the full HTTP integration tests. After the F3 split, the limiter takes
  * `clientKeyFor: HttpExchange => String` as a constructor callback, so we can
  * pass a stub that pretends every request is from the same/different client
  * without spinning up a server.
  */
class WebRateLimiterTest extends FunSuite:

  // We never call any HttpExchange method in these tests -- the limiter only
  // forwards it to the supplied clientKeyFor callback, which we override.
  // null is acceptable as a sentinel; the tests use a controlled client key.
  private val anyExchange: HttpExchange = null

  /** Build a limiter with a controllable virtual clock and a fixed client key. */
  private def fixed(
      submitsPerMinute: Int,
      statusPerMinute: Int = 100,
      clientKey: String = "client-1",
      windowMs: Long = 60_000L
  ): (WebRateLimiter, AtomicLong) =
    val clock = new AtomicLong(0L)
    val limiter = new WebRateLimiter(
      submitsPerMinute = submitsPerMinute,
      statusPerMinute = statusPerMinute,
      clientKeyFor = _ => clientKey,
      nowMillis = () => clock.get(),
      windowMs = windowMs
    )
    (limiter, clock)

  // ---- bucket id + description metadata ----

  test("RateLimitBucket.id is stable across enum cases") {
    assertEquals(WebRateLimiter.RateLimitBucket.Submit.id, "submit")
    assertEquals(WebRateLimiter.RateLimitBucket.JobStatus.id, "job-status")
  }

  test("RateLimitBucket.description is human-readable") {
    assertEquals(WebRateLimiter.RateLimitBucket.Submit.description, "submit")
    assertEquals(WebRateLimiter.RateLimitBucket.JobStatus.description, "job status")
  }

  // ---- per-bucket independence ----

  test("Submit and JobStatus buckets are tracked independently") {
    val (limiter, _) = fixed(submitsPerMinute = 1, statusPerMinute = 1)
    // First submit accepted, second rejected.
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).isEmpty)
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).nonEmpty)
    // JobStatus is still on its first window -> first hit accepted regardless.
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.JobStatus).isEmpty)
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.JobStatus).nonEmpty)
  }

  // ---- limit enforcement + rejection content ----

  test("submitsPerMinute=0 disables Submit (every request accepted)") {
    val (limiter, _) = fixed(submitsPerMinute = 0)
    (1 to 100).foreach { _ =>
      assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).isEmpty)
    }
  }

  test("first request crossing the limit returns RateLimitRejection with bucket info") {
    val (limiter, _) = fixed(submitsPerMinute = 3)
    (1 to 3).foreach { _ =>
      assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).isEmpty)
    }
    val rejection = limiter
      .check(anyExchange, WebRateLimiter.RateLimitBucket.Submit)
      .getOrElse(fail("expected rejection on 4th call"))
    assertEquals(rejection.bucket, WebRateLimiter.RateLimitBucket.Submit)
    assertEquals(rejection.limitPerMinute, 3)
    assertEquals(rejection.clientKey, "client-1")
    assert(rejection.retryAfterMs > 0L && rejection.retryAfterMs <= 60_000L)
  }

  // ---- principalKey override ----

  test("principalKey overrides clientKeyFor (per-user buckets)") {
    val (limiter, _) = fixed(submitsPerMinute = 1, clientKey = "shared-ip")
    // Two different users behind the same IP each get their own quota.
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit, Some("user:alice")).isEmpty)
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit, Some("user:bob")).isEmpty)
    // alice exhausts her quota; bob still has his window.
    val aliceSecond = limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit, Some("user:alice"))
    val bobSecond = limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit, Some("user:bob"))
    assert(aliceSecond.exists(_.clientKey == "user:alice"), s"alice second: $aliceSecond")
    assert(bobSecond.exists(_.clientKey == "user:bob"), s"bob second: $bobSecond")
  }

  // ---- window expiry ----

  test("window resets after windowMs elapses") {
    val (limiter, clock) = fixed(submitsPerMinute = 1, windowMs = 1_000L)
    // First request in window 0..999 ms -> accepted; second -> rejected.
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).isEmpty)
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).nonEmpty)
    // Advance past the window boundary.
    clock.set(1_500L)
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).isEmpty,
      "first request in the next window must be accepted")
  }

  test("retryAfterMs reflects the remaining window time") {
    val (limiter, clock) = fixed(submitsPerMinute = 1, windowMs = 60_000L)
    clock.set(0L)
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).isEmpty)
    clock.set(20_000L)
    val rejection = limiter
      .check(anyExchange, WebRateLimiter.RateLimitBucket.Submit)
      .getOrElse(fail("expected rejection"))
    // Window started at 0, reject at 20_000. Remaining = 60_000 - 20_000 = 40_000.
    assertEquals(rejection.retryAfterMs, 40_000L)
  }

  test("retryAfterMs is at least 1 even when the window is exactly exhausted") {
    val (limiter, clock) = fixed(submitsPerMinute = 1, windowMs = 100L)
    clock.set(0L)
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).isEmpty)
    clock.set(100L) // exactly windowMs -- counts as a fresh window per the >= check
    assert(limiter.check(anyExchange, WebRateLimiter.RateLimitBucket.Submit).isEmpty,
      "exact-window-boundary should be a new window")
    val rejection = limiter
      .check(anyExchange, WebRateLimiter.RateLimitBucket.Submit)
      .getOrElse(fail("expected rejection"))
    assert(rejection.retryAfterMs >= 1L, s"retryAfterMs floor=1 but got ${rejection.retryAfterMs}")
  }
