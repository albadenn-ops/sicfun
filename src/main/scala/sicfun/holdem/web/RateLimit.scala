package sicfun.holdem.web

import com.sun.net.httpserver.HttpExchange

import java.net.InetAddress
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import scala.jdk.CollectionConverters.*

private[web] object RateLimit:
  private val WindowMs = 60L * 1000L
  val ClientIpSourceRemoteAddress = "remote-address"

  enum RateLimitBucket:
    case Submit, JobStatus, Auth

    def id: String = this match
      case Submit => "submit"
      case JobStatus => "job-status"
      case Auth => "auth"

    def description: String = this match
      case Submit => "submit"
      case JobStatus => "job status"
      case Auth => "auth"

  final case class RateLimitRejection(
      bucket: RateLimitBucket,
      limitPerMinute: Int,
      retryAfterMs: Long,
      clientKey: String
  )

  final class RequestRateLimiter(
      submitsPerMinute: Int,
      statusPerMinute: Int,
      authPerMinute: Int,
      trustedClientIpHeader: Option[String],
      trustedProxyIps: Set[String],
      nowMillis: () => Long = () => System.currentTimeMillis()
  ):
    private val windows = new ConcurrentHashMap[String, RateLimitState]()
    private val lastCleanupAtMs = new AtomicLong(0L)

    def check(
        exchange: HttpExchange,
        bucket: RateLimitBucket,
        principalKey: Option[String] = None
    ): Option[RateLimitRejection] =
      val limitPerMinute = bucket match
        case RateLimitBucket.Submit => submitsPerMinute
        case RateLimitBucket.JobStatus => statusPerMinute
        case RateLimitBucket.Auth => authPerMinute
      if limitPerMinute <= 0 then None
      else
        val now = nowMillis()
        cleanupIfDue(now)
        val clientKey = principalKey.getOrElse(rateLimitClientKey(exchange, trustedClientIpHeader, trustedProxyIps))
        val key = s"${bucket.id}|$clientKey"
        var rejection = Option.empty[RateLimitRejection]
        windows.compute(
          key,
          (_, existing) =>
            if existing == null || now - existing.windowStartedAtMs >= WindowMs then
              RateLimitState(windowStartedAtMs = now, requestCount = 1)
            else if existing.requestCount < limitPerMinute then
              existing.copy(requestCount = existing.requestCount + 1)
            else
              rejection = Some(
                RateLimitRejection(
                  bucket = bucket,
                  limitPerMinute = limitPerMinute,
                  retryAfterMs = math.max(1L, WindowMs - (now - existing.windowStartedAtMs)),
                  clientKey = clientKey
                )
              )
              existing
        )
        rejection

    private def cleanupIfDue(now: Long): Unit =
      val lastCleanup = lastCleanupAtMs.get()
      if now - lastCleanup >= WindowMs && lastCleanupAtMs.compareAndSet(lastCleanup, now) then
        val cutoff = now - (WindowMs * 2L)
        val iterator = windows.entrySet().iterator()
        while iterator.hasNext do
          val entry = iterator.next()
          if entry.getValue.windowStartedAtMs < cutoff then
            iterator.remove()

  def rateLimitClientIpSource(
      trustedClientIpHeader: Option[String],
      trustedProxyIps: Set[String]
  ): String =
    trustedClientIpHeader match
      case Some(header) if trustedProxyIps.nonEmpty =>
        s"header:$header via loopback-or-allowlisted-proxy"
      case Some(header) =>
        s"header:$header via loopback-only"
      case None =>
        ClientIpSourceRemoteAddress

  /** Resolve the audit-display client address for the given request, applying the
    * same trusted-proxy policy as rate limiting so audit `remote=` log fields and
    * rate-limit `clientKey=` values agree on who the client is.
    *
    * Behind a trusted reverse proxy (peer is loopback or in `trustedProxyIps` and a
    * single-valued `trustedClientIpHeader` parses as an IP), returns the resolved
    * client IP without a port (X-Forwarded-For carries no port). Otherwise returns
    * the formatted direct TCP peer `host:port` with IPv6 brackets per RFC 3986 §3.2.2.
    *
    * IPv6 hosts get bracketed in both cases so a log-line parser splitting on the
    * trailing `:port` cannot mistake the address's internal `:` for a port delimiter. */
  def resolveAuditClientAddress(
      exchange: HttpExchange,
      trustedClientIpHeader: Option[String],
      trustedProxyIps: Set[String]
  ): String =
    trustedClientIpHeader
      .filter(_ => trustsRateLimitClientIpHeader(remoteInetAddress(exchange), trustedProxyIps))
      .flatMap(headerName => forwardedClientKey(exchange, headerName))
      .map(formatAuditHostOnly)
      .getOrElse(formatAuditPeer(exchange))

  private def formatAuditHostOnly(host: String): String =
    if host.contains(':') then s"[$host]" else host

  private[web] def formatAuditPeer(exchange: HttpExchange): String =
    Option(exchange.getRemoteAddress).map { address =>
      val host = address.getHostString
      val port = address.getPort
      if host.contains(':') then s"[$host]:$port" else s"$host:$port"
    }.getOrElse("-")

  def trustedProxyIpSummary(trustedProxyIps: Set[String]): String =
    trustedProxyIps.toVector.sorted match
      case Vector() => "-"
      case values => values.mkString(",")

  private[web] def parseTrustedProxyIps(raw: Option[String]): Either[String, Set[String]] =
    raw match
      case None => Right(Set.empty)
      case Some(value) =>
        val entries = value.split(",").toVector.map(_.trim).filter(_.nonEmpty)
        entries.foldLeft[Either[String, Vector[String]]](Right(Vector.empty)) { (acc, entry) =>
          for
            parsed <- acc
            normalized <- parseTrustedClientIpLiteral(entry)
              .toRight(s"--rateLimitTrustedProxyIps must contain comma-separated IP literals; invalid entry: $entry")
          yield parsed :+ normalized
        }.map(_.toSet)

  private[web] def trustsRateLimitClientIpHeader(
      remoteAddress: Option[InetAddress],
      trustedProxyIps: Set[String]
  ): Boolean =
    remoteAddress.exists(address => address.isLoopbackAddress || trustedProxyIps.contains(address.getHostAddress))

  private final case class RateLimitState(
      windowStartedAtMs: Long,
      requestCount: Int
  )

  private def rateLimitClientKey(
      exchange: HttpExchange,
      trustedClientIpHeader: Option[String],
      trustedProxyIps: Set[String]
  ): String =
    trustedClientIpHeader
      .filter(_ => trustsRateLimitClientIpHeader(remoteInetAddress(exchange), trustedProxyIps))
      .flatMap(headerName => forwardedClientKey(exchange, headerName).map(value => s"header:$value"))
      .getOrElse(s"remote:${clientAddressKey(exchange)}")

  private def forwardedClientKey(exchange: HttpExchange, headerName: String): Option[String] =
    Option(exchange.getRequestHeaders.get(headerName))
      .map(_.asScala.toVector.map(_.trim).filter(_.nonEmpty))
      .collect { case Vector(singleValue) if !singleValue.contains(',') => singleValue }
      .flatMap(parseTrustedClientIpLiteral)

  private def parseTrustedClientIpLiteral(value: String): Option[String] =
    val looksLikeIpLiteral =
      value.nonEmpty &&
        value.exists(_.isDigit) &&
        (value.contains(".") || value.contains(":")) &&
        value.forall(ch =>
          ch.isDigit ||
            ch == '.' ||
            ch == ':' ||
            ch == '%' ||
            (ch >= 'a' && ch <= 'f') ||
            (ch >= 'A' && ch <= 'F')
        )
    if !looksLikeIpLiteral then None
    else
      try
        Some(InetAddress.getByName(value).getHostAddress)
      catch
        case _: Exception => None

  private def remoteInetAddress(exchange: HttpExchange): Option[InetAddress] =
    Option(exchange.getRemoteAddress).flatMap(address => Option(address.getAddress))

  private def clientAddressKey(exchange: HttpExchange): String =
    Option(exchange.getRemoteAddress)
      .flatMap(address => Option(address.getAddress).map(_.getHostAddress).orElse(Option(address.getHostString)))
      .filter(_.nonEmpty)
      .getOrElse("unknown")
