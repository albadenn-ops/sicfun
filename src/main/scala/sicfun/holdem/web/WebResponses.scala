package sicfun.holdem.web

import com.sun.net.httpserver.HttpExchange
import ujson.Value

import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import java.util.zip.GZIPOutputStream

/** Shared HTTP response writers and security headers.
  */
private[web] object WebResponses:

  private val ContentSecurityPolicy =
    "default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; img-src 'self' data:; object-src 'none'; script-src 'self'; style-src 'self'"

  // Defense-in-depth: explicitly deny browser features the app does not use, so any
  // future inline-script-induced exploit (or compromised vendored library) cannot
  // promote itself into the user's hardware or browser-level capabilities. `browsing-
  // topics=()` is the current Chrome opt-out; `interest-cohort=()` is the legacy FLoC
  // name kept for backwards compatibility with older browser builds.
  private val PermissionsPolicy =
    "accelerometer=(), ambient-light-sensor=(), autoplay=(), battery=(), bluetooth=(), " +
    "browsing-topics=(), camera=(), display-capture=(), document-domain=(), " +
    "encrypted-media=(), fullscreen=(), geolocation=(), gyroscope=(), hid=(), " +
    "idle-detection=(), interest-cohort=(), local-fonts=(), magnetometer=(), " +
    "microphone=(), midi=(), otp-credentials=(), payment=(), picture-in-picture=(), " +
    "publickey-credentials-get=(), screen-wake-lock=(), serial=(), storage-access=(), " +
    "usb=(), web-share=(), xr-spatial-tracking=()"

  // Threshold below which gzip overhead can exceed the savings. A 256-byte JSON
  // typically compresses to 200-250 bytes once the gzip header (~20 bytes) is
  // included -- not worth the CPU. Standard nginx/apache default is 256-1024.
  private val MinGzipSize = 256

  /** Text-shaped MIME types we'll gzip when the client opts in via Accept-Encoding.
    * Shared with [[StaticAssetsHandler]] so the policy is consistent across static
    * file serving and API JSON responses. */
  def isCompressibleType(contentType: String): Boolean =
    val lower = contentType.toLowerCase
    lower.startsWith("text/") ||
      lower.startsWith("application/javascript") ||
      lower.startsWith("application/json") ||
      lower.startsWith("image/svg+xml")

  /** RFC 7231 sec 5.3.4: an Accept-Encoding value is acceptable iff the matching
    * codec is listed with no q parameter (default q=1) or with q>0. A value of
    * "gzip;q=0" explicitly _rejects_ gzip and must be honored. Shared with
    * [[StaticAssetsHandler]] so the policy is consistent. */
  def clientAcceptsGzip(exchange: HttpExchange): Boolean =
    Option(exchange.getRequestHeaders.getFirst("Accept-Encoding")).exists { raw =>
      raw.split(',').iterator.map(_.trim.toLowerCase).exists(tokenAcceptsGzip)
    }

  private def tokenAcceptsGzip(token: String): Boolean =
    if token == "gzip" then true
    else if !token.startsWith("gzip;") && !token.startsWith("gzip ") then false
    else
      // gzip with parameters; q-value optional. Parse "q=..." param if present.
      val params = token.substring(4).stripPrefix(";").stripPrefix(" ").split(';')
      val qParam = params.iterator.map(_.trim).find(_.startsWith("q="))
      qParam match
        case None => true
        case Some(q) =>
          try q.substring(2).toDouble > 0.0
          catch case _: NumberFormatException => true

  def applySecurityHeaders(exchange: HttpExchange): Unit =
    val headers = exchange.getResponseHeaders
    headers.set("Cache-Control", "no-store")
    headers.set("Content-Security-Policy", ContentSecurityPolicy)
    headers.set("Permissions-Policy", PermissionsPolicy)
    headers.set("Referrer-Policy", "no-referrer")
    headers.set("X-Content-Type-Options", "nosniff")
    headers.set("X-Frame-Options", "DENY")
    // CSP `frame-ancestors 'none'` + X-Frame-Options already block framing.
    // COOP/CORP cover orthogonal threats: cross-origin window references
    // (XS-Leaks, Spectre process isolation) and cross-origin resource
    // embedding via <img>/<script>/fetch. Both are additive and harmless
    // for a same-origin app.
    headers.set("Cross-Origin-Opener-Policy", "same-origin")
    headers.set("Cross-Origin-Resource-Policy", "same-origin")
    // Private review tool: search engines must not index it even if it
    // accidentally ends up reachable from the public internet (the operator
    // forgets to firewall it, the reverse-proxy ACL is too permissive, etc.).
    // `noindex` blocks indexing of the response; `nofollow` blocks crawling
    // any URLs the response references.
    headers.set("X-Robots-Tag", "noindex, nofollow")

  def writeJson(exchange: HttpExchange, status: Int, value: Value): Unit =
    val bytes = ujson.write(value, indent = 2).getBytes(StandardCharsets.UTF_8)
    writeBytes(exchange, status, bytes, "application/json; charset=utf-8")

  def writePlain(exchange: HttpExchange, status: Int, body: String, contentType: String): Unit =
    writeBytes(exchange, status, body.getBytes(StandardCharsets.UTF_8), contentType)

  def writeRedirect(exchange: HttpExchange, status: Int, location: String): Unit =
    exchange.getResponseHeaders.set("Location", location)
    exchange.sendResponseHeaders(status, -1L)

  def writeBytes(
      exchange: HttpExchange,
      status: Int,
      bytes: Array[Byte],
      contentType: String
  ): Unit =
    val compressible = isCompressibleType(contentType)
    exchange.getResponseHeaders.set("Content-Type", contentType)
    exchange.getResponseHeaders.set("Cache-Control", "no-store")
    if compressible then
      exchange.getResponseHeaders.set("Vary", "Accept-Encoding")
    val isHead = exchange.getRequestMethod.equalsIgnoreCase("HEAD")
    if isHead then
      // RFC 7231 sec 4.3.2: HEAD response has the same headers as GET but
      // MUST NOT include a body. The static handler routes HEAD to a dedicated
      // branch for happy-path file serving; this guard catches all the error
      // paths (404 / 403 / 401 / 405 / 500) that flow through writePlain so
      // those also stop emitting a body on HEAD requests. Content-Length is
      // the uncompressed byte count -- error bodies are short enough that
      // they're never gzipped (under MinGzipSize), so no variant mismatch.
      exchange.sendResponseHeaders(status, bytes.length.toLong)
    else
      val shouldCompress = compressible && bytes.length >= MinGzipSize && clientAcceptsGzip(exchange)
      if shouldCompress then
        val buffer = new ByteArrayOutputStream(math.max(256, bytes.length / 4))
        val gz = new GZIPOutputStream(buffer)
        try gz.write(bytes) finally gz.close()
        val compressed = buffer.toByteArray
        exchange.getResponseHeaders.set("Content-Encoding", "gzip")
        exchange.sendResponseHeaders(status, compressed.length.toLong)
        val body = exchange.getResponseBody
        body.write(compressed)
        body.flush()
      else
        exchange.sendResponseHeaders(status, bytes.length.toLong)
        val body = exchange.getResponseBody
        body.write(bytes)
        body.flush()

  def contentTypeFor(path: Path): String =
    path.getFileName.toString.toLowerCase match
      case name if name.endsWith(".html") => "text/html; charset=utf-8"
      case name if name.endsWith(".css") => "text/css; charset=utf-8"
      case name if name.endsWith(".js") => "application/javascript; charset=utf-8"
      case name if name.endsWith(".mjs") => "application/javascript; charset=utf-8"
      case name if name.endsWith(".map") => "application/json; charset=utf-8"
      case name if name.endsWith(".svg") => "image/svg+xml"
      case name if name.endsWith(".png") => "image/png"
      case name if name.endsWith(".jpg") || name.endsWith(".jpeg") => "image/jpeg"
      case name if name.endsWith(".webp") => "image/webp"
      case name if name.endsWith(".ico") => "image/x-icon"
      case name if name.endsWith(".wasm") => "application/wasm"
      case name if name.endsWith(".woff2") => "font/woff2"
      case name if name.endsWith(".json") => "application/json; charset=utf-8"
      case name if name.endsWith(".txt") => "text/plain; charset=utf-8"
      case _ => "application/octet-stream"
