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
  // promote itself into the user's hardware. `interest-cohort=()` opts out of FLoC.
  private val PermissionsPolicy =
    "accelerometer=(), camera=(), display-capture=(), encrypted-media=(), geolocation=(), gyroscope=(), interest-cohort=(), magnetometer=(), microphone=(), payment=(), usb=()"

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
