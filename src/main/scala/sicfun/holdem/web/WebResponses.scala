package sicfun.holdem.web

import com.sun.net.httpserver.HttpExchange
import ujson.Value

import java.nio.charset.StandardCharsets
import java.nio.file.Path

/** Response-writing utilities for the embedded `com.sun.net.httpserver` surface.
  *
  * Stateless writers used by every handler in [[HandHistoryReviewServer]] to emit
  * JSON, plain-text, redirect, or raw-byte responses, plus a content-type lookup
  * for static-asset paths.
  *
  * Contract: every body-writing call sets `Cache-Control: no-store`. Static-asset
  * handlers wanting different cache semantics must override `Cache-Control` after
  * calling [[writeBytes]].
  */
object WebResponses:
  /** Writes a JSON value at the given status, indenting at 2 spaces. */
  def writeJson(exchange: HttpExchange, status: Int, value: Value): Unit =
    val bytes = ujson.write(value, indent = 2).getBytes(StandardCharsets.UTF_8)
    writeBytes(exchange, status, bytes, "application/json; charset=utf-8")

  /** Writes a UTF-8 plain string at the given status with the supplied content-type. */
  def writePlain(exchange: HttpExchange, status: Int, body: String, contentType: String): Unit =
    writeBytes(exchange, status, body.getBytes(StandardCharsets.UTF_8), contentType)

  /** Writes a redirect with `Location: location` and an empty body. */
  def writeRedirect(exchange: HttpExchange, status: Int, location: String): Unit =
    exchange.getResponseHeaders.set("Location", location)
    exchange.sendResponseHeaders(status, -1L)

  /** Writes raw bytes with the supplied content-type and `Cache-Control: no-store`. */
  def writeBytes(
      exchange: HttpExchange,
      status: Int,
      bytes: Array[Byte],
      contentType: String
  ): Unit =
    exchange.getResponseHeaders.set("Content-Type", contentType)
    exchange.getResponseHeaders.set("Cache-Control", "no-store")
    exchange.sendResponseHeaders(status, bytes.length.toLong)
    val body = exchange.getResponseBody
    body.write(bytes)
    body.flush()

  /** Maps a file path's extension to a MIME type for static-asset serving.
    * Falls back to `application/octet-stream` for unknown extensions.
    */
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
