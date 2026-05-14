package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpHandler}

import java.io.ByteArrayOutputStream
import java.nio.file.{Files, Path, Paths}
import java.time.{Instant, ZoneOffset, ZonedDateTime}
import java.time.format.DateTimeFormatter
import java.util.zip.GZIPOutputStream
import scala.util.control.NonFatal

import sicfun.holdem.web.AuthStack.ensureAuthenticatedStatic
import sicfun.holdem.web.HandHistoryReviewServer.BasicAuthConfig
import sicfun.holdem.web.WebResponses.{applySecurityHeaders, contentTypeFor, isCompressibleType, writePlain}

private[web] final class StaticAssetsHandler(
    staticDir: Path,
    basicAuth: Option[BasicAuthConfig] = None,
    platformAuth: Option[PlatformUserAuth.Service] = None
) extends HttpHandler:

  override def handle(exchange: HttpExchange): Unit =
    try
      applySecurityHeaders(exchange)
      val method = exchange.getRequestMethod
      val isHead = method.equalsIgnoreCase("HEAD")
      val isGet = method.equalsIgnoreCase("GET")
      val isOptions = method.equalsIgnoreCase("OPTIONS")
      if !ensureAuthenticatedStatic(exchange, basicAuth, platformAuth) then ()
      else if isOptions then
        exchange.getResponseHeaders.set("Allow", "GET, HEAD, OPTIONS")
        exchange.sendResponseHeaders(200, -1L)
      else if !isGet && !isHead then
        exchange.getResponseHeaders.set("Allow", "GET, HEAD, OPTIONS")
        writePlain(exchange, 405, "GET, HEAD, or OPTIONS required", "text/plain; charset=utf-8")
      else
        val requestPath = Option(exchange.getRequestURI.getPath).getOrElse("/")
        val relative = if requestPath == "/" then Paths.get("index.html") else Paths.get(requestPath.dropWhile(_ == '/'))
        val resolved = staticDir.resolve(relative).normalize()
        if !resolved.startsWith(staticDir) then
          writePlain(exchange, 403, "forbidden", "text/plain; charset=utf-8")
        else
          val target =
            if Files.isDirectory(resolved) then resolved.resolve("index.html")
            else resolved
          if Files.exists(target) && Files.isRegularFile(target) then
            val size = Files.size(target)
            val lastModified = Files.getLastModifiedTime(target).toMillis
            // HTTP-date is second-resolution. Truncate file mtime so the value we emit
            // can be parsed and compared losslessly by clients on the way back.
            val lastModifiedSecond = (lastModified / 1000L) * 1000L
            val lastModifiedHttp = DateTimeFormatter.RFC_1123_DATE_TIME
              .format(ZonedDateTime.ofInstant(Instant.ofEpochMilli(lastModifiedSecond), ZoneOffset.UTC))
            val contentType = contentTypeFor(target)
            val compressible = isCompressibleType(contentType)
            val acceptsGzip = Option(exchange.getRequestHeaders.getFirst("Accept-Encoding"))
              .exists { raw =>
                raw.split(',').iterator.map(_.trim.toLowerCase).exists { token =>
                  token == "gzip" || token.startsWith("gzip;")
                }
              }
            val willCompress = compressible && acceptsGzip && isGet
            // Variant ETag: gzipped and uncompressed are different representations.
            // RFC 7232 sec 2.3.1: weak ETags MAY indicate equivalent representations,
            // but conservative caches that key only on ETag (ignoring Vary) need
            // distinct values to avoid serving the wrong encoding.
            val etag = s"""W/"$size-$lastModified${if willCompress then "-gz" else ""}""""
            val cacheControl =
              if requestPath.startsWith("/vendor/") then "public, max-age=31536000"
              else "public, max-age=0, must-revalidate"
            exchange.getResponseHeaders.set("Cache-Control", cacheControl)
            exchange.getResponseHeaders.set("ETag", etag)
            exchange.getResponseHeaders.set("Last-Modified", lastModifiedHttp)
            // Vary: Accept-Encoding for compressible types so caches store gzipped and
            // uncompressed variants separately even when keying on URL + Vary headers.
            if compressible then
              exchange.getResponseHeaders.set("Vary", "Accept-Encoding")
            val ifNoneMatch = Option(exchange.getRequestHeaders.getFirst("If-None-Match"))
            val ifModifiedSince = Option(exchange.getRequestHeaders.getFirst("If-Modified-Since"))
            // RFC 7232 sec 3.3: ignore If-Modified-Since when If-None-Match is present.
            // Otherwise, 304 if the file has not been modified after the client's date.
            val notModified =
              if ifNoneMatch.isDefined then
                ifNoneMatch.exists { raw =>
                  val parts = raw.split(',').iterator.map(_.trim).filter(_.nonEmpty).toVector
                  parts.contains("*") || parts.contains(etag)
                }
              else
                ifModifiedSince.exists { raw =>
                  try
                    val clientMs = ZonedDateTime
                      .parse(raw.trim, DateTimeFormatter.RFC_1123_DATE_TIME)
                      .toInstant
                      .toEpochMilli
                    lastModifiedSecond <= clientMs
                  catch
                    case NonFatal(_) => false
                }
            if notModified then
              exchange.sendResponseHeaders(304, -1L)
            else if isHead then
              exchange.getResponseHeaders.set("Content-Type", contentType)
              if willCompress then
                exchange.getResponseHeaders.set("Content-Encoding", "gzip")
              exchange.sendResponseHeaders(200, -1L)
            else if willCompress then
              // Compress in memory: static files are small (top of bundle ~50 KB),
              // and the in-memory buffer is simpler than streaming gzip + chunked transfer.
              val buffer = new ByteArrayOutputStream(math.max(1024, (size / 4).toInt))
              val gz = new GZIPOutputStream(buffer)
              val input = Files.newInputStream(target)
              try input.transferTo(gz) finally input.close()
              gz.close()
              val compressed = buffer.toByteArray
              exchange.getResponseHeaders.set("Content-Type", contentType)
              exchange.getResponseHeaders.set("Content-Encoding", "gzip")
              exchange.sendResponseHeaders(200, compressed.length.toLong)
              val body = exchange.getResponseBody
              body.write(compressed)
              body.flush()
            else
              exchange.getResponseHeaders.set("Content-Type", contentType)
              exchange.sendResponseHeaders(200, size)
              val body = exchange.getResponseBody
              val input = Files.newInputStream(target)
              try
                input.transferTo(body)
                body.flush()
              finally input.close()
          else
            writePlain(exchange, 404, "not found", "text/plain; charset=utf-8")
    catch
      case NonFatal(e) =>
        writePlain(exchange, 500, s"internal server error: ${e.getMessage}", "text/plain; charset=utf-8")
    finally
      exchange.close()
