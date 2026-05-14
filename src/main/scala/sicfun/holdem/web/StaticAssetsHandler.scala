package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpHandler}

import java.nio.file.{Files, Path, Paths}
import java.time.{Instant, ZoneOffset, ZonedDateTime}
import java.time.format.DateTimeFormatter
import scala.util.control.NonFatal

import sicfun.holdem.web.AuthStack.ensureAuthenticatedStatic
import sicfun.holdem.web.HandHistoryReviewServer.BasicAuthConfig
import sicfun.holdem.web.WebResponses.{applySecurityHeaders, contentTypeFor, writePlain}

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
            val etag = s"""W/"$size-$lastModified""""
            val cacheControl =
              if requestPath.startsWith("/vendor/") then "public, max-age=31536000"
              else "public, max-age=0, must-revalidate"
            exchange.getResponseHeaders.set("Cache-Control", cacheControl)
            exchange.getResponseHeaders.set("ETag", etag)
            exchange.getResponseHeaders.set("Last-Modified", lastModifiedHttp)
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
              exchange.getResponseHeaders.set("Content-Type", contentTypeFor(target))
              exchange.sendResponseHeaders(200, -1L)
            else
              exchange.getResponseHeaders.set("Content-Type", contentTypeFor(target))
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
