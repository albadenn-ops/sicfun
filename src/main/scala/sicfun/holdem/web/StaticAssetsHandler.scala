package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpHandler}

import java.io.ByteArrayOutputStream
import java.nio.file.{Files, LinkOption, Path, Paths}
import java.time.{Instant, ZoneOffset, ZonedDateTime}
import java.time.format.DateTimeFormatter
import java.util.zip.GZIPOutputStream
import scala.util.control.NonFatal

import sicfun.holdem.web.AuthStack.ensureAuthenticatedStatic
import sicfun.holdem.web.HandHistoryReviewServer.BasicAuthConfig
import sicfun.holdem.web.HandHistoryReviewServerRuntime.logHandlerException
import sicfun.holdem.web.WebResponses.{applySecurityHeaders, clientAcceptsGzip, contentTypeFor, isCompressibleType, writePlain}

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
      else if hasDotPrefixedSegment(exchange) then
        // Reject any path segment starting with `.` as defense in depth. The static
        // dir should never contain dot-prefixed entries (.git/, .env, .htaccess,
        // .DS_Store), but a misconfigured deployment that points staticDir at an
        // unsanitized bundle or a repo checkout would otherwise expose them. 404
        // (not 403) so the response is indistinguishable from a missing file and
        // does not confirm the rule exists.
        writePlain(exchange, 404, "not found", "text/plain; charset=utf-8")
      else
        val requestPath = Option(exchange.getRequestURI.getPath).getOrElse("/")
        val relative = if requestPath == "/" then Paths.get("index.html") else Paths.get(requestPath.dropWhile(_ == '/'))
        val resolved = staticDir.resolve(relative).normalize()
        if !resolved.startsWith(staticDir) then
          writePlain(exchange, 403, "forbidden", "text/plain; charset=utf-8")
        else
          val target =
            if Files.isDirectory(resolved, LinkOption.NOFOLLOW_LINKS) then resolved.resolve("index.html")
            else resolved
          // NOFOLLOW_LINKS: a symlink under the static dir whose target lies
          // outside it (or anywhere on the filesystem) would otherwise be
          // followed by Files.isRegularFile/size/getLastModifiedTime, so a
          // misconfigured deploy that includes such a symlink would serve
          // arbitrary host files even though the normalized path check at
          // line 49 says "resolved.startsWith(staticDir)". Treating symlinks
          // as non-regular files makes the static dir self-contained: only
          // genuine files inside it get served.
          if Files.exists(target, LinkOption.NOFOLLOW_LINKS) && Files.isRegularFile(target, LinkOption.NOFOLLOW_LINKS) then
            val size = Files.size(target)
            val lastModified = Files.getLastModifiedTime(target).toMillis
            // HTTP-date is second-resolution. Truncate file mtime so the value we emit
            // can be parsed and compared losslessly by clients on the way back.
            val lastModifiedSecond = (lastModified / 1000L) * 1000L
            val lastModifiedHttp = DateTimeFormatter.RFC_1123_DATE_TIME
              .format(ZonedDateTime.ofInstant(Instant.ofEpochMilli(lastModifiedSecond), ZoneOffset.UTC))
            val contentType = contentTypeFor(target)
            val compressible = isCompressibleType(contentType)
            val acceptsGzip = clientAcceptsGzip(exchange)
            // `wouldCompressIfGet` describes the VARIANT (gzipped or plain) the
            // resource is going to be served as; HEAD must report the same
            // variant headers and ETag as GET would for the same request shape
            // so client caches that store HEAD-validated entries don't choke
            // when the same client then issues GET.
            val wouldCompressIfGet = compressible && acceptsGzip
            // `willCompress` controls whether THIS response actually contains
            // compressed bytes. HEAD has no body either way, so we skip the
            // compression work for HEAD even when the variant is gzipped.
            val willCompress = wouldCompressIfGet && isGet
            // Variant ETag: gzipped and uncompressed are different representations.
            // RFC 7232 sec 2.3.1: weak ETags MAY indicate equivalent representations,
            // but conservative caches that key only on ETag (ignoring Vary) need
            // distinct values to avoid serving the wrong encoding.
            val etag = s"""W/"$size-$lastModified${if wouldCompressIfGet then "-gz" else ""}""""
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
              // Advertise the same variant headers a GET would emit, so client
              // caches that validate via HEAD then fetch via GET see consistent
              // headers and don't invalidate. wouldCompressIfGet -- NOT
              // willCompress -- because willCompress is gated on `isGet`.
              if wouldCompressIfGet then
                exchange.getResponseHeaders.set("Content-Encoding", "gzip")
              exchange.sendResponseHeaders(200, -1L)
            else if willCompress then
              // Compress in memory: static files are small (top of bundle ~50 KB),
              // and the in-memory buffer is simpler than streaming gzip + chunked transfer.
              val buffer = new ByteArrayOutputStream(math.max(1024, (size / 4).toInt))
              val gz = new GZIPOutputStream(buffer)
              try
                val input = Files.newInputStream(target)
                try input.transferTo(gz)
                finally input.close()
              finally gz.close()
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
        logHandlerException(exchange, e, "unhandled exception in StaticAssetsHandler")
        try writePlain(exchange, 500, "internal server error", "text/plain; charset=utf-8")
        catch case NonFatal(_) => ()
    finally
      exchange.close()

  private def hasDotPrefixedSegment(exchange: HttpExchange): Boolean =
    val raw = Option(exchange.getRequestURI.getPath).getOrElse("/")
    // Split on BOTH '/' and '\' so an attacker sending `/%5C.git/HEAD`
    // (percent-encoded backslash, which Windows treats as a path separator)
    // cannot smuggle a dot-prefixed segment past the check. The path-traversal
    // guard at staticDir.resolve(...).startsWith(staticDir) catches escapes
    // outside the static dir, but a backslash-prefixed dotfile inside the
    // static dir would otherwise serve up `staticDir\.git\HEAD` content.
    //
    // Skip the path-navigation primitives `.` and `..` so the path-traversal
    // check downstream still produces its more informative 403 response.
    raw.split('/').flatMap(_.split('\\')).exists(segment =>
      segment.nonEmpty && segment.startsWith(".") && segment != "." && segment != ".."
    )
