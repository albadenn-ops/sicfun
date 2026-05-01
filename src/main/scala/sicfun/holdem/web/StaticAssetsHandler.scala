package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpHandler}

import java.nio.file.{Files, Path, Paths}
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
      if !ensureAuthenticatedStatic(exchange, basicAuth, platformAuth) then ()
      else if !exchange.getRequestMethod.equalsIgnoreCase("GET") then
        writePlain(exchange, 405, "GET required", "text/plain; charset=utf-8")
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
            exchange.getResponseHeaders.set("Content-Type", contentTypeFor(target))
            exchange.sendResponseHeaders(200, Files.size(target))
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
