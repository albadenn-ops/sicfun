package sicfun.holdem.web

import munit.FunSuite

import java.nio.file.Paths

class WebResponsesTest extends FunSuite:

  test("contentTypeFor returns charset for text formats") {
    assertEquals(WebResponses.contentTypeFor(Paths.get("page.html")), "text/html; charset=utf-8")
    assertEquals(WebResponses.contentTypeFor(Paths.get("style.css")), "text/css; charset=utf-8")
    assertEquals(WebResponses.contentTypeFor(Paths.get("app.js")), "application/javascript; charset=utf-8")
    assertEquals(WebResponses.contentTypeFor(Paths.get("module.mjs")), "application/javascript; charset=utf-8")
    assertEquals(WebResponses.contentTypeFor(Paths.get("data.json")), "application/json; charset=utf-8")
    assertEquals(WebResponses.contentTypeFor(Paths.get("README.txt")), "text/plain; charset=utf-8")
    assertEquals(WebResponses.contentTypeFor(Paths.get("app.js.map")), "application/json; charset=utf-8")
  }

  test("contentTypeFor returns image/binary types without charset") {
    assertEquals(WebResponses.contentTypeFor(Paths.get("logo.svg")), "image/svg+xml")
    assertEquals(WebResponses.contentTypeFor(Paths.get("photo.png")), "image/png")
    assertEquals(WebResponses.contentTypeFor(Paths.get("photo.jpg")), "image/jpeg")
    assertEquals(WebResponses.contentTypeFor(Paths.get("photo.jpeg")), "image/jpeg")
    assertEquals(WebResponses.contentTypeFor(Paths.get("photo.webp")), "image/webp")
    assertEquals(WebResponses.contentTypeFor(Paths.get("favicon.ico")), "image/x-icon")
    assertEquals(WebResponses.contentTypeFor(Paths.get("module.wasm")), "application/wasm")
    assertEquals(WebResponses.contentTypeFor(Paths.get("font.woff2")), "font/woff2")
  }

  test("contentTypeFor falls back to octet-stream for unknown extensions") {
    assertEquals(WebResponses.contentTypeFor(Paths.get("archive.zip")), "application/octet-stream")
    assertEquals(WebResponses.contentTypeFor(Paths.get("data.bin")), "application/octet-stream")
    assertEquals(WebResponses.contentTypeFor(Paths.get("noext")), "application/octet-stream")
  }

  test("contentTypeFor lowercases the file name before matching") {
    assertEquals(WebResponses.contentTypeFor(Paths.get("PAGE.HTML")), "text/html; charset=utf-8")
    assertEquals(WebResponses.contentTypeFor(Paths.get("Logo.SVG")), "image/svg+xml")
    assertEquals(WebResponses.contentTypeFor(Paths.get("Photo.JPG")), "image/jpeg")
  }

  test("contentTypeFor uses only the file name, not the directory") {
    assertEquals(
      WebResponses.contentTypeFor(Paths.get("/var/www/html/about/index.html")),
      "text/html; charset=utf-8"
    )
  }
