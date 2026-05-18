// site-charts.js — chart primitives for SICFUN Playing Hall v2.
// Each primitive is a pure (el, data, options?) => void. Renders SVG / uPlot into el.
(function (global) {
  "use strict";

  const COLOR_SIGNAL = "#4af626";
  const COLOR_WARN = "#ff4d00";
  const COLOR_MUTED = "#8b9794";
  const COLOR_INK = "#edf6f0";
  const FONT = '"Consolas","Courier New",monospace';

  function svg(tag, attrs, children) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (children) for (const c of children) el.appendChild(c);
    return el;
  }

  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  function fmtSigned(v) { const n = Number(v) || 0; return (n >= 0 ? "+" : "") + n.toFixed(2); }
  function fmtInt(v) { return (Number(v) || 0).toLocaleString("en-US"); }

  function renderBarH(el, data, options) {
    clear(el);
    const labels = data.labels || [];
    const values = (data.values || []).map(Number);
    const signed = !!data.signed;
    const width = el.clientWidth || 480;
    const rowH = 28;
    const height = labels.length * rowH + 12;
    const pad = 120;
    const maxAbs = Math.max(1, ...values.map(Math.abs));
    const svgEl = svg("svg", {width, height, role: "img", "aria-label": (options && options.title) || "bar chart"});
    const midX = signed ? pad + (width - pad - 12) / 2 : pad;
    labels.forEach((label, i) => {
      const v = values[i];
      const y = i * rowH + 6;
      svgEl.appendChild(svg("text", {x: pad - 8, y: y + rowH - 14, "text-anchor": "end", fill: COLOR_MUTED, "font-family": FONT, "font-size": 11}, [document.createTextNode(label)]));
      const barLen = (Math.abs(v) / maxAbs) * (width - pad - 12) / (signed ? 2 : 1);
      const barX = signed ? (v >= 0 ? midX : midX - barLen) : pad;
      const fill = signed ? (v >= 0 ? COLOR_SIGNAL : COLOR_WARN) : COLOR_SIGNAL;
      svgEl.appendChild(svg("rect", {x: barX, y: y + 4, width: Math.max(1, barLen), height: rowH - 12, fill, opacity: 0.82}));
      svgEl.appendChild(svg("text", {
        x: signed ? (v >= 0 ? barX + barLen + 6 : barX - 6) : barX + barLen + 6,
        y: y + rowH - 14,
        "text-anchor": signed && v < 0 ? "end" : "start",
        fill: COLOR_INK, "font-family": FONT, "font-size": 11
      }, [document.createTextNode(fmtSigned(v))]));
    });
    if (signed) svgEl.appendChild(svg("line", {x1: midX, y1: 0, x2: midX, y2: height, stroke: COLOR_MUTED, "stroke-width": 1, opacity: 0.3}));
    el.appendChild(svgEl);
  }

  function renderDonut(el, data, options) {
    clear(el);
    const labels = data.labels || [];
    const values = (data.values || []).map(Number);
    const total = values.reduce((a, b) => a + b, 0) || 1;
    const width = el.clientWidth || 360;
    const size = Math.min(width, 220);
    const cx = size / 2, cy = size / 2, rOuter = size * 0.45, rInner = size * 0.28;
    const svgEl = svg("svg", {width: "100%", height: size + labels.length * 16, viewBox: `0 0 ${size} ${size + labels.length * 16}`, role: "img", "aria-label": (options && options.title) || "donut chart"});
    const palette = [COLOR_SIGNAL, "#6fa8dc", "#f6c84a", COLOR_WARN, "#b385ff", "#4fd1c5"];
    let acc = -Math.PI / 2;
    values.forEach((v, i) => {
      const angle = (v / total) * 2 * Math.PI;
      if (angle <= 0) return;
      const x1 = cx + Math.cos(acc) * rOuter, y1 = cy + Math.sin(acc) * rOuter;
      const x2 = cx + Math.cos(acc + angle) * rOuter, y2 = cy + Math.sin(acc + angle) * rOuter;
      const xi2 = cx + Math.cos(acc + angle) * rInner, yi2 = cy + Math.sin(acc + angle) * rInner;
      const xi1 = cx + Math.cos(acc) * rInner, yi1 = cy + Math.sin(acc) * rInner;
      const large = angle > Math.PI ? 1 : 0;
      const path = `M ${x1} ${y1} A ${rOuter} ${rOuter} 0 ${large} 1 ${x2} ${y2} L ${xi2} ${yi2} A ${rInner} ${rInner} 0 ${large} 0 ${xi1} ${yi1} Z`;
      svgEl.appendChild(svg("path", {d: path, fill: palette[i % palette.length], opacity: 0.88}));
      acc += angle;
    });
    let ly = size + 2;
    labels.forEach((label, i) => {
      const pct = Math.round((values[i] / total) * 100);
      svgEl.appendChild(svg("rect", {x: 10, y: ly - 10, width: 10, height: 10, fill: palette[i % palette.length]}));
      svgEl.appendChild(svg("text", {x: 26, y: ly - 1, fill: COLOR_INK, "font-family": FONT, "font-size": 11}, [document.createTextNode(`${label}: ${fmtInt(values[i])} (${pct}%)`)]));
      ly += 16;
    });
    el.appendChild(svgEl);
  }

  function renderStackedBarH(el, data, options) {
    clear(el);
    const labels = data.labels || [];
    const values = (data.values || []).map(Number);
    const total = values.reduce((a, b) => a + b, 0) || 1;
    const palette = [COLOR_SIGNAL, COLOR_WARN, COLOR_MUTED];
    const width = el.clientWidth || 360;
    const height = 48;
    const svgEl = svg("svg", {width: "100%", height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": (options && options.title) || "stacked bar"});
    let x = 0;
    values.forEach((v, i) => {
      const w = (v / total) * width;
      svgEl.appendChild(svg("rect", {x, y: 4, width: Math.max(0, w), height: 22, fill: palette[i % palette.length], opacity: 0.82}));
      if (w > 30) {
        svgEl.appendChild(svg("text", {x: x + w / 2, y: 20, "text-anchor": "middle", fill: "#041106", "font-family": FONT, "font-size": 10, "font-weight": 700}, [document.createTextNode(`${labels[i]} ${fmtInt(v)}`)]));
      }
      x += w;
    });
    el.appendChild(svgEl);
  }

  function renderKpiCard(el, data, options) {
    clear(el);
    const label = data.label || "";
    const value = data.value == null ? "-" : String(data.value);
    const note = data.note || "";
    const sparklineValues = Array.isArray(data.sparklineValues) ? data.sparklineValues : null;
    const width = el.clientWidth || 180;
    const height = 92;
    const card = svg("svg", {width: "100%", height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label});
    card.appendChild(svg("text", {x: 8, y: 16, fill: COLOR_MUTED, "font-family": FONT, "font-size": 10, "letter-spacing": 1.4}, [document.createTextNode(label.toUpperCase())]));
    card.appendChild(svg("text", {x: 8, y: 44, fill: COLOR_INK, "font-family": FONT, "font-size": 22, "font-weight": 700}, [document.createTextNode(value)]));
    if (note) card.appendChild(svg("text", {x: 8, y: 84, fill: COLOR_MUTED, "font-family": FONT, "font-size": 10}, [document.createTextNode(note)]));
    if (sparklineValues && sparklineValues.length >= 2) {
      const yMin = Math.min(...sparklineValues), yMax = Math.max(...sparklineValues);
      const yRange = (yMax - yMin) || 1;
      const sx = (i) => 8 + (i / (sparklineValues.length - 1)) * (width - 16);
      const sy = (v) => 72 - ((v - yMin) / yRange) * 20;
      let d = `M ${sx(0)} ${sy(sparklineValues[0])}`;
      for (let i = 1; i < sparklineValues.length; i++) d += ` L ${sx(i)} ${sy(sparklineValues[i])}`;
      const last = sparklineValues[sparklineValues.length - 1];
      card.appendChild(svg("path", {d, fill: "none", stroke: last >= 0 ? COLOR_SIGNAL : COLOR_WARN, "stroke-width": 1.5, opacity: 0.9}));
    }
    el.appendChild(card);
  }

  function renderMatrix(el, data, options) {
    clear(el);
    const rows = data.rows || [];
    const cols = data.cols || [];
    const cells = data.cells || {};
    const cellW = 64, cellH = 26, labelW = 100, headerH = 20;
    const width = labelW + cellW * cols.length;
    const height = headerH + cellH * rows.length;
    const svgEl = svg("svg", {width, height, role: "img", "aria-label": (options && options.title) || "matrix"});
    cols.forEach((c, i) => svgEl.appendChild(svg("text", {x: labelW + i * cellW + cellW / 2, y: 14, "text-anchor": "middle", fill: COLOR_MUTED, "font-family": FONT, "font-size": 10}, [document.createTextNode(c)])));
    rows.forEach((r, ri) => {
      svgEl.appendChild(svg("text", {x: labelW - 6, y: headerH + ri * cellH + cellH - 8, "text-anchor": "end", fill: COLOR_MUTED, "font-family": FONT, "font-size": 11}, [document.createTextNode(r)]));
      cols.forEach((c, ci) => {
        const v = cells[r] && cells[r][c];
        const x = labelW + ci * cellW, y = headerH + ri * cellH;
        const color = typeof v === "number" ? (v >= 0 ? COLOR_SIGNAL : COLOR_WARN) : COLOR_MUTED;
        svgEl.appendChild(svg("rect", {x, y, width: cellW - 1, height: cellH - 1, fill: color, opacity: typeof v === "number" ? 0.18 : 0.04}));
        svgEl.appendChild(svg("text", {x: x + cellW / 2, y: y + cellH - 8, "text-anchor": "middle", fill: COLOR_INK, "font-family": FONT, "font-size": 11}, [document.createTextNode(typeof v === "number" ? fmtSigned(v) : "-")]));
      });
    });
    el.appendChild(svgEl);
  }

  function uPlotTheme(extra) {
    const base = {
      class: "sicfun-uplot",
      width: 600,
      height: 200,
      axes: [
        {stroke: COLOR_MUTED, grid: {stroke: "rgba(74,246,38,0.08)"}, ticks: {stroke: "rgba(74,246,38,0.14)"}, font: `11px ${FONT}`},
        {stroke: COLOR_MUTED, grid: {stroke: "rgba(74,246,38,0.08)"}, ticks: {stroke: "rgba(74,246,38,0.14)"}, font: `11px ${FONT}`}
      ],
      cursor: {drag: {x: false, y: false}}
    };
    return Object.assign(base, extra || {});
  }

  function renderLine(el, data, options) {
    clear(el);
    const xs = (data.xs || []).map(Number);
    const ys = (data.ys || []).map(Number);
    if (xs.length < 2 || ys.length < 2) {
      el.textContent = "No per-hand data";
      return;
    }
    // Defensive fallback when vendor/uPlot.iife.min.js fails to load
    // (404 from a misconfigured proxy, blocked by a CSP-tightening
    // browser extension, network blip during page load). Without this
    // guard `new uPlot(...)` throws "uPlot is not defined", which the
    // surrounding renderHallResults / renderResults does not catch,
    // and the whole results board half-renders -- the KPI cards (raw
    // SVG, no uPlot dependency) and the renderBarH / renderDonut /
    // renderStackedBarH / renderMatrix charts all silently disappear
    // because the uncaught exception aborts the call chain. A text
    // placeholder keeps the rest of the panel intact.
    if (typeof uPlot === "undefined") {
      el.textContent = "Chart library unavailable.";
      return;
    }
    const opts = uPlotTheme({
      width: el.clientWidth || 600,
      height: (options && options.height) || 220,
      scales: {x: {time: false}},
      series: [
        {},
        {label: (options && options.yLabel) || "cumulative chips", stroke: COLOR_SIGNAL, width: 1.6, fill: "rgba(74,246,38,0.08)"}
      ]
    });
    new uPlot(opts, [xs, ys], el);
    // Label the container so screen readers announce a meaningful chart
    // name. uPlot renders a <canvas>; canvas elements have no inherent
    // semantics and would otherwise leave the dashboard chart slot
    // silent to assistive tech. The 5 SVG chart primitives above
    // (renderBarH / renderDonut / renderStackedBarH / renderMatrix /
    // renderKpiCard) all set role + aria-label on their <svg> root
    // directly; uPlot doesn't expose that hook, so attach to the
    // wrapper el instead. Default "line chart" mirrors the other
    // primitives' silent fallback so a caller omitting options.title
    // still gets a non-empty accessible name.
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", (options && options.title) || "line chart");
  }

  function renderHistogram(el, data, options) {
    clear(el);
    const values = (data.values || []).map(Number);
    if (values.length === 0) {
      el.textContent = "No equity data";
      return;
    }
    // Same vendor-script-missing fallback as renderLine -- see comment
    // there for the failure modes that drop window.uPlot.
    if (typeof uPlot === "undefined") {
      el.textContent = "Chart library unavailable.";
      return;
    }
    const bucketCount = (options && options.bucketCount) || 10;
    const min = options && options.min != null ? options.min : 0;
    const max = options && options.max != null ? options.max : 1;
    const width = (max - min) / bucketCount;
    const edges = Array.from({length: bucketCount + 1}, (_, i) => min + i * width);
    const counts = new Array(bucketCount).fill(0);
    values.forEach(v => {
      let idx = Math.floor((v - min) / width);
      if (idx < 0) idx = 0;
      if (idx >= bucketCount) idx = bucketCount - 1;
      counts[idx]++;
    });
    const centers = edges.slice(0, -1).map((e) => e + width / 2);
    const opts = uPlotTheme({
      width: el.clientWidth || 600,
      height: (options && options.height) || 200,
      scales: {x: {time: false, range: [min, max]}},
      series: [
        {},
        {
          label: (options && options.yLabel) || "decisions",
          stroke: COLOR_SIGNAL,
          fill: "rgba(74,246,38,0.22)",
          paths: (u) => {
            const pts = new Path2D();
            for (let i = 0; i < counts.length; i++) {
              const xL = u.valToPos(edges[i], "x", true);
              const xR = u.valToPos(edges[i + 1], "x", true);
              const yT = u.valToPos(counts[i], "y", true);
              const yB = u.valToPos(0, "y", true);
              pts.rect(xL, yT, xR - xL, yB - yT);
            }
            return {fill: pts, stroke: pts};
          }
        }
      ]
    });
    new uPlot(opts, [centers, counts], el);
    // Same screen-reader-labeling treatment as renderLine -- uPlot's
    // canvas is opaque to assistive tech, so attach role + aria-label
    // to the container element. Default "histogram" mirrors the other
    // primitives' silent fallback.
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", (options && options.title) || "histogram");
  }

  global.SicfunCharts = {
    renderBarH,
    renderDonut,
    renderStackedBarH,
    renderKpiCard,
    renderMatrix,
    renderLine,
    renderHistogram
  };
})(window);
