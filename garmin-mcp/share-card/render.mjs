// Render a run share card (1080x1350 PNG) from fetch_run.py's JSON.
//   node share-card/render.mjs share-card/run.json out.png [copy.json]
// copy.json (optional) overrides any of: headline, subline, tag, timeline.
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [, , inPath = "share-card/run.json", outPath = "share-card/run.png", copyPath] = process.argv;
const run = JSON.parse(readFileSync(inPath, "utf8"));
const copy = {
  headline: "LEGS ON.",
  subline: "Tri on Sunday. Marathon up next. Today was just about moving.",
  tag: "SHAKEOUT",
  timeline: ["TRI", "SHAKEOUT", "MARATHON"], // middle item is "you are here"
  ...(copyPath ? JSON.parse(readFileSync(copyPath, "utf8")) : {}),
};

const imperial = (run.unit_system || "").startsWith("statute");
const unit = imperial ? { d: 1609.344, dl: "MI", p: "/MI", e: 3.28084, el: "FT" } : { d: 1000, dl: "KM", p: "/KM", e: 1, el: "M" };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const fmtTime = (s) => {
  s = Math.round(s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
};
const fmtPace = (mps) => (mps > 0 ? fmtTime(unit.d / mps) : "–");

const dist = (run.distance_m || 0) / unit.d;
const start = run.start_local ? new Date(run.start_local.replace(" ", "T")) : null;
const dateLine = start
  ? start.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" }).toUpperCase() +
    " · " + start.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })
  : "";

const stats = [
  ["PACE", fmtPace(run.avg_speed), unit.p],
  ["TIME", fmtTime(run.moving_s || run.duration_s), ""],
  ["AVG HR", run.avg_hr ? Math.round(run.avg_hr) : "–", run.avg_hr ? "BPM" : ""],
  ["ELEV", run.elev_gain_m != null ? Math.round(run.elev_gain_m * unit.e) : "–", run.elev_gain_m != null ? unit.el : ""],
];

// Route → SVG path, equirectangular with latitude correction, fitted into the box.
function routeSvg(points, W, H, pad) {
  if (!points || points.length < 2) return "";
  const lat0 = points.reduce((a, p) => a + p[0], 0) / points.length;
  const k = Math.cos((lat0 * Math.PI) / 180);
  const xy = points.map(([la, lo]) => [lo * k, -la]);
  const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const s = Math.min((W - 2 * pad) / (maxX - minX || 1), (H - 2 * pad) / (maxY - minY || 1));
  const ox = (W - (maxX - minX) * s) / 2, oy = (H - (maxY - minY) * s) / 2;
  const pts = xy.map(([x, y]) => [ox + (x - minX) * s, oy + (y - minY) * s]);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const [sx, sy] = pts[0], [ex, ey] = pts[pts.length - 1];
  return `
  <svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="route">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#FFB21F"/><stop offset=".5" stop-color="#FF5A1F"/><stop offset="1" stop-color="#FF2E88"/>
      </linearGradient>
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="14"/></filter>
    </defs>
    <path d="${d}" fill="none" stroke="url(#g)" stroke-width="22" stroke-linecap="round" stroke-linejoin="round" opacity=".55" filter="url(#glow)"/>
    <path d="${d}" fill="none" stroke="url(#g)" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="${sx}" cy="${sy}" r="13" fill="#0B0B0D" stroke="#FFB21F" stroke-width="6"/>
    <circle cx="${ex}" cy="${ey}" r="13" fill="#FF2E88"/>
  </svg>`;
}

// Splits: full laps only (drop a short trailing partial), bar height = relative speed.
const laps = (run.laps || []).filter((l) => l.avg_speed > 0 && l.distance_m > unit.d * 0.6);
const speeds = laps.map((l) => l.avg_speed);
const hi = Math.max(...speeds);
const bars = laps
  .map((l, i) => {
    // Proportional to speed, so an even easy run reads as even rather than exaggerated.
    const h = 100 * (l.avg_speed / hi);
    return `<div class="bar"><div class="fill" style="height:${h}%"></div><span>${fmtPace(l.avg_speed)}</span><em>${i + 1}</em></div>`;
  })
  .join("");

const tl = copy.timeline
  .map((t, i) => `<div class="tl ${i === 1 ? "now" : i === 0 ? "done" : "next"}"><i></i><b>${esc(t)}</b></div>`)
  .join('<div class="tl-line"></div>');

const html = `<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@500;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
  :root { --bg:#0B0B0D; --ink:#F4F1EA; --mute:#8C8A86; --hot:#FF5A1F; --pink:#FF2E88; --gold:#FFB21F; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { width:1080px; height:1350px; background:var(--bg); color:var(--ink); font-family:Inter,sans-serif; overflow:hidden; position:relative; }
  .glow { position:absolute; inset:0; background:
      radial-gradient(600px 480px at 70% 38%, rgba(255,90,31,.20), transparent 70%),
      radial-gradient(500px 400px at 25% 60%, rgba(255,46,136,.12), transparent 70%); }
  .grain { position:absolute; inset:0; opacity:.07; mix-blend-mode:overlay;
      background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
  .wrap { position:relative; padding:64px 72px; height:100%; display:flex; flex-direction:column; }
  .top { display:flex; justify-content:space-between; align-items:center; font:700 22px 'JetBrains Mono',monospace; letter-spacing:.08em; color:var(--mute); }
  .tag { color:var(--bg); background:var(--gold); padding:10px 18px; border-radius:999px; letter-spacing:.14em; }
  .hero { position:relative; height:610px; margin-top:20px; }
  .route { position:absolute; inset:0; }
  .dist { position:absolute; left:-6px; bottom:-10px; font-family:Anton,sans-serif; font-size:260px; line-height:.82; letter-spacing:-.01em; }
  .dist small { font-size:84px; color:var(--hot); margin-left:14px; letter-spacing:.02em; }
  h1 { font-family:Anton,sans-serif; font-size:92px; line-height:1; margin-top:34px; letter-spacing:.01em;
       background:linear-gradient(90deg,var(--gold),var(--hot) 45%,var(--pink)); -webkit-background-clip:text; color:transparent; }
  .sub { font-size:30px; font-weight:500; color:#CFCBC4; margin-top:12px; max-width:936px; text-wrap:balance; line-height:1.3; }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); margin-top:40px; border-top:2px solid #26252A; padding-top:26px; }
  .stat label { font:700 18px 'JetBrains Mono',monospace; letter-spacing:.14em; color:var(--mute); }
  .stat div { font:800 52px Inter,sans-serif; margin-top:6px; letter-spacing:-.02em; }
  .stat div sup { font-size:20px; font-weight:700; color:var(--mute); margin-left:6px; vertical-align:10px; letter-spacing:.06em; }
  .bottom { margin-top:auto; display:flex; align-items:flex-end; justify-content:space-between; gap:40px; }
  .splits-wrap label { display:block; font:700 18px 'JetBrains Mono',monospace; letter-spacing:.14em; color:var(--mute); margin-bottom:14px; }
  .splits { display:flex; gap:10px; height:120px; align-items:flex-end; }
  .bar { width:44px; height:100%; display:flex; flex-direction:column; justify-content:flex-end; align-items:center; position:relative; }
  .bar .fill { width:100%; border-radius:8px 8px 3px 3px; background:linear-gradient(180deg,var(--hot),rgba(255,90,31,.25)); }
  .bar span { position:absolute; top:-26px; font:700 13px 'JetBrains Mono',monospace; color:var(--mute); white-space:nowrap; display:none; }
  .bar em { font:700 14px 'JetBrains Mono',monospace; color:var(--mute); font-style:normal; margin-top:8px; }
  .timeline { display:flex; align-items:center; font:700 18px 'JetBrains Mono',monospace; letter-spacing:.12em; }
  .tl { display:flex; flex-direction:column; align-items:center; gap:10px; }
  .tl i { width:20px; height:20px; border-radius:50%; border:3px solid var(--mute); }
  .tl.done i { background:var(--mute); } .tl.done b { color:var(--mute); text-decoration:line-through; text-decoration-thickness:2px; }
  .tl.now i { background:var(--hot); border-color:var(--hot); box-shadow:0 0 0 8px rgba(255,90,31,.22); }
  .tl.now b { color:var(--ink); } .tl.next b { color:var(--gold); } .tl.next i { border-color:var(--gold); }
  .tl-line { width:54px; height:3px; background:#34323A; margin:0 8px 30px; }
</style></head><body>
<div class="glow"></div><div class="grain"></div>
<div class="wrap">
  <div class="top"><span>${esc(dateLine)}${run.location ? " · " + esc(run.location.toUpperCase()) : ""}</span><span class="tag">${esc(copy.tag)}</span></div>
  <div class="hero">
    ${routeSvg(run.route, 936, 610, 40)}
    <div class="dist">${dist.toFixed(dist >= 10 ? 1 : 2)}<small>${unit.dl}</small></div>
  </div>
  <h1>${esc(copy.headline)}</h1>
  <p class="sub">${esc(copy.subline)}</p>
  <div class="stats">${stats.map(([l, v, u]) => `<div class="stat"><label>${l}</label><div>${esc(v)}${u ? `<sup>${u}</sup>` : ""}</div></div>`).join("")}</div>
  <div class="bottom">
    <div class="splits-wrap">${laps.length > 1 ? `<label>SPLITS</label><div class="splits">${bars}</div>` : ""}</div>
    <div class="timeline">${tl}</div>
  </div>
</div></body></html>`;

writeFileSync(outPath.replace(/\.png$/, ".html"), html);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1080, height: 1350 }, deviceScaleFactor: 2 });
await page.setContent(html, { waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);
await page.screenshot({ path: outPath });
await browser.close();
console.log(`Wrote ${outPath}`);
