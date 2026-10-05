/* SimAuthor project page. Data come from docs/data (built by scripts/build_page_data.py
   and scripts/render_generations.py from the released traces). */
(() => {
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f3 = (x) => (x == null ? "–" : Number(x).toFixed(3));
const sgn = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
const pad3 = (i) => String(i).padStart(3, "0");
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const getJSON = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(p + " " + r.status); return r.json(); });

const themeListeners = [];
const onTheme = (fn) => themeListeners.push(fn);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => themeListeners.forEach((f) => f()));
new MutationObserver(() => themeListeners.forEach((f) => f())).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

let INDEX = [], GENS = {}, CURVES = {}, MECHS = [];
const RUNS = {};

/* ───────────────────────── paper tables (main.tex) ───────────────────────── */
const KEYS = ["VSD", "AS", "COPD", "AF", "LQT", "WPW"];
const T_MAIN = [
  ["Root S(0)", [.258, .139, .165, .216, .286, .364]],
  ["Sampling (best of 100)", [.477, .184, .391, .492, .418, .456]],
  ["PUCT score search", [.356, .247, .455, .560, .532, .550]],
  ["Text-Opt", [.416, .412, .549, .801, .554, .526]],
  ["SimAuthor", [.587, .557, .672, .738, .568, .581], true],
];
const T_ABL = [
  ["VSD", .587, .366, .399, .395, .356, .348], ["AS", .557, .152, .383, .458, .143, .324],
  ["COPD", .672, .609, .662, .544, .585, .529], ["AF", .738, .589, .719, .717, .556, .686],
  ["LQT", .568, .460, .540, .539, .439, .514], ["WPW", .581, .492, .573, .574, .486, .547],
];
const T_HELD = [
  ["Root S(0)", [.223, .103, .088, .104, .277, .385]],
  ["Sampling", [.425, .146, .362, .117, .341, .443]],
  ["SimAuthor", [.577, .440, .507, .204, .480, .538], true],
];
const T_DOWN = [
  ["Real only", [.406, .633, .857, .691]],
  ["+ Conv. aug.", [.405, .642, .810, .733]],
  ["+ SimAuthor", [.404, .747, .861, .804], true],
];
const T_REP = [
  ["Root S(0)", [.317, .192, .152, .354]],
  ["Sampling", [.123, .106, .162, .434]],
  ["Text-Opt", [.443, .258, .468, .631]],
  ["SimAuthor", [.204, .306, .561, .665], true],
];

function bestIdx(rows, col) {
  let b = -1, v = -1;
  rows.forEach((r, i) => { if (r[1][col] > v) { v = r[1][col]; b = i; } });
  return b;
}
function simpleTable(el, head, rows, skipBest = 0) {
  const best = head.slice(1).map((_, c) => bestIdx(rows.slice(skipBest), c) + skipBest);
  el.innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r, i) => `<tr class="${r[2] ? "ours" : ""}"><td>${r[0]}</td>${r[1].map((v, c) =>
      `<td class="${best[c] === i ? "best" : ""}">${f3(v)}</td>`).join("")}</tr>`).join("") + "</tbody>";
}
/* dot charts for the generalization subsection: one row per task, one dot per method */
function dotChart(el, rowLabels, series) {
  const W = 360, rowH = 26, m = { l: 62, r: 14, t: 8, b: 26 }, H = m.t + m.b + rowH * rowLabels.length;
  const all = series.flatMap((s) => s.values);
  const lo = Math.max(0, Math.floor((d3.min(all) - .05) * 10) / 10), hi = Math.min(1, Math.ceil((d3.max(all) + .03) * 10) / 10);
  const x = d3.scaleLinear([lo, hi], [m.l, W - m.r]);
  let h = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  x.ticks(4).forEach((t) => { h += `<line class="gridline" x1="${x(t)}" x2="${x(t)}" y1="${m.t}" y2="${H - m.b}"/><text x="${x(t)}" y="${H - 8}" text-anchor="middle" style="fill:${css("--muted")};font:11px var(--f-sans)">${t.toFixed(1)}</text>`; });
  rowLabels.forEach((lab, i) => {
    const y = m.t + rowH * i + rowH / 2, vals = series.map((s) => s.values[i]);
    h += `<text x="${m.l - 10}" y="${y + 4}" text-anchor="end" style="fill:${css("--fg-2")};font:12px var(--f-sans)">${lab}</text>`;
    h += `<line x1="${x(d3.min(vals))}" x2="${x(d3.max(vals))}" y1="${y}" y2="${y}" stroke="${css("--line")}" stroke-width="3" stroke-linecap="round"/>`;
    series.forEach((s) => {
      const ours = s.ours;
      h += `<circle cx="${x(s.values[i])}" cy="${y}" r="${ours ? 6 : 4.5}" fill="${s.color}" stroke="${css("--bg")}" stroke-width="2"><title>${lab} · ${s.name}: ${f3(s.values[i])}</title></circle>`;
    });
  });
  h += "</svg>";
  el.innerHTML = `<div class="gen-legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div>` + h;
}
function renderGenCharts() {
  const cols = { root: css("--muted"), samp: css("--s4"), text: css("--s3"), ours: css("--s1"), conv: css("--s2") };
  dotChart($("#chart-rep"), ["VSD", "AS", "COPD", "AF"], [
    { name: "Root", values: T_REP[0][1], color: cols.root }, { name: "Sampling", values: T_REP[1][1], color: cols.samp },
    { name: "Text-Opt", values: T_REP[2][1], color: cols.text }, { name: "SimAuthor", values: T_REP[3][1], color: cols.ours, ours: true }]);
  dotChart($("#chart-held"), KEYS, [
    { name: "Root", values: T_HELD[0][1], color: cols.root }, { name: "Sampling", values: T_HELD[1][1], color: cols.samp },
    { name: "SimAuthor", values: T_HELD[2][1], color: cols.ours, ours: true }]);
  dotChart($("#chart-down"), ["LQT ID", "LQT OOD", "WPW ID", "WPW OOD"], [
    { name: "Real only", values: T_DOWN[0][1], color: cols.root }, { name: "+ Conv. aug.", values: T_DOWN[1][1], color: cols.conv },
    { name: "+ SimAuthor", values: T_DOWN[2][1], color: cols.ours, ours: true }]);
}
function renderTables() {
  simpleTable($("#table-held"), ["Method", ...KEYS], T_HELD, 1);
  simpleTable($("#table-down"), ["Training data", "ID", "OOD", "ID", "OOD"], T_DOWN);
  $("#table-down thead").insertAdjacentHTML("afterbegin", '<tr><th></th><th colspan="2" style="text-align:center">LQT</th><th colspan="2" style="text-align:center">WPW</th></tr>');
  simpleTable($("#table-rep"), ["Method", "VSD", "AS", "COPD", "AF"], T_REP);
  const d = (a, b) => `<span class="dlt">${b - a >= 0 ? "+" : "−"}${Math.abs(b - a).toFixed(3).slice(1)}</span>`;

}

/* ───────────────────────── hero: three scope channels ───────────────────────── */
const hero = { real: 0, audio: null, playing: null };
function melImage(mel) {
  const raw = atob(mel.b64), img = new ImageData(mel.w, mel.h);
  for (let i = 0; i < raw.length; i++) {
    const c = d3.rgb(d3.interpolateMagma(raw.charCodeAt(i) / 255));
    img.data[4 * i] = c.r; img.data[4 * i + 1] = c.g; img.data[4 * i + 2] = c.b; img.data[4 * i + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = mel.w; off.height = mel.h;
  off.getContext("2d").putImageData(img, 0, 0); return off;
}
function fit(cv) {
  const dpr = devicePixelRatio || 1; cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return [ctx, cv.clientWidth, cv.clientHeight];
}
function graticule(ctx, w, h, cols, rows, color) {
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let i = 1; i < cols; i++) { const x = Math.round(i * w / cols) + .5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let j = 1; j < rows; j++) { const y = Math.round(j * h / rows) + .5; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
}
function drawScope(prefix, smp, color) {
  const [mc, mw, mh] = fit($(`#${prefix}-mel`));
  mc.imageSmoothingEnabled = true; mc.drawImage(melImage(smp.mel), 0, 0, mw, mh); graticule(mc, mw, mh, 10, 4, "rgba(255,255,255,.14)");
  const [tc, w, h] = fit($(`#${prefix}-trace`)); graticule(tc, w, h, 10, 2, getComputedStyle($(".band")).getPropertyValue("--grat").trim());
  const env = smp.env, n = env.length / 2, mid = h / 2;
  tc.strokeStyle = color; tc.lineWidth = 1.1; tc.shadowColor = color; tc.shadowBlur = 3;
  tc.beginPath();
  for (let i = 0; i < n; i++) { const x = i / (n - 1) * w; tc.moveTo(x, mid - env[2 * i + 1] * mid * .92); tc.lineTo(x, mid - env[2 * i] * mid * .92); }
  tc.stroke();
}
const HERO_SRC = { VSD: "ZCHSound", COPD: "ICBHI 2017" };
const HERO_DESC = { VSD: "Heart sounds with a ventricular septal defect", COPD: "Lung sounds in COPD" };
hero.task = 0; hero.ex = 0;
function heroDraw() {
  const ex = GENS._hero && GENS._hero.examples[hero.task]; if (!ex) return;
  const it = ex.items[hero.ex], cs = getComputedStyle($(".band"));
  drawScope("real", it.real, cs.getPropertyValue("--real").trim());
  drawScope("root", it.root, cs.getPropertyValue("--sim").trim());
  drawScope("best", it.authored, cs.getPropertyValue("--sim").trim());
}
function heroRender() {
  const ex = GENS._hero.examples[hero.task];
  $$("#hero-task button").forEach((b, i) => b.setAttribute("aria-pressed", String(i === hero.task)));
  $("#real-name").textContent = `Real recording (${HERO_SRC[ex.task]})`;
  $("#best-name").textContent = `Authored program #${ex.best}`;
  $("#root-read").textContent = `score ${f3(ex.root_score)}`;
  $("#best-read").textContent = `score ${f3(ex.best_score)}`;
  $("#hero-cap").textContent = `${HERO_DESC[ex.task]}, 10 s. A real recording, the zero-shot root program and the program after 100 authoring attempts. Real recordings are shown as images only.`;
  heroDraw();
}
function stopHeroAudio() {
  if (hero.audio) hero.audio.pause();
  hero.playing = null;
  $$("[data-play]").forEach((x) => { x.textContent = "Play"; x.setAttribute("aria-pressed", "false"); });
}
function initHero() {
  if (!GENS._hero) return;
  $("#hero-task").innerHTML = GENS._hero.examples.map((e, i) => `<button data-t="${i}" aria-pressed="${i === 0}">${e.task === "VSD" ? "Heart sounds (VSD)" : "Lung sounds (COPD)"}</button>`).join("");
  $$("#hero-task button").forEach((b) => b.addEventListener("click", () => { hero.task = +b.dataset.t; stopHeroAudio(); heroRender(); }));
  $$("[data-play]").forEach((b) => b.addEventListener("click", () => {
    const id = b.dataset.play, same = hero.playing === id && hero.audio && !hero.audio.paused;
    stopHeroAudio();
    if (same) return;
    const it = GENS._hero.examples[hero.task].items[hero.ex];
    hero.audio = new Audio(it[id].audio); hero.playing = id;
    hero.audio.play().catch(() => {}); b.textContent = "Stop"; b.setAttribute("aria-pressed", "true");
    hero.audio.onended = () => stopHeroAudio();
  }));
  heroRender(); onTheme(heroDraw);
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(heroDraw, 150); });
}

/* ───────────────────────── harness lab ───────────────────────── */
const S = { key: "LQT", run: null, step: 0, sel: 0, tab: "change", playing: false, timer: 0, phaseTimers: [] };

const DG_BOX = {
  blueprint: [16, 12, 170, 52, "Scientific blueprint", "written once"],
  library: [454, 12, 170, 52, "Mechanism library", "≤ 8 recent entries"],
  refiner: [232, 96, 176, 56, "Refiner (LLM)", "one focused edit"],
  execute: [454, 96, 170, 56, "Execute program", "100 records"],
  orch: [16, 186, 170, 52, "Orchestrator", "flat PUCT"],
  evaluator: [454, 186, 170, 52, "Evaluator", "score + report"],
};
const DG_EDGE = {
  bp_ref: ["M186 38 C 225 38, 255 66, 285 96", ""],
  lib_ref: ["M454 32 C 410 32, 378 60, 362 96", ""],
  ref_lib: ["M408 106 C 430 88, 460 76, 500 64", "extract if Δ ≥ .02"],
  orch_ref: ["M110 186 C 140 150, 190 128, 232 124", "parent"],
  ref_exe: ["M408 124 L 454 124", "child"],
  exe_eval: ["M539 152 L 539 186", ""],
  eval_orch: ["M454 222 L 186 222", "score"],
  eval_ref: ["M454 200 C 410 192, 362 176, 338 152", "report"],
};
const PHASES = [
  ["select", "Select a parent", ["orch"], ["orch_ref"]],
  ["refine", "Refine with blueprint, report and library", ["refiner", "blueprint", "library"], ["bp_ref", "lib_ref", "eval_ref"]],
  ["execute", "Execute the child program", ["execute"], ["ref_exe"]],
  ["evaluate", "Score against real recordings", ["evaluator"], ["exe_eval"]],
  ["archive", "Archive the child in the tree", ["orch"], ["eval_orch"]],
  ["extract", "Distill a mechanism into the library", ["library"], ["ref_lib"]],
];
function buildDiagram() {
  const svg = $("#diagram");
  let h = `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z"/></marker></defs>`;
  for (const [id, [d, lab]] of Object.entries(DG_EDGE)) {
    h += `<path class="dg-edge" id="e-${id}" d="${d}" marker-end="url(#arrow)"${id === "ref_lib" ? ' stroke-dasharray="4 4"' : ""}/>`;
    if (lab) {
      const m = d.match(/[-\d.]+/g).map(Number), x = (m[0] + m[m.length - 2]) / 2, y = (m[1] + m[m.length - 1]) / 2;
      const off = { eval_orch: [0, -8], orch_ref: [-26, -6], ref_lib: [44, -4], eval_ref: [-10, 16], ref_exe: [0, -8] }[id] || [0, -6];
      h += `<text class="dg-label" x="${x + off[0]}" y="${y + off[1]}" text-anchor="middle">${lab}</text>`;
    }
  }
  for (const [id, [x, y, w, hh, t, s]] of Object.entries(DG_BOX)) {
    h += `<g class="dg-box" id="b-${id}"><rect x="${x}" y="${y}" width="${w}" height="${hh}" rx="9"/>
      <text x="${x + w / 2}" y="${y + hh / 2 - 2}" text-anchor="middle">${t}</text>
      <text class="sub" x="${x + w / 2}" y="${y + hh / 2 + 15}" text-anchor="middle">${s}</text></g>`;
  }
  svg.innerHTML = h;
  Object.entries(BOX_TAB).forEach(([box, tab]) => {
    const g = $("#b-" + box); g.style.cursor = "pointer";
    g.setAttribute("role", "button"); g.setAttribute("tabindex", "0");
    g.addEventListener("click", () => setTab(tab));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setTab(tab); } });
  });
}
// Each inspector tab corresponds to one part of the harness diagram.
const TAB_PARTS = {
  change: { label: "Code change: the refiner's revision", boxes: ["refiner"], edges: ["ref_exe"] },
  feedback: { label: "Evaluator feedback: score and report", boxes: ["evaluator"], edges: ["eval_orch", "eval_ref"] },
  context: { label: "Refiner context: what the refiner was given", boxes: ["blueprint", "library", "orch"], edges: ["bp_ref", "lib_ref", "orch_ref", "eval_ref"] },
  code: { label: "Program: the executed simulator", boxes: ["execute"], edges: ["exe_eval"] },
};
const BOX_TAB = { refiner: "change", evaluator: "feedback", blueprint: "context", library: "context", orch: "context", execute: "code" };
function lightTab(tab) {
  if (S.phaseTimers.length) return; // an attempt animation is running
  $$(".dg-box,.dg-edge").forEach((e) => e.classList.remove("on", "hot"));
  const part = TAB_PARTS[tab]; if (!part) return;
  part.boxes.forEach((b) => $("#b-" + b).classList.add("on"));
  part.edges.forEach((e) => $("#e-" + e).classList.add("on"));
  $("#phase-label").textContent = part.label;
}
function lightPhase(i, hot) {
  $$(".dg-box,.dg-edge").forEach((e) => e.classList.remove("on", "hot"));
  if (i < 0) { $("#phase-label").textContent = ""; return; }
  const [, label, boxes, edges] = PHASES[i];
  boxes.forEach((b) => $("#b-" + b).classList.add(hot ? "hot" : "on"));
  edges.forEach((e) => $("#e-" + e).classList.add(hot ? "hot" : "on"));
  $("#phase-label").textContent = label;
}

function prepRun(run) {
  const N = run.nodes.length;
  run.children = Array.from({ length: N }, () => []);
  run.nodes.forEach((n) => { if (n.parent != null) run.children[n.parent].push(n.id); });
  run.linSet = new Set(run.lineage);
  // replay visit backpropagation exactly as the orchestrator does
  const v = new Array(N).fill(0); run.visitsAt = [v.slice()];
  for (let t = 1; t < N; t++) {
    let c = t; v[c]++;
    while (run.nodes[c].parent != null) { c = run.nodes[c].parent; v[c]++; }
    run.visitsAt.push(v.slice());
  }
  run.bsf = []; let b = -1;
  run.nodes.forEach((n) => { b = Math.max(b, n.score); run.bsf.push(b); });
  const root = d3.stratify().id((d) => d.id).parentId((d) => d.parent)(run.nodes);
  // top-down tidy tree: x = breadth, y = depth (revisions from the root)
  root.sort((a, b) => a.data.id - b.data.id);
  d3.tree().nodeSize([10, 30]).separation((a, c) => (a.parent === c.parent ? 1 : 1.5))(root);
  const xs = root.descendants().map((d) => d.x);
  run.x0 = d3.min(xs); run.x1 = d3.max(xs); run.depth = d3.max(root.descendants(), (d) => d.depth) || 1;
  run.pos = {};
  root.each((d) => { run.pos[d.data.id] = { x: d.x, y: d.depth * 30 }; });
  run.maxScore = d3.max(run.nodes, (n) => n.score);
  run.minScore = d3.min(run.nodes, (n) => n.score);
}

async function loadRun(key) {
  if (!RUNS[key]) { RUNS[key] = await getJSON(`data/runs/${key}.json`); prepRun(RUNS[key]); }
  return RUNS[key];
}

function taskTabs(el, onPick, current) {
  el.innerHTML = INDEX.map((t) => `<button class="tab" role="tab" data-key="${t.key}" aria-selected="${t.key === current}" title="${t.name} (${t.modality_label})">${t.key}</button>`).join("");
  el.addEventListener("click", (e) => { const b = e.target.closest(".tab"); if (b) onPick(b.dataset.key); });
}
function markTabs(el, key) { $$(".tab", el).forEach((b) => b.setAttribute("aria-selected", String(b.dataset.key === key))); }

async function pickTask(key, step = null) {
  stopPlay();
  S.key = key; markTabs($("#task-tabs"), key);
  $("#insp-body").innerHTML = '<p class="loading">Loading the run…</p>';
  S.run = await loadRun(key);
  const N = S.run.nodes.length - 1;
  $("#step").max = N;
  drawTree();
  if (step == null) { setStep(N, false); select(S.run.best); }
  else setStep(step, true);
}

const colorScale = () => d3.scaleSequential(d3.interpolateRgb(css("--seq-lo"), css("--seq-hi"))).domain([S.run.minScore, S.run.maxScore]);

function drawTree() {
  const run = S.run, el = $("#tree"), pad = 22;
  const W = run.x1 - run.x0 + 2 * pad, H = run.depth * 30 + 2 * pad + 16;
  const P = (id) => [run.pos[id].x - run.x0 + pad, run.pos[id].y + pad + 8];
  const col = colorScale();
  let h = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Search tree of ${run.nodes.length} programs">`;
  for (let d = 1; d <= run.depth; d++) h += `<line x1="0" x2="${W}" y1="${d * 30 + pad + 8}" y2="${d * 30 + pad + 8}" class="gridline" opacity=".5"/>`;
  const link = d3.linkVertical();
  run.nodes.forEach((n) => {
    if (n.parent == null) return;
    const lin = run.linSet.has(n.id);
    h += `<path class="t-link${lin ? " lin" : ""}" data-id="${n.id}" d="${link({ source: P(n.parent), target: P(n.id) })}"/>`;
  });
  run.nodes.forEach((n) => {
    const [x, y] = P(n.id), lin = run.linSet.has(n.id);
    const r = n.id === 0 || n.id === run.best ? 6.5 : lin ? 5 : 4;
    h += `<circle class="t-node${lin ? " lin" : ""}" data-id="${n.id}" cx="${x}" cy="${y}" r="${r}" fill="${col(n.score)}"/>`;
  });
  const [rx, ry] = P(0), [bx, by] = P(run.best);
  h += `<text class="t-label" x="${rx}" y="${ry - 11}" text-anchor="middle">S(0) ${f3(run.nodes[0].score)}</text>`;
  h += `<text class="t-label" id="best-label" x="${bx}" y="${by + 18}" text-anchor="middle">best #${run.best} ${f3(run.nodes[run.best].score)}</text>`;
  h += `<circle class="t-now" id="t-now" r="9"/><circle class="t-ring" id="t-ring" r="8"/></svg><div class="tip" hidden></div>`;
  el.innerHTML = h;
  run.P = P;
  const tip = $(".tip", el);
  $$(".t-node", el).forEach((c) => {
    c.addEventListener("click", () => select(+c.dataset.id));
    c.addEventListener("mouseenter", () => {
      const n = run.nodes[+c.dataset.id], box = el.getBoundingClientRect(), cb = c.getBoundingClientRect();
      tip.innerHTML = `#${n.id} · ${f3(n.score)}${n.parent != null ? ` · Δ ${sgn(n.score - run.nodes[n.parent].score)}` : " · root"}`;
      tip.style.left = cb.left - box.left + cb.width / 2 + "px"; tip.style.top = cb.top - box.top + "px"; tip.hidden = false;
    });
    c.addEventListener("mouseleave", () => { tip.hidden = true; });
  });
  $("#ramp-lo").textContent = f3(run.minScore); $("#ramp-hi").textContent = f3(run.maxScore);
}

function setStep(t, follow = true) {
  const run = S.run, N = run.nodes.length - 1;
  S.step = Math.max(0, Math.min(N, t));
  $("#step").value = S.step;
  $$("#tree .t-node").forEach((c) => { c.style.display = +c.dataset.id <= S.step ? "" : "none"; });
  $$("#tree .t-link").forEach((c) => { c.style.display = +c.dataset.id <= S.step ? "" : "none"; });
  const bl = $("#best-label"); if (bl) bl.style.display = run.best <= S.step ? "" : "none";
  const [nx, ny] = run.P(S.step); const now = $("#t-now"); now.setAttribute("cx", nx); now.setAttribute("cy", ny);
  $("#tree-count").textContent = `${S.step + 1} of ${N + 1} programs`;
  updateStatus();
  if (follow) select(S.step);
}

function updateStatus() {
  const run = S.run, n = run.nodes[S.step];
  if (S.step === 0) {
    $("#status").innerHTML = `<b>Root S(0)</b> written in one call from the condition name and the output contract, before any recording was seen. Search score <b>${f3(n.score)}</b>. Drag the slider to follow all ${run.attempts} authoring attempts.`;
    return;
  }
  if (S.step === run.nodes.length - 1 && S.sel === run.best) {
    const b = run.nodes[run.best];
    $("#status").innerHTML = `All ${run.attempts} attempts shown. The best program, <b>#${b.id}</b> (${f3(b.score)}), was reached at attempt ${b.attempt}, ${run.lineage.length - 1} revisions away from S(0) (${f3(run.nodes[0].score)}). Drag the slider to replay the run.`;
    return;
  }
  const p = run.nodes[n.parent], d = n.score - p.score;
  const prevAttempt = run.nodes[S.step - 1].attempt || 0, failed = Math.max(0, n.attempt - prevAttempt - 1);
  const newBest = run.bsf[S.step] > run.bsf[S.step - 1];
  $("#status").innerHTML = `Attempt <b>${n.attempt}</b> of ${run.attempts}${failed ? ` <span class="muted">(after ${failed} invalid attempt${failed > 1 ? "s" : ""} that produced no usable program)</span>` : ""}: revised <b>#${p.id}</b> (${f3(p.score)}) into <b>#${n.id}</b> (${f3(n.score)}, <span class="${d >= 0 ? "up" : "down"}">${sgn(d)}</span>).` +
    (newBest ? ` <b>New best.</b>` : "") + (n.mech.length ? ` Mechanism mined: <code>${esc(n.mech[0].name)}</code>.` : "");
}

/* playback */
function stopPlay() {
  S.phaseTimers.forEach(clearTimeout); S.phaseTimers = [];
}
function setTab(tab) { S.tab = tab; renderInspector(); }
/* Walk the diagram through one authoring attempt (select → … → archive), then rest. */
function animateAttempt() {
  stopPlay();
  const n = S.run.nodes[S.step];
  if (!n || n.parent == null || REDUCED) { lightTab(S.tab); return; }
  const phases = [0, 1, 2, 3, 4].concat(n.mech.length ? [5] : []);
  phases.forEach((ph, k) => S.phaseTimers.push(setTimeout(() => lightPhase(ph, ph === 5), k * 260)));
  S.phaseTimers.push(setTimeout(() => { S.phaseTimers = []; lightTab(S.tab); }, phases.length * 260 + 900));
}
function buildMarks() {
  const run = S.run, N = run.nodes.length - 1;
  $("#scrub-marks").innerHTML = run.nodes.map((n, i) => (i > 0 && run.bsf[i] > run.bsf[i - 1])
    ? `<i style="left:${(i / N) * 100}%" title="#${i}: new best ${f3(n.score)}"></i>` : "").join("");
}

/* selection + inspector */
function markSelection() {
  const run = S.run, [x, y] = run.P(S.sel), ring = $("#t-ring");
  ring.setAttribute("cx", x); ring.setAttribute("cy", y);
}
function select(id) { S.sel = id; markSelection(); renderInspector(); updateStatus(); }

function lineDiff(a, b) {
  a = a.split("\n"); b = b.split("\n");
  let s = 0; while (s < a.length && s < b.length && a[s] === b[s]) s++;
  let e = 0; while (e < a.length - s && e < b.length - s && a[a.length - 1 - e] === b[b.length - 1 - e]) e++;
  const A = a.slice(s, a.length - e), B = b.slice(s, b.length - e), n = A.length, m = B.length;
  const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const ops = [];
  for (let k = 0; k < s; k++) ops.push(["c", a[k], k + 1]);
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && A[i] === B[j]) { ops.push(["c", A[i], s + j + 1]); i++; j++; }
    else if (j < m && (i >= n || L[i][j + 1] >= L[i + 1][j])) { ops.push(["a", B[j], s + j + 1]); j++; }
    else { ops.push(["d", A[i], null]); i++; }
  }
  for (let k = a.length - e; k < a.length; k++) ops.push(["c", a[k], null]);
  return ops;
}
function diffHTML(ops, ctx = 3) {
  const keep = new Array(ops.length).fill(false);
  ops.forEach((o, i) => { if (o[0] !== "c") for (let k = Math.max(0, i - ctx); k <= Math.min(ops.length - 1, i + ctx); k++) keep[k] = true; });
  let h = "", gap = false, add = 0, del = 0;
  ops.forEach((o, i) => {
    if (o[0] === "a") add++; if (o[0] === "d") del++;
    if (!keep[i]) { gap = true; return; }
    if (gap || (i > 0 && !keep[i - 1])) { h += `<div class="h">⋯ ${o[2] ? "line " + o[2] : ""}</div>`; gap = false; }
    h += `<div class="${o[0]}">${esc(o[1])}</div>`;
  });
  return { html: h || '<div class="h">No line changes.</div>', add, del };
}

function reportHTML(rep) {
  if (!rep) return '<p class="muted">No report stored.</p>';
  const overlapCol = rep.header.findIndex((h) => /overlap|value/i.test(h));
  let h = `<p style="font-weight:600;margin-bottom:4px">${esc(rep.title)}</p><p class="muted" style="font-size:.8rem;margin-bottom:8px">${esc(rep.meta)}</p>`;
  h += `<div class="table-scroll"><table class="report-table"><thead><tr>${rep.header.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>`;
  rep.rows.forEach((r) => {
    h += `<tr class="${r.n ? "noted" : ""}">${r.c.map((c, i) => {
      const v = parseFloat(c);
      const bar = i === overlapCol && v >= 0 && v <= 1 ? `<span class="bar" style="width:${Math.round(v * 44)}px"></span>` : "";
      return `<td>${bar}${esc(c)}</td>`;
    }).join("")}</tr>`;
    if (r.n) h += `<tr><td class="note" colspan="${r.c.length}">${esc(r.n)}</td></tr>`;
  });
  h += "</tbody></table></div>";
  if (rep.obs.length) h += `<ul class="obs">${rep.obs.slice(0, 8).map((o) => `<li>${esc(o.replace(/^[-*]\s*/, ""))}</li>`).join("")}</ul>`;
  return h;
}
function mechCard(m, extra = "") {
  return `<div class="mech-card"><span class="nm">${esc(m.name)}</span><p>${esc(m.mechanism)}</p>${m.snippet ? `<code>${esc(m.snippet)}</code>` : ""}
    <span class="mech-meta"><span>Δ ${m.delta != null ? sgn(m.delta) : "–"}</span><span>observed ${m.count || 1}×</span>${extra}</span></div>`;
}

function renderInspector() {
  const run = S.run, n = run.nodes[S.sel], p = n.parent != null ? run.nodes[n.parent] : null;
  const d = p ? n.score - p.score : null;
  const tags = [run.linSet.has(n.id) ? "on the best lineage" : "", n.id === run.best ? "best program of the run" : ""].filter(Boolean).join(", ");
  $("#insp-head").innerHTML = `<div class="insp-title"><b>${n.id === 0 ? "Root program S(0)" : "Program #" + n.id}</b>
    <span class="num">score ${f3(n.score)}</span>
    ${p ? `<span class="num ${d >= 0 ? "up" : "down"}">${sgn(d)} vs #${p.id}</span>` : ""}</div>
    <span class="muted" style="font-size:.86rem">${run.name}, ${run.evaluator}, seed ${run.seed}${n.attempt ? `, attempt ${n.attempt} of ${run.attempts}` : ""}${tags ? `, ${tags}` : ""}</span>`;
  $$("#insp-tabs .subtab").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === S.tab)));
  const body = $("#insp-body"); let h = "";
  if (S.tab === "change") {
    if (!p) {
      h = `<h4>Root program</h4><p style="font-size:.92rem;color:var(--fg-2)">S(0) was generated in a single call from the condition name, the modality and the immutable output contract. Every later program descends from it. Open <b>Program</b> to read it, or pick any node in the tree to see its edit.</p>`;
    } else {
      const df = diffHTML(lineDiff(p.code, n.code));
      if (n.mech.length) h += `<h4>Mechanism extracted from this revision</h4>${n.mech.map((m) => mechCard(m)).join("")}`;
      if (n.notes) h += `<h4>Revision note in the program docstring</h4><div class="prose-box">${esc(n.notes)}</div>`;
      h += `<h4>Diff against parent #${p.id} <span style="text-transform:none;letter-spacing:0">· <span style="color:var(--good)">+${df.add}</span> <span style="color:var(--bad)">−${df.del}</span> lines</span></h4><div class="diff">${df.html}</div>`;
    }
  } else if (S.tab === "feedback") {
    h = `<h4>Discrepancy report for this program</h4>${reportHTML(n.report)}`;
    if (n.img) h += `<h4>Visual feedback</h4><img class="visual-img" loading="lazy" alt="Real recordings (top) and this program's outputs (bottom)" src="assets/visual/${run.key}/node_${pad3(n.id)}.jpg">`;
    else h += `<p class="muted" style="font-size:.82rem;margin-top:12px">The comparison figure is published for the best lineage and the ten top-scoring programs.</p>`;
    h += `<p class="muted" style="font-size:.82rem;margin-top:10px">The score drives selection; the report and figure are what the refiner sees when it revises this program.</p>`;
  } else if (S.tab === "context") {
    if (!p) {
      h = `<h4>Initialization</h4><p style="font-size:.9rem;color:var(--fg-2)">The blueprint and the root are independent one-shot calls. The blueprint is written at temperature 0.2 without access to any recording and is then shown to the refiner on every attempt.</p><details class="fold" open><summary>Scientific blueprint (${run.shared.blueprint.length.toLocaleString()} characters)</summary><div class="prose-box">${esc(run.shared.blueprint)}</div></details>`;
    } else {
      h = `<h4>What the refiner received to write #${n.id}</h4>
        <p style="font-size:.88rem;color:var(--fg-2);margin-bottom:10px">One prompt of ${n.prompt_chars.toLocaleString()} characters, plus parent #${p.id}'s comparison figure.</p>
        <details class="fold"><summary>Task identity and scientific blueprint</summary><div><div class="prose-box">${esc(run.shared.task + "\n\n" + run.shared.blueprint)}</div></div></details>
        <details class="fold"><summary>Immutable output contract</summary><div><div class="prose-box">${esc(run.shared.contract)}</div></div></details>
        <details class="fold"><summary>Current simulator: parent #${p.id} (${p.code.split("\n").length} lines)</summary><div><pre class="codebox">${esc(p.code)}</pre></div></details>
        <details class="fold" open><summary>Numerical discrepancy report of #${p.id}</summary><div>${reportHTML(p.report)}</div></details>
        <details class="fold" open><summary>Mechanism library (${n.prompt_mech.length} entr${n.prompt_mech.length === 1 ? "y" : "ies"})</summary><div>${n.prompt_mech.length ? `<div class="tag-list">${n.prompt_mech.map((m) => `<span class="tag">${esc(m)}</span>`).join("")}</div>` : '<p class="muted" style="font-size:.85rem">Empty at this point of the run.</p>'}</div></details>
        <details class="fold"><summary>Refinement policy</summary><div><div class="prose-box">${esc(run.shared.policy + "\n\n" + run.shared.visual)}</div></div></details>`;
    }
  } else {
    h = `<pre class="codebox"><code class="language-python" id="code-el">${esc(n.code)}</code></pre>`;
  }
  body.innerHTML = h;
  if (S.tab === "code" && window.hljs && n.code.length < 60000) hljs.highlightElement($("#code-el"));
  lightTab(S.tab);
  $$("[data-goto]", $("#insp-head")).forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); select(+a.dataset.goto); }));
}

function initLab() {
  buildDiagram();
  taskTabs($("#task-tabs"), (k) => pickTask(k), S.key);
  $("#btn-next").addEventListener("click", () => { setStep(S.step + 1); animateAttempt(); });
  $("#btn-prev").addEventListener("click", () => { setStep(S.step - 1); animateAttempt(); });
  $("#step").addEventListener("input", (e) => { stopPlay(); setStep(+e.target.value); });
  $("#step").addEventListener("change", animateAttempt);
  $("#step").addEventListener("keydown", (e) => { if (e.key.startsWith("Arrow")) setTimeout(animateAttempt, 0); });
  $("#insp-tabs").addEventListener("click", (e) => { const b = e.target.closest(".subtab"); if (b) setTab(b.dataset.tab); });
  onTheme(() => { if (S.run) { drawTree(); setStep(S.step, false); markSelection(); } });
}

/* ───────────────────────── generations ───────────────────────── */
const L = { key: "COPD", pos: null, sample: 0, audio: null };
function drawMel(cv, mel) {
  const raw = atob(mel.b64), img = new ImageData(mel.w, mel.h);
  for (let i = 0; i < raw.length; i++) {
    const c = d3.rgb(d3.interpolateMagma(raw.charCodeAt(i) / 255));
    img.data[4 * i] = c.r; img.data[4 * i + 1] = c.g; img.data[4 * i + 2] = c.b; img.data[4 * i + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = mel.w; off.height = mel.h; off.getContext("2d").putImageData(img, 0, 0);
  const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d"); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, cv.width, cv.height);
}
function drawEnv(cv, env) {
  const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight; cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr); ctx.fillStyle = css("--bg"); ctx.fillRect(0, 0, w, h);
  const n = env.length / 2, mid = h / 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) ctx.lineTo(i / (n - 1) * w, mid - env[2 * i + 1] * mid * .9);
  for (let i = n - 1; i >= 0; i--) ctx.lineTo(i / (n - 1) * w, mid - env[2 * i] * mid * .9);
  ctx.fillStyle = css("--trace"); ctx.fill();
}
function drawTraces(cv, series, sr, labels) {
  const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight; cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  ctx.fillStyle = css("--bg"); ctx.fillRect(0, 0, w, h);
  const secs = series[0].length / sr, pxs = w / secs;
  for (let s = 0; s <= secs * 25; s++) { // ECG paper: 1 mm = 40 ms, bold every 200 ms
    const x = s * pxs * 0.04; ctx.strokeStyle = s % 5 ? css("--grid") : css("--grid-major");
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  const mm = pxs * 0.04;
  for (let y = 0, k = 0; y <= h; y += mm, k++) { ctx.strokeStyle = k % 5 ? css("--grid") : css("--grid-major"); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  const band = h / series.length;
  series.forEach((sig, k) => {
    const lo = d3.min(sig), hi = d3.max(sig), sc = (band * .8) / ((hi - lo) || 1), base = band * k + band * .9;
    ctx.strokeStyle = css("--fg"); ctx.lineWidth = 1.3; ctx.beginPath();
    sig.forEach((v, i) => ctx.lineTo(i / sig.length * w, base - (v - lo) * sc)); ctx.stroke();
    ctx.fillStyle = css("--muted"); ctx.font = "600 11px JetBrains Mono, monospace"; ctx.fillText(labels[k], 8, band * k + 16);
  });
}
/* ───────────────────────── results chart ───────────────────────── */
const METHODS = [
  ["SimAuthor", "--s1", true], ["PUCT score search", "--s2", true], ["Text-Opt", "--s3", true],
  ["Sampling", "--s4", true], ["− Report", "--s5", false], ["− Library", "--s6", false],
];
const R = { key: "COPD", on: Object.fromEntries(METHODS.map((m) => [m[0], m[2]])) };
const bsf = (a) => { let b = -1; return a.map((v) => (b = Math.max(b, v))); };
function renderChart() {
  const data = CURVES[R.key], el = $("#res-chart"), W = 1000, H = 400, m = { l: 52, r: 16, t: 14, b: 44 };
  const series = METHODS.filter(([n]) => data[n]).map(([n, c]) => {
    const runs = data[n].map(bsf), finals = runs.map((r) => r[r.length - 1]);
    const order = finals.map((v, i) => i).sort((a, b) => finals[a] - finals[b]);
    const med = runs[order[Math.floor(order.length / 2)]], len = d3.max(runs, (r) => r.length);
    const band = d3.range(len).map((i) => { const vs = runs.map((r) => r[Math.min(i, r.length - 1)]); return [i, d3.min(vs), d3.max(vs)]; });
    return { n, color: css(c), med, band, multi: runs.length > 1 };
  });
  const vis = series.filter((s) => R.on[s.n]);
  const xmax = d3.max(series, (s) => s.med.length - 1), ymax = Math.min(1, Math.ceil((d3.max(vis.length ? vis : series, (s) => d3.max(s.band, (b) => b[2])) + 0.05) * 10) / 10);
  const x = d3.scaleLinear([0, xmax], [m.l, W - m.r]), y = d3.scaleLinear([0, ymax], [H - m.b, m.t]);
  let h = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Best-so-far score by archived program for ${R.key}">`;
  y.ticks(5).forEach((t) => { h += `<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${m.l - 8}" y="${y(t) + 3}" text-anchor="end" style="fill:${css("--muted")};font:12px var(--f-sans)">${t.toFixed(1)}</text>`; });
  x.ticks(5).forEach((t) => { h += `<text x="${x(t)}" y="${H - m.b + 16}" text-anchor="middle" style="fill:${css("--muted")};font:12px var(--f-sans)">${t}</text>`; });
  h += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" style="fill:${css("--muted")};font:13px var(--f-sans)">authoring attempt</text>`;
  h += `<text transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle" style="fill:${css("--muted")};font:13px var(--f-sans)">best search score so far</text>`;
  vis.forEach((s) => { if (s.multi) h += `<path d="${d3.area().x((b) => x(b[0])).y0((b) => y(b[1])).y1((b) => y(b[2])).curve(d3.curveStepAfter)(s.band)}" fill="${s.color}" opacity=".13"/>`; });
  vis.forEach((s) => {
    h += `<path d="${d3.line().x((v, i) => x(i)).y((v) => y(v)).curve(d3.curveStepAfter)(s.med)}" fill="none" stroke="${s.color}" stroke-width="${s.n === "SimAuthor" ? 2.6 : 2}"/>`;
    const last = s.med.length - 1;
    h += `<circle cx="${x(last)}" cy="${y(s.med[last])}" r="4" fill="${s.color}" stroke="${css("--bg")}" stroke-width="2"/>`;
  });
  h += `<line id="xh" y1="${m.t}" y2="${H - m.b}" stroke="${css("--fg")}" stroke-dasharray="3 3" opacity="0"/><rect id="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/></svg><div class="tip" hidden style="transform:translate(10px,-50%)"></div>`;
  el.innerHTML = h;
  const svg = $("svg", el), tip = $(".tip", el), xh = $("#xh", el);
  $("#hit", el).addEventListener("mousemove", (e) => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse()), i = Math.max(0, Math.min(xmax, Math.round(x.invert(p.x))));
    xh.setAttribute("x1", x(i)); xh.setAttribute("x2", x(i)); xh.setAttribute("opacity", 1);
    tip.innerHTML = `program ${i}<br>` + vis.map((s) => `<span style="color:${s.color}">■</span> ${s.n}: ${f3(s.med[Math.min(i, s.med.length - 1)])}`).join("<br>");
    const box = el.getBoundingClientRect(); let lx = e.clientX - box.left; if (lx > box.width - 200) lx -= 220;
    tip.style.left = lx + "px"; tip.style.top = e.clientY - box.top + "px"; tip.hidden = false;
  });
  $("#hit", el).addEventListener("mouseleave", () => { tip.hidden = true; xh.setAttribute("opacity", 0); });
  $("#series-legend").innerHTML = series.map((s) => `<button aria-pressed="${R.on[s.n]}" data-s="${s.n}"><i style="background:${s.color}"></i>${s.n} <span class="num muted">${f3(s.med[s.med.length - 1])}</span></button>`).join("");
}
function initResults() {
  taskTabs($("#res-tabs"), (k) => { R.key = k; markTabs($("#res-tabs"), k); renderChart(); }, R.key);
  $("#series-legend").addEventListener("click", (e) => { const b = e.target.closest("[data-s]"); if (b) { R.on[b.dataset.s] = !R.on[b.dataset.s]; renderChart(); } });
  renderChart(); onTheme(renderChart);
}

/* ───────────────────────── misc ───────────────────────── */
function initCopy() {
  $$(".copy").forEach((b) => b.addEventListener("click", () => {
    const text = b.parentElement.innerText.replace(/^Cop(y|ied)\s*/, "");
    const done = () => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); };
    navigator.clipboard?.writeText(text).then(done, () => {
      const r = document.createRange(); r.selectNodeContents(b.parentElement); const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    });
  }));
}

function initLightbox() {
  const box = document.createElement("div"); box.className = "lightbox"; box.hidden = true;
  box.innerHTML = '<img alt=""><button class="lb-close" aria-label="Close">×</button>';
  document.body.appendChild(box);
  const close = () => { box.hidden = true; };
  box.addEventListener("click", close);
  addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  $$(".split-fig img").forEach((im) => {
    im.style.cursor = "zoom-in"; im.tabIndex = 0;
    const open = () => { $("img", box).src = im.src; $("img", box).alt = im.alt; box.hidden = false; };
    im.addEventListener("click", open);
    im.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  });
}

async function main() {
  initLightbox(); renderTables(); renderGenCharts(); onTheme(renderGenCharts); initCopy();
  try {
    [INDEX, GENS, CURVES] = await Promise.all(["data/index.json", "data/generations.json", "data/curves.json"].map(getJSON));
  } catch (e) {
    $("#insp-body").innerHTML = `<p class="loading">Could not load the run data (${esc(e.message)}). Serve the docs/ folder over HTTP, for example with <code>python -m http.server</code>.</p>`;
    return;
  }
  initHero(); initLab(); initResults();
  await pickTask(S.key);
}
main();
})();
