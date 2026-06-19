/* BetterTogether dashboard — renders window.BT_DATA (from bundle.js) into 3 tabs.
   No build step, no fetch(); ECharts + highlight.js are vendored locally. */
(function () {
  "use strict";
  const D = window.BT_DATA;
  const PROF = D.profiling;

  // ---------- tiny DOM helper ----------
  function el(tag, attrs) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      if (v == null) continue;
      if (k === "class") n.className = v;
      else if (k === "html") n.innerHTML = v;
      else if (k === "style") n.setAttribute("style", v);
      else if (k.slice(0, 2) === "on" && typeof v === "function") n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (let i = 2; i < arguments.length; i++) add(n, arguments[i]);
    return n;
  }
  function add(n, kid) {
    if (kid == null) return;
    if (Array.isArray(kid)) { kid.forEach(k => add(n, k)); return; }
    n.appendChild(typeof kid === "object" ? kid : document.createTextNode(String(kid)));
  }
  const byId = id => document.getElementById(id);
  const fmt = (v, d) => (v == null || isNaN(v)) ? "—" : Number(v).toFixed(d == null ? 3 : d);
  // compact per-view reading guide: "Shows: <story> · Read: <how>"
  const guide = (shows, read) => el("p", { class: "guide" }, el("strong", null, "Shows: "), shows, "  ·  ", el("strong", null, "Read: "), read);

  // ---------- indexes ----------
  const devById = {}; D.devices.forEach(d => devById[d.id] = d);
  const appById = {}; D.apps.forEach(a => appById[a.id] = a);
  const CELLS = new Map();
  const ckey = (dev, app, sc, st, pu) => dev + "|" + app + "|" + sc + "|" + st + "|" + pu;
  PROF.cells.forEach(c => CELLS.set(ckey(c.device, c.app, c.scenario, c.stage, c.pu), c));
  const getCell = (dev, app, sc, st, pu) => CELLS.get(ckey(dev, app, sc, st, pu));
  const puRank = pu => { const i = PROF.pus.indexOf(pu); return i < 0 ? 99 : i; };
  const interferenceDevices = D.devices
    .filter(d => d.scenarios_available.indexOf("isolated") >= 0 && d.scenarios_available.indexOf("interference") >= 0)
    .map(d => d.id);

  // schedules (Section 4) — index by device|app|backend
  const SCHED = D.schedules || { table_types: [], modes: [], cells: [], measured: { rows: [] } };
  const schedByKey = new Map();
  SCHED.cells.forEach(c => schedByKey.set(c.device + "|" + c.app + "|" + c.backend, c));
  // light-theme chart palette + shared figure styling (axis/grid greys, a
  // light->red sequential ramp, and one font family for every figure so the
  // ECharts views read like consistent paper figures).
  const CHART = {
    axis: "#5b6675", grid: "#e3e8f0", cell: "#1f2328",
    ramp: ["#e7f5e9", "#7fc98a", "#f3d250", "#f0883e", "#d6453a"],
    font: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    fs: 12,
  };
  // reusable axis/text styles so titles, ticks and names share one look
  CHART.textStyle = { color: CHART.axis, fontFamily: CHART.font, fontSize: CHART.fs };
  CHART.nameStyle = { color: CHART.axis, fontFamily: CHART.font, fontSize: CHART.fs, fontWeight: 500 };
  CHART.axisLabel = { color: CHART.axis, fontFamily: CHART.font, fontSize: CHART.fs };
  CHART.axisLine = { lineStyle: { color: "#d0d7de" } };
  CHART.split = { lineStyle: { color: CHART.grid } };
  CHART.tooltip = {
    backgroundColor: "#ffffff", borderColor: "#d0d7de", borderWidth: 1,
    textStyle: { color: CHART.cell, fontFamily: CHART.font, fontSize: CHART.fs },
    extraCssText: "box-shadow:0 1px 4px rgba(31,35,40,0.12);border-radius:6px;",
  };
  CHART.legend = { textStyle: CHART.textStyle, top: 4, type: "scroll", icon: "roundRect", itemWidth: 14, itemHeight: 8, itemGap: 16 };
  // one PU colour map shared by the Compare bars (S3) and the Schedule chunks (S4).
  // Luminance is deliberately spread (vulkan darkest → little lightest) so the
  // series stay distinguishable in greyscale / for colour-blind viewers.
  const PU_COLOR = { cuda: "#1a9e76", vulkan: "#2f6feb", big: "#e0760b", medium: "#a07bf0", little: "#f0857d" };
  // distinct marker per PU so line/scatter series don't rely on colour alone (a11y)
  const PU_SYMBOL = { vulkan: "circle", cuda: "triangle", big: "rect", medium: "diamond", little: "roundRect" };
  const puSymbol = pu => PU_SYMBOL[pu] || "emptyCircle";
  // a schedule chunk's core_type ('GPU'/'Big'/..) + hardware -> colour. OLD-format
  // GPU chunks carry no hardware, so fall back to the cell's backend (cu/vk).
  function chunkColor(core_type, hardware, cellBackend) {
    if (core_type === "GPU") {
      const hw = hardware || (cellBackend === "cu" ? "gpu_cuda" : "gpu_vulkan");
      return hw === "gpu_cuda" ? PU_COLOR.cuda : PU_COLOR.vulkan;
    }
    return PU_COLOR[core_type.toLowerCase()] || CHART.axis;
  }

  function topoPUs(dev) {
    const d = devById[dev]; if (!d) return [];
    const pus = [];
    if (d.gpu) pus.push(d.gpu.backend);
    ["big", "medium", "little"].forEach(t => { if (d.cpu_tiers[t].count > 0) pus.push(t); });
    return pus;
  }
  // PUs to show for (device, app, scenario): topology ∪ measured, ordered
  function rowPUs(dev, app, sc) {
    const set = new Set(topoPUs(dev));
    PROF.cells.forEach(c => { if (c.device === dev && c.app === app && c.scenario === sc) set.add(c.pu); });
    return [...set].sort((a, b) => puRank(a) - puRank(b));
  }
  function rawMetric(cell, metric) {
    let w = 0, s = 0; cell.raw.forEach(r => { w += r.count; s += r[metric] * r.count; });
    return w ? s / w : null;
  }
  const cellValue = (cell, metric) => !cell ? null : (cell.agg ? cell.agg[metric] : rawMetric(cell, metric));
  function cellTip(c) {
    const runs = c.raw.map(r => `  run ${r.run}: p50=${fmt(r.p50)} cv=${fmt(r.cv)} (n=${r.count})`).join("\n");
    const p = c.raw[0].provenance;
    return `${c.device} · ${c.app} · ${c.scenario} · stage ${c.stage} · ${c.pu}\n`
      + `${c.raw.length} run(s):\n${runs}\n`
      + (c.flags.high_cv ? "⚠ high CV (>0.1) — excluded by the solver's gate\n" : "")
      + (c.agg ? "" : "✗ all runs cv>0.5 — raw count-weighted value shown\n")
      + `host=${p.host} git=${p.git_sha} ts=${p.ts} gov=${p.freq_governor}`;
  }
  function coverageTip(dev, app, sc) {
    const cov = (PROF.coverage[dev] || {})[app] || {};
    const have = Object.keys(cov).filter(be => cov[be][sc]).map(be => `${be}:${cov[be][sc]} runs`);
    return have.length
      ? `collected here (${have.join(", ")}) but this PU was not measured`
      : `uncollected: no run files under data/profiling/${dev}/${app}/*/${sc}/`;
  }

  // ---------- shared controls ----------
  function segCtrl(label, opts, cur, on) {
    const seg = el("div", { class: "segctrl" });
    opts.forEach(o => seg.appendChild(el("button", { class: o[0] === cur ? "on" : "", onclick: () => on(o[0]) }, o[1])));
    return el("div", { class: "ctrl" }, el("label", null, label), seg);
  }
  function selectCtrl(label, opts, cur, on) {
    const s = el("select", { onchange: e => on(e.target.value) });
    opts.forEach(o => { const op = el("option", { value: o }, String(o)); if (String(o) === String(cur)) op.selected = true; s.appendChild(op); });
    return el("div", { class: "ctrl" }, el("label", null, label), s);
  }
  function toggCtrl(label, opts, set, on) {
    const wrap = el("div", { class: "togglist" });
    opts.forEach(o => wrap.appendChild(el("button", { class: set.has(o) ? "on" : "", onclick: () => { set.has(o) ? set.delete(o) : set.add(o); on(); } }, o)));
    return el("div", { class: "ctrl" }, el("label", null, label), wrap);
  }
  const reprof = () => renderProfiling(byId("tab-profiling"));

  // ============================================================ SECTION 1
  function renderDevices(root) {
    root.innerHTML = "";
    root.appendChild(guide(
      "the test fleet — every device the framework can target, and the PUs each one has.",
      "each card = one device: CPU tiers (per-core blocks) + GPU + supported backends. Greyed = no profiling yet. OMP runs everywhere, CUDA on NVIDIA, Vulkan on integrated GPUs."));
    const grid = el("div", { class: "cards" });
    [...D.devices].sort((a, b) => (b.has_data - a.has_data) || a.id.localeCompare(b.id)).forEach(d => grid.appendChild(deviceCard(d)));
    root.appendChild(grid);
  }
  function deviceCard(d) {
    const kv = el("div", { class: "kv" });
    const addkv = (k, v) => { kv.appendChild(el("div", { class: "k" }, k)); kv.appendChild(el("div", null, v)); };
    addkv("CPU", `${d.cores.length} cores`);
    if (d.gpu) addkv("GPU", `${d.gpu.name || d.gpu.backend} · subgroup ${d.gpu.subgroup_size}`);
    if (d.freq) addkv("clocks", d.freq.cpu);
    if (d.scenarios_available.length) addkv("profiled", d.scenarios_available.join(" + "));

    const tierWrap = el("div", null);
    ["big", "medium", "little", "super"].forEach(t => {
      const ti = d.cpu_tiers[t]; if (!ti.count) return;
      const bar = el("div", { class: "tierbar" }, el("span", { class: "tag tier-" + t }, `${t} ×${ti.count}`));
      ti.ids.forEach((id, i) => bar.appendChild(el("span", {
        class: "coreblock " + (ti.pinnable[i] ? "" : "np"),
        title: "core " + id + (ti.pinnable[i] ? " (pinnable)" : " (not pinnable)")
      }, "c" + id)));
      tierWrap.appendChild(bar);
    });

    const chips = el("div", { class: "chips" });
    d.backends_supported.forEach(b => chips.appendChild(el("span", { class: "badge " + b }, b.toUpperCase())));
    if (!d.has_data) chips.appendChild(el("span", { class: "badge nodata" }, "no profiling data"));

    return el("div", { class: "card" + (d.has_data ? "" : " dim") },
      el("h3", null, d.id),
      el("div", { class: "desc" }, d.description || ""),
      kv, tierWrap, chips);
  }

  // ============================================================ SECTION 2
  let curApp = D.apps[0].id;
  function renderApps(root) {
    root.innerHTML = "";
    const bar = el("div", { class: "appbar" });
    D.apps.forEach(a => bar.appendChild(el("button", { class: "tab" + (a.id === curApp ? " active" : ""), onclick: () => { curApp = a.id; renderApps(root); } }, a.title)));
    root.appendChild(bar);
    const a = appById[curApp];
    root.appendChild(guide(
      `${a.characteristic}`,
      `${a.input} · ${a.n_stages} stages. The matrix shows which backends implement each stage; open a stage below to see its kernel source per backend and its buffer shapes.`));
    root.appendChild(supportMatrix(a));
    a.stages.forEach(s => root.appendChild(stageCard(s)));
    root.querySelectorAll("pre code").forEach(b => { try { window.hljs.highlightElement(b); } catch (e) {} });
  }
  function supportMatrix(a) {
    const t = el("table", { class: "matrix" });
    t.appendChild(el("tr", null, el("th", { class: "lbl" }, "stage"), el("th", null, "OMP"), el("th", null, "CUDA"), el("th", null, "Vulkan")));
    a.stages.forEach(s => {
      const r = el("tr", null, el("td", { class: "lbl" }, `S${s.stage} · ${s.op}`));
      ["omp", "cuda", "vulkan"].forEach(b => r.appendChild(el("td", { class: s.support[b] ? "ok" : "empty" }, s.support[b] ? "✓" : "—")));
      t.appendChild(r);
    });
    return el("div", null, el("div", { class: "matrixhdr" }, el("strong", null, "Stage → backend support")), el("div", { class: "tablewrap" }, t));
  }
  function stageCard(s) {
    const tabs = el("div", { class: "btabs" });
    const panes = el("div", null);
    const paneByB = {};
    const order = ["omp", "cuda", "vulkan"], labels = { omp: "OMP", cuda: "CUDA", vulkan: "Vulkan" };
    let first = null;
    order.forEach(b => {
      const has = s.support[b];
      const pane = el("div", { style: "display:none" }); renderKernel(pane, s.kernels[b]); panes.appendChild(pane); paneByB[b] = pane;
      const btn = el("button", { class: has ? "off" : "off", onclick: has ? () => show(b) : null }, labels[b]);
      if (!has) btn.disabled = true;
      tabs.appendChild(btn);
      if (has && !first) first = b;
    });
    function show(b) {
      order.forEach((k, i) => { paneByB[k].style.display = k === b ? "block" : "none"; tabs.children[i].className = (k === b && s.support[k]) ? "on" : (s.support[k] ? "off" : "off"); });
    }
    if (first) show(first);
    return el("div", { class: "stage" },
      el("div", { class: "head" }, el("div", null, el("span", { class: "num" }, "S" + s.stage), el("span", { class: "op" }, s.op)), el("div", { class: "desc" }, s.desc)),
      el("div", { class: "body" }, tabs, panes, appdataTable(s.appdata)));
  }
  function renderKernel(root, list) {
    if (!list || !list.length) { root.appendChild(el("div", { class: "muted small" }, "not implemented for this backend")); return; }
    list.forEach(k => {
      const label = el("div", { class: "srcfile" });
      if (k.symbol) label.appendChild(el("code", null, k.symbol + "()  "));
      label.appendChild(el("code", null, k.path));
      const code = el("code", { class: "language-" + k.lang }); code.textContent = k.code;
      root.appendChild(el("div", null, label, el("pre", null, code)));
    });
  }
  function appdataTable(rows) {
    if (!rows || !rows.length) return el("div", null);
    const keys = [], order = ["buffer", "role", "type", "shape", "alloc", "used"];
    rows.forEach(r => Object.keys(r).forEach(k => { if (keys.indexOf(k) < 0) keys.push(k); }));
    keys.sort((a, b) => { const ia = order.indexOf(a), ib = order.indexOf(b); return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib); });
    const t = el("table", { class: "appdata" });
    t.appendChild(el("tr", null, keys.map(k => el("th", null, k))));
    rows.forEach(r => t.appendChild(el("tr", null, keys.map(k => el("td", null, r[k] != null ? String(r[k]) : "")))));
    return el("div", { class: "appdata" }, el("div", { class: "srcfile" }, "AppData — buffers / shapes for this stage"), el("div", { class: "tablewrap" }, t));
  }

  // ============================================================ SECTION 3
  let curSub = "profile";
  // profiling sub-tabs hidden from the bar (views kept intact — clear this set to re-enable)
  const PROF_HIDDEN = new Set(["shift", "compare"]);
  let charts = [];
  const disposeCharts = () => { charts.forEach(c => { try { c.dispose(); } catch (e) {} }); charts = []; };
  const resizeCharts = () => charts.forEach(c => { try { c.resize(); } catch (e) {} });
  window.addEventListener("resize", resizeCharts);

  const sel = {
    profile: { device: null, app: "tree", scenario: "isolated", metric: "p50", rel: false },
    shift: { device: null, app: "tree", metric: "p50" },
    table: { app: "tree", scenario: "isolated", metric: "p50", devices: null, pus: null, sortStage: 0 },
    heat: { device: null, app: "tree", scenario: "isolated", metric: "p50" },
    cmp: { app: "tree", stage: 1, scenario: "isolated", metric: "p50", rel: false },
    intf: { device: null, app: "tree", metric: "p50" }
  };
  // shared axes: changing device/app/scenario/metric in any profiling sub-tab
  // propagates to the others so you don't re-pick them every time (each view
  // still clamps to its own valid set). View-specific keys (log, stage, sets) stay local.
  function syncAxis(k, v) { Object.keys(sel).forEach(s => { if (k in sel[s]) sel[s][k] = v; }); }
  const scenForDevice = dev => { const d = devById[dev]; return (d && d.scenarios_available.length) ? d.scenarios_available : ["isolated"]; };
  const dataDevices = app => D.devices.filter(d => (PROF.coverage[d.id] || {})[app]).map(d => d.id);

  function renderProfiling(root) {
    root.innerHTML = "";
    if (!PROF.cells.length) { root.appendChild(el("p", { class: "note" }, "No profiling data found under data/profiling/. Run the bm-prof-* profilers, then regenerate.")); return; }
    const sub = el("div", { class: "subtabs" });
    const subs = [["profile", "Stage profile"], ["shift", "Interference shift"], ["heatmap", "Heatmap"], ["compare", "Compare"], ["table", "Detail table"], ["interference", "Interference ratio"]]
      .filter(o => !PROF_HIDDEN.has(o[0]));
    if (!subs.some(o => o[0] === curSub)) curSub = subs[0][0];   // fall back if the active tab was hidden
    subs.forEach(o => sub.appendChild(el("button", { class: o[0] === curSub ? "on" : "", onclick: () => { curSub = o[0]; renderProfiling(root); } }, o[1])));
    root.appendChild(sub);
    const view = el("div", null); root.appendChild(view);
    disposeCharts();
    ({ profile: viewProfile, shift: viewShift, table: viewTable, heatmap: viewHeatmap, compare: viewCompare, interference: viewInterference }[curSub])(view);
  }

  function legend() {
    return el("div", { class: "legend" },
      el("span", null, el("span", { class: "sw", style: "background:var(--warn)" }), "high CV (>0.1) — excluded by solver gate"),
      el("span", null, el("span", { class: "sw", style: "background:var(--drop)" }), "all runs cv>0.5 (raw value shown)"),
      el("span", null, el("span", { class: "sw", style: "background:var(--empty)" }), "— uncollected (hover for path)"));
  }

  // The "human" view: the heterogeneity story. One line per PU across stages
  // (where lines cross, the best PU changes) + a best-PU-per-stage ribbon +
  // a single-PU-vs-ideal cost bar (the headroom the scheduler chases).
  function viewProfile(view) {
    const st = sel.profile, app = appById[st.app];
    const devs = dataDevices(st.app);
    if (!devs.length) { view.appendChild(el("p", { class: "note" }, "No profiled device for this app.")); return; }
    if (!st.device || devs.indexOf(st.device) < 0) st.device = devs[0];
    const scens = scenForDevice(st.device); if (scens.indexOf(st.scenario) < 0) st.scenario = scens[0];
    const isTime = st.metric !== "cv";

    const ctr = el("div", { class: "controls" });
    ctr.appendChild(selectCtrl("Device", devs, st.device, v => { syncAxis("device", v); reprof(); }));
    ctr.appendChild(segCtrl("App", D.apps.map(a => [a.id, a.title]), st.app, v => { syncAxis("app", v); syncAxis("device", null); reprof(); }));
    ctr.appendChild(segCtrl("Scenario", scens.map(s => [s, s]), st.scenario, v => { syncAxis("scenario", v); reprof(); }));
    ctr.appendChild(selectCtrl("Metric", PROF.metrics, st.metric, v => { syncAxis("metric", v); reprof(); }));
    ctr.appendChild(segCtrl("Bars", [["abs", "absolute (ms)"], ["rel", "× vs fastest"]], st.rel ? "rel" : "abs", v => { st.rel = (v === "rel"); reprof(); }));
    view.appendChild(ctr);

    const pus = rowPUs(st.device, st.app, st.scenario);
    const stages = []; for (let i = 1; i <= app.n_stages; i++) stages.push(i);
    const val = (s, pu) => cellValue(getCell(st.device, st.app, st.scenario, s, pu), st.metric);
    // is the (stage,pu) cell solver-gated? "" ok, "warn" high-CV, "drop" all runs cv>0.5
    const flagOf = (s, pu) => { const c = getCell(st.device, st.app, st.scenario, s, pu); return c ? (!c.agg ? "drop" : (c.flags.high_cv ? "warn" : "")) : ""; };
    const best = stages.map(s => {
      let m = Infinity, who = null;
      pus.forEach(pu => { const v = val(s, pu); if (v != null && v < m) { m = v; who = pu; } });
      return who ? { v: m, pu: who } : null;
    });

    if (!pus.length) { view.appendChild(el("p", { class: "note" }, "No PUs measured here.")); return; }

    // --- Chart A: per-stage cost, grouped bars (one bar per PU within each stage).
    // Stages are discrete, so bars (not a line) — the read is "which PU wins this
    // stage" (shortest bar in the group); a shared y keeps stages comparable too. ---
    const rel = st.rel;   // bars: relative (× vs the stage's fastest PU) vs absolute ms
    const bestV = stages.map((s, i) => best[i] ? best[i].v : null);
    view.appendChild(el("div", { class: "figtitle" }, `Per-stage cost by processing unit — ${app.title} on ${st.device}`));
    view.appendChild(guide(
      "no single PU is fastest on every stage — the winner shifts stage to stage (the heterogeneity the scheduler exploits).",
      rel
        ? `${st.scenario} · ${st.metric}, each stage normalised to its own fastest PU. Shortest bar = fastest (1×); a bar at 4× is 4× slower than the best on that stage. Hover for absolute ms.`
        : `${st.scenario} · ${st.metric}${isTime ? " (ms)" : ""}, bars from 0 (linear). Shortest bar in a stage = fastest PU; bar height ∝ time. Cheap stages look tiny next to expensive ones — switch to "× vs fastest" to compare PUs regardless of magnitude.`));
    const divA = el("div", { class: "chart" }); view.appendChild(divA);
    const series = pus.map(pu => ({
      name: pu, type: "bar", barMaxWidth: 22,
      itemStyle: { color: PU_COLOR[pu] || CHART.axis, borderRadius: [2, 2, 0, 0] },
      emphasis: { focus: "series" },
      data: stages.map((s, i) => {
        const v = val(s, pu); if (v == null) return null;
        return rel ? (bestV[i] ? +(v / bestV[i]).toFixed(3) : null) : +v.toFixed(4);
      }),
    }));
    const chartA = echarts.init(divA);
    chartA.setOption({
      backgroundColor: "transparent", textStyle: CHART.textStyle,
      tooltip: Object.assign({
        trigger: "axis", axisPointer: { type: "shadow" },
        formatter: params => {
          const i = params[0].dataIndex, s = stages[i];
          const rows = params.filter(p => p.value != null).map(p => {
            const ms = val(s, p.seriesName);
            return { m: p.marker, pu: p.seriesName, ms, rat: bestV[i] ? ms / bestV[i] : null };
          }).sort((a, b) => a.ms - b.ms);
          return `<b>stage ${s}</b>` + rows.map(r => `<br/>${r.m}${r.pu}: ${fmt(r.ms, 3)} ms${r.rat != null ? ` · ${fmt(r.rat, 2)}×` : ""}`).join("");
        },
      }, CHART.tooltip),
      legend: CHART.legend,
      grid: { left: 62, right: 24, top: 44, bottom: 44, containLabel: false },
      xAxis: { type: "category", name: "stage", nameLocation: "middle", nameGap: 26, nameTextStyle: CHART.nameStyle, data: stages.map(s => "S" + s), axisLabel: CHART.axisLabel, axisLine: CHART.axisLine, axisTick: { show: false } },
      yAxis: {
        type: "value", name: rel ? "× vs fastest PU" : st.metric + (isTime ? " (ms)" : ""),
        nameTextStyle: CHART.nameStyle, axisLabel: CHART.axisLabel, splitLine: CHART.split, axisLine: { show: false }, axisTick: { show: false },
      },
      series,
    });
    charts.push(chartA); chartA.resize();

    // --- best-PU-per-stage ribbon (a visual preview of what a schedule wants) ---
    const ribbon = el("div", { class: "puribbon" });
    stages.forEach((s, i) => {
      const b = best[i], fl = b ? flagOf(s, b.pu) : "";
      ribbon.appendChild(el("div", {
        class: "pucell" + (fl ? " flagged" : ""), style: b ? `background:${PU_COLOR[b.pu] || CHART.axis}` : "background:var(--empty)",
        title: b ? `stage ${s}: fastest on ${b.pu} (${fmt(b.v, 3)}${isTime ? " ms" : ""})${fl ? " — ⚠ high-CV / solver-gated cell (see Detail table)" : ""}` : `stage ${s}: no data`,
      }, el("span", { class: "s" }, "S" + s), el("span", { class: "p" }, b ? b.pu : "—")));
    });
    view.appendChild(el("div", { class: "ribbonwrap" },
      el("div", { class: "note" }, "Best PU per stage — a preview of the heterogeneous assignment a schedule wants (⚠ = rests on a high-CV cell):"), ribbon));

    // --- Chart B: single-PU whole-app cost vs the ideal (only meaningful for times) ---
    if (!isTime) return;
    const rows = [];
    pus.filter(pu => stages.every(s => val(s, pu) != null))
      .map(pu => ({ pu, total: stages.reduce((a, s) => a + val(s, pu), 0) }))
      .sort((a, b) => b.total - a.total)
      .forEach(t => rows.push({ cat: "all " + t.pu, val: +t.total.toFixed(3), color: PU_COLOR[t.pu] || CHART.axis }));
    const bestSingle = rows.length ? Math.min(...rows.map(r => r.val)) : null;
    const idealTotal = best.every(b => b) ? best.reduce((a, b) => a + b.v, 0) : null;
    if (idealTotal != null) rows.push({ cat: "ideal (best per stage, serial — not achievable)", val: +idealTotal.toFixed(3), color: CHART.cell });
    if (!rows.length) return;
    const nFlag = best.filter((b, i) => b && flagOf(stages[i], b.pu)).length;
    const headroom = (bestSingle != null && idealTotal) ? bestSingle / idealTotal : null;
    if (headroom != null)
      view.appendChild(el("p", { class: "hero" },
        `${fmt(headroom, 2)}× headroom`, el("span", { class: "sub" },
          ` (static placement, pre-pipeline) — best single PU ${fmt(bestSingle, 1)} ms vs ideal per-stage ${fmt(idealTotal, 1)} ms`
          + (nFlag ? ` · ⚠ ${nFlag} stage${nFlag > 1 ? "s" : ""} rest on high-CV cells` : ""))));
    view.appendChild(el("div", { class: "figtitle" }, "Whole-app cost: single-PU vs. ideal per-stage placement"));
    view.appendChild(guide(
      "the heterogeneity headroom — how much per-stage placement could save vs running everything on one PU.",
      "bars = total ms on one PU; 'ideal' = each stage on its fastest PU (serial, not achievable). Headroom is a static reference, not the measured speedup — see the Schedule tab for that."));
    const disp = rows.slice().reverse();   // ECharts category axis renders bottom-up
    const divB = el("div", { class: "chart", style: "height:" + (88 + rows.length * 42) + "px" }); view.appendChild(divB);
    const chartB = echarts.init(divB);
    chartB.setOption({
      backgroundColor: "transparent", textStyle: CHART.textStyle,
      tooltip: Object.assign({ trigger: "axis", axisPointer: { type: "shadow" }, formatter: p => `${p[0].name}<br/><b>${fmt(p[0].value, 2)} ms</b>` }, CHART.tooltip),
      grid: { left: 170, right: 72, top: 12, bottom: 36 },
      xAxis: { type: "value", name: "whole-app cost (ms)", nameLocation: "middle", nameGap: 24, axisLabel: CHART.axisLabel, nameTextStyle: CHART.nameStyle, splitLine: CHART.split, axisLine: { show: false }, axisTick: { show: false } },
      yAxis: { type: "category", data: disp.map(r => r.cat), axisLabel: CHART.axisLabel, axisLine: CHART.axisLine, axisTick: { show: false } },
      series: [{
        type: "bar", barWidth: "52%",
        data: disp.map(r => ({ value: r.val, itemStyle: { color: r.color, borderRadius: [0, 3, 3, 0] } })),
        label: { show: true, position: "right", color: CHART.axis, fontFamily: CHART.font, formatter: p => fmt(p.value, 1) },
      }],
    });
    charts.push(chartB); chartB.resize();
  }

  // The interference story: contention not only slows stages, it changes WHICH
  // PU is fastest. Two best-PU ribbons (isolated vs interference, flips outlined)
  // + a slope chart of each stage's best time isolated->interference.
  function viewShift(view) {
    const st = sel.shift, app = appById[st.app];
    const devs = dataDevices(st.app).filter(d => interferenceDevices.indexOf(d) >= 0);
    if (!devs.length) {
      view.appendChild(el("p", { class: "note" },
        `No device has both isolated and interference data for ${app.title}. Interference collected on: ${interferenceDevices.join(", ") || "—"}.`));
      return;
    }
    if (!st.device || devs.indexOf(st.device) < 0) st.device = devs[0];
    const isTime = st.metric !== "cv";

    const ctr = el("div", { class: "controls" });
    ctr.appendChild(selectCtrl("Device", devs, st.device, v => { syncAxis("device", v); reprof(); }));
    ctr.appendChild(segCtrl("App", D.apps.map(a => [a.id, a.title]), st.app, v => { syncAxis("app", v); syncAxis("device", null); reprof(); }));
    ctr.appendChild(selectCtrl("Metric", PROF.metrics, st.metric, v => { syncAxis("metric", v); reprof(); }));
    view.appendChild(ctr);

    const stages = []; for (let i = 1; i <= app.n_stages; i++) stages.push(i);
    const pusIso = rowPUs(st.device, st.app, "isolated");
    const pusInt = rowPUs(st.device, st.app, "interference");
    const bestIn = (sc, pus, s) => {
      let m = Infinity, who = null;
      pus.forEach(pu => { const v = cellValue(getCell(st.device, st.app, sc, s, pu), st.metric); if (v != null && v < m) { m = v; who = pu; } });
      return who ? { v: m, pu: who } : null;
    };
    const iso = stages.map(s => bestIn("isolated", pusIso, s));
    const intf = stages.map(s => bestIn("interference", pusInt, s));
    const isFlip = i => iso[i] && intf[i] && iso[i].pu !== intf[i].pu;
    const flips = stages.filter((s, i) => isFlip(i)).length;
    const ratios = stages.map((s, i) => (iso[i] && intf[i] && iso[i].v > 0) ? intf[i].v / iso[i].v : null)
      .filter(x => x != null).sort((a, b) => a - b);
    const medR = ratios.length ? ratios[Math.floor(ratios.length / 2)] : null;

    view.appendChild(el("p", { class: "hero" },
      `${flips}/${stages.length} stages change their fastest PU`,
      el("span", { class: "sub" }, " under contention")));
    view.appendChild(el("p", { class: "note" },
      `${st.device} · ${app.title} · ${st.metric}${isTime ? " (ms)" : ""} — contention doesn't just change timings, it can move the sweet spot to a different PU. `
      + "That's why the solver profiles under interference, not in isolation."
      + (medR != null ? ` Median per-stage best-time ratio interference÷isolated: ${fmt(medR, 2)}×.` : "")
      + (medR != null && medR < 1 ? " (A median <1 here is small-sample / high-CV noise — see the Detail-table CV flags — not a real contention speedup.)" : "")));

    function ribbon(label, arr, markFlip) {
      const r = el("div", { class: "puribbon" });
      stages.forEach((s, i) => {
        const b = arr[i], flip = markFlip && isFlip(i);
        r.appendChild(el("div", {
          class: "pucell" + (flip ? " flip" : ""),
          style: b ? `background:${PU_COLOR[b.pu] || CHART.axis}` : "background:var(--empty)",
          title: b ? `stage ${s}: fastest on ${b.pu} (${fmt(b.v, 3)}${isTime ? " ms" : ""})${flip ? ` — was ${iso[i].pu} when isolated` : ""}` : `stage ${s}: no data`,
        }, el("span", { class: "s" }, "S" + s), el("span", { class: "p" }, b ? b.pu : "—")));
      });
      return el("div", { class: "ribbonwrap" }, el("div", { class: "note" }, label), r);
    }
    view.appendChild(ribbon("Fastest PU per stage — isolated:", iso, false));
    view.appendChild(ribbon("Fastest PU per stage — under interference (outlined = changed):", intf, true));

    // slope chart: each stage's best time, isolated -> interference (orange line = PU flipped)
    const series = stages.map((s, i) => {
      if (!iso[i] || !intf[i]) return null;
      const flip = isFlip(i);
      return {
        name: "S" + s, type: "line", symbolSize: 10, z: flip ? 5 : 2,
        lineStyle: { width: flip ? 3 : 1.5, color: flip ? "#e8833a" : "#c2c9d4" },
        data: [
          { value: +iso[i].v.toFixed(4), symbol: puSymbol(iso[i].pu), itemStyle: { color: PU_COLOR[iso[i].pu] || CHART.axis } },
          { value: +intf[i].v.toFixed(4), symbol: puSymbol(intf[i].pu), itemStyle: { color: PU_COLOR[intf[i].pu] || CHART.axis }, label: { show: flip, position: "right", color: "#e8833a", fontWeight: 700, fontSize: 11, formatter: "S" + s + " ⇄" } },
        ],
      };
    }).filter(Boolean);
    if (!series.length) { view.appendChild(el("p", { class: "note" }, "No stage has both isolated and interference data to compare.")); return; }
    view.appendChild(el("div", { class: "figtitle" }, "Per-stage best time: isolated vs. under contention"));
    const div = el("div", { class: "chart" }); view.appendChild(div);
    const chart = echarts.init(div);
    chart.setOption({
      backgroundColor: "transparent", textStyle: CHART.textStyle,
      tooltip: Object.assign({
        trigger: "item",
        formatter: p => {
          const i = (+p.seriesName.slice(1)) - 1;
          const r = (iso[i] && intf[i] && iso[i].v > 0) ? fmt(intf[i].v / iso[i].v, 2) + "×" : "—";
          return `stage ${i + 1}<br/>isolated: ${iso[i].pu} ${fmt(iso[i].v, 3)}<br/>interference: ${intf[i].pu} ${fmt(intf[i].v, 3)}`
            + `<br/>slowdown ${r}${isFlip(i) ? " · <b style='color:#e8833a'>fastest PU changed</b>" : ""}`;
        },
      }, CHART.tooltip),
      grid: { left: 64, right: 64, top: 20, bottom: 40 },
      xAxis: { type: "category", data: ["isolated", "interference (BTPM)"], boundaryGap: true, axisLabel: CHART.axisLabel, axisLine: CHART.axisLine, axisTick: { show: false } },
      yAxis: { type: "log", name: st.metric + (isTime ? " (ms)" : ""), nameTextStyle: CHART.nameStyle, axisLabel: CHART.axisLabel, splitLine: CHART.split, axisLine: { show: false }, axisTick: { show: false } },
      series,
    });
    charts.push(chart); chart.resize();
    view.appendChild(guide(
      "contention can move the sweet spot — the fastest PU for a stage may change under load.",
      "two ribbons = fastest PU isolated vs under load (outlined = it changed). Each slope line is one stage: up = slower under load, orange = the winner flipped."));
  }

  function viewTable(view) {
    const st = sel.table, app = appById[st.app];
    const devs = dataDevices(st.app);
    if (!st.devices) st.devices = new Set(D.devices.filter(d => d.has_data).map(d => d.id));
    if (!st.pus) st.pus = new Set(PROF.pus);
    const ctr = el("div", { class: "controls" });
    ctr.appendChild(segCtrl("App", D.apps.map(a => [a.id, a.title]), st.app, v => { syncAxis("app", v); reprof(); }));
    ctr.appendChild(segCtrl("Scenario", [["isolated", "isolated"], ["interference", "interference"]], st.scenario, v => { syncAxis("scenario", v); reprof(); }));
    ctr.appendChild(selectCtrl("Metric", PROF.metrics, st.metric, v => { syncAxis("metric", v); reprof(); }));
    ctr.appendChild(toggCtrl("Devices", D.devices.filter(d => d.has_data).map(d => d.id), st.devices, reprof));
    ctr.appendChild(toggCtrl("PUs", PROF.pus, st.pus, reprof));
    view.appendChild(ctr);
    view.appendChild(legend());
    view.appendChild(el("p", { class: "note" }, `${app.title} · ${st.scenario} · ${st.metric}${st.metric === "cv" ? "" : " (ms)"} · click a stage header to sort rows by it`));

    const n = app.n_stages;
    const t = el("table", { class: "matrix" });
    const head = el("tr", null, el("th", { class: "lbl" }, "device / PU"));
    for (let s = 1; s <= n; s++) { const sortS = () => { st.sortStage = st.sortStage === s ? 0 : s; reprof(); }; head.appendChild(el("th", { class: "stagehdr" + (st.sortStage === s ? " on" : ""), role: "button", tabindex: "0", "aria-sort": st.sortStage === s ? "ascending" : "none", title: "sort rows by stage " + s, onclick: sortS, onkeydown: e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sortS(); } } }, "S" + s)); }
    t.appendChild(head);

    let rows = [];
    [...st.devices].filter(d => devs.indexOf(d) >= 0).sort().forEach(dev =>
      rowPUs(dev, st.app, st.scenario).filter(pu => st.pus.has(pu)).forEach(pu => rows.push({ dev, pu })));
    if (st.sortStage) rows.sort((a, b) => {
      const va = cellValue(getCell(a.dev, st.app, st.scenario, st.sortStage, a.pu), st.metric);
      const vb = cellValue(getCell(b.dev, st.app, st.scenario, st.sortStage, b.pu), st.metric);
      return (va == null ? Infinity : va) - (vb == null ? Infinity : vb);
    });
    if (!rows.length) { view.appendChild(el("p", { class: "note" }, "Nothing selected.")); return; }
    rows.forEach(({ dev, pu }) => {
      const tr = el("tr", null, el("td", { class: "lbl" }, dev, " / ", el("span", { class: "tier-" + pu }, pu)));
      for (let s = 1; s <= n; s++) {
        const c = getCell(dev, st.app, st.scenario, s, pu);
        if (!c) { tr.appendChild(el("td", { class: "empty", title: coverageTip(dev, st.app, st.scenario) }, "—")); continue; }
        const cls = !c.agg ? "drop" : (c.flags.high_cv ? "warn" : "");
        tr.appendChild(el("td", { class: cls, title: cellTip(c) }, fmt(cellValue(c, st.metric))));
      }
      t.appendChild(tr);
    });
    view.appendChild(el("div", { class: "tablewrap" }, t));
  }

  function viewHeatmap(view) {
    const st = sel.heat, devs = D.devices.filter(d => d.has_data).map(d => d.id);
    if (!st.device || devs.indexOf(st.device) < 0) st.device = devs[0];
    const scens = scenForDevice(st.device); if (scens.indexOf(st.scenario) < 0) st.scenario = scens[0];
    const ctr = el("div", { class: "controls" });
    ctr.appendChild(selectCtrl("Device", devs, st.device, v => { syncAxis("device", v); reprof(); }));
    ctr.appendChild(segCtrl("App", D.apps.map(a => [a.id, a.title]), st.app, v => { syncAxis("app", v); syncAxis("device", null); reprof(); }));
    ctr.appendChild(segCtrl("Scenario", scens.map(s => [s, s]), st.scenario, v => { syncAxis("scenario", v); reprof(); }));
    ctr.appendChild(selectCtrl("Metric", PROF.metrics, st.metric, v => { syncAxis("metric", v); reprof(); }));
    view.appendChild(ctr);

    const app = appById[st.app], pus = rowPUs(st.device, st.app, st.scenario);
    const stages = []; for (let i = 1; i <= app.n_stages; i++) stages.push(i);
    const data = []; let vmax = 0;
    pus.forEach((pu, yi) => stages.forEach((s, xi) => {
      const v = cellValue(getCell(st.device, st.app, st.scenario, s, pu), st.metric);
      if (v != null) { data.push([xi, yi, +v.toFixed(4)]); vmax = Math.max(vmax, v); }
    }));
    view.appendChild(el("div", { class: "figtitle" }, `Stage × PU cost heatmap — ${app.title} on ${st.device}`));
    view.appendChild(guide(
      "the full per-stage × per-PU cost for one config, at a glance.",
      `${st.scenario} · ${st.metric}. Darker / red = slower. Read a row for one PU across stages, a column for one stage's options.`));
    if (!data.length) { view.appendChild(el("p", { class: "note" }, "No data for this selection.")); return; }
    const div = el("div", { class: "chart" }); view.appendChild(div);
    const chart = echarts.init(div);
    chart.setOption({
      backgroundColor: "transparent", textStyle: CHART.textStyle,
      tooltip: Object.assign({ position: "top", formatter: p => `stage ${stages[p.value[0]]} · ${pus[p.value[1]]}<br/>${st.metric} = <b>${p.value[2]}</b>` }, CHART.tooltip),
      grid: { left: 80, right: 30, top: 16, bottom: 60 },
      xAxis: { type: "category", name: "stage", nameLocation: "middle", nameGap: 26, nameTextStyle: CHART.nameStyle, data: stages.map(s => "S" + s), axisLabel: CHART.axisLabel, axisLine: CHART.axisLine, axisTick: { show: false }, splitArea: { show: false } },
      yAxis: { type: "category", data: pus, axisLabel: CHART.axisLabel, axisLine: CHART.axisLine, axisTick: { show: false }, splitArea: { show: false } },
      visualMap: { min: 0, max: +vmax.toFixed(3), calculable: true, orient: "horizontal", left: "center", bottom: 6, itemHeight: 110, textStyle: CHART.axisLabel, inRange: { color: CHART.ramp } },
      series: [{ type: "heatmap", data, itemStyle: { borderColor: "#fff", borderWidth: 1 }, label: { show: true, fontSize: 10, fontFamily: CHART.font, color: "#0b1220", formatter: p => p.value[2] } }]
    });
    charts.push(chart); chart.resize();
  }

  function viewCompare(view) {
    const st = sel.cmp, app = appById[st.app];
    if (st.stage > app.n_stages) st.stage = 1;
    const stageOpts = []; for (let i = 1; i <= app.n_stages; i++) stageOpts.push(i);
    const ctr = el("div", { class: "controls" });
    ctr.appendChild(segCtrl("App", D.apps.map(a => [a.id, a.title]), st.app, v => { syncAxis("app", v); reprof(); }));
    ctr.appendChild(selectCtrl("Stage", stageOpts, st.stage, v => { st.stage = +v; reprof(); }));
    ctr.appendChild(segCtrl("Scenario", [["isolated", "isolated"], ["interference", "interference"]], st.scenario, v => { syncAxis("scenario", v); reprof(); }));
    ctr.appendChild(selectCtrl("Metric", PROF.metrics, st.metric, v => { syncAxis("metric", v); reprof(); }));
    ctr.appendChild(segCtrl("Bars", [["abs", "absolute (ms)"], ["rel", "× vs fastest"]], st.rel ? "rel" : "abs", v => { st.rel = (v === "rel"); reprof(); }));
    view.appendChild(ctr);

    const rel = st.rel;
    const devs = dataDevices(st.app);
    const puSet = new Set(); devs.forEach(dev => rowPUs(dev, st.app, st.scenario).forEach(pu => puSet.add(pu)));
    const pus = [...puSet].sort((a, b) => puRank(a) - puRank(b));
    const cmpVal = (dev, pu) => cellValue(getCell(dev, st.app, st.scenario, st.stage, pu), st.metric);
    const bestPerDev = devs.map(dev => { let m = Infinity; pus.forEach(pu => { const v = cmpVal(dev, pu); if (v != null && v < m) m = v; }); return m === Infinity ? null : m; });
    const series = pus.map(pu => ({
      name: pu, type: "bar", itemStyle: { color: PU_COLOR[pu] },
      data: devs.map((dev, di) => { const v = cmpVal(dev, pu); if (v == null) return null; return rel ? (bestPerDev[di] ? +(v / bestPerDev[di]).toFixed(3) : null) : +v.toFixed(4); })
    }));
    view.appendChild(el("div", { class: "figtitle" }, `Stage ${st.stage} (${app.stages[st.stage - 1].op}) across devices & PUs — ${app.title}`));
    view.appendChild(guide(
      "how one stage's cost compares across the whole fleet.",
      `stage ${st.stage} · ${st.scenario} · ${st.metric}. One bar group per device; shortest bar = fastest PU there. ${rel ? "Each device normalised to its own fastest PU (1×) — compares PU choice across fast and slow devices." : "Bars from 0; switch to × vs fastest to compare across very different devices."}`));
    if (!devs.length) { view.appendChild(el("p", { class: "note" }, "No data.")); return; }
    const div = el("div", { class: "chart" }); view.appendChild(div);
    const chart = echarts.init(div);
    chart.setOption({
      backgroundColor: "transparent", textStyle: CHART.textStyle,
      tooltip: Object.assign({ trigger: "axis", axisPointer: { type: "shadow" } }, CHART.tooltip),
      legend: CHART.legend,
      grid: { left: 62, right: 20, top: 44, bottom: 72 },
      xAxis: { type: "category", data: devs, axisLabel: Object.assign({ rotate: 18 }, CHART.axisLabel), axisLine: CHART.axisLine, axisTick: { show: false } },
      yAxis: { type: "value", name: rel ? "× vs fastest PU" : st.metric + (st.metric === "cv" ? "" : " (ms)"), axisLabel: CHART.axisLabel, nameTextStyle: CHART.nameStyle, splitLine: CHART.split, axisLine: { show: false }, axisTick: { show: false } },
      series: series.map(s => Object.assign({}, s, { barMaxWidth: 26, itemStyle: Object.assign({ borderRadius: [2, 2, 0, 0] }, s.itemStyle) }))
    });
    charts.push(chart); chart.resize();
  }

  function viewInterference(view) {
    const st = sel.intf;
    if (!interferenceDevices.length) { view.appendChild(el("p", { class: "note" }, "No device has interference data collected.")); return; }
    if (!st.device || interferenceDevices.indexOf(st.device) < 0) st.device = interferenceDevices[0];
    if (["p50", "p95", "p99", "mean"].indexOf(st.metric) < 0) st.metric = "p50";  // ratio needs a time metric
    const ctr = el("div", { class: "controls" });
    ctr.appendChild(selectCtrl("Device", interferenceDevices, st.device, v => { syncAxis("device", v); reprof(); }));
    ctr.appendChild(segCtrl("App", D.apps.map(a => [a.id, a.title]), st.app, v => { syncAxis("app", v); syncAxis("device", null); reprof(); }));
    ctr.appendChild(selectCtrl("Metric", ["p50", "p95", "p99", "mean"], st.metric, v => { syncAxis("metric", v); reprof(); }));
    view.appendChild(ctr);
    const missing = D.devices.filter(d => d.has_data && interferenceDevices.indexOf(d.id) < 0).map(d => d.id);
    view.appendChild(el("div", { class: "figtitle" }, `Interference / isolated ratio — ${appById[st.app].title} on ${st.device}`));
    view.appendChild(guide(
      "how much each stage·PU slows down (or speeds up) under contention.",
      `${st.metric} ratio = contention ÷ isolated. >1 = slower (red), <1 = faster (green); the number is the ×. No interference on: ${missing.join(", ") || "—"}.`));

    const app = appById[st.app], pus = rowPUs(st.device, st.app, "interference");
    const stages = []; for (let i = 1; i <= app.n_stages; i++) stages.push(i);
    // colour by log2(ratio) so the scale is balanced around 1 (a 4× slowdown and a
    // 4× speedup sit equidistant from neutral); label/tooltip show the raw ratio.
    const data = []; let k = 0.2;
    pus.forEach((pu, yi) => stages.forEach((s, xi) => {
      const vi = cellValue(getCell(st.device, st.app, "interference", s, pu), st.metric);
      const vz = cellValue(getCell(st.device, st.app, "isolated", s, pu), st.metric);
      if (vi != null && vi > 0 && vz != null && vz > 0) {
        const r = vi / vz, lg = Math.log2(r);
        data.push([xi, yi, +lg.toFixed(4), +r.toFixed(3)]);
        k = Math.max(k, Math.abs(lg));
      }
    }));
    if (!data.length) { view.appendChild(el("p", { class: "note" }, "No paired isolated+interference data for this selection.")); return; }
    const div = el("div", { class: "chart" }); view.appendChild(div);
    const chart = echarts.init(div);
    chart.setOption({
      backgroundColor: "transparent", textStyle: CHART.textStyle,
      tooltip: Object.assign({ position: "top", formatter: p => `stage ${stages[p.value[0]]} · ${pus[p.value[1]]}<br/>ratio = <b>${p.value[3]}×</b>` }, CHART.tooltip),
      grid: { left: 80, right: 30, top: 16, bottom: 60 },
      xAxis: { type: "category", name: "stage", nameLocation: "middle", nameGap: 26, nameTextStyle: CHART.nameStyle, data: stages.map(s => "S" + s), axisLabel: CHART.axisLabel, axisLine: CHART.axisLine, axisTick: { show: false }, splitArea: { show: false } },
      yAxis: { type: "category", data: pus, axisLabel: CHART.axisLabel, axisLine: CHART.axisLine, axisTick: { show: false }, splitArea: { show: false } },
      visualMap: { min: +(-k).toFixed(3), max: +k.toFixed(3), dimension: 2, calculable: true, orient: "horizontal", left: "center", bottom: 6, itemHeight: 110, textStyle: CHART.axisLabel, formatter: v => Math.pow(2, v).toFixed(2) + "×", inRange: { color: ["#3fb950", "#e3c341", "#d6453a"] } },
      series: [{ type: "heatmap", data, itemStyle: { borderColor: "#fff", borderWidth: 1 }, label: { show: true, fontSize: 10, fontFamily: CHART.font, color: "#0b1220", formatter: p => p.value[3] } }]
    });
    charts.push(chart); chart.resize();
  }

  // ============================================================ SECTION 4
  // The framework's payoff: the z3 stage->PU schedule + the measured speedup.
  let curSchedSub = "explorer";
  const ssel = { device: null, app: null, backend: null, table_type: "btpm", mode: "tmax" };
  let schedExpand = false;   // explorer: show only ★best until the user expands
  const schedApps = D.apps.filter(a => SCHED.cells.some(c => c.app === a.id));
  // own chart lifecycle (kept separate from the profiling tab's `charts`)
  let schedCharts = [];
  const disposeSchedCharts = () => { schedCharts.forEach(c => { try { c.dispose(); } catch (e) {} }); schedCharts = []; };
  const resizeSchedCharts = () => schedCharts.forEach(c => { try { c.resize(); } catch (e) {} });
  window.addEventListener("resize", resizeSchedCharts);

  function renderSchedule(root) {
    root.innerHTML = "";
    if (!SCHED.cells.length) {
      root.appendChild(el("p", { class: "note" }, "No schedules found under data/schedules/. Run the z3 optimizer (optimizer/orchestrate), then regenerate."));
      return;
    }
    root.appendChild(el("p", { class: "note" },
      "Profiling (tab 3) feeds the z3 SMT solver, which partitions the stage sequence into contiguous per-PU "
      + "chunks (the schedule explorer). Pipelining those chunks across PUs yields the measured end-to-end "
      + "speedup. Chunk widths/labels are the solver's predicted ms — the headline speedup is the measured "
      + "figure from speedup-summary.md; the per-schedule predicted speedup_over_* is bogus and not shown."));
    const sub = el("div", { class: "subtabs" });
    [["explorer", "Schedule explorer"], ["speedup", "Measured speedup"]]
      .forEach(o => sub.appendChild(el("button", { class: o[0] === curSchedSub ? "on" : "", onclick: () => { curSchedSub = o[0]; renderSchedule(root); } }, o[1])));
    root.appendChild(sub);
    const view = el("div", null); root.appendChild(view);
    disposeSchedCharts();
    (curSchedSub === "speedup" ? viewScheduleSpeedup : viewScheduleExplorer)(view);
  }
  const resched = () => renderSchedule(byId("tab-schedule"));

  function viewScheduleExplorer(view) {
    // resolve the (device, app, backend) selection against what actually exists
    if (!ssel.app || !schedApps.some(a => a.id === ssel.app)) ssel.app = schedApps[0].id;
    const devs = [...new Set(SCHED.cells.filter(c => c.app === ssel.app).map(c => c.device))].sort();
    if (!ssel.device || devs.indexOf(ssel.device) < 0) ssel.device = devs[0];
    const bes = SCHED.cells.filter(c => c.device === ssel.device && c.app === ssel.app).map(c => c.backend);
    if (bes.indexOf(ssel.backend) < 0) ssel.backend = bes[0];

    const ctr = el("div", { class: "controls" });
    ctr.appendChild(selectCtrl("Device", devs, ssel.device, v => { ssel.device = v; resched(); }));
    ctr.appendChild(segCtrl("App", schedApps.map(a => [a.id, a.title]), ssel.app, v => { ssel.app = v; ssel.device = null; resched(); }));
    ctr.appendChild(segCtrl("Backend", bes.map(b => [b, b.toUpperCase()]), ssel.backend, v => { ssel.backend = v; resched(); }));
    ctr.appendChild(segCtrl("Profiling table", [["isolated", "isolated"], ["btpm", "interference (BTPM)"]], ssel.table_type, v => { ssel.table_type = v; resched(); }));
    ctr.appendChild(segCtrl("Objective", [["tmax", "tmax (makespan)"], ["gapness", "gapness"]], ssel.mode, v => { ssel.mode = v; resched(); }));
    view.appendChild(ctr);

    const cell = schedByKey.get(ssel.device + "|" + ssel.app + "|" + ssel.backend);
    const variant = cell && cell.variants.find(v => v.table_type === ssel.table_type && v.mode === ssel.mode);
    const n = cell ? cell.n_stages : 0;

    view.appendChild(el("p", { class: "note" },
      `${appById[ssel.app].title} · ${ssel.device} · ${ssel.backend.toUpperCase()} · `
      + `z3 solved on the ${ssel.table_type === "btpm" ? "interference (BTPM)" : "isolated"} table, ${ssel.mode} objective. `
      + `Stages S1–S${n}; chunk colours: GPU = backend hue, Big/Medium/Little = CPU tiers.`));

    if (!variant) {
      const avail = cell ? cell.variants.map(v => `${v.table_type}/${v.mode}`) : [];
      view.appendChild(el("p", { class: "note" },
        avail.length ? `No schedule for ${ssel.table_type}/${ssel.mode}. Available for this cell: ${avail.join(", ")}.`
          : "No schedules for this device/app/backend."));
      return;
    }

    // baseline reference (predicted, isolated) — same units as predicted chunk time
    if (cell.baseline && cell.baseline.fastest != null) {
      const b = cell.baseline;
      view.appendChild(el("div", { class: "sched-ref" },
        `fastest single-PU baseline (isolated, predicted): ${fmt(b.fastest, 2)} ms`
        + (b.omp != null ? ` · all-CPU ${fmt(b.omp, 2)}` : "")
        + (b[ssel.backend] != null ? ` · all-${ssel.backend.toUpperCase()} ${fmt(b[ssel.backend], 2)}` : "")));
    }
    if (!variant.validated)
      view.appendChild(el("p", { class: "sched-badge" }, `⚠ ${variant.format} serialization — not schema-validated; stage ranges derived from the legacy 0-based chunk list.`));

    // best first, then by predicted makespan (nulls last)
    const order = variant.schedules.slice().sort((a, b) => {
      if (a.uid === variant.best_uid) return -1;
      if (b.uid === variant.best_uid) return 1;
      return (a.makespan == null ? Infinity : a.makespan) - (b.makespan == null ? Infinity : b.makespan);
    });
    // one shared ms axis for every candidate so bar lengths compare directly
    const baseFastest = cell.baseline ? cell.baseline.fastest : null;
    const maxMk = Math.max(...order.map(s => s.makespan || 0), baseFastest || 0) || 1;
    view.appendChild(guide(
      "how z3 splits the stages across PUs for this cell, and whether the pipeline beats one PU.",
      "each row = a chunk a PU runs; bar length = its predicted time on the shared ms axis. Chunks run in parallel, so makespan = the longest lane (outlined). Dashed line = single-PU baseline; a longest lane ending left of it = faster. Candidates compare across cards."));

    const list = el("div", { class: "sched-list" });
    list.appendChild(schedRuler(maxMk, baseFastest));
    list.appendChild(scheduleCard(order[0], variant, cell, maxMk, baseFastest));
    const rest = order.slice(1);
    if (rest.length) {
      list.appendChild(el("button", { class: "morebtn", onclick: () => { schedExpand = !schedExpand; resched(); } },
        (schedExpand ? "▾ Hide " : "▸ Show ") + rest.length + " other candidate" + (rest.length > 1 ? "s" : "") + " (ranked by predicted makespan)"));
      if (schedExpand) rest.forEach(s => list.appendChild(scheduleCard(s, variant, cell, maxMk, baseFastest)));
    }
    view.appendChild(list);
  }

  // shared ms axis ruler (0..maxMk). 104px = the lane-label gutter so ticks line up with lane tracks.
  function schedRuler(maxMk, baseFastest) {
    const r = el("div", { class: "sched-ruler" });
    for (let i = 0; i <= 4; i++)
      r.appendChild(el("span", { class: "tick" + (i === 0 ? " zero" : "") + (i === 4 ? " end" : ""), style: `left:calc(104px + (100% - 104px) * ${i / 4})` },
        fmt(maxMk * i / 4, maxMk < 10 ? 1 : 0) + (i === 4 ? " ms" : "")));
    if (baseFastest != null && baseFastest <= maxMk)
      r.appendChild(el("span", { class: "baseline-tick", style: `left:calc(104px + (100% - 104px) * ${baseFastest / maxMk})` }, "single-PU"));
    return r;
  }

  // One card = one candidate. Each chunk is a parallel LANE (a PU runs it),
  // bar length = its predicted time on the shared ms axis. The chunks pipeline
  // concurrently, so the pipeline makespan IS the longest lane (outlined).
  function scheduleCard(s, variant, cell, maxMk, baseFastest) {
    const isBest = s.uid === variant.best_uid;
    const head = el("div", { class: "sched-head" },
      el("span", { class: "uid" }, s.uid),
      isBest ? el("span", { class: "tag best" }, "★ best") : null,
      el("span", { class: "muted small" },
        (s.makespan != null ? `makespan ${fmt(s.makespan, 2)} ms` : "makespan —")
        + (s.covers ? "" : " · ⚠ partial coverage")));
    const maxChunk = s.chunks.length && s.chunks.every(c => c.time != null) ? Math.max(...s.chunks.map(c => c.time)) : null;
    const lanes = el("div", { class: "lanes" });
    s.chunks.forEach(c => {
      const span = c.start_stage === c.end_stage ? "S" + c.start_stage : "S" + c.start_stage + "–S" + c.end_stage;
      const hw = c.core_type === "GPU" ? (c.hardware || (cell.backend === "cu" ? "gpu_cuda" : "gpu_vulkan")) : null;
      const tier = c.core_type === "GPU" ? (hw === "gpu_cuda" ? "cuda" : "vulkan") : c.core_type.toLowerCase();
      const isBottleneck = maxChunk != null && c.time === maxChunk;
      const w = (c.time != null && maxMk) ? (c.time / maxMk * 100) : 100;
      const bar = el("div", {
        class: "lane-bar", style: `width:${w}%;background:${chunkColor(c.core_type, c.hardware, cell.backend)}`,
        title: `${c.core_type}${hw ? " (" + hw + ")" : ""} · stages ${span}`
          + (c.time != null ? ` · predicted ${fmt(c.time, 2)} ms` : "") + (isBottleneck ? " · ← makespan (longest lane)" : ""),
      }, c.time != null ? fmt(c.time, 1) + " ms" : "");
      lanes.appendChild(el("div", { class: "lane" + (isBottleneck ? " bottleneck" : "") },
        el("div", { class: "lane-label", title: `${c.core_type} · stages ${span}` },
          el("span", { class: "tier-" + tier }, c.core_type), " " + span),
        el("div", { class: "lane-track" }, bar)));
    });
    // one baseline guide spanning the card's lane tracks (104px = label gutter)
    if (baseFastest != null && maxMk && baseFastest <= maxMk)
      lanes.appendChild(el("div", { class: "baseline-guide", style: `left:calc(104px + (100% - 104px) * ${baseFastest / maxMk})`, title: `fastest single-PU baseline ${fmt(baseFastest, 2)} ms` }));
    return el("div", { class: "sched" + (isBest ? " best" : "") }, head, lanes);
  }

  function speedColor(x) { return x == null ? CHART.axis : x >= 1.1 ? "#3fb950" : x >= 0.95 ? "#e3c341" : "#d6453a"; }

  function viewScheduleSpeedup(view) {
    const M = SCHED.measured, rows = M.rows || [];
    if (!rows.length) { view.appendChild(el("p", { class: "note" }, "No measured speedup rows parsed from speedup-summary.md.")); return; }
    view.appendChild(el("div", { class: "figtitle" }, "Measured end-to-end speedup vs. fastest single-PU baseline"));
    view.appendChild(guide(
      "the real payoff: the scheduled pipeline vs the fastest single PU, both measured on device.",
      ">1× (green) = pipeline wins, ≈1 (amber) = tie, <1 (red) = loss. Dashed line = 1.0× baseline. Hover a bar for its caveat."));
    const div = el("div", { class: "chart", style: "height:480px" }); view.appendChild(div);
    const cats = rows.map(r => `${r.device_label} · ${r.app} · ${r.backend}`);
    const chart = echarts.init(div);
    chart.setOption({
      backgroundColor: "transparent", textStyle: CHART.textStyle,
      tooltip: Object.assign({
        trigger: "axis", axisPointer: { type: "shadow" },
        formatter: p => {
          const r = rows[p[0].dataIndex];
          return `${r.device_label} · ${r.app} · ${r.backend}<br/>`
            + `baseline: ${r.baseline_label}<br/>best schedule: ${r.best_label}<br/>`
            + `<b>speedup ${fmt(r.speedup, 2)}×</b>`
            + (r.caveat ? `<br/><span style="color:#9a6700">⚠ ${r.caveat}</span>` : "");
        }
      }, CHART.tooltip),
      grid: { left: 52, right: 20, top: 16, bottom: 132 },
      xAxis: { type: "category", data: cats, axisLabel: Object.assign({}, CHART.axisLabel, { rotate: 32, fontSize: 10 }), axisLine: CHART.axisLine, axisTick: { show: false } },
      yAxis: { type: "value", name: "speedup (×)", axisLabel: CHART.axisLabel, nameTextStyle: CHART.nameStyle, splitLine: CHART.split, axisLine: { show: false }, axisTick: { show: false } },
      series: [{
        type: "bar", barMaxWidth: 34,
        data: rows.map(r => ({ value: r.speedup, itemStyle: { color: speedColor(r.speedup), borderRadius: [2, 2, 0, 0] } })),
        markLine: {
          symbol: "none", silent: true,
          data: [{ yAxis: 1 }],
          lineStyle: { color: CHART.axis, type: "dashed" },
          label: { formatter: "fastest single-PU = 1.0×", color: CHART.axis, fontFamily: CHART.font, position: "insideEndTop" }
        }
      }]
    });
    schedCharts.push(chart); chart.resize();

    view.appendChild(el("p", { class: "note" },
      "Local fleet measured: 0.78×–1.80× vs fastest single-PU. Paper full eval: 2.14× geomean, up to 7.59× over GPU-only "
      + "(IISWC'25). The in-schedule predicted speedup_over_* metric is bogus (units mismatch) and is not used here."));
    if (M.reading && M.reading.length) {
      const d = el("details", { class: "caveats" }, el("summary", null, "Reading the results · caveats"));
      M.reading.forEach(t => d.appendChild(el("p", { class: "note" }, "• " + t)));
      if (M.tree_losses) d.appendChild(el("p", { class: "note" }, "Why the two tree losses: " + M.tree_losses));
      (M.caveats || []).forEach(t => d.appendChild(el("p", { class: "note" }, "⚠ " + t)));
      view.appendChild(d);
    }
  }

  // ---------- boot ----------
  byId("meta").innerHTML = `generated from <code>${D.git_sha}</code> · ${D.devices.length} devices · ${D.apps.length} apps · ${PROF.cells.length} profiling cells · ${SCHED.cells.length} schedule cells · interference on: ${interferenceDevices.join(", ") || "—"}`;
  document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => { const on = b === btn; b.classList.toggle("active", on); b.setAttribute("aria-selected", on ? "true" : "false"); });
    const id = btn.dataset.tab;
    document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + id));
    if (id === "profiling") resizeCharts();
    if (id === "schedule") resizeSchedCharts();
  }));
  renderDevices(byId("tab-devices"));
  renderApps(byId("tab-apps"));
  renderProfiling(byId("tab-profiling"));
  renderSchedule(byId("tab-schedule"));
})();
