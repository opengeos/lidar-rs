// Interactive browser playground for lidar-rs compiled to WebAssembly (lidar-js).
// Runs entirely client-side on Uint8Array GeoTIFF buffers — zero server uploads.
(function () {
  "use strict";

  const SCRIPT_SRC = document.currentScript ? document.currentScript.src : window.location.href;
  const SITE_ROOT = new URL("../", SCRIPT_SRC);
  const WASM_URL = new URL("wasm/lidar_js.js", SITE_ROOT).href;
  const SAMPLE_URLS = {
    dem: new URL("samples/dem.tif", SITE_ROOT).href,
    dsm: new URL("samples/dsm.tif", SITE_ROOT).href,
  };

  let modulePromise = null;
  let loadedFile = null; // { bytes: Uint8Array, name: string }
  let activeObjectUrls = [];
  let currentSim = null; // retained for lazy on-demand GIF encoding; freed on replace/unload
  let currentSimId = 0;
  let currentSimStem = "";
  let cachedGifUrl = null;
  let cachedGifKey = null;
  let pendingGif = null; // { key: string, promise: Promise<string | null> }
  let loadGeneration = 0;
  let runToken = 0;

  // Interactive inundation player state
  let player = null;
  let playTimer = null;

  const $ = (id) => document.getElementById(id);

  function loadWasm() {
    if (!modulePromise) {
      modulePromise = import(WASM_URL).then(async (m) => {
        await m.default();
        return m;
      });
    }
    return modulePromise;
  }

  function setStatus(text, isError) {
    const el = $("lp-status");
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("lp-error", !!isError);
  }

  function errMsg(e) {
    return e && e.message ? e.message : String(e);
  }

  function revokeObjectUrl(url) {
    if (!url) return;
    URL.revokeObjectURL(url);
    const idx = activeObjectUrls.indexOf(url);
    if (idx !== -1) activeObjectUrls.splice(idx, 1);
  }

  function currentGifOptions() {
    const pingPong = $("lp-pingpong") ? $("lp-pingpong").checked : true;
    const showSpills = $("lp-spill-paths") ? $("lp-spill-paths").checked : true;
    return {
      pingPong,
      showSpills,
      key: `${currentSimId}:${pingPong ? 1 : 0}:${showSpills ? 1 : 0}`,
    };
  }

  function invalidateCachedGif() {
    if (cachedGifUrl) {
      const oldUrl = cachedGifUrl;
      cachedGifUrl = null;
      cachedGifKey = null;
      revokeObjectUrl(oldUrl);
    } else {
      cachedGifKey = null;
    }
  }

  function clearObjectUrls() {
    for (const url of activeObjectUrls) {
      URL.revokeObjectURL(url);
    }
    activeObjectUrls = [];
    cachedGifUrl = null;
    cachedGifKey = null;
    pendingGif = null;
  }

  function releaseSim() {
    if (currentSim) {
      currentSim.free();
      currentSim = null;
    }
    currentSimId++;
    currentSimStem = "";
    invalidateCachedGif();
    pendingGif = null;
  }

  function stopPlayer() {
    if (playTimer !== null) {
      clearInterval(playTimer);
      playTimer = null;
    }
    if (player) {
      player.playing = false;
    }
    const btn = $("lp-play-btn");
    if (btn) btn.textContent = "Play";
  }

  // Yields to the browser event loop without depending on requestAnimationFrame
  // (which pauses in background tabs).
  function nextTick() {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  /** Infers linear-squared area unit label from a WKT/PROJ projection string. */
  function detectAreaUnit(projection) {
    const p = (projection || "").toLowerCase();
    if (!p) return "units²";
    const isProjected =
      p.includes("projcs[") || p.includes("projcrs[") || p.includes("+proj=utm") || p.includes("+proj=aea") || p.includes("+proj=lcc") || p.includes("+proj=merc");
    if (!isProjected && (p.includes("geogcs[") || p.includes("geogcrs[") || p.includes("epsg:4326") || p.includes("+proj=longlat"))) {
      return "deg²";
    }
    if (p.includes("foot") || p.includes("feet") || p.includes("us_survey_foot")) {
      return "ft²";
    }
    if (p.includes("metre") || p.includes("meter") || p.includes("unit[\"m\"") || isProjected) {
      return "m²";
    }
    return "units²";
  }

  function formatArea(area, unit) {
    if (!Number.isFinite(area)) return `— ${unit}`;
    if (unit === "deg²" || (area > 0 && area < 1)) {
      return `${area.toPrecision(4)} ${unit}`;
    }
    return `${area.toLocaleString(undefined, { maximumFractionDigits: 0 })} ${unit}`;
  }

  function drawRgba(canvas, width, height, rgba, baseRgba) {
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const out = new Uint8ClampedArray(width * height * 4);
    if (baseRgba) {
      for (let i = 0; i < out.length; i += 4) {
        const a = rgba[i + 3];
        if (a > 0) {
          const alpha = 0.78;
          const inv = 1 - alpha;
          out[i] = Math.round(baseRgba[i] * inv + rgba[i] * alpha);
          out[i + 1] = Math.round(baseRgba[i + 1] * inv + rgba[i + 1] * alpha);
          out[i + 2] = Math.round(baseRgba[i + 2] * inv + rgba[i + 2] * alpha);
          out[i + 3] = 255;
        } else {
          out[i] = baseRgba[i];
          out[i + 1] = baseRgba[i + 1];
          out[i + 2] = baseRgba[i + 2];
          out[i + 3] = 255;
        }
      }
    } else {
      out.set(rgba);
    }
    ctx.putImageData(new ImageData(out, width, height), 0, 0);
  }

  function drawActiveSpillPaths(ctx, width, height, fillTimes, tHours, spillPaths, stepIdx) {
    if (!spillPaths || !spillPaths.length || tHours <= 0) return 0;
    let activeCount = 0;
    for (const path of spillPaths) {
      if (
        tHours < path.spill_time_h ||
        (path.merge_time_h != null && tHours >= path.merge_time_h) ||
        !path.cells ||
        path.cells.length < 2
      ) {
        continue;
      }

      // Draw each contiguous unsubmerged (dry) overland portion separately so channels never
      // bridge across submerged intermediate pockets; count the path once if any portion renders.
      const nCells = path.cells.length;
      let pathActive = false;
      let idx = 0;
      while (idx < nCells) {
        const [c, r] = path.cells[idx];
        const ft = fillTimes[r * width + c];
        if (ft >= 0 && ft <= tHours) {
          idx++;
          continue;
        }
        const runStart = idx;
        while (idx + 1 < nCells) {
          const [nc, nr] = path.cells[idx + 1];
          const nft = fillTimes[nr * width + nc];
          if (!(nft >= 0 && nft <= tHours)) {
            idx++;
          } else {
            break;
          }
        }
        const runEnd = idx;
        idx++;

        const start = Math.max(0, runStart - 1);
        const end = Math.min(nCells - 1, runEnd + 1);
        if (end <= start) continue;

        pathActive = true;
        ctx.save();
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        // Dark halo for crisp contrast over hillshade
        ctx.beginPath();
        ctx.moveTo(path.cells[start][0] + 0.5, path.cells[start][1] + 0.5);
        for (let i = start + 1; i <= end; i++) {
          ctx.lineTo(path.cells[i][0] + 0.5, path.cells[i][1] + 0.5);
        }
        ctx.strokeStyle = "rgba(10, 26, 62, 0.85)";
        ctx.lineWidth = 4.2;
        ctx.stroke();

        // Solid sky-blue stream channel
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.2;
        ctx.stroke();

        // Animated bright-white flowing dashes showing downstream spill direction
        ctx.setLineDash([4, 6]);
        ctx.lineDashOffset = -(stepIdx * 3);
        ctx.strokeStyle = "rgba(224, 242, 254, 0.92)";
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.setLineDash([]);

        // Directional arrowhead at downstream shoreline tip
        const [tx, ty] = [path.cells[end][0] + 0.5, path.cells[end][1] + 0.5];
        const tailIdx = Math.max(start, end - 6);
        const [bx, by] = [path.cells[tailIdx][0] + 0.5, path.cells[tailIdx][1] + 0.5];
        const dx = tx - bx;
        const dy = ty - by;
        const len = Math.hypot(dx, dy);
        if (len > 0.5) {
          const ux = dx / len;
          const uy = dy / len;
          const headLen = 8.5;
          const spread = 0.52;
          const cosS = Math.cos(spread);
          const sinS = Math.sin(spread);
          const lx = tx - headLen * (ux * cosS - uy * sinS);
          const ly = ty - headLen * (ux * sinS + uy * cosS);
          const rx = tx - headLen * (ux * cosS + uy * sinS);
          const ry = ty - headLen * (-ux * sinS + uy * cosS);

          ctx.beginPath();
          ctx.moveTo(tx, ty);
          ctx.lineTo(lx, ly);
          ctx.lineTo(rx, ry);
          ctx.closePath();
          ctx.fillStyle = "#38bdf8";
          ctx.fill();
          ctx.strokeStyle = "rgba(10, 26, 62, 0.9)";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
        ctx.restore();
      }
      if (pathActive) activeCount++;
    }
    return activeCount;
  }

  function renderPlayerFrame() {
    if (!player) return;
    const {
      width,
      height,
      hillshade,
      fillTimes,
      steps,
      spillPaths,
      rainIntensity,
      areaUnit,
      frameBuf,
      stepIdx,
      direction,
    } = player;
    const step = steps[stepIdx];
    const tHours = step ? step.time_hours : 0;

    // Composite live water inundation over shaded-relief DEM using full-precision f64 fill times
    const alpha = 0.72;
    const inv = 1 - alpha;
    const wr = 20;
    const wg = 115;
    const wb = 225;
    const n = width * height;
    for (let i = 0; i < n; i++) {
      const p = i * 4;
      const ft = fillTimes[i];
      if (tHours > 0 && ft >= 0 && ft <= tHours) {
        const freshness = Math.min(1, Math.max(0, (tHours - ft) / Math.max(tHours, 1)));
        const r = Math.round(wr * (1 - 0.35 * freshness));
        const g = Math.round(wg * (1 - 0.3 * freshness));
        const b = Math.round(wb * (1 - 0.12 * freshness));
        frameBuf[p] = Math.round(hillshade[p] * inv + r * alpha);
        frameBuf[p + 1] = Math.round(hillshade[p + 1] * inv + g * alpha);
        frameBuf[p + 2] = Math.round(hillshade[p + 2] * inv + b * alpha);
        frameBuf[p + 3] = 255;
      } else {
        frameBuf[p] = hillshade[p];
        frameBuf[p + 1] = hillshade[p + 1];
        frameBuf[p + 2] = hillshade[p + 2];
        frameBuf[p + 3] = 255;
      }
    }

    const canvas = $("lp-player-canvas");
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.putImageData(new ImageData(frameBuf, width, height), 0, 0);

    const showSpills = $("lp-spill-paths") ? $("lp-spill-paths").checked : true;
    const activeSpills = showSpills
      ? drawActiveSpillPaths(ctx, width, height, fillTimes, tHours, spillPaths, stepIdx)
      : 0;

    const slider = $("lp-slider");
    if (slider) slider.value = String(stepIdx);

    const telemetry = $("lp-player-telemetry");
    if (telemetry && step) {
      const phase = direction >= 0 ? "RISING" : "FALLING";
      const totalHours = steps[steps.length - 1].time_hours;
      const cumRain = (tHours * rainIntensity).toFixed(1);
      const pct = (step.inundated_fraction * 100).toFixed(1);
      const areaStr = formatArea(step.inundated_area, areaUnit);
      const spillSuffix = showSpills ? ` · Active spills: ${activeSpills}` : "";
      telemetry.textContent =
        `${phase} · T = ${tHours.toFixed(1)} / ${totalHours.toFixed(1)} h · Rain: ${cumRain} cm · Inundated: ${step.inundated_cells.toLocaleString()} cells (${areaStr}, ${pct}% of sinks)${spillSuffix}`;
    }

    renderInundationChart();
  }

  function renderInundationChart() {
    if (!player) return;
    const canvas = $("lp-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const { steps, stepIdx } = player;
    if (!steps.length) return;

    const pad = { left: 46, right: 16, top: 16, bottom: 32 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    const fg = getComputedStyle(document.body).color || "#333";
    const gridColor = "rgba(128,128,128,0.22)";
    const maxT = Math.max(steps[steps.length - 1].time_hours, 1e-6);

    const xOf = (t) => pad.left + (t / maxT) * pw;
    const yOf = (frac) => pad.top + ph * (1 - Math.min(1, Math.max(0, frac)));

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.font = "11px sans-serif";
    ctx.fillStyle = fg;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
      const y = yOf(frac);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + pw, y);
      ctx.stroke();
      ctx.fillText(`${Math.round(frac * 100)}%`, pad.left - 6, y);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const t = (maxT * i) / ticks;
      ctx.fillText(`${t.toFixed(0)}h`, xOf(t), pad.top + ph + 6);
    }

    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(0));
    for (const s of steps) {
      ctx.lineTo(xOf(s.time_hours), yOf(s.inundated_fraction));
    }
    ctx.lineTo(xOf(steps[steps.length - 1].time_hours), yOf(0));
    ctx.closePath();
    ctx.fillStyle = "rgba(20, 115, 225, 0.18)";
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(0));
    for (const s of steps) {
      ctx.lineTo(xOf(s.time_hours), yOf(s.inundated_fraction));
    }
    ctx.strokeStyle = "#1473e1";
    ctx.lineWidth = 2.2;
    ctx.stroke();

    const cur = steps[stepIdx];
    if (cur) {
      const cx = xOf(cur.time_hours);
      const cy = yOf(cur.inundated_fraction);
      ctx.strokeStyle = "#e65100";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(cx, pad.top);
      ctx.lineTo(cx, pad.top + ph);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "#e65100";
      ctx.fill();
    }
  }

  function stepPlayer() {
    if (!player || player.steps.length <= 1) return;
    const pingPong = $("lp-pingpong") ? $("lp-pingpong").checked : true;
    const last = player.steps.length - 1;
    if (pingPong) {
      let next = player.stepIdx + player.direction;
      if (next > last) {
        player.direction = -1;
        next = Math.max(0, last - 1);
      } else if (next < 0) {
        player.direction = 1;
        next = Math.min(last, 1);
      }
      player.stepIdx = next;
    } else {
      player.direction = 1;
      player.stepIdx = (player.stepIdx + 1) % player.steps.length;
    }
    renderPlayerFrame();
  }

  function startPlayer() {
    if (!player) return;
    stopPlayer();
    player.playing = true;
    const btn = $("lp-play-btn");
    if (btn) btn.textContent = "Pause";
    const delayMs = parseInt($("lp-speed")?.value || "200", 10) || 200;
    playTimer = setInterval(stepPlayer, delayMs);
  }

  function initPlayer(
    width,
    height,
    hillshade,
    fillTimes,
    steps,
    spillPaths,
    rainIntensity,
    areaUnit,
  ) {
    stopPlayer();
    // Prepend the dry initial state (T = 0 h) so playback, scrubbing, and chart clicks start at T0.
    const playerSteps =
      steps.length > 0 && steps[0].time_hours > 0
        ? [
            {
              step: 0,
              time_hours: 0,
              inundated_cells: 0,
              inundated_area: 0,
              inundated_fraction: 0,
            },
            ...steps,
          ]
        : steps;
    player = {
      width,
      height,
      hillshade,
      fillTimes,
      steps: playerSteps,
      spillPaths,
      rainIntensity,
      areaUnit,
      frameBuf: new Uint8ClampedArray(width * height * 4),
      stepIdx: 0,
      direction: 1,
      playing: false,
    };
    const slider = $("lp-slider");
    if (slider) {
      slider.min = "0";
      slider.max = String(Math.max(0, playerSteps.length - 1));
      slider.value = "0";
    }
    renderPlayerFrame();
    if (playerSteps.length > 1) {
      startPlayer();
    }
  }

  async function ensureGifUrl() {
    const simRef = currentSim;
    if (!simRef) throw new Error("No active inundation simulation");
    const opts = currentGifOptions();
    if (cachedGifUrl && cachedGifKey === opts.key) return cachedGifUrl;
    if (pendingGif && pendingGif.key === opts.key) return pendingGif.promise;

    const taskKey = opts.key;
    const promise = (async () => {
      setStatus("Encoding animated inundation GIF in WebAssembly…");
      await nextTick();
      if (currentSim !== simRef) {
        return null;
      }
      if (currentGifOptions().key !== taskKey) {
        return ensureGifUrl();
      }
      const t0 = performance.now();
      const gifBytes = simRef.encodeGif(150, opts.pingPong, 400, opts.showSpills);
      if (currentSim !== simRef) {
        return null;
      }
      if (currentGifOptions().key !== taskKey) {
        return ensureGifUrl();
      }
      const elapsed = performance.now() - t0;
      const url = URL.createObjectURL(new Blob([gifBytes], { type: "image/gif" }));
      invalidateCachedGif();
      activeObjectUrls.push(url);
      cachedGifUrl = url;
      cachedGifKey = taskKey;
      const gifImg = $("lp-gif");
      if (gifImg) gifImg.src = url;
      setStatus(
        `Encoded animated inundation GIF (${(gifBytes.length / 1024).toFixed(0)} KB) in ${elapsed.toFixed(0)} ms.`,
      );
      return url;
    })();

    const entry = { key: taskKey, promise };
    pendingGif = entry;
    try {
      return await promise;
    } finally {
      if (pendingGif === entry) {
        pendingGif = null;
      }
    }
  }

  function renderSummary(items) {
    const el = $("lp-summary");
    el.innerHTML = "";
    for (const [label, value] of items) {
      const card = document.createElement("div");
      card.className = "lp-stat";
      const valEl = document.createElement("strong");
      valEl.textContent = value;
      const lblEl = document.createElement("span");
      lblEl.textContent = label;
      card.append(valEl, lblEl);
      el.appendChild(card);
    }
  }

  function renderDownloads(files, includeLazyGif) {
    const el = $("lp-downloads");
    el.innerHTML = "";
    if (includeLazyGif) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "md-button";
      btn.textContent = "Export Inundation GIF";
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
          const url = await ensureGifUrl();
          if (!url) return;
          const a = document.createElement("a");
          a.href = url;
          a.download = `${currentSimStem}_inundation.gif`;
          a.click();
        } catch (e) {
          setStatus(`GIF export failed: ${errMsg(e)}`, true);
        } finally {
          btn.disabled = false;
        }
      });
      el.appendChild(btn);
    }
    for (const { label, filename, data, getData, mime } of files) {
      if (typeof getData === "function") {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "md-button";
        btn.textContent = label;
        btn.addEventListener("click", () => {
          try {
            const payload = getData();
            const url = URL.createObjectURL(new Blob([payload], { type: mime }));
            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            a.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
          } catch (e) {
            setStatus(`${label} export failed: ${errMsg(e)}`, true);
          }
        });
        el.appendChild(btn);
        continue;
      }
      const blob = new Blob([data], { type: mime });
      const url = URL.createObjectURL(blob);
      activeObjectUrls.push(url);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.className = "md-button";
      a.textContent = label;
      el.appendChild(a);
    }
  }

  function renderTable(features) {
    const tbody = $("lp-tbody");
    tbody.innerHTML = "";
    const limit = 100;
    const shown = features.slice(0, limit);
    $("lp-table-note").textContent =
      features.length > limit
        ? `(showing first ${limit} of ${features.length} features)`
        : `(${features.length} total)`;

    const fmt = (n) => (typeof n === "number" && Number.isFinite(n) ? n.toFixed(2) : "—");
    for (const d of shown) {
      const tr = document.createElement("tr");
      const cols = [
        d.id,
        d.level,
        d.region_id,
        d.count,
        fmt(d.area),
        fmt(d.volume),
        fmt(d.avg_depth),
        fmt(d.max_depth),
      ];
      for (const val of cols) {
        const td = document.createElement("td");
        td.textContent = String(val);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }

  function syncModeFields() {
    const isInundation = $("lp-mode").value === "inundation";
    document.querySelectorAll("#lidar-playground .lp-inundation-only").forEach((el) => {
      el.hidden = !isInundation;
    });
  }

  async function runAnalysis() {
    if (!loadedFile) return;
    const token = ++runToken;
    const mode = $("lp-mode").value;
    const kernelSize = Math.max(1, parseInt($("lp-kernel").value, 10) || 1);
    const minSize = Math.max(1, parseInt($("lp-min-size").value, 10) || 1000);
    const minDepth = Math.max(0.01, parseFloat($("lp-min-depth").value) || 0.3);
    const interval = Math.max(0.01, parseFloat($("lp-interval").value) || 0.3);
    const rain = Math.max(0.1, parseFloat($("lp-rain").value) || 5.0);
    const duration = Math.max(0.5, parseFloat($("lp-duration").value) || 70.0);
    const step = Math.max(0.1, parseFloat($("lp-step").value) || 1.0);

    const runBtn = $("lp-run");
    runBtn.disabled = true;
    setStatus(
      `Running ${mode === "inundation" ? "inundation simulation" : "mount delineation"} on ${loadedFile.name} in WebAssembly…`,
    );
    await nextTick();

    try {
      const wasm = await loadWasm();
      if (token !== runToken) return;

      const t0 = performance.now();
      const stem = loadedFile.name.replace(/\.[^.]+$/, "");

      if (mode === "inundation") {
        const sim = wasm.simulateInundation(
          loadedFile.bytes,
          loadedFile.name,
          kernelSize,
          minSize,
          minDepth,
          interval,
          rain,
          duration,
          step,
        );
        if (token !== runToken) {
          sim.free();
          return;
        }
        // Only discard previous results and object URLs once the new run succeeds.
        stopPlayer();
        releaseSim();
        clearObjectUrls();
        currentSim = sim;
        currentSimStem = stem;

        const elapsed = performance.now() - t0;
        const w = sim.width;
        const h = sim.height;
        const areaUnit = detectAreaUnit(sim.projection);
        const hill = sim.hillshadeRgba();
        const depth = sim.depthRgba();
        const level = sim.levelRgba();
        const fill = sim.fillTimeRgba();
        const fillTimes = sim.fillTimeValues();
        const depressions = JSON.parse(sim.depressionsJson());
        const steps = JSON.parse(sim.stepsJson());
        const spillPaths = JSON.parse(sim.spillPathsJson());
        const lastStep = steps.length > 0 ? steps[steps.length - 1] : null;

        $("lp-results").hidden = false;
        $("lp-player-section").hidden = false;
        $("lp-fig-depth").hidden = false;
        $("lp-fig-fill").hidden = false;
        $("lp-cap-level").textContent = "Nested Depression Levels (over hillshade)";
        const gifDetails = $("lp-gif-details");
        if (gifDetails) gifDetails.open = false;
        const gifImg = $("lp-gif");
        if (gifImg) gifImg.removeAttribute("src");

        initPlayer(w, h, hill, fillTimes, steps, spillPaths, rain, areaUnit);

        drawRgba($("lp-c-hillshade"), w, h, hill, null);
        drawRgba($("lp-c-depth"), w, h, depth, hill);
        drawRgba($("lp-c-level"), w, h, level, hill);
        drawRgba($("lp-c-fill"), w, h, fill, hill);

        renderSummary([
          ["Raster Size", `${w} × ${h}`],
          ["Sink Regions", String(sim.numRegions)],
          ["Nested Depressions", String(depressions.length)],
          ["Max Level", String(sim.maxLevel)],
          ["Time Steps", String(steps.length)],
          ["Spill Paths", String(spillPaths.length)],
          ["Final Inundated", lastStep ? `${(lastStep.inundated_fraction * 100).toFixed(1)}%` : "0%"],
          ["WASM Runtime", `${elapsed.toFixed(0)} ms`],
        ]);

        renderDownloads(
          [
            { label: "Depressions GeoJSON", filename: `${stem}_depressions.geojson`, data: sim.depressionsGeojson(), mime: "application/geo+json" },
            { label: "Spill Flow Paths GeoJSON", filename: `${stem}_spill_paths.geojson`, getData: () => sim.spillPathsGeojson(), mime: "application/geo+json" },
            { label: "Depression Level GeoTIFF", filename: `${stem}_level.tif`, data: sim.levelTif(), mime: "image/tiff" },
            { label: "Fill Time GeoTIFF", filename: `${stem}_fill_time.tif`, data: sim.fillTimeTif(), mime: "image/tiff" },
            { label: "Catchment Hierarchy GeoTIFF", filename: `${stem}_catchment_hir.tif`, data: sim.catchmentHirTif(), mime: "image/tiff" },
            { label: "Sink GeoTIFF", filename: `${stem}_sink.tif`, data: sim.sinkTif(), mime: "image/tiff" },
          ],
          true,
        );

        renderTable(depressions);
        setStatus(
          `lidar-rs ${wasm.version()}: simulated ${steps.length} inundation steps across ${depressions.length} nested depressions (${sim.maxLevel} levels) in ${elapsed.toFixed(0)} ms.`,
        );
      } else {
        const hill = wasm.hillshadeRgba(loadedFile.bytes, loadedFile.name, 315.0, 45.0, 1.5);
        const mounts = wasm.delineateMounts(
          loadedFile.bytes,
          loadedFile.name,
          minSize,
          minDepth,
          interval,
        );
        try {
          if (token !== runToken) return;
          stopPlayer();
          releaseSim();
          clearObjectUrls();
          player = null;

          const elapsed = performance.now() - t0;
          const w = mounts.width;
          const h = mounts.height;
          const level = mounts.levelRgba();
          const features = JSON.parse(mounts.depressionsJson());

          $("lp-results").hidden = false;
          $("lp-player-section").hidden = true;
          $("lp-fig-depth").hidden = true;
          $("lp-fig-fill").hidden = true;
          $("lp-cap-level").textContent = "Nested Mount / Canopy Levels (over hillshade)";

          drawRgba($("lp-c-hillshade"), w, h, hill, null);
          drawRgba($("lp-c-level"), w, h, level, hill);

          renderSummary([
            ["Raster Size", `${w} × ${h}`],
            ["Mount Regions", String(mounts.numRegions)],
            ["Nested Mounts", String(features.length)],
            ["Max Level", String(mounts.maxLevel)],
            ["WASM Runtime", `${elapsed.toFixed(0)} ms`],
          ]);

          renderDownloads(
            [
              { label: "Mounts GeoJSON", filename: `${stem}_mounts.geojson`, data: mounts.geojson(), mime: "application/geo+json" },
              { label: "Mount Level GeoTIFF", filename: `${stem}_mount_level.tif`, data: mounts.levelTif(), mime: "image/tiff" },
              { label: "Mount ID GeoTIFF", filename: `${stem}_mount_id.tif`, data: mounts.idTif(), mime: "image/tiff" },
            ],
            false,
          );

          renderTable(features);
          setStatus(
            `lidar-rs ${wasm.version()}: delineated ${features.length} nested mounts across ${mounts.maxLevel} levels in ${elapsed.toFixed(0)} ms.`,
          );
        } finally {
          mounts.free();
        }
      }
    } catch (e) {
      if (token === runToken) {
        setStatus(`Analysis failed: ${errMsg(e)}`, true);
      }
    } finally {
      if ($("lp-run")) {
        $("lp-run").disabled = !loadedFile;
      }
    }
  }

  async function loadFile(file) {
    const gen = ++loadGeneration;
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      if (gen !== loadGeneration) return;
      loadedFile = { bytes, name: file.name };
      $("lp-run").disabled = false;
      await runAnalysis();
    } catch (e) {
      if (gen === loadGeneration) {
        setStatus(`Could not read ${file.name}: ${errMsg(e)}`, true);
      }
    }
  }

  async function loadSample(kind) {
    const gen = ++loadGeneration;
    const url = SAMPLE_URLS[kind];
    const name = `${kind}.tif`;
    try {
      setStatus(`Downloading sample ${name}…`);
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const bytes = new Uint8Array(await resp.arrayBuffer());
      if (gen !== loadGeneration) return;
      loadedFile = { bytes, name };
      $("lp-mode").value = kind === "dsm" ? "mounts" : "inundation";
      syncModeFields();
      $("lp-run").disabled = false;
      await runAnalysis();
    } catch (e) {
      if (gen === loadGeneration) {
        setStatus(`Could not load sample ${name}: ${errMsg(e)}`, true);
      }
    }
  }

  function init() {
    const root = $("lidar-playground");
    if (!root) {
      loadGeneration++;
      runToken++;
      stopPlayer();
      releaseSim();
      player = null;
      clearObjectUrls();
      loadedFile = null;
      return;
    }
    if (root.dataset.ready) return;
    root.dataset.ready = "1";

    syncModeFields();
    loadWasm().then(
      (m) =>
        setStatus(
          `lidar-rs ${m.version()} WebAssembly is ready. Drop a GeoTIFF or click a sample below.`,
        ),
      (e) => setStatus(`Could not load lidar-rs WebAssembly module: ${errMsg(e)}`, true),
    );

    $("lp-mode").addEventListener("change", syncModeFields);
    $("lp-run").addEventListener("click", runAnalysis);
    $("lp-sample-dem").addEventListener("click", () => loadSample("dem"));
    $("lp-sample-dsm").addEventListener("click", () => loadSample("dsm"));
    $("lp-file").addEventListener("change", (e) => {
      if (e.target.files[0]) loadFile(e.target.files[0]);
    });

    // Lazy GIF preview when user opens the collapsible preview
    const gifDetails = $("lp-gif-details");
    if (gifDetails) {
      gifDetails.addEventListener("toggle", () => {
        if (
          gifDetails.open &&
          currentSim &&
          (!cachedGifUrl || cachedGifKey !== currentGifOptions().key)
        ) {
          ensureGifUrl().catch((e) => setStatus(`GIF preview failed: ${errMsg(e)}`, true));
        }
      });
    }

    // Invalidate cached GIF when ping-pong or spill-paths options change; refresh if open
    const refreshGifOptions = () => {
      const gifImg = $("lp-gif");
      if (gifImg) gifImg.removeAttribute("src");
      invalidateCachedGif();
      if (gifDetails && gifDetails.open && currentSim) {
        ensureGifUrl().catch((e) => setStatus(`GIF preview failed: ${errMsg(e)}`, true));
      }
    };
    $("lp-pingpong").addEventListener("change", refreshGifOptions);
    $("lp-spill-paths").addEventListener("change", () => {
      if (player) renderPlayerFrame();
      refreshGifOptions();
    });

    // Interactive inundation player controls
    $("lp-play-btn").addEventListener("click", () => {
      if (!player) return;
      if (player.playing) {
        stopPlayer();
      } else {
        startPlayer();
      }
    });
    $("lp-prev-btn").addEventListener("click", () => {
      if (!player) return;
      stopPlayer();
      player.direction = -1;
      player.stepIdx = Math.max(0, player.stepIdx - 1);
      renderPlayerFrame();
    });
    $("lp-next-btn").addEventListener("click", () => {
      if (!player) return;
      stopPlayer();
      player.direction = 1;
      player.stepIdx = Math.min(player.steps.length - 1, player.stepIdx + 1);
      renderPlayerFrame();
    });
    $("lp-slider").addEventListener("input", (e) => {
      if (!player) return;
      stopPlayer();
      player.stepIdx = Math.min(
        player.steps.length - 1,
        Math.max(0, parseInt(e.target.value, 10) || 0),
      );
      renderPlayerFrame();
    });
    $("lp-speed").addEventListener("change", () => {
      if (player && player.playing) {
        startPlayer();
      }
    });
    $("lp-chart").addEventListener("click", (e) => {
      if (!player || !player.steps.length) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const padLeft = 46;
      const padRight = 16;
      const scaleX = e.currentTarget.width / rect.width;
      const x = (e.clientX - rect.left) * scaleX;
      const frac = Math.min(
        1,
        Math.max(0, (x - padLeft) / (e.currentTarget.width - padLeft - padRight)),
      );
      stopPlayer();
      const maxT = Math.max(player.steps[player.steps.length - 1].time_hours, 1e-6);
      const targetT = frac * maxT;
      let bestIdx = 0;
      let bestDiff = Infinity;
      for (let i = 0; i < player.steps.length; i++) {
        const diff = Math.abs(player.steps[i].time_hours - targetT);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestIdx = i;
        }
      }
      player.stepIdx = bestIdx;
      renderPlayerFrame();
    });

    const drop = $("lp-drop");
    drop.addEventListener("dragover", (e) => {
      e.preventDefault();
      drop.classList.add("lp-over");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("lp-over"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("lp-over");
      if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
    });
  }

  if (window.document$) {
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
