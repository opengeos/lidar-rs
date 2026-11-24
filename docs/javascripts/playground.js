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
    if (!Number.isFinite(area)) return `N/A ${unit}`;
    if (unit === "deg²" || (area > 0 && area < 1)) {
      return `${area.toPrecision(4)} ${unit}`;
    }
    return `${area.toLocaleString(undefined, { maximumFractionDigits: 0 })} ${unit}`;
  }

  /**
   * Detects the vertical elevation unit suffix from an explicit vertical CRS definition
   * (`VERT_CS` / `VERTCRS` in WKT or `+vunits=` in PROJ). Returns `""` when no vertical unit
   * is declared so horizontal units never mislabel vertical elevations.
   */
  function detectElevationUnit(projection) {
    const raw = projection || "";
    if (!raw) return "";
    const projVunits = raw.match(/\+vunits=([a-z0-9_-]+)/i);
    if (projVunits) {
      const vu = projVunits[1].toLowerCase();
      if (vu === "m" || vu === "meter" || vu === "metre") return "m";
      if (vu === "ft" || vu === "us-ft" || vu.includes("foot")) return "ft";
    }
    const vertStart = raw.search(/VERT_CS\[|VERTCRS\[/i);
    if (vertStart === -1) return "";
    const vertSlice = raw.slice(vertStart).toLowerCase();
    const unitMatch = vertSlice.match(/(?:lengthunit|unit)\[\s*"([^"]+)"/);
    if (!unitMatch) return "";
    const u = unitMatch[1];
    if (u.includes("foot") || u.includes("feet") || u === "ft" || u === "us_survey_foot") {
      return "ft";
    }
    if (u.includes("metre") || u.includes("meter") || u === "m") {
      return "m";
    }
    return "";
  }

  const LEVEL_COLORS = [
    "#1d4ed8", // Level 1 (Leaf): Royal Blue
    "#06b6d4", // Level 2: Cyan
    "#22c55e", // Level 3: Emerald Green
    "#f59e0b", // Level 4: Amber / Orange
    "#dc2626", // Level 5: Red
    "#9333ea", // Level 6: Purple
    "#db2777", // Level 7: Pink
    "#4f46e5", // Level 8: Indigo
  ];

  function levelColor(level) {
    const idx = Math.max(0, (level || 1) - 1) % LEVEL_COLORS.length;
    return LEVEL_COLORS[idx];
  }

  function hexToRgba(hex, alpha) {
    const h = hex.replace("#", "");
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function isDarkScheme() {
    return document.body && document.body.getAttribute("data-md-color-scheme") === "slate";
  }

  function volumeUnitFor(areaUnit) {
    if (areaUnit === "m²") return "m³";
    if (areaUnit === "ft²") return "ft³";
    return `${areaUnit || "units²"}·z`;
  }

  function formatCompactVolume(vol, areaUnit) {
    const volUnit = volumeUnitFor(areaUnit);
    if (!Number.isFinite(vol)) return `N/A ${volUnit}`;
    const abs = Math.abs(vol);
    if (abs >= 1e6) return `${(vol / 1e6).toFixed(2)}M ${volUnit}`;
    if (abs >= 1e4) return `${(vol / 1e3).toFixed(1)}k ${volUnit}`;
    if (abs >= 1e3) return `${(vol / 1e3).toFixed(2)}k ${volUnit}`;
    if (abs > 0 && abs < 1) return `${vol.toPrecision(3)} ${volUnit}`;
    return `${vol.toFixed(0)} ${volUnit}`;
  }

  function upperBound(sortedArr, value) {
    let lo = 0;
    let hi = sortedArr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sortedArr[mid] <= value) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  /**
   * Computes per-depression sorted owned-cell fill times and guaranteed interior badge anchor
   * pixels (via a 2-pass L1 distance-to-boundary transform on `depIds`) using dense ID-indexed
   * typed arrays to avoid per-pixel Map hashing on large DEMs.
   */
  function precomputeDepressionGridStats(width, height, depIds, fillTimes, depressions) {
    const fillStats = new Map();
    const badgePos = new Map();
    if (!depIds || !fillTimes || !depressions || depressions.length === 0) {
      return { fillStats, badgePos };
    }

    let maxId = 0;
    for (const d of depressions) {
      if (d.id > maxId) maxId = d.id;
    }
    if (maxId <= 0) return { fillStats, badgePos };

    const n = width * height;
    const totalCounts = new Uint32Array(maxId + 1);
    const wetCounts = new Uint32Array(maxId + 1);
    const sumCol = new Float64Array(maxId + 1);
    const sumRow = new Float64Array(maxId + 1);

    for (let r = 0; r < height; r++) {
      const rowOff = r * width;
      for (let c = 0; c < width; c++) {
        const idx = rowOff + c;
        const id = depIds[idx];
        if (id <= 0 || id > maxId) continue;
        totalCounts[id]++;
        sumCol[id] += c;
        sumRow[id] += r;
        const ft = fillTimes[idx];
        if (Number.isFinite(ft) && ft >= 0) {
          wetCounts[id]++;
        }
      }
    }

    // Allocate and populate per-depression fill-time arrays, then sort numerically.
    const timesById = new Array(maxId + 1);
    const cursors = new Uint32Array(maxId + 1);
    for (const d of depressions) {
      const id = d.id;
      const arr = new Float64Array(wetCounts[id]);
      timesById[id] = arr;
      fillStats.set(id, { total: totalCounts[id], times: arr });
    }
    for (let i = 0; i < n; i++) {
      const id = depIds[i];
      if (id <= 0 || id > maxId) continue;
      const arr = timesById[id];
      if (!arr) continue;
      const ft = fillTimes[i];
      if (!Number.isFinite(ft) || ft < 0) continue;
      arr[cursors[id]++] = ft;
    }
    for (const entry of fillStats.values()) {
      entry.times.sort();
    }

    // 2-pass L1 distance-to-boundary transform to place each ID badge in the widest interior
    // pocket of its depression band (avoiding crescent/ring border collisions).
    const dist = new Uint16Array(n);
    const maxDist = 65534;
    for (let r = 0; r < height; r++) {
      const rowOff = r * width;
      for (let c = 0; c < width; c++) {
        const idx = rowOff + c;
        const id = depIds[idx];
        if (id <= 0 || id > maxId) continue;
        if (
          c === 0 ||
          r === 0 ||
          c + 1 === width ||
          r + 1 === height ||
          depIds[idx - 1] !== id ||
          depIds[idx + 1] !== id ||
          depIds[idx - width] !== id ||
          depIds[idx + width] !== id
        ) {
          dist[idx] = 1;
        } else {
          const d = Math.min(dist[idx - 1], dist[idx - width]) + 1;
          dist[idx] = d < maxDist ? d : maxDist;
        }
      }
    }
    for (let r = height - 1; r >= 0; r--) {
      const rowOff = r * width;
      for (let c = width - 1; c >= 0; c--) {
        const idx = rowOff + c;
        const cur = dist[idx];
        if (cur <= 1) continue;
        const d = Math.min(dist[idx + 1], dist[idx + width]) + 1;
        if (d < cur) dist[idx] = d;
      }
    }

    const meanCol = new Float64Array(maxId + 1);
    const meanRow = new Float64Array(maxId + 1);
    const bestD = new Uint16Array(maxId + 1);
    const bestDistSq = new Float64Array(maxId + 1).fill(Infinity);
    const bestCol = new Int32Array(maxId + 1).fill(-1);
    const bestRow = new Int32Array(maxId + 1).fill(-1);
    for (let id = 1; id <= maxId; id++) {
      if (totalCounts[id] > 0) {
        meanCol[id] = sumCol[id] / totalCounts[id];
        meanRow[id] = sumRow[id] / totalCounts[id];
      }
    }

    for (let r = 0; r < height; r++) {
      const rowOff = r * width;
      for (let c = 0; c < width; c++) {
        const idx = rowOff + c;
        const id = depIds[idx];
        if (id <= 0 || id > maxId) continue;
        const d = dist[idx];
        const dc = c - meanCol[id];
        const dr = r - meanRow[id];
        const distSq = dc * dc + dr * dr;
        const prevD = bestD[id];
        if (d > prevD || (d === prevD && distSq < bestDistSq[id])) {
          bestD[id] = d;
          bestDistSq[id] = distSq;
          bestCol[id] = c;
          bestRow[id] = r;
        }
      }
    }
    for (const d of depressions) {
      const id = d.id;
      if (bestCol[id] >= 0) {
        badgePos.set(id, { col: bestCol[id] + 0.5, row: bestRow[id] + 0.5 });
      }
    }

    return { fillStats, badgePos };
  }

  /**
   * Builds per-region depression tree layouts (leaves left-to-right via deterministic DFS,
   * internal nodes centered above their children).
   */
  function buildDepressionTrees(depressions) {
    const byRegion = new Map();
    for (const d of depressions || []) {
      let arr = byRegion.get(d.region_id);
      if (!arr) {
        arr = [];
        byRegion.set(d.region_id, arr);
      }
      arr.push(d);
    }

    const trees = [];
    for (const [regionId, nodes] of byRegion.entries()) {
      const byId = new Map();
      for (const d of nodes) byId.set(d.id, d);

      const childrenMap = new Map();
      for (const d of nodes) childrenMap.set(d.id, []);
      for (const d of nodes) {
        if (Array.isArray(d.children_ids) && d.children_ids.length > 0) {
          for (const cid of d.children_ids) {
            if (byId.has(cid) && cid !== d.id && !childrenMap.get(d.id).includes(cid)) {
              childrenMap.get(d.id).push(cid);
            }
          }
        }
      }
      for (const d of nodes) {
        if (d.parent_id > 0 && byId.has(d.parent_id) && d.parent_id !== d.id) {
          const arr = childrenMap.get(d.parent_id);
          if (!arr.includes(d.id)) arr.push(d.id);
        }
      }
      for (const arr of childrenMap.values()) {
        arr.sort((a, b) => a - b);
      }

      const roots = nodes
        .filter((d) => !(d.parent_id > 0 && byId.has(d.parent_id)))
        .sort((a, b) => b.level - a.level || a.id - b.id);
      if (roots.length === 0 && nodes.length > 0) {
        roots.push(nodes.slice().sort((a, b) => b.level - a.level || a.id - b.id)[0]);
      }

      const slotX = new Map();
      const visited = new Set();
      let leafCursor = 0;

      function assignSlots(id) {
        if (visited.has(id)) return slotX.get(id) ?? 0;
        visited.add(id);
        const kids = (childrenMap.get(id) || []).filter((cid) => !visited.has(cid));
        if (kids.length === 0) {
          const x = leafCursor++;
          slotX.set(id, x);
          return x;
        }
        let sum = 0;
        for (const cid of kids) {
          sum += assignSlots(cid);
        }
        const x = sum / kids.length;
        slotX.set(id, x);
        return x;
      }

      for (const r of roots) {
        assignSlots(r.id);
      }
      // Include any disconnected nodes just in case.
      for (const d of nodes) {
        if (!visited.has(d.id)) assignSlots(d.id);
      }

      const edges = [];
      for (const d of nodes) {
        for (const cid of childrenMap.get(d.id) || []) {
          edges.push({ childId: cid, parentId: d.id });
        }
      }

      let maxLevel = 1;
      for (const d of nodes) {
        if (d.level > maxLevel) maxLevel = d.level;
      }
      const totalVolume = roots.reduce((acc, r) => acc + (Number.isFinite(r.volume) ? r.volume : 0), 0);

      trees.push({
        regionId,
        nodes,
        byId,
        childrenMap,
        roots,
        edges,
        slotX,
        leafCount: Math.max(1, leafCursor),
        maxLevel,
        totalVolume,
      });
    }

    trees.sort(
      (a, b) =>
        b.maxLevel - a.maxLevel ||
        b.nodes.length - a.nodes.length ||
        b.totalVolume - a.totalVolume ||
        a.regionId - b.regionId,
    );
    return trees;
  }

  /**
   * Derives live UI fill state (`dry`, `filling`, `full`) for one depression at `tHours`.
   * Presentation-layer projection of the WASM outputs from `crates/lidar/src/inundation.rs`
   * (`fill_time` cell grid) and `crates/lidar/src/fsm.rs` (`fill_time_h` / `merge_time_h`);
   * keep aligned with those Rust definitions (or replace with WASM-exported per-step state).
   */
  function computeDepressionFillState(depId, tHours, fillStats, childrenMap, memo, fsmBasinsById) {
    if (memo.has(depId)) return memo.get(depId);
    const stat = fillStats.get(depId);
    let frac = 0;
    let allCellsWet = false;
    if (stat && stat.total > 0) {
      const wet = tHours > 0 ? upperBound(stat.times, tHours) : 0;
      frac = wet / stat.total;
      allCellsWet = wet === stat.total;
    } else if (childrenMap && childrenMap.has(depId)) {
      const kids = childrenMap.get(depId);
      if (kids.length > 0) {
        let allChildrenFull = true;
        for (const cid of kids) {
          const ks = computeDepressionFillState(
            cid,
            tHours,
            fillStats,
            childrenMap,
            memo,
            fsmBasinsById,
          );
          if (ks.status !== "full") allChildrenFull = false;
        }
        frac = allChildrenFull ? 1 : 0;
        allCellsWet = allChildrenFull;
      }
    }

    let status;
    const fsmBasin = fsmBasinsById ? fsmBasinsById.get(depId) : null;
    if (fsmBasin) {
      const isFull =
        tHours > 0 &&
        typeof fsmBasin.fill_time_h === "number" &&
        Number.isFinite(fsmBasin.fill_time_h) &&
        tHours >= fsmBasin.fill_time_h;
      const isMerged =
        tHours > 0 &&
        typeof fsmBasin.merge_time_h === "number" &&
        Number.isFinite(fsmBasin.merge_time_h) &&
        tHours >= fsmBasin.merge_time_h;
      const hasWater = frac > 0 || isFull || isMerged;
      status = !hasWater ? "dry" : isFull || isMerged ? "full" : "filling";
    } else {
      status = frac <= 0 ? "dry" : allCellsWet ? "full" : "filling";
    }

    const clampedFrac = Math.min(1, Math.max(0, frac));
    const res = { frac: clampedFrac, status };
    memo.set(depId, res);
    return res;
  }

  function populateTreeNodeSelect(tree, selectedDepId) {
    const sel = $("lp-tree-node-select");
    const wrap = $("lp-tree-node-wrap");
    if (!sel || !wrap) return;
    sel.innerHTML = "";
    if (!tree || !tree.nodes || tree.nodes.length === 0) {
      wrap.hidden = true;
      return;
    }
    wrap.hidden = false;
    const placeholder = document.createElement("option");
    placeholder.value = "0";
    placeholder.textContent = "Inspect node…";
    sel.appendChild(placeholder);
    const ordered = tree.nodes
      .slice()
      .sort((a, b) => b.level - a.level || a.id - b.id);
    for (const d of ordered) {
      const opt = document.createElement("option");
      opt.value = String(d.id);
      opt.textContent = `#${d.id} (Lvl ${d.level})`;
      sel.appendChild(opt);
    }
    sel.value = selectedDepId && tree.byId.has(selectedDepId) ? String(selectedDepId) : "0";
  }

  function drawDepressionBadges(ctx, width, height, playerState, tHours) {
    const {
      depressions,
      badgePos,
      fillStats,
      fsmBasinsById,
      selectedDepId,
      activeRegionId,
      treeRegions,
    } = playerState;
    if (!depressions || depressions.length === 0 || !badgePos || badgePos.size === 0) return;

    const showAll = depressions.length <= 40;
    const r = Math.max(7, Math.min(11, Math.round(Math.min(width, height) * 0.025)));
    const memo = new Map();
    const hitBadges = [];

    // Draw higher levels first, then lower levels, and selected badge last so it sits on top.
    const ordered = depressions
      .filter((d) => showAll || d.region_id === activeRegionId || d.id === selectedDepId)
      .slice()
      .sort((a, b) => {
        if (a.id === selectedDepId) return 1;
        if (b.id === selectedDepId) return -1;
        return b.level - a.level || a.id - b.id;
      });

    const activeTree = (treeRegions || []).find((t) => t.regionId === activeRegionId);
    const childrenMap = activeTree ? activeTree.childrenMap : null;

    for (const d of ordered) {
      const pos = badgePos.get(d.id);
      if (!pos) continue;
      const state = computeDepressionFillState(
        d.id,
        tHours,
        fillStats,
        childrenMap,
        memo,
        fsmBasinsById,
      );
      const isSelected = d.id === selectedDepId;
      const color = levelColor(d.level);
      const alpha = isSelected || state.status !== "dry" ? 0.96 : 0.62;

      ctx.save();
      ctx.globalAlpha = alpha;

      // Subtle dark shadow for contrast over both bright hillshade and deep water
      ctx.beginPath();
      ctx.arc(pos.col, pos.row, r + (isSelected ? 2 : 0.5), 0, Math.PI * 2);
      ctx.fillStyle = "rgba(15, 23, 42, 0.55)";
      ctx.fill();

      // Filled colored circle
      ctx.beginPath();
      ctx.arc(pos.col, pos.row, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Crisp white ring (gold when selected)
      ctx.lineWidth = isSelected ? 2.4 : 1.5;
      ctx.strokeStyle = isSelected ? "#fbbf24" : "#ffffff";
      ctx.stroke();

      // Bold white ID label
      const label = String(d.id);
      const fontSize = label.length >= 3 ? Math.max(7, Math.round(r * 0.78)) : Math.max(8, Math.round(r * 1.05));
      ctx.font = `700 ${fontSize}px sans-serif`;
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, pos.col, pos.row + 0.5);
      ctx.restore();

      const drawnRadius = r + (isSelected ? 2 : 0.75);
      hitBadges.push({ id: d.id, x: pos.col, y: pos.row, r: drawnRadius });
    }

    playerState.badgeHitTargets = hitBadges;
  }

  function renderTreeLegend(tree) {
    const el = $("lp-tree-legend");
    if (!el || !tree) return;
    el.innerHTML = "";
    for (let lvl = 1; lvl <= tree.maxLevel; lvl++) {
      const item = document.createElement("span");
      item.className = "lp-tree-legend-item";
      const sw = document.createElement("span");
      sw.className = "lp-tree-swatch";
      sw.style.background = levelColor(lvl);
      const suffix = lvl === 1 ? " (Leaf)" : lvl === tree.maxLevel && tree.maxLevel > 1 ? " (Root)" : "";
      item.append(sw, document.createTextNode(`Level ${lvl}${suffix}`));
      el.appendChild(item);
    }
    const states = [
      ["lp-tree-swatch-state lp-tree-swatch-dry", "Dry"],
      ["lp-tree-swatch-state lp-tree-swatch-filling", "Filling"],
      ["lp-tree-swatch-state lp-tree-swatch-full", "Full / Merged"],
    ];
    for (const [cls, lbl] of states) {
      const item = document.createElement("span");
      item.className = "lp-tree-legend-item";
      const sw = document.createElement("span");
      sw.className = cls;
      item.append(sw, document.createTextNode(lbl));
      el.appendChild(item);
    }
  }

  const MAX_TREE_CANVAS_WIDTH = 8192;

  function computeTreeLayoutSize(tree, availW) {
    const pad = { left: 74, right: 66, top: 52, bottom: 54 };
    const nodeR = tree.leafCount > 12 || availW < 680 ? 15 : 17;
    const innerPadX = nodeR + 18;
    const minSlotPx = tree.leafCount <= 9 ? 64 : 58;
    const neededW =
      pad.left + pad.right + innerPadX * 2 + Math.max(0, tree.leafCount - 1) * minSlotPx;
    const logicalW = Math.min(MAX_TREE_CANVAS_WIDTH, Math.max(availW, neededW));
    const levelGap =
      tree.maxLevel <= 1
        ? 0
        : tree.maxLevel <= 5
          ? 60
          : Math.max(44, Math.min(56, Math.floor(340 / (tree.maxLevel - 1))));
    const logicalH = Math.max(210, pad.top + pad.bottom + Math.max(1, tree.maxLevel - 1) * levelGap);
    return { pad, nodeR, innerPadX, logicalW, logicalH };
  }

  function drawDepressionTreeScene(ctx, layout, tree, tHours, opts) {
    const { pad, nodeR, innerPadX, logicalW, logicalH } = layout;
    const {
      showOutline,
      dark,
      fg,
      canvasBg,
      areaUnit,
      elevUnit = "",
      fillStats,
      fsmBasinsById = null,
      selectedDepId = 0,
      hoveredTreeDepId = 0,
      offsetY = 0,
    } = opts;

    const states = new Map();
    for (const d of tree.nodes) {
      computeDepressionFillState(
        d.id,
        tHours,
        fillStats,
        tree.childrenMap,
        states,
        fsmBasinsById,
      );
    }

    let activeCount = 0;
    let fullCount = 0;
    let maxActiveLevel = 0;
    for (const d of tree.nodes) {
      const st = states.get(d.id);
      if (st && st.status !== "dry") {
        activeCount++;
        if (d.level > maxActiveLevel) maxActiveLevel = d.level;
        if (st.status === "full") fullCount++;
      }
    }

    const mutedText = dark ? "rgba(226, 232, 240, 0.42)" : "rgba(30, 41, 59, 0.38)";
    const gridColor = dark ? "rgba(148, 163, 184, 0.2)" : "rgba(100, 116, 139, 0.22)";
    const axisColor = dark ? "rgba(203, 213, 225, 0.55)" : "rgba(30, 41, 59, 0.65)";
    const edgeMergedColor = dark ? "#cbd5e1" : "#334155";
    const edgeGhostColor = dark ? "rgba(148, 163, 184, 0.24)" : "rgba(100, 116, 139, 0.24)";
    const nodeBorderColor = dark ? "#f8fafc" : "#0f172a";

    const axisX = pad.left - 8;
    const plotLeft = pad.left + innerPadX;
    const plotW = Math.max(1, logicalW - plotLeft - pad.right - innerPadX);
    const plotH = logicalH - pad.top - pad.bottom;

    const yOfLevel = (lvl) => {
      if (tree.maxLevel <= 1) return offsetY + pad.top + plotH * 0.5;
      return offsetY + pad.top + plotH * (1 - (lvl - 1) / (tree.maxLevel - 1));
    };
    const slotSpan = Math.max(1, tree.leafCount - 1);
    const xOfSlot = (sx) => {
      if (tree.leafCount <= 1) return plotLeft + plotW * 0.5;
      return plotLeft + (sx / slotSpan) * plotW;
    };

    // Y-axis spine and horizontal level guide lines
    ctx.save();
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(axisX, offsetY + pad.top - 16);
    ctx.lineTo(axisX, offsetY + pad.top + plotH + 18);
    ctx.stroke();

    ctx.font = "10.5px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let lvl = 1; lvl <= tree.maxLevel; lvl++) {
      const y = yOfLevel(lvl);
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(axisX, y);
      ctx.lineTo(logicalW - 10, y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = axisColor;
      ctx.beginPath();
      ctx.moveTo(axisX - 5, y);
      ctx.lineTo(axisX, y);
      ctx.stroke();

      const levelActive = lvl <= maxActiveLevel;
      ctx.fillStyle = levelActive ? fg : mutedText;
      if (lvl === 1) {
        ctx.fillText("Level 1", axisX - 9, y - 6);
        ctx.fillText("(Leaf)", axisX - 9, y + 7);
      } else if (lvl === tree.maxLevel && tree.maxLevel > 1) {
        ctx.fillText(`Level ${lvl}`, axisX - 9, y - 6);
        ctx.fillText("(Root)", axisX - 9, y + 7);
      } else {
        ctx.fillText(`Level ${lvl}`, axisX - 9, y);
      }
    }
    ctx.restore();

    // Compute screen positions for every node
    const coords = new Map();
    for (const d of tree.nodes) {
      const sx = tree.slotX.get(d.id) ?? 0;
      coords.set(d.id, { x: xOfSlot(sx), y: yOfLevel(d.level) });
    }

    // 1. Draw child -> parent edges
    for (const { childId, parentId } of tree.edges) {
      const cPos = coords.get(childId);
      const pPos = coords.get(parentId);
      if (!cPos || !pPos) continue;
      const cSt = states.get(childId) || { frac: 0, status: "dry" };
      const pSt = states.get(parentId) || { frac: 0, status: "dry" };
      const isRelatedToSelected =
        selectedDepId > 0 && (childId === selectedDepId || parentId === selectedDepId);

      const childFsm = fsmBasinsById ? fsmBasinsById.get(childId) : null;
      const isMerged = childFsm
        ? tHours > 0 &&
          typeof childFsm.merge_time_h === "number" &&
          Number.isFinite(childFsm.merge_time_h) &&
          tHours >= childFsm.merge_time_h
        : pSt.status !== "dry";

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cPos.x, cPos.y);
      ctx.lineTo(pPos.x, pPos.y);

      if (isMerged) {
        ctx.strokeStyle = isRelatedToSelected ? "#f59e0b" : edgeMergedColor;
        ctx.lineWidth = isRelatedToSelected ? 3.0 : 2.3;
        ctx.stroke();
      } else if (cSt.status === "full" && (showOutline || parentId === selectedDepId)) {
        ctx.strokeStyle = isRelatedToSelected ? "#fbbf24" : "#38bdf8";
        ctx.lineWidth = 2.0;
        ctx.setLineDash([5, 4]);
        ctx.stroke();
      } else if (showOutline) {
        ctx.strokeStyle = isRelatedToSelected ? "rgba(245, 158, 11, 0.55)" : edgeGhostColor;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([2, 4]);
        ctx.stroke();
      }
      ctx.restore();
    }

    // 2. Draw nodes & elevation/volume annotations
    const hitNodes = [];
    for (const d of tree.nodes) {
      const pos = coords.get(d.id);
      if (!pos) continue;
      const st = states.get(d.id) || { frac: 0, status: "dry" };
      const color = levelColor(d.level);
      const isSelected = d.id === selectedDepId;
      const isHovered = d.id === hoveredTreeDepId;

      if (st.status === "dry" && !showOutline && !isSelected) {
        continue;
      }

      hitNodes.push({ id: d.id, x: pos.x, y: pos.y, r: nodeR + 6 });

      ctx.save();
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR + 5, 0, Math.PI * 2);
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 3.2;
        ctx.stroke();
      } else if (isHovered) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR + 4, 0, Math.PI * 2);
        ctx.strokeStyle = hexToRgba(color, 0.65);
        ctx.lineWidth = 2.2;
        ctx.stroke();
      }

      const idFontSize = nodeR >= 17 ? 12 : 11;
      if (st.status === "dry") {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = canvasBg;
        ctx.fill();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = hexToRgba(color, 0.48);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.font = `600 ${idFontSize - 1}px sans-serif`;
        ctx.fillStyle = mutedText;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(d.id), pos.x, pos.y + 0.5);
      } else if (st.status === "filling") {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = canvasBg;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(color, dark ? 0.22 : 0.14);
        ctx.fill();

        ctx.save();
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
        ctx.clip();
        const visualFrac = Math.min(0.92, Math.max(0.08, st.frac));
        const fillY = pos.y + nodeR - 2 * nodeR * visualFrac;
        ctx.fillStyle = color;
        ctx.fillRect(pos.x - nodeR, fillY, nodeR * 2, pos.y + nodeR - fillY + 1);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(pos.x - nodeR, fillY);
        ctx.lineTo(pos.x + nodeR, fillY);
        ctx.stroke();
        ctx.restore();

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
        ctx.strokeStyle = nodeBorderColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.font = `700 ${idFontSize}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = visualFrac >= 0.55 ? "#ffffff" : dark ? "#f8fafc" : "#0f172a";
        ctx.fillText(String(d.id), pos.x, pos.y + 0.5);
      } else {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = nodeBorderColor;
        ctx.lineWidth = 1.8;
        ctx.stroke();

        ctx.font = `700 ${idFontSize}px sans-serif`;
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(d.id), pos.x, pos.y + 0.5);
      }

      const fillingPct = Math.min(99, Math.max(1, Math.round(st.frac * 100)));
      const elevLine =
        st.status === "filling"
          ? `${d.max_elev.toFixed(1)}${elevUnit} (${fillingPct}%)`
          : `${d.max_elev.toFixed(1)}${elevUnit}`;
      const volLine = formatCompactVolume(d.volume, areaUnit);

      ctx.font = "10px sans-serif";
      ctx.fillStyle = st.status === "dry" ? mutedText : fg;

      const isRoot = d.parent_id === 0 || !tree.byId.has(d.parent_id) || d.level === tree.maxLevel;
      if (d.level === 1 && !isRoot) {
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(elevLine, pos.x, pos.y + nodeR + 6);
        ctx.fillText(volLine, pos.x, pos.y + nodeR + 18);
      } else if (isRoot) {
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(volLine, pos.x, pos.y - nodeR - 5);
        ctx.fillText(elevLine, pos.x, pos.y - nodeR - 17);
      } else {
        const pPos = coords.get(d.parent_id);
        let placeLeft = !pPos || pos.x <= pPos.x;
        const labelW = Math.max(ctx.measureText(elevLine).width, ctx.measureText(volLine).width);
        if (placeLeft && pos.x - nodeR - 7 - labelW < axisX + 6) {
          placeLeft = false;
        }
        const lx = placeLeft ? pos.x - nodeR - 7 : pos.x + nodeR + 7;
        ctx.textAlign = placeLeft ? "right" : "left";
        ctx.textBaseline = "middle";
        ctx.fillText(elevLine, lx, pos.y - 6);
        ctx.fillText(volLine, lx, pos.y + 6);
      }
      ctx.restore();
    }

    return { hitNodes, activeCount, fullCount, maxActiveLevel };
  }

  const MAX_TREE_GIF_FRAME_PIXELS = 40_000_000;
  const MAX_TREE_GIF_DIM = 1600;

  /**
   * Minimal pure-JS streaming GIF89a encoder with shared 256-color Global Color Table and
   * LSB-first LZW. Calls `renderFrameRgba(frameIdx)` lazily so only one RGBA buffer is live
   * at a time.
   */
  function encodeStreamedFramesToGif(frameCount, renderFrameRgba, width, height, delayMs) {
    const nPix = width * height;
    const delayCs = Math.max(2, Math.round((delayMs || 150) / 10));

    // 1. Sample up to 16 evenly spaced frames to build a 5-5-5 RGB frequency histogram
    const counts = new Uint32Array(32768);
    const sumR = new Float64Array(32768);
    const sumG = new Float64Array(32768);
    const sumB = new Float64Array(32768);
    const sampleCount = Math.min(frameCount, 16);
    for (let s = 0; s < sampleCount; s++) {
      const fi =
        sampleCount === 1 ? 0 : Math.round((s * (frameCount - 1)) / (sampleCount - 1));
      const rgba = renderFrameRgba(fi);
      for (let i = 0; i < rgba.length; i += 4) {
        const r = rgba[i];
        const g = rgba[i + 1];
        const b = rgba[i + 2];
        const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
        counts[key]++;
        sumR[key] += r;
        sumG[key] += g;
        sumB[key] += b;
      }
    }

    const buckets = [];
    for (let k = 0; k < 32768; k++) {
      if (counts[k] > 0) buckets.push(k);
    }
    buckets.sort((a, b) => counts[b] - counts[a]);

    const palette = new Uint8Array(256 * 3);
    let palLen = 0;
    const maxDataColors = 224;
    for (let i = 0; i < buckets.length && palLen < maxDataColors; i++) {
      const k = buckets[i];
      const c = counts[k];
      palette[palLen * 3] = Math.round(sumR[k] / c);
      palette[palLen * 3 + 1] = Math.round(sumG[k] / c);
      palette[palLen * 3 + 2] = Math.round(sumB[k] / c);
      palLen++;
    }
    for (let i = maxDataColors; i < buckets.length && palLen < 256; i++) {
      const k = buckets[i];
      const c = counts[k];
      palette[palLen * 3] = Math.round(sumR[k] / c);
      palette[palLen * 3 + 1] = Math.round(sumG[k] / c);
      palette[palLen * 3 + 2] = Math.round(sumB[k] / c);
      palLen++;
    }
    const levels = [0, 85, 170, 255];
    for (let r = 0; r < 4 && palLen < 256; r++) {
      for (let g = 0; g < 4 && palLen < 256; g++) {
        for (let b = 0; b < 4 && palLen < 256; b++) {
          palette[palLen * 3] = levels[r];
          palette[palLen * 3 + 1] = levels[g];
          palette[palLen * 3 + 2] = levels[b];
          palLen++;
        }
      }
    }

    // Lazy O(1) 15-bit RGB -> palette index lookup cache
    const lut = new Int16Array(32768).fill(-1);
    function nearestColor(key, r, g, b) {
      const cached = lut[key];
      if (cached >= 0) return cached;
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let p = 0; p < palLen; p++) {
        const dr = r - palette[p * 3];
        const dg = g - palette[p * 3 + 1];
        const db = b - palette[p * 3 + 2];
        const d = dr * dr + dg * dg + db * db;
        if (d < bestDist) {
          bestDist = d;
          bestIdx = p;
          if (d === 0) break;
        }
      }
      lut[key] = bestIdx;
      return bestIdx;
    }

    // Growable byte writer
    const chunks = [];
    let buf = new Uint8Array(65536);
    let pos = 0;
    function ensure(extra) {
      if (pos + extra > buf.length) {
        chunks.push(buf.subarray(0, pos));
        buf = new Uint8Array(Math.max(65536, extra * 2));
        pos = 0;
      }
    }
    function writeByte(b) {
      ensure(1);
      buf[pos++] = b & 0xff;
    }
    function writeU16(v) {
      ensure(2);
      buf[pos++] = v & 0xff;
      buf[pos++] = (v >> 8) & 0xff;
    }
    function writeBytes(arr) {
      ensure(arr.length);
      buf.set(arr, pos);
      pos += arr.length;
    }

    // Header + Logical Screen Descriptor + Global Color Table + NETSCAPE2.0 loop
    writeBytes([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]); // GIF89a
    writeU16(width);
    writeU16(height);
    writeByte(0xf7); // GCT flag=1, 8-bit color res, 256-entry GCT
    writeByte(0);
    writeByte(0);
    writeBytes(palette);
    writeBytes([
      0x21, 0xff, 0x0b, 0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30,
      0x03, 0x01, 0x00, 0x00, 0x00,
    ]);

    // Reusable LZW open-addressed hash table (zero per-frame GC)
    const HT_SIZE = 16384;
    const HT_MASK = HT_SIZE - 1;
    const htKey = new Int32Array(HT_SIZE);
    const htVal = new Int32Array(HT_SIZE);
    const htStamp = new Uint32Array(HT_SIZE);
    let stamp = 1;
    const subBlock = new Uint8Array(255);
    const indexed = new Uint8Array(nPix);

    for (let f = 0; f < frameCount; f++) {
      const rgba = renderFrameRgba(f);
      for (let i = 0, p = 0; i < nPix; i++, p += 4) {
        const r = rgba[p];
        const g = rgba[p + 1];
        const b = rgba[p + 2];
        const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
        indexed[i] = nearestColor(key, r, g, b);
      }

      // Graphic Control Extension
      writeBytes([0x21, 0xf9, 0x04, 0x04, delayCs & 0xff, (delayCs >> 8) & 0xff, 0x00, 0x00]);
      // Image Descriptor
      writeByte(0x2c);
      writeU16(0);
      writeU16(0);
      writeU16(width);
      writeU16(height);
      writeByte(0x00);
      // LZW minimum code size = 8
      writeByte(8);

      const clearCode = 256;
      const eoiCode = 257;
      let nextCode = 258;
      let codeSize = 9;
      let bitBuf = 0;
      let bitCount = 0;
      let subLen = 0;

      function flushSub() {
        if (subLen > 0) {
          ensure(subLen + 1);
          buf[pos++] = subLen;
          buf.set(subBlock.subarray(0, subLen), pos);
          pos += subLen;
          subLen = 0;
        }
      }
      function emit(code) {
        bitBuf |= code << bitCount;
        bitCount += codeSize;
        while (bitCount >= 8) {
          subBlock[subLen++] = bitBuf & 0xff;
          if (subLen === 255) flushSub();
          bitBuf >>>= 8;
          bitCount -= 8;
        }
      }

      stamp++;
      if (stamp === 0xffffffff) {
        htStamp.fill(0);
        stamp = 1;
      }
      emit(clearCode);

      let prefix = indexed[0];
      for (let i = 1; i < nPix; i++) {
        const k = indexed[i];
        const combined = (prefix << 8) | k;
        let slot = ((combined * 2654435761) >>> 18) & HT_MASK;
        let found = -1;
        while (htStamp[slot] === stamp) {
          if (htKey[slot] === combined) {
            found = htVal[slot];
            break;
          }
          slot = (slot + 1) & HT_MASK;
        }
        if (found >= 0) {
          prefix = found;
        } else {
          emit(prefix);
          if (nextCode < 4096) {
            htStamp[slot] = stamp;
            htKey[slot] = combined;
            htVal[slot] = nextCode;
            if (nextCode === 1 << codeSize && codeSize < 12) {
              codeSize++;
            }
            nextCode++;
          } else {
            emit(clearCode);
            stamp++;
            if (stamp === 0xffffffff) {
              htStamp.fill(0);
              stamp = 1;
            }
            nextCode = 258;
            codeSize = 9;
          }
          prefix = k;
        }
      }

      emit(prefix);
      if (nextCode === 1 << codeSize && codeSize < 12) {
        codeSize++;
      }
      emit(eoiCode);
      if (bitCount > 0) {
        subBlock[subLen++] = bitBuf & 0xff;
      }
      flushSub();
      writeByte(0x00); // block terminator
    }

    writeByte(0x3b); // GIF trailer
    chunks.push(buf.subarray(0, pos));
    const totalLen = chunks.reduce((s, c) => s + c.length, 0);
    const out = new Uint8Array(totalLen);
    let off = 0;
    for (const c of chunks) {
      out.set(c, off);
      off += c.length;
    }
    return out;
  }

  async function exportDepressionTreeGif() {
    const playerRef = player;
    const simIdRef = currentSimId;
    const stemRef = currentSimStem;
    if (!playerRef || !playerRef.treeRegions || playerRef.treeRegions.length === 0) return;

    const tree =
      playerRef.treeRegions.find((t) => t.regionId === playerRef.activeRegionId) ||
      playerRef.treeRegions[0];
    const pingPong = $("lp-pingpong") ? $("lp-pingpong").checked : true;
    const showOutline = $("lp-tree-outline") ? $("lp-tree-outline").checked : true;
    const delayMs = Math.max(60, Math.min(400, parseInt($("lp-speed")?.value || "150", 10) || 150));

    setStatus("Encoding animated depression tree GIF in browser…");
    await nextTick();
    if (player !== playerRef || currentSimId !== simIdRef) return;
    const t0 = performance.now();

    const wrap = $("lp-tree-canvas-wrap");
    const baseW = Math.max(640, Math.min(960, (wrap ? wrap.clientWidth : 720) - 2));
    const layout = computeTreeLayoutSize(tree, baseW);
    const bannerH = 38;
    const legendH = 30;
    const rawW = layout.logicalW;
    const rawH = bannerH + layout.logicalH + legendH;
    const scale = Math.min(1, MAX_TREE_GIF_DIM / rawW, MAX_TREE_GIF_DIM / rawH);
    const gifW = Math.max(1, Math.round(rawW * scale));
    const gifH = Math.max(1, Math.round(rawH * scale));

    const framePixels = gifW * gifH;
    const maxBudgetFrames = Math.floor(MAX_TREE_GIF_FRAME_PIXELS / Math.max(1, framePixels));
    if (maxBudgetFrames < 2) {
      setStatus(
        `Depression tree GIF exceeds browser pixel budget (${gifW}×${gifH}); try a smaller region.`,
        true,
      );
      return;
    }

    const steps = playerRef.steps;
    const totalHours = steps.length > 0 ? steps[steps.length - 1].time_hours : 0;

    // Build frame schedule bounded by both max frame count and total frame-pixel budget
    let indices = steps.map((_, i) => i);
    const maxTotalFrames = Math.min(160, maxBudgetFrames);
    const maxForward = pingPong ? Math.max(2, Math.floor((maxTotalFrames + 2) / 2)) : maxTotalFrames;
    if (indices.length > maxForward) {
      const sampled = [];
      for (let s = 0; s < maxForward; s++) {
        sampled.push(Math.round((s * (indices.length - 1)) / (maxForward - 1)));
      }
      indices = sampled;
    }
    const schedule = indices.map((idx) => ({ idx, phase: "RISING" }));
    if (pingPong && indices.length > 2) {
      for (let j = indices.length - 2; j >= 1; j--) {
        if (schedule.length >= maxTotalFrames) break;
        schedule.push({ idx: indices[j], phase: "FALLING" });
      }
    }

    const offscreen = document.createElement("canvas");
    offscreen.width = gifW;
    offscreen.height = gifH;
    const ctx = offscreen.getContext("2d", { willReadFrequently: true });

    const dark = isDarkScheme();
    const fg = dark ? "#e2e8f0" : "#1e293b";
    const subFg = dark ? "#94a3b8" : "#64748b";
    const canvasBg = dark ? "#161922" : "#ffffff";
    const bannerBg = dark ? "#1f2430" : "#f1f5f9";
    const borderCol = dark ? "#2e3545" : "#e2e8f0";

    const renderFrameRgba = (schedIdx) => {
      const { idx, phase } = schedule[schedIdx];
      const step = steps[idx];
      const tHours = step ? step.time_hours : 0;
      const fracTime = totalHours > 0 ? Math.min(1, Math.max(0, tHours / totalHours)) : 0;

      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.fillStyle = canvasBg;
      ctx.fillRect(0, 0, rawW, rawH);

      const res = drawDepressionTreeScene(ctx, layout, tree, tHours, {
        showOutline,
        dark,
        fg,
        canvasBg,
        areaUnit: playerRef.areaUnit,
        elevUnit: playerRef.elevUnit,
        fillStats: playerRef.fillStats,
        fsmBasinsById: playerRef.fsmBasinsById,
        selectedDepId: 0,
        hoveredTreeDepId: 0,
        offsetY: bannerH,
      });

      // Top telemetry banner
      ctx.fillStyle = bannerBg;
      ctx.fillRect(0, 0, rawW, bannerH);
      ctx.strokeStyle = borderCol;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, bannerH - 0.5);
      ctx.lineTo(rawW, bannerH - 0.5);
      ctx.stroke();

      // Animated progress bar at bottom of banner
      ctx.fillStyle = "#1473e1";
      ctx.fillRect(0, bannerH - 3, Math.round(rawW * fracTime), 3);

      ctx.font = "700 12.5px sans-serif";
      ctx.fillStyle = fg;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const regionSuffix = playerRef.treeRegions.length > 1 ? ` (Region ${tree.regionId})` : "";
      ctx.fillText(
        `Depression Tree${regionSuffix} · ${phase} · T = ${tHours.toFixed(1)} / ${totalHours.toFixed(1)} h`,
        14,
        bannerH * 0.48,
      );

      ctx.font = "600 11.5px sans-serif";
      ctx.fillStyle = subFg;
      ctx.textAlign = "right";
      const rightLabel =
        res.activeCount === 0
          ? `0/${tree.nodes.length} active · Dry`
          : res.fullCount === tree.nodes.length
            ? `${tree.nodes.length}/${tree.nodes.length} full · Complete`
            : `${res.activeCount}/${tree.nodes.length} active · ${res.fullCount} full · Lvl ${res.maxActiveLevel}/${tree.maxLevel}`;
      ctx.fillText(rightLabel, rawW - 14, bannerH * 0.48);

      // Bottom legend bar
      const legY = bannerH + layout.logicalH;
      ctx.strokeStyle = borderCol;
      ctx.beginPath();
      ctx.moveTo(0, legY + 0.5);
      ctx.lineTo(rawW, legY + 0.5);
      ctx.stroke();

      ctx.font = "10.5px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const cy = legY + legendH * 0.5;
      let lx = 16;
      for (let lvl = 1; lvl <= tree.maxLevel; lvl++) {
        ctx.beginPath();
        ctx.arc(lx + 5, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = levelColor(lvl);
        ctx.fill();
        ctx.strokeStyle = dark ? "#cbd5e1" : "#334155";
        ctx.lineWidth = 1;
        ctx.stroke();
        const txt =
          lvl === 1 ? "Level 1 (Leaf)" : lvl === tree.maxLevel && tree.maxLevel > 1 ? `Level ${lvl} (Root)` : `Level ${lvl}`;
        ctx.fillStyle = subFg;
        ctx.fillText(txt, lx + 14, cy + 0.5);
        lx += 14 + ctx.measureText(txt).width + 14;
        if (lx > rawW - 160) break;
      }
      const stateItems = [
        { label: "Full", kind: "full" },
        { label: "Filling", kind: "filling" },
        { label: "Dry", kind: "dry" },
      ];
      let rx = rawW - 14;
      for (const stItem of stateItems) {
        const tw = ctx.measureText(stItem.label).width;
        rx -= tw;
        ctx.fillStyle = subFg;
        ctx.fillText(stItem.label, rx, cy + 0.5);
        rx -= 15;
        ctx.save();
        ctx.beginPath();
        ctx.arc(rx + 5, cy, 5, 0, Math.PI * 2);
        if (stItem.kind === "full") {
          ctx.fillStyle = LEVEL_COLORS[0];
          ctx.fill();
        } else if (stItem.kind === "filling") {
          ctx.fillStyle = hexToRgba(LEVEL_COLORS[0], 0.2);
          ctx.fill();
          ctx.clip();
          ctx.fillStyle = LEVEL_COLORS[0];
          ctx.fillRect(rx, cy, 10, 6);
          ctx.restore();
          ctx.save();
          ctx.beginPath();
          ctx.arc(rx + 5, cy, 5, 0, Math.PI * 2);
        } else {
          ctx.setLineDash([2, 2]);
        }
        ctx.strokeStyle = stItem.kind === "dry" ? subFg : dark ? "#cbd5e1" : "#334155";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
        rx -= 12;
      }

      return ctx.getImageData(0, 0, gifW, gifH).data;
    };

    const gifBytes = encodeStreamedFramesToGif(
      schedule.length,
      renderFrameRgba,
      gifW,
      gifH,
      delayMs,
    );
    if (player !== playerRef || currentSimId !== simIdRef) return;

    const elapsed = performance.now() - t0;
    const blobUrl = URL.createObjectURL(new Blob([gifBytes], { type: "image/gif" }));
    const a = document.createElement("a");
    a.href = blobUrl;
    const regionPart = playerRef.treeRegions.length > 1 ? `_region${tree.regionId}` : "";
    a.download = `${stemRef || "lidar"}_depression_tree${regionPart}.gif`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 2000);
    setStatus(
      `Exported depression tree GIF (${(gifBytes.length / 1024).toFixed(0)} KB, ${schedule.length} frames) in ${elapsed.toFixed(0)} ms.`,
    );
  }

  function renderDepressionTree() {
    if (!player) return;
    const panel = $("lp-tree-panel");
    const canvas = $("lp-tree-canvas");
    if (!panel || !canvas) return;
    const trees = player.treeRegions || [];
    if (trees.length === 0) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;

    const tree = trees.find((t) => t.regionId === player.activeRegionId) || trees[0];
    player.activeRegionId = tree.regionId;
    const step = player.steps[player.stepIdx];
    const tHours = step ? step.time_hours : 0;
    const showOutline = $("lp-tree-outline") ? $("lp-tree-outline").checked : true;

    const wrap = $("lp-tree-canvas-wrap");
    const availW = Math.max(320, (wrap ? wrap.clientWidth : 800) - 2);
    const layout = computeTreeLayoutSize(tree, availW);
    const { logicalW, logicalH } = layout;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.style.width = `${logicalW}px`;
    canvas.style.height = `${logicalH}px`;
    const targetW = Math.round(logicalW * dpr);
    const targetH = Math.round(logicalH * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, logicalW, logicalH);

    const dark = isDarkScheme();
    const fg = getComputedStyle(document.body).color || (dark ? "#e2e8f0" : "#1e293b");
    const canvasBg =
      (wrap && getComputedStyle(wrap).backgroundColor) || (dark ? "#1e2129" : "#ffffff");

    const { hitNodes, activeCount, fullCount, maxActiveLevel } = drawDepressionTreeScene(
      ctx,
      layout,
      tree,
      tHours,
      {
        showOutline,
        dark,
        fg,
        canvasBg,
        areaUnit: player.areaUnit,
        elevUnit: player.elevUnit,
        fillStats: player.fillStats,
        fsmBasinsById: player.fsmBasinsById,
        selectedDepId: player.selectedDepId,
        hoveredTreeDepId: player.hoveredTreeDepId,
        offsetY: 0,
      },
    );

    const statusEl = $("lp-tree-status");
    if (statusEl) {
      if (activeCount === 0) {
        statusEl.textContent = `(0/${tree.nodes.length} active · Dry at T = ${tHours.toFixed(1)} h)`;
      } else if (fullCount === tree.nodes.length) {
        statusEl.textContent = `(${tree.nodes.length}/${tree.nodes.length} full · Levels 1–${tree.maxLevel} complete)`;
      } else {
        statusEl.textContent = `(${activeCount}/${tree.nodes.length} active · ${fullCount} full · Active level ${maxActiveLevel}/${tree.maxLevel})`;
      }
    }

    player.treeLogicalW = logicalW;
    player.treeLogicalH = logicalH;
    player.treeHitNodes = hitNodes;
  }

  function renderSelectedOverlay(ctx, width, height, depIds, selectedId, depsById, spillPaths) {
    if (!selectedId || !depIds || !depsById) return;
    const sel = depsById.get(selectedId);
    if (!sel) return;
    const related = new Set([selectedId]);
    if (sel.parent_id > 0) related.add(sel.parent_id);
    if (Array.isArray(sel.children_ids)) {
      for (const cid of sel.children_ids) related.add(cid);
    }

    // Draw crisp gold contour around the selected depression (and soft amber around parent/children)
    const img = ctx.getImageData(0, 0, width, height);
    const d = img.data;
    for (let r = 0; r < height; r++) {
      const rowOff = r * width;
      for (let c = 0; c < width; c++) {
        const idx = rowOff + c;
        const id = depIds[idx];
        if (!related.has(id)) continue;
        const isBorder =
          c === 0 ||
          c + 1 === width ||
          r === 0 ||
          r + 1 === height ||
          depIds[idx - 1] !== id ||
          depIds[idx + 1] !== id ||
          depIds[idx - width] !== id ||
          depIds[idx + width] !== id;
        const p = idx * 4;
        if (id === selectedId) {
          if (isBorder) {
            d[p] = 251;
            d[p + 1] = 191;
            d[p + 2] = 36;
            d[p + 3] = 255;
          } else {
            d[p] = Math.min(255, Math.round(d[p] * 0.72 + 251 * 0.28));
            d[p + 1] = Math.min(255, Math.round(d[p + 1] * 0.72 + 191 * 0.28));
            d[p + 2] = Math.min(255, Math.round(d[p + 2] * 0.72 + 36 * 0.28));
          }
        } else if (isBorder) {
          d[p] = 249;
          d[p + 1] = 115;
          d[p + 2] = 22;
          d[p + 3] = 255;
        }
      }
    }
    ctx.putImageData(img, 0, 0);

    // Highlight outgoing spill path from selected depression in vivid amber
    if (Array.isArray(spillPaths)) {
      for (const sp of spillPaths) {
        if (sp.from_dep_id !== selectedId || !sp.cells || sp.cells.length < 2) continue;
        ctx.save();
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(sp.cells[0][0] + 0.5, sp.cells[0][1] + 0.5);
        for (let i = 1; i < sp.cells.length; i++) {
          ctx.lineTo(sp.cells[i][0] + 0.5, sp.cells[i][1] + 0.5);
        }
        ctx.strokeStyle = "rgba(120, 53, 15, 0.9)";
        ctx.lineWidth = 4.6;
        ctx.stroke();
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 2.4;
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  function renderRatingChart(points) {
    const canvas = $("lp-rating-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!points || points.length === 0) return;

    const pad = { left: 54, right: 16, top: 18, bottom: 28 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;
    const fg = getComputedStyle(document.body).color || "#333";
    const maxD = Math.max(...points.map((p) => p.depth), 1e-6);
    const maxV = Math.max(...points.map((p) => p.volume), 1e-6);

    const xOf = (d) => pad.left + (d / maxD) * pw;
    const yOf = (v) => pad.top + ph * (1 - v / maxV);

    ctx.strokeStyle = "rgba(128,128,128,0.22)";
    ctx.lineWidth = 1;
    ctx.font = "10px sans-serif";
    ctx.fillStyle = fg;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const frac of [0, 0.5, 1]) {
      const y = pad.top + ph * (1 - frac);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + pw, y);
      ctx.stroke();
      const val = maxV * frac;
      ctx.fillText(val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val.toFixed(0), pad.left - 5, y);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const frac of [0, 0.5, 1]) {
      const d = maxD * frac;
      ctx.fillText(`${d.toFixed(2)}m`, xOf(d), pad.top + ph + 5);
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    const volUnit = volumeUnitFor(player && player.areaUnit);
    ctx.fillText(`Stage-Volume Rating Curve (${volUnit} vs depth)`, pad.left, 3);

    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(0));
    for (const p of points) {
      ctx.lineTo(xOf(p.depth), yOf(p.volume));
    }
    ctx.lineTo(xOf(points[points.length - 1].depth), yOf(0));
    ctx.closePath();
    ctx.fillStyle = "rgba(245, 158, 11, 0.22)";
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(0));
    for (const p of points) {
      ctx.lineTo(xOf(p.depth), yOf(p.volume));
    }
    ctx.strokeStyle = "#d97706";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function selectDepression(depId) {
    if (!player) return;
    player.selectedDepId = depId || 0;
    const titleEl = $("lp-inspector-title");
    const emptyEl = $("lp-inspector-empty");
    const bodyEl = $("lp-inspector-body");
    const clearBtn = $("lp-inspector-clear");
    const statsEl = $("lp-inspector-stats");

    document.querySelectorAll("#lp-tbody tr").forEach((tr) => {
      tr.classList.toggle("lp-row-selected", Number(tr.dataset.depId) === player.selectedDepId);
    });

    if (!player.selectedDepId || !player.depsById.has(player.selectedDepId)) {
      if (titleEl) titleEl.textContent = "Depression Inspector";
      if (emptyEl) emptyEl.hidden = false;
      if (bodyEl) bodyEl.hidden = true;
      if (clearBtn) clearBtn.hidden = true;
      const nodeSel = $("lp-tree-node-select");
      if (nodeSel) nodeSel.value = "0";
      renderPlayerFrame();
      return;
    }

    const d = player.depsById.get(player.selectedDepId);
    if (d && d.region_id && player.activeRegionId !== d.region_id) {
      const targetTree = (player.treeRegions || []).find((t) => t.regionId === d.region_id);
      if (targetTree) {
        player.activeRegionId = d.region_id;
        const regSel = $("lp-tree-region");
        if (regSel) regSel.value = String(d.region_id);
        renderTreeLegend(targetTree);
        populateTreeNodeSelect(targetTree, d.id);
      }
    } else {
      const nodeSel = $("lp-tree-node-select");
      if (nodeSel) nodeSel.value = String(d.id);
    }
    const outSpill = (player.spillPaths || []).find((s) => s.from_dep_id === d.id);
    const effMmH = player.rainIntensity * player.runoffCoeff * 10.0;
    const spillRunoff = outSpill ? `${(outSpill.spill_time_h * effMmH).toFixed(1)} mm` : "N/A";
    const childrenStr =
      Array.isArray(d.children_ids) && d.children_ids.length > 0 ? d.children_ids.join(", ") : "None";
    const parentStr = d.parent_id > 0 ? `#${d.parent_id}` : "Root";

    if (titleEl) titleEl.textContent = `Depression #${d.id} (Level ${d.level}, Region ${d.region_id})`;
    if (emptyEl) emptyEl.hidden = true;
    if (bodyEl) bodyEl.hidden = false;
    if (clearBtn) clearBtn.hidden = false;

    if (statsEl) {
      statsEl.innerHTML = "";
      const volUnit = volumeUnitFor(player.areaUnit);
      const items = [
        ["Area", formatArea(d.area, player.areaUnit)],
        ["Volume", `${d.volume.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${volUnit}`],
        ["Max / Mean Depth", `${d.max_depth.toFixed(2)} / ${d.avg_depth.toFixed(2)} m`],
        ["Parent / Children", `${parentStr} / ${childrenStr}`],
        ["Circularity / HI", `${d.circularity.toFixed(2)} / ${d.hypsometric_integral.toFixed(2)}`],
        ["Rim Slope / Spill", `${d.rim_slope_deg.toFixed(1)}° / ${spillRunoff}`],
      ];
      for (const [k, v] of items) {
        const div = document.createElement("div");
        div.className = "lp-ins-item";
        const strong = document.createElement("strong");
        strong.textContent = v;
        const span = document.createElement("span");
        span.textContent = k;
        div.append(strong, span);
        statsEl.appendChild(div);
      }
    }

    const rcPts = (player.ratingCurvesByDep.get(d.id) || []).slice().sort((a, b) => a.depth - b.depth);
    renderRatingChart(rcPts);
    renderPlayerFrame();
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

    if (player.selectedDepId > 0) {
      renderSelectedOverlay(
        ctx,
        width,
        height,
        player.depIds,
        player.selectedDepId,
        player.depsById,
        spillPaths,
      );
    }

    const showBadges = $("lp-id-badges") ? $("lp-id-badges").checked : true;
    if (showBadges) {
      drawDepressionBadges(ctx, width, height, player, tHours);
    } else {
      player.badgeHitTargets = [];
    }

    const slider = $("lp-slider");
    if (slider) slider.value = String(stepIdx);

    const telemetry = $("lp-player-telemetry");
    if (telemetry && step) {
      const phase = direction >= 0 ? "RISING" : "FALLING";
      const totalHours = steps[steps.length - 1].time_hours;
      const effRunoffMm = (tHours * rainIntensity * player.runoffCoeff * 10.0).toFixed(0);
      const pct = (step.inundated_fraction * 100).toFixed(1);
      const areaStr = formatArea(step.inundated_area, areaUnit);
      const spillSuffix = showSpills ? ` · Active spills: ${activeSpills}` : "";
      const cNote = player.runoffCoeff < 0.999 ? ` (C=${player.runoffCoeff.toFixed(2)})` : "";
      telemetry.textContent =
        `${phase} · T = ${tHours.toFixed(1)} / ${totalHours.toFixed(1)} h · Runoff: ${effRunoffMm} mm${cNote} · Inundated: ${step.inundated_cells.toLocaleString()} cells (${areaStr}, ${pct}%)${spillSuffix}`;
    }

    renderInundationChart();
    renderDepressionTree();
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
    depIds,
    depressions,
    ratingCurves,
    steps,
    spillPaths,
    rainIntensity,
    runoffCoeff,
    areaUnit,
    elevUnit,
    fsmBasinsById,
  ) {
    stopPlayer();
    const depsById = new Map();
    for (const d of depressions || []) {
      depsById.set(d.id, d);
    }
    const ratingCurvesByDep = new Map();
    for (const pt of ratingCurves || []) {
      let arr = ratingCurvesByDep.get(pt.dep_id);
      if (!arr) {
        arr = [];
        ratingCurvesByDep.set(pt.dep_id, arr);
      }
      arr.push(pt);
    }
    const { fillStats, badgePos } = precomputeDepressionGridStats(
      width,
      height,
      depIds,
      fillTimes,
      depressions,
    );
    const treeRegions = buildDepressionTrees(depressions);
    const activeRegionId = treeRegions.length > 0 ? treeRegions[0].regionId : 0;

    const regWrap = $("lp-tree-region-wrap");
    const regSel = $("lp-tree-region");
    if (regSel && regWrap) {
      regSel.innerHTML = "";
      for (const t of treeRegions) {
        const opt = document.createElement("option");
        opt.value = String(t.regionId);
        opt.textContent = `Region ${t.regionId} (${t.nodes.length} nodes, ${t.maxLevel} lvls)`;
        regSel.appendChild(opt);
      }
      regSel.value = String(activeRegionId);
      regWrap.hidden = treeRegions.length <= 1;
    }
    if (treeRegions.length > 0) {
      renderTreeLegend(treeRegions[0]);
      populateTreeNodeSelect(treeRegions[0], 0);
    } else {
      populateTreeNodeSelect(null, 0);
    }

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
      depIds,
      depressions: depressions || [],
      depsById,
      ratingCurvesByDep,
      fillStats,
      fsmBasinsById: fsmBasinsById || null,
      badgePos,
      badgeHitTargets: [],
      treeRegions,
      activeRegionId,
      treeHitNodes: [],
      hoveredTreeDepId: 0,
      selectedDepId: 0,
      steps: playerSteps,
      spillPaths,
      rainIntensity,
      runoffCoeff,
      areaUnit,
      elevUnit: elevUnit || "",
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
    selectDepression(0);
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

  async function runTreeGifExport(button) {
    if (button) button.disabled = true;
    try {
      await exportDepressionTreeGif();
    } catch (e) {
      setStatus(`Depression tree GIF export failed: ${errMsg(e)}`, true);
    } finally {
      if (button) button.disabled = false;
    }
  }

  function renderDownloads(files, includeLazyGif, includeTreeGif) {
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
    if (includeTreeGif) {
      const treeBtn = document.createElement("button");
      treeBtn.type = "button";
      treeBtn.className = "md-button";
      treeBtn.textContent = "Export Depression Tree GIF";
      treeBtn.addEventListener("click", () => runTreeGifExport(treeBtn));
      el.appendChild(treeBtn);
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

    const fmt = (n, dec = 2) =>
      typeof n === "number" && Number.isFinite(n) ? n.toFixed(dec) : "N/A";
    for (const d of shown) {
      const tr = document.createElement("tr");
      tr.dataset.depId = String(d.id);
      const cols = [
        d.id,
        d.level,
        d.parent_id > 0 ? d.parent_id : "Root",
        d.region_id,
        d.count,
        fmt(d.area, 1),
        fmt(d.volume, 1),
        fmt(d.max_depth, 2),
        fmt(d.circularity, 2),
        fmt(d.hypsometric_integral, 2),
        typeof d.rim_slope_deg === "number" && Number.isFinite(d.rim_slope_deg)
          ? `${d.rim_slope_deg.toFixed(1)}°`
          : "N/A",
      ];
      for (const val of cols) {
        const td = document.createElement("td");
        td.textContent = String(val);
        tr.appendChild(td);
      }
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      const activateRow = () => {
        if (player) {
          stopPlayer();
          selectDepression(d.id);
        }
      };
      tr.addEventListener("click", activateRow);
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activateRow();
        }
      });
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
    const routingMode = $("lp-routing") ? $("lp-routing").value : "static";
    const kernelSize = Math.max(1, parseInt($("lp-kernel").value, 10) || 1);
    const breachVal = Math.max(0, parseFloat($("lp-breach")?.value) || 0);
    const breachMaxDepth = breachVal > 0 ? breachVal : undefined;
    const minSize = Math.max(1, parseInt($("lp-min-size").value, 10) || 1000);
    const minDepth = Math.max(0.01, parseFloat($("lp-min-depth").value) || 0.3);
    const interval = Math.max(0.01, parseFloat($("lp-interval").value) || 0.3);
    const rain = Math.max(0.1, parseFloat($("lp-rain").value) || 5.0);
    const runoffCoeff = Math.min(1.0, Math.max(0.05, parseFloat($("lp-runoff-coeff")?.value) || 1.0));
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
        const sim =
          typeof wasm.simulateInundationAdvanced === "function"
            ? wasm.simulateInundationAdvanced(
                loadedFile.bytes,
                loadedFile.name,
                kernelSize,
                minSize,
                minDepth,
                interval,
                rain,
                duration,
                step,
                breachMaxDepth,
                runoffCoeff,
                undefined,
                routingMode,
              )
            : wasm.simulateInundation(
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
        const elevUnit = detectElevationUnit(sim.projection);
        const hill = sim.hillshadeRgba();
        const depth = sim.depthRgba();
        const level = sim.levelRgba();
        const fill = sim.fillTimeRgba();
        const fillTimes = sim.fillTimeValues();
        const depIds =
          typeof sim.depressionIdValues === "function" ? sim.depressionIdValues() : null;
        const depressions = JSON.parse(sim.depressionsJson());
        const ratingCurves =
          typeof sim.ratingCurvesJson === "function" ? JSON.parse(sim.ratingCurvesJson()) : [];
        const steps = JSON.parse(sim.stepsJson());
        const spillPaths = JSON.parse(sim.spillPathsJson());
        const fsmBasins =
          typeof sim.fsmBasinsJson === "function" ? JSON.parse(sim.fsmBasinsJson()) : [];
        const fsmBasinsById =
          Array.isArray(fsmBasins) && fsmBasins.length > 0
            ? new Map(fsmBasins.map((b) => [b.dep_id, b]))
            : null;
        const lastStep = steps.length > 0 ? steps[steps.length - 1] : null;

        $("lp-results").hidden = false;
        $("lp-player-section").hidden = false;
        $("lp-fig-depth").hidden = false;
        $("lp-fig-fill").hidden = false;
        $("lp-cap-level").textContent = "Nested Depression Levels (over hillshade)";
        const gifLinkRow = $("lp-gif-link-row");
        if (gifLinkRow) gifLinkRow.hidden = false;
        const gifPreviewLink = $("lp-gif-preview-link");
        if (gifPreviewLink) {
          gifPreviewLink.textContent = "Preview exported animated GIF";
          gifPreviewLink.setAttribute("aria-expanded", "false");
        }
        const gifWrap = $("lp-gif-wrap");
        if (gifWrap) gifWrap.hidden = true;
        const gifImg = $("lp-gif");
        if (gifImg) gifImg.removeAttribute("src");

        initPlayer(
          w,
          h,
          hill,
          fillTimes,
          depIds,
          depressions,
          ratingCurves,
          steps,
          spillPaths,
          rain,
          runoffCoeff,
          areaUnit,
          elevUnit,
          fsmBasinsById,
        );

        drawRgba($("lp-c-hillshade"), w, h, hill, null);
        drawRgba($("lp-c-depth"), w, h, depth, hill);
        drawRgba($("lp-c-level"), w, h, level, hill);
        drawRgba($("lp-c-fill"), w, h, fill, hill);

        const fsmMassBalance =
          typeof sim.fsmMassBalanceJson === "function"
            ? JSON.parse(sim.fsmMassBalanceJson())
            : null;
        const summaryCards = [
          ["Raster Size", `${w} × ${h}`],
          ["Routing", fsmMassBalance !== null ? "Fill-Spill-Merge" : "Static"],
          ["Sink Regions", String(sim.numRegions)],
          ["Nested Depressions", String(depressions.length)],
          ["Max Level", String(sim.maxLevel)],
          ["Time Steps", String(steps.length)],
          ["Spill Paths", String(spillPaths.length)],
          ["Final Inundated", lastStep ? `${(lastStep.inundated_fraction * 100).toFixed(1)}%` : "0%"],
          ["WASM Runtime", `${elapsed.toFixed(0)} ms`],
        ];
        if (fsmMassBalance) {
          summaryCards.push([
            "FSM Stored / Runoff",
            `${(fsmMassBalance.total_stored_m3 / 1000).toFixed(1)}k / ${(fsmMassBalance.total_runoff_m3 / 1000).toFixed(1)}k m³`,
          ]);
          summaryCards.push([
            "FSM Boundary Outflow",
            `${(fsmMassBalance.total_boundary_outflow_m3 / 1000).toFixed(1)}k m³`,
          ]);
          summaryCards.push([
            "Mass Balance Err",
            `${fsmMassBalance.mass_balance_error_m3.toExponential(1)} m³`,
          ]);
        }
        renderSummary(summaryCards);

        const downloadItems = [
          {
            label: "Export PNG Snapshot (Scale + Legend)",
            filename: `${stem}_snapshot.png`,
            getData: () => {
              const showSpills = $("lp-spill-paths") ? $("lp-spill-paths").checked : true;
              const idx = player ? player.stepIdx : 0;
              return sim.renderFramePng(idx, true, true, showSpills);
            },
            mime: "image/png",
          },
          { label: "Depressions GeoJSON", filename: `${stem}_depressions.geojson`, data: sim.depressionsGeojson(), mime: "application/geo+json" },
          { label: "Spill Flow Paths GeoJSON", filename: `${stem}_spill_paths.geojson`, getData: () => sim.spillPathsGeojson(), mime: "application/geo+json" },
          { label: "Connectivity Graph JSON", filename: `${stem}_connectivity_graph.json`, getData: () => sim.connectivityGraphJson(), mime: "application/json" },
          { label: "Connectivity GraphML", filename: `${stem}_connectivity_graph.graphml`, getData: () => sim.connectivityGraphml(), mime: "application/xml" },
          { label: "Rating Curves CSV", filename: `${stem}_rating_curves.csv`, getData: () => sim.ratingCurvesCsv(), mime: "text/csv" },
          { label: "Inundation Curve CSV", filename: `${stem}_inundation_curve.csv`, getData: () => sim.inundationCurveCsv(), mime: "text/csv" },
          { label: "Depression Level GeoTIFF", filename: `${stem}_level.tif`, data: sim.levelTif(), mime: "image/tiff" },
          { label: "Fill Time GeoTIFF", filename: `${stem}_fill_time.tif`, data: sim.fillTimeTif(), mime: "image/tiff" },
          { label: "Catchment Hierarchy GeoTIFF", filename: `${stem}_catchment_hir.tif`, data: sim.catchmentHirTif(), mime: "image/tiff" },
          { label: "Sink GeoTIFF", filename: `${stem}_sink.tif`, data: sim.sinkTif(), mime: "image/tiff" },
        ];
        if (fsmMassBalance && typeof sim.fsmBasinsCsv === "function") {
          const fsmItem = {
            label: "FSM Basins CSV",
            filename: `${stem}_fsm_basins.csv`,
            getData: () => sim.fsmBasinsCsv(),
            mime: "text/csv",
          };
          const curveIdx = downloadItems.findIndex((item) => item.label === "Inundation Curve CSV");
          if (curveIdx >= 0) {
            downloadItems.splice(curveIdx + 1, 0, fsmItem);
          } else {
            downloadItems.push(fsmItem);
          }
        }
        renderDownloads(downloadItems, true, depressions.length > 0);

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
          const gifLinkRow = $("lp-gif-link-row");
          if (gifLinkRow) gifLinkRow.hidden = true;
          const gifPreviewLink = $("lp-gif-preview-link");
          if (gifPreviewLink) {
            gifPreviewLink.textContent = "Preview exported animated GIF";
            gifPreviewLink.setAttribute("aria-expanded", "false");
          }
          const gifWrap = $("lp-gif-wrap");
          if (gifWrap) gifWrap.hidden = true;
          const gifImg = $("lp-gif");
          if (gifImg) gifImg.removeAttribute("src");

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

    // Lazy GIF preview toggled by disclosure button under Downloads
    const gifPreviewLink = $("lp-gif-preview-link");
    const gifWrap = $("lp-gif-wrap");
    if (gifPreviewLink && gifWrap) {
      gifPreviewLink.addEventListener("click", () => {
        if (!currentSim) return;
        const opening = gifWrap.hidden;
        gifWrap.hidden = !opening;
        gifPreviewLink.setAttribute("aria-expanded", opening ? "true" : "false");
        gifPreviewLink.textContent = opening
          ? "Hide animated GIF preview"
          : "Preview exported animated GIF";
        if (opening && (!cachedGifUrl || cachedGifKey !== currentGifOptions().key)) {
          ensureGifUrl().catch((err) => setStatus(`GIF preview failed: ${errMsg(err)}`, true));
        }
      });
    }

    // Invalidate cached GIF when ping-pong or spill-paths options change; refresh if open
    const refreshGifOptions = () => {
      const gifImg = $("lp-gif");
      if (gifImg) gifImg.removeAttribute("src");
      invalidateCachedGif();
      if (gifWrap && !gifWrap.hidden && currentSim) {
        ensureGifUrl().catch((e) => setStatus(`GIF preview failed: ${errMsg(e)}`, true));
      }
    };
    $("lp-pingpong").addEventListener("change", refreshGifOptions);
    $("lp-spill-paths").addEventListener("change", () => {
      if (player) renderPlayerFrame();
      refreshGifOptions();
    });
    const idBadgesCheck = $("lp-id-badges");
    if (idBadgesCheck) {
      idBadgesCheck.addEventListener("change", () => {
        if (player) renderPlayerFrame();
      });
    }
    const treeOutlineCheck = $("lp-tree-outline");
    if (treeOutlineCheck) {
      treeOutlineCheck.addEventListener("change", () => {
        if (player) renderDepressionTree();
      });
    }
    const treeGifBtn = $("lp-tree-gif-btn");
    if (treeGifBtn) {
      treeGifBtn.addEventListener("click", () => runTreeGifExport(treeGifBtn));
    }
    const treeRegionSel = $("lp-tree-region");
    if (treeRegionSel) {
      treeRegionSel.addEventListener("change", () => {
        if (!player) return;
        const rid = parseInt(treeRegionSel.value, 10) || 0;
        player.activeRegionId = rid;
        const target = (player.treeRegions || []).find((t) => t.regionId === rid);
        if (target) {
          renderTreeLegend(target);
          populateTreeNodeSelect(target, player.selectedDepId);
        }
        renderPlayerFrame();
      });
    }
    const treeNodeSel = $("lp-tree-node-select");
    if (treeNodeSel) {
      treeNodeSel.addEventListener("change", () => {
        if (!player) return;
        const depId = parseInt(treeNodeSel.value, 10) || 0;
        stopPlayer();
        selectDepression(depId);
      });
    }

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
    const inspectorClear = $("lp-inspector-clear");
    if (inspectorClear) {
      inspectorClear.addEventListener("click", () => selectDepression(0));
    }
    $("lp-player-canvas").addEventListener("click", (e) => {
      if (!player || !player.depIds) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const fx = ((e.clientX - rect.left) / rect.width) * player.width;
      const fy = ((e.clientY - rect.top) / rect.height) * player.height;
      const col = Math.floor(fx);
      const row = Math.floor(fy);
      if (col < 0 || col >= player.width || row < 0 || row >= player.height) return;

      // Prefer clicking directly on an ID badge circle if visible
      let depId = 0;
      const badges = player.badgeHitTargets || [];
      for (let i = badges.length - 1; i >= 0; i--) {
        const b = badges[i];
        const dx = fx - b.x;
        const dy = fy - b.y;
        if (dx * dx + dy * dy <= b.r * b.r) {
          depId = b.id;
          break;
        }
      }
      if (!depId) {
        depId = player.depIds[row * player.width + col];
      }
      stopPlayer();
      selectDepression(depId > 0 ? depId : 0);
    });
    const treeCanvas = $("lp-tree-canvas");
    if (treeCanvas) {
      const hitTestTree = (e) => {
        if (!player || !player.treeHitNodes || !player.treeHitNodes.length) return 0;
        const rect = treeCanvas.getBoundingClientRect();
        const scaleX = rect.width > 0 ? (player.treeLogicalW || rect.width) / rect.width : 1;
        const scaleY = rect.height > 0 ? (player.treeLogicalH || rect.height) / rect.height : 1;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        for (let i = player.treeHitNodes.length - 1; i >= 0; i--) {
          const n = player.treeHitNodes[i];
          const dx = x - n.x;
          const dy = y - n.y;
          if (dx * dx + dy * dy <= n.r * n.r) {
            return n.id;
          }
        }
        return 0;
      };
      treeCanvas.addEventListener("click", (e) => {
        if (!player) return;
        const hitId = hitTestTree(e);
        if (hitId > 0) {
          stopPlayer();
          selectDepression(hitId);
        }
      });
      treeCanvas.addEventListener("mousemove", (e) => {
        if (!player) return;
        const hitId = hitTestTree(e);
        treeCanvas.style.cursor = hitId > 0 ? "pointer" : "default";
        if (player.hoveredTreeDepId !== hitId) {
          player.hoveredTreeDepId = hitId;
          renderDepressionTree();
        }
      });
      treeCanvas.addEventListener("mouseleave", () => {
        if (!player) return;
        treeCanvas.style.cursor = "default";
        if (player.hoveredTreeDepId !== 0) {
          player.hoveredTreeDepId = 0;
          renderDepressionTree();
        }
      });
      treeCanvas.addEventListener("keydown", (e) => {
        if (!player || !player.treeRegions || !player.treeRegions.length) return;
        const tree =
          player.treeRegions.find((t) => t.regionId === player.activeRegionId) ||
          player.treeRegions[0];
        if (!tree || !tree.nodes.length) return;
        const ordered = tree.nodes
          .slice()
          .sort(
            (a, b) =>
              (tree.slotX.get(a.id) ?? 0) - (tree.slotX.get(b.id) ?? 0) ||
              a.level - b.level ||
              a.id - b.id,
          );
        const curId = player.hoveredTreeDepId || player.selectedDepId;
        const curIdx = ordered.findIndex((n) => n.id === curId);
        if (e.key === "ArrowRight" || e.key === "ArrowUp") {
          e.preventDefault();
          const next = ordered[(curIdx + 1 + ordered.length) % ordered.length];
          player.hoveredTreeDepId = next.id;
          renderDepressionTree();
        } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
          e.preventDefault();
          const prev = ordered[(curIdx - 1 + ordered.length) % ordered.length];
          player.hoveredTreeDepId = prev.id;
          renderDepressionTree();
        } else if (e.key === "Home") {
          e.preventDefault();
          player.hoveredTreeDepId = ordered[0].id;
          renderDepressionTree();
        } else if (e.key === "End") {
          e.preventDefault();
          player.hoveredTreeDepId = ordered[ordered.length - 1].id;
          renderDepressionTree();
        } else if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          const targetId = player.hoveredTreeDepId || ordered[0].id;
          stopPlayer();
          selectDepression(targetId);
        } else if (e.key === "Escape") {
          if (player.selectedDepId > 0) {
            e.preventDefault();
            selectDepression(0);
          }
        }
      });
    }
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

  let themeObserverAttached = false;
  function ensureThemeObserver() {
    if (themeObserverAttached || typeof MutationObserver === "undefined" || !document.body) return;
    themeObserverAttached = true;
    new MutationObserver(() => {
      if (!player) return;
      renderInundationChart();
      renderDepressionTree();
      if (player.selectedDepId > 0) {
        const rcPts = (player.ratingCurvesByDep.get(player.selectedDepId) || [])
          .slice()
          .sort((a, b) => a.depth - b.depth);
        renderRatingChart(rcPts);
      }
    }).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
  }

  let resizeRaf = 0;
  window.addEventListener("resize", () => {
    if (!player || resizeRaf) return;
    resizeRaf = window.requestAnimationFrame(() => {
      resizeRaf = 0;
      if (player) renderDepressionTree();
    });
  });

  if (window.document$) {
    window.document$.subscribe(() => {
      ensureThemeObserver();
      init();
    });
  } else {
    document.addEventListener("DOMContentLoaded", () => {
      ensureThemeObserver();
      init();
    });
  }
})();
