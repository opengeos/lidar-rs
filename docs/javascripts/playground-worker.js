// Dedicated ES module Web Worker for the lidar-rs WebAssembly Playground.
// Keeps DEM delineation, inundation simulation, GeoJSON/GeoTIFF exports, and
// GIF encoding completely off the browser main thread with progress reporting,
// zero-copy ArrayBuffer transfers, and instant cancellation via termination.

let wasm = null;
let wasmInitOutput = null;
let wasmUrlLoaded = null;
let activeSim = null;
let activeSimRunId = 0;
let activeMounts = null;
let activeMountsRunId = 0;
const cancelledIds = new Set();

function postProgress(id, stage, fraction, message) {
  self.postMessage({
    type: "progress",
    id,
    stage,
    fraction: Math.max(0, Math.min(1, fraction)),
    message,
  });
}

function errMsg(err) {
  if (err && typeof err === "object" && "message" in err && err.message) {
    return String(err.message);
  }
  return String(err);
}

function releaseOutputs() {
  if (activeSim) {
    try {
      activeSim.free();
    } catch (_) {
      // Ignore already-freed WASM handle
    }
    activeSim = null;
  }
  activeSimRunId = 0;
  if (activeMounts) {
    try {
      activeMounts.free();
    } catch (_) {
      // Ignore already-freed WASM handle
    }
    activeMounts = null;
  }
  activeMountsRunId = 0;
}

function collectTransferable(list, seen, typedArray) {
  if (!typedArray || !(typedArray.buffer instanceof ArrayBuffer)) return;
  const wasmMemBuf = wasmInitOutput && wasmInitOutput.memory ? wasmInitOutput.memory.buffer : null;
  if (wasmMemBuf && typedArray.buffer === wasmMemBuf) return;
  if (!seen.has(typedArray.buffer)) {
    seen.add(typedArray.buffer);
    list.push(typedArray.buffer);
  }
}

async function ensureWasm(wasmUrl, id) {
  if (wasm && wasmUrlLoaded === wasmUrl) return wasm;
  postProgress(id, "init", 0.05, "Loading lidar-rs WebAssembly module...");
  const mod = await import(wasmUrl);
  wasmInitOutput = await mod.default();
  wasm = mod;
  wasmUrlLoaded = wasmUrl;
  return wasm;
}

async function handleRun(msg) {
  const { id, runId, wasmUrl, mode, bytes, filename, params } = msg;
  const mod = await ensureWasm(wasmUrl, id);
  if (cancelledIds.has(id)) {
    cancelledIds.delete(id);
    self.postMessage({ type: "cancelled", id });
    return;
  }

  const inputBytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const t0 = performance.now();

  let dimsLabel = "";
  postProgress(id, "inspect", 0.12, `Reading ${filename} metadata...`);
  try {
    const info = JSON.parse(mod.rasterInfo(inputBytes, filename));
    if (info && info.width && info.height) {
      dimsLabel = ` (${info.width}x${info.height})`;
    }
  } catch (_) {
    // Continue; the main pipeline call will surface any parse error cleanly.
  }

  if (cancelledIds.has(id)) {
    cancelledIds.delete(id);
    self.postMessage({ type: "cancelled", id });
    return;
  }

  const transfer = [];
  const seen = new Set();

  if (mode === "inundation") {
    const {
      kernelSize,
      minSize,
      minDepth,
      interval,
      rain,
      duration,
      step,
      breachMaxDepth,
      runoffCoeff,
      scsType2Mm,
      routingMode,
    } = params;

    const routingLabel = routingMode === "fsm" ? "Fill-Spill-Merge" : "level-set";
    postProgress(
      id,
      "simulate",
      0.28,
      `Running DEM filtering, depression delineation, and ${routingLabel} simulation on ${filename}${dimsLabel}...`,
    );

    const sim =
      typeof mod.simulateInundationAdvanced === "function"
        ? mod.simulateInundationAdvanced(
            inputBytes,
            filename,
            kernelSize,
            minSize,
            minDepth,
            interval,
            rain,
            duration,
            step,
            breachMaxDepth,
            runoffCoeff,
            scsType2Mm,
            routingMode,
          )
        : mod.simulateInundation(
            inputBytes,
            filename,
            kernelSize,
            minSize,
            minDepth,
            interval,
            rain,
            duration,
            step,
          );

    if (cancelledIds.has(id)) {
      sim.free();
      cancelledIds.delete(id);
      self.postMessage({ type: "cancelled", id });
      return;
    }

    postProgress(id, "extract", 0.84, "Extracting raster layers and hydrological telemetry...");

    releaseOutputs();
    activeSim = sim;
    activeSimRunId = runId;

    const hill = sim.hillshadeRgba();
    const depth = sim.depthRgba();
    const level = sim.levelRgba();
    const fill = sim.fillTimeRgba();
    const fillTimes = sim.fillTimeValues();
    const depIds = typeof sim.depressionIdValues === "function" ? sim.depressionIdValues() : null;
    const rawBounds = typeof sim.boundsWgs84 === "function" ? sim.boundsWgs84() : undefined;
    const boundsWgs84 = rawBounds && rawBounds.length === 4 ? Array.from(rawBounds) : null;

    collectTransferable(transfer, seen, hill);
    collectTransferable(transfer, seen, depth);
    collectTransferable(transfer, seen, level);
    collectTransferable(transfer, seen, fill);
    collectTransferable(transfer, seen, fillTimes);
    collectTransferable(transfer, seen, depIds);

    const result = {
      width: sim.width,
      height: sim.height,
      projection: sim.projection,
      boundsWgs84,
      maxLevel: sim.maxLevel,
      numRegions: sim.numRegions,
      numSteps: sim.numSteps,
      runoffCoeff: typeof sim.runoffCoeff === "number" ? sim.runoffCoeff : runoffCoeff,
      hill,
      depth,
      level,
      fill,
      fillTimes,
      depIds,
      depressionsJson: sim.depressionsJson(),
      ratingCurvesJson: typeof sim.ratingCurvesJson === "function" ? sim.ratingCurvesJson() : "[]",
      stepsJson: sim.stepsJson(),
      spillPathsJson: sim.spillPathsJson(),
      inundationCurveCsv:
        typeof sim.inundationCurveCsv === "function" ? sim.inundationCurveCsv() : "",
      fsmMassBalanceJson:
        typeof sim.fsmMassBalanceJson === "function" ? sim.fsmMassBalanceJson() : "null",
      fsmBasinsJson: typeof sim.fsmBasinsJson === "function" ? sim.fsmBasinsJson() : "[]",
      hasFsmBasinsCsv: typeof sim.fsmBasinsCsv === "function",
    };

    const elapsedMs = performance.now() - t0;
    postProgress(id, "done", 1.0, "Simulation complete.");
    self.postMessage(
      {
        type: "runResult",
        id,
        runId,
        mode: "inundation",
        version: mod.version(),
        elapsedMs,
        result,
      },
      transfer,
    );
  } else {
    const { minSize, minDepth, interval } = params;

    postProgress(id, "hillshade", 0.24, `Computing shaded relief for ${filename}${dimsLabel}...`);
    const hill = mod.hillshadeRgba(inputBytes, filename, 315.0, 45.0, 1.5);

    if (cancelledIds.has(id)) {
      cancelledIds.delete(id);
      self.postMessage({ type: "cancelled", id });
      return;
    }

    postProgress(
      id,
      "delineate",
      0.48,
      `Delineating nested mounts and canopies on ${filename}${dimsLabel}...`,
    );
    const mounts = mod.delineateMounts(inputBytes, filename, minSize, minDepth, interval);

    if (cancelledIds.has(id)) {
      mounts.free();
      cancelledIds.delete(id);
      self.postMessage({ type: "cancelled", id });
      return;
    }

    postProgress(id, "extract", 0.85, "Extracting mount hierarchy layers and attributes...");

    releaseOutputs();
    activeMounts = mounts;
    activeMountsRunId = runId;

    const level = mounts.levelRgba();
    const mountIds =
      typeof mounts.mountIdValues === "function"
        ? mounts.mountIdValues()
        : typeof mounts.depressionIdValues === "function"
          ? mounts.depressionIdValues()
          : null;
    const rawBounds = typeof mounts.boundsWgs84 === "function" ? mounts.boundsWgs84() : undefined;
    const boundsWgs84 = rawBounds && rawBounds.length === 4 ? Array.from(rawBounds) : null;
    const featuresJson =
      typeof mounts.mountsJson === "function" ? mounts.mountsJson() : mounts.depressionsJson();

    collectTransferable(transfer, seen, hill);
    collectTransferable(transfer, seen, level);
    collectTransferable(transfer, seen, mountIds);

    const result = {
      width: mounts.width,
      height: mounts.height,
      projection: typeof mounts.projection === "string" ? mounts.projection : "",
      boundsWgs84,
      maxLevel: mounts.maxLevel,
      numRegions: mounts.numRegions,
      hill,
      level,
      mountIds,
      featuresJson,
    };

    const elapsedMs = performance.now() - t0;
    postProgress(id, "done", 1.0, "Mount delineation complete.");
    self.postMessage(
      {
        type: "runResult",
        id,
        runId,
        mode: "mounts",
        version: mod.version(),
        elapsedMs,
        result,
      },
      transfer,
    );
  }
}

function handleExportGif(msg) {
  const { id, runId, delayMs, pingPong, maxDim, showSpills } = msg;
  if (cancelledIds.has(id)) {
    cancelledIds.delete(id);
    self.postMessage({ type: "cancelled", id });
    return;
  }
  if (!activeSim || activeSimRunId !== runId) {
    throw new Error("No active inundation simulation in worker; click Run simulation first.");
  }
  const t0 = performance.now();
  postProgress(id, "gif", 0.25, "Encoding animated inundation GIF in Web Worker...");
  const bytes = activeSim.encodeGif(delayMs, pingPong, maxDim, showSpills);
  if (cancelledIds.has(id)) {
    cancelledIds.delete(id);
    self.postMessage({ type: "cancelled", id });
    return;
  }
  const elapsedMs = performance.now() - t0;
  postProgress(id, "done", 1.0, "Animated GIF ready.");
  const transfer = [];
  collectTransferable(transfer, new Set(), bytes);
  self.postMessage(
    {
      type: "exportGifResult",
      id,
      runId,
      bytes,
      elapsedMs,
    },
    transfer,
  );
}

function yieldWorkerTick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Encodes pre-rendered RGBA frames (transferred zero-copy as ArrayBuffers) into a
 * looping GIF89a off the main thread, yielding periodically so cooperative cancel
 * messages can interrupt mid-encode without terminating the worker or losing WASM state.
 */
async function handleEncodeGifFrames(msg) {
  const { id, width, height, delayMs, frames, levelColors = [] } = msg;
  if (!Array.isArray(frames) || frames.length === 0) {
    throw new Error("No frames supplied for GIF encoding.");
  }
  const t0 = performance.now();
  const frameCount = frames.length;
  const nPix = width * height;
  const delayCs = Math.max(2, Math.round((delayMs || 150) / 10));

  const frameViews = frames.map((buf) => (buf instanceof Uint8Array ? buf : new Uint8Array(buf)));

  postProgress(id, "gif", 0.48, "Quantizing depression tree GIF palette in Web Worker...");

  // 1. Sample up to 16 evenly spaced frames to build a 5-5-5 RGB frequency histogram
  const counts = new Uint32Array(32768);
  const sumR = new Float64Array(32768);
  const sumG = new Float64Array(32768);
  const sumB = new Float64Array(32768);
  const sampleCount = Math.min(frameCount, 16);
  for (let s = 0; s < sampleCount; s++) {
    if (cancelledIds.has(id)) {
      cancelledIds.delete(id);
      self.postMessage({ type: "cancelled", id });
      return;
    }
    const fi = sampleCount === 1 ? 0 : Math.round((s * (frameCount - 1)) / (sampleCount - 1));
    const rgba = frameViews[fi];
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
    if ((s & 3) === 3) {
      await yieldWorkerTick();
    }
  }

  const buckets = [];
  for (let k = 0; k < 32768; k++) {
    if (counts[k] > 0) buckets.push(k);
  }
  buckets.sort((a, b) => counts[b] - counts[a]);

  const palette = new Uint8Array(256 * 3);
  let palLen = 0;
  for (const hex of levelColors) {
    if (palLen >= 256) break;
    const h = String(hex).replace("#", "");
    palette[palLen * 3] = parseInt(h.slice(0, 2), 16);
    palette[palLen * 3 + 1] = parseInt(h.slice(2, 4), 16);
    palette[palLen * 3 + 2] = parseInt(h.slice(4, 6), 16);
    palLen++;
  }
  const maxDataColors = 192;
  for (let i = 0; i < buckets.length && palLen < maxDataColors; i++) {
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

  writeBytes([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]); // GIF89a
  writeU16(width);
  writeU16(height);
  writeByte(0xf7);
  writeByte(0);
  writeByte(0);
  writeBytes(palette);
  writeBytes([
    0x21, 0xff, 0x0b, 0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30,
    0x03, 0x01, 0x00, 0x00, 0x00,
  ]);

  const HT_SIZE = 16384;
  const HT_MASK = HT_SIZE - 1;
  const htKey = new Int32Array(HT_SIZE);
  const htVal = new Int32Array(HT_SIZE);
  const htStamp = new Uint32Array(HT_SIZE);
  let stamp = 1;
  const subBlock = new Uint8Array(255);
  const indexed = new Uint8Array(nPix);

  for (let f = 0; f < frameCount; f++) {
    if (cancelledIds.has(id)) {
      cancelledIds.delete(id);
      self.postMessage({ type: "cancelled", id });
      return;
    }

    const rgba = frameViews[f];
    for (let i = 0, p = 0; i < nPix; i++, p += 4) {
      const r = rgba[p];
      const g = rgba[p + 1];
      const b = rgba[p + 2];
      const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
      indexed[i] = nearestColor(key, r, g, b);
    }

    writeBytes([0x21, 0xf9, 0x04, 0x04, delayCs & 0xff, (delayCs >> 8) & 0xff, 0x00, 0x00]);
    writeByte(0x2c);
    writeU16(0);
    writeU16(0);
    writeU16(width);
    writeU16(height);
    writeByte(0x00);
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
    writeByte(0x00);

    // Yield periodically so incoming cancel messages can preempt encoding and progress stays smooth.
    if ((f & 3) === 3 || f === frameCount - 1) {
      const frac = 0.5 + 0.48 * ((f + 1) / frameCount);
      postProgress(
        id,
        "gif",
        frac,
        `Encoding depression tree GIF frame ${f + 1}/${frameCount} in Web Worker...`,
      );
      await yieldWorkerTick();
    }
  }

  writeByte(0x3b);
  chunks.push(buf.subarray(0, pos));
  const totalLen = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(totalLen);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }

  const elapsedMs = performance.now() - t0;
  postProgress(id, "done", 1.0, "Depression tree GIF ready.");
  const transfer = [];
  collectTransferable(transfer, new Set(), out);
  self.postMessage(
    {
      type: "encodeGifFramesResult",
      id,
      bytes: out,
      elapsedMs,
    },
    transfer,
  );
}

function handleExportAsset(msg) {
  const { id, runId, asset, options = {} } = msg;
  if (cancelledIds.has(id)) {
    cancelledIds.delete(id);
    self.postMessage({ type: "cancelled", id });
    return;
  }
  let data;

  if (asset === "mountsGeojson" || asset === "mountLevelTif" || asset === "mountIdTif") {
    if (!activeMounts || activeMountsRunId !== runId) {
      throw new Error("No active mount delineation in worker; click Run simulation first.");
    }
    if (asset === "mountsGeojson") {
      postProgress(id, "export", 0.35, "Exporting mounts GeoJSON...");
      data = activeMounts.geojson();
    } else if (asset === "mountLevelTif") {
      postProgress(id, "export", 0.35, "Encoding mount level GeoTIFF...");
      data = activeMounts.levelTif();
    } else {
      postProgress(id, "export", 0.35, "Encoding mount ID GeoTIFF...");
      data = activeMounts.idTif();
    }
  } else {
    if (!activeSim || activeSimRunId !== runId) {
      throw new Error("No active inundation simulation in worker; click Run simulation first.");
    }
    switch (asset) {
      case "pngSnapshot":
        postProgress(id, "export", 0.35, "Rendering PNG snapshot in Web Worker...");
        data = activeSim.renderFramePng(
          options.stepIdx || 0,
          options.showScaleBar !== false,
          options.showLegend !== false,
          options.showSpills !== false,
        );
        break;
      case "depressionsGeojson":
        postProgress(id, "export", 0.35, "Polygonizing depressions to WGS 84 GeoJSON...");
        data = activeSim.depressionsGeojson();
        break;
      case "spillPathsGeojson":
        postProgress(id, "export", 0.35, "Reprojecting spill flow paths to WGS 84 GeoJSON...");
        data = activeSim.spillPathsGeojson();
        break;
      case "connectivityGraphJson":
        data = activeSim.connectivityGraphJson();
        break;
      case "connectivityGraphml":
        data = activeSim.connectivityGraphml();
        break;
      case "ratingCurvesCsv":
        data = activeSim.ratingCurvesCsv();
        break;
      case "inundationCurveCsv":
        data = activeSim.inundationCurveCsv();
        break;
      case "fsmBasinsCsv":
        data = activeSim.fsmBasinsCsv();
        break;
      case "levelTif":
        postProgress(id, "export", 0.35, "Encoding depression level GeoTIFF...");
        data = activeSim.levelTif();
        break;
      case "fillTimeTif":
        postProgress(id, "export", 0.35, "Encoding fill time GeoTIFF...");
        data = activeSim.fillTimeTif();
        break;
      case "catchmentHirTif":
        postProgress(id, "export", 0.35, "Encoding catchment hierarchy GeoTIFF...");
        data = activeSim.catchmentHirTif();
        break;
      case "sinkTif":
        postProgress(id, "export", 0.35, "Encoding sink GeoTIFF...");
        data = activeSim.sinkTif();
        break;
      default:
        throw new Error(`Unknown export asset: ${asset}`);
    }
  }

  const transfer = [];
  if (data instanceof Uint8Array) {
    collectTransferable(transfer, new Set(), data);
  }
  self.postMessage(
    {
      type: "exportAssetResult",
      id,
      runId,
      asset,
      data,
    },
    transfer,
  );
}

self.addEventListener("message", async (event) => {
  const msg = event.data;
  if (!msg || typeof msg !== "object") return;
  const { type, id } = msg;

  try {
    if (type === "init") {
      const mod = await ensureWasm(msg.wasmUrl, id || 0);
      self.postMessage({ type: "ready", id, version: mod.version() });
    } else if (type === "cancel") {
      if (typeof msg.targetId === "number") {
        cancelledIds.add(msg.targetId);
      }
    } else if (type === "release") {
      releaseOutputs();
      self.postMessage({ type: "released", id });
    } else if (type === "run") {
      await handleRun(msg);
    } else if (type === "exportGif") {
      handleExportGif(msg);
    } else if (type === "encodeGifFrames") {
      await handleEncodeGifFrames(msg);
    } else if (type === "exportAsset") {
      handleExportAsset(msg);
    } else {
      throw new Error(`Unknown worker message type: ${type}`);
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      id,
      message: errMsg(err),
    });
  } finally {
    if (typeof id === "number") {
      cancelledIds.delete(id);
    }
  }
});
