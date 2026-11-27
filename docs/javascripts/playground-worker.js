// Dedicated ES module Web Worker for the lidar-rs WebAssembly Playground.
// Keeps DEM delineation, inundation simulation, GeoJSON/GeoTIFF exports, and
// GIF encoding completely off the browser main thread with progress reporting,
// zero-copy ArrayBuffer transfers, and instant cancellation via termination.

import { encodeGif89a } from "./gif-encoder.js";

let wasm = null;
let wasmInitOutput = null;
let wasmUrlLoaded = null;
let activeSim = null;
let activeSimRunId = 0;
let activeMounts = null;
let activeMountsRunId = 0;
const inFlightIds = new Set();
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

    let result;
    try {
      postProgress(id, "extract", 0.84, "Extracting raster layers and hydrological telemetry...");

      const hill = sim.hillshadeRgba();
      const depth = sim.depthRgba();
      const level = sim.levelRgba();
      const fill = sim.fillTimeRgba();
      const fillTimes = sim.fillTimeValues();
      const depIds =
        typeof sim.depressionIdValues === "function" ? sim.depressionIdValues() : null;
      const rawBounds = typeof sim.boundsWgs84 === "function" ? sim.boundsWgs84() : undefined;
      const boundsWgs84 = rawBounds && rawBounds.length === 4 ? Array.from(rawBounds) : null;

      collectTransferable(transfer, seen, hill);
      collectTransferable(transfer, seen, depth);
      collectTransferable(transfer, seen, level);
      collectTransferable(transfer, seen, fill);
      collectTransferable(transfer, seen, fillTimes);
      collectTransferable(transfer, seen, depIds);

      result = {
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
        ratingCurvesJson:
          typeof sim.ratingCurvesJson === "function" ? sim.ratingCurvesJson() : "[]",
        stepsJson: sim.stepsJson(),
        spillPathsJson: sim.spillPathsJson(),
        inundationCurveCsv:
          typeof sim.inundationCurveCsv === "function" ? sim.inundationCurveCsv() : "",
        fsmMassBalanceJson:
          typeof sim.fsmMassBalanceJson === "function" ? sim.fsmMassBalanceJson() : "null",
        fsmBasinsJson: typeof sim.fsmBasinsJson === "function" ? sim.fsmBasinsJson() : "[]",
        hasFsmBasinsCsv: typeof sim.fsmBasinsCsv === "function",
      };
    } catch (err) {
      sim.free();
      throw err;
    }

    releaseOutputs();
    activeSim = sim;
    activeSimRunId = runId;

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

    let result;
    try {
      postProgress(id, "extract", 0.85, "Extracting mount hierarchy layers and attributes...");

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

      result = {
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
    } catch (err) {
      mounts.free();
      throw err;
    }

    releaseOutputs();
    activeMounts = mounts;
    activeMountsRunId = runId;

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
  const frameViews = frames.map((buf) => (buf instanceof Uint8Array ? buf : new Uint8Array(buf)));

  const out = await encodeGif89a({
    frameCount: frameViews.length,
    getFrameRgba: (fi) => frameViews[fi],
    width,
    height,
    delayMs,
    levelColors,
    isCancelled: () => cancelledIds.has(id),
    onProgress: (frac, message) => postProgress(id, "gif", frac, message),
    yieldTick: yieldWorkerTick,
  });

  if (!out || cancelledIds.has(id)) {
    cancelledIds.delete(id);
    self.postMessage({ type: "cancelled", id });
    return;
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

  if (type === "cancel") {
    if (typeof msg.targetId === "number" && inFlightIds.has(msg.targetId)) {
      cancelledIds.add(msg.targetId);
    }
    return;
  }

  if (typeof id === "number") {
    inFlightIds.add(id);
  }
  try {
    if (type === "init") {
      const mod = await ensureWasm(msg.wasmUrl, id || 0);
      self.postMessage({ type: "ready", id, version: mod.version() });
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
      inFlightIds.delete(id);
      cancelledIds.delete(id);
    }
  }
});
