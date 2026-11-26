# `lidar-wasm` (WebAssembly bindings for `lidar-rs`)

[![npm](https://img.shields.io/npm/v/lidar-wasm.svg)](https://www.npmjs.com/package/lidar-wasm)

Run DEM filtering, selective depression breaching, nested surface depression and mount delineation, catchment hierarchy analysis, fill-and-spill inundation simulation (static and dynamic Fill-Spill-Merge routing), stage-storage rating curves, connectivity graphs, and PNG/GIF rendering directly in **browsers, Web Workers, and Node.js** via WebAssembly with zero GDAL, zero Python, and zero server dependencies.

## Installation

```sh
npm install lidar-wasm
```

## Building

```sh
# Browser ES module (outputs to ./pkg)
npm run build:web

# Node.js module (outputs to ./pkg-node)
npm run build:node

# Bundler target for Vite / Webpack / Rollup (outputs to ./pkg-bundler)
npm run build:bundler

# Build all targets and run the Node.js verification suite
npm run build
npm test
```

## Browser Usage (ES Module)

```ts
import init, {
  simulateInundationAdvanced,
  type NestedDepression,
  type DepressionsFeatureCollection,
} from "lidar-wasm";

await init();

const demBytes = new Uint8Array(await (await fetch("dem.tif")).arrayBuffer());
const sim = simulateInundationAdvanced(
  demBytes,
  "dem.tif",
  3,       // median filter kernel size (<= 1 to skip)
  1000,    // min_size (pixels)
  0.3,     // min_depth (m)
  0.3,     // slicing interval (m)
  5.0,     // rain intensity (cm/h)
  20.0,    // rain duration (h)
  1.0,     // time step (h)
  null,    // optional selective breach max_depth (m)
  1.0,     // runoff coefficient C in (0, 1]
  null,    // optional SCS Type II storm depth (mm)
  "fsm",   // "static" or "fsm"
);

try {
  const depressions: NestedDepression[] = JSON.parse(sim.depressionsJson());
  const geojson: DepressionsFeatureCollection = JSON.parse(sim.depressionsGeojson());
  const bounds = sim.boundsWgs84(); // [minLon, minLat, maxLon, maxLat] or undefined

  const gifBytes = sim.encodeGif(150, true, 400, true);
  const gifUrl = URL.createObjectURL(new Blob([gifBytes], { type: "image/gif" }));
  console.log({ count: depressions.length, features: geojson.features.length, bounds, gifUrl });
} finally {
  sim.free();
}
```

## Dedicated Web Worker Usage

For large DEMs and GIF encoding, run `lidar-wasm` inside a module Web Worker so the browser UI stays responsive and long runs can be cancelled cleanly with `worker.terminate()`.

```ts
// worker.ts
import init, { simulateInundationAdvanced } from "lidar-wasm";

const ready = init();

self.addEventListener("message", async (event: MessageEvent) => {
  const { id, demBytes, filename } = event.data;
  await ready;
  self.postMessage({ type: "progress", id, stage: "simulate", fraction: 0.3 });

  const sim = simulateInundationAdvanced(
    demBytes, filename, 3, 1000, 0.3, 0.3, 5.0, 20.0, 1.0, null, 1.0, null, "static",
  );
  try {
    const hillshade = sim.hillshadeRgba();
    const fillTimes = sim.fillTimeValues();
    self.postMessage(
      {
        type: "result",
        id,
        width: sim.width,
        height: sim.height,
        hillshade,
        fillTimes,
        depressionsJson: sim.depressionsJson(),
      },
      { transfer: [hillshade.buffer, fillTimes.buffer] },
    );
  } finally {
    sim.free();
  }
});
```

```ts
// main.ts
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
worker.postMessage({ id: 1, demBytes, filename: "dem.tif" });
// Instant cancellation at any time:
// worker.terminate();
```

## Node.js Usage

```ts
import fs from "node:fs";
import * as lidar from "lidar-wasm/node";

const demBytes = new Uint8Array(fs.readFileSync("../test_data/dem.tif"));
const sim = lidar.simulateInundation(demBytes, "dem.tif", 3, 1000, 0.2, 0.3, 5.0, 10.0, 2.0);
try {
  fs.writeFileSync("depressions.geojson", sim.depressionsGeojson());
  fs.writeFileSync("spill_paths.geojson", sim.spillPathsGeojson());
  fs.writeFileSync("inundation.gif", sim.encodeGif(150, true, 400, true));
} finally {
  sim.free();
}
```

## Memory Management

`SinksOutput`, `DepressionsOutput`, `CatchmentsOutput`, and `InundationOutput` wrap WebAssembly linear memory allocations. Always call `.free()` (ideally inside a `finally` block) once you have copied the buffers or strings you need.

## TypeScript Definitions

Complete TypeScript declarations for every exported function, output class, and parsed JSON/GeoJSON structure live in [`lidar_js.d.ts`](./lidar_js.d.ts) and are referenced via `"types": "./lidar_js.d.ts"` in [`package.json`](./package.json).

## License

`lidar-rs` source code is MIT-licensed. Compiled `.wasm` binaries statically link `richdem` (`GPL-3.0-only`) and are distributed under the GNU General Public License v3.0 (`GPL-3.0-only`). See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
