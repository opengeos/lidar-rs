# JavaScript / WebAssembly

`lidar-js` compiles [`lidar-rs`](https://github.com/opengeos/lidar-rs) to WebAssembly via `wasm-bindgen` for browsers and Node.js. All operations run completely in memory on `Uint8Array` GeoTIFF, GeoJSON, and GIF buffers with zero filesystem, GDAL, or C/C++ dependencies.

Try it right now with no installation on the [Playground](playground.md).

## Quick Start (Browser)

```js
import init, {
  version,
  rasterInfo,
  extractSinks,
  delineateDepressions,
  delineateMounts,
  simulateInundation,
} from "./wasm/lidar_js.js";

await init();
console.log("lidar-rs", version());

const demBytes = new Uint8Array(await (await fetch("dem.tif")).arrayBuffer());
const info = JSON.parse(rasterInfo(demBytes, "dem.tif"));
console.log(`${info.width}×${info.height}`, info.min, "→", info.max);

// Run the full hydrology & inundation pipeline in memory
const sim = simulateInundation(
  demBytes,
  "dem.tif",
  3,     // median filter kernel size (<= 1 to skip)
  1000,  // min_size (cells)
  0.3,   // min_depth (m)
  0.3,   // slicing interval (m)
  5.0,   // rain intensity (cm/h)
  20.0,  // rain duration (hours)
  1.0,   // time step (hours)
);

try {
  // Encode the animated shaded-relief inundation GIF on demand
  const gifBytes = sim.encodeGif(150, true, 400); // delay ms, ping-pong, max dim px
  const gifUrl = URL.createObjectURL(new Blob([gifBytes], { type: "image/gif" }));
  document.querySelector("#inundation-gif").src = gifUrl;

  // Render the shaded-relief preview onto an HTML5 <canvas>
  const canvas = document.querySelector("#hillshade");
  canvas.width = sim.width;
  canvas.height = sim.height;
  const ctx = canvas.getContext("2d");
  ctx.putImageData(
    new ImageData(new Uint8ClampedArray(sim.hillshadeRgba()), sim.width, sim.height),
    0,
    0,
  );

  // Inspect structured JSON outputs
  const depressions = JSON.parse(sim.depressionsJson());
  const steps = JSON.parse(sim.stepsJson());
  console.log(`Delineated ${depressions.length} depressions across ${sim.maxLevel} levels`);
  console.log(`Final inundated cells: ${steps.at(-1).inundated_cells}`);

  // Download or pass GeoJSON / GeoTIFF buffers directly to Leaflet, MapLibre, etc.
  const geojson = JSON.parse(sim.depressionsGeojson());
  const fillTimeTif = sim.fillTimeTif();
} finally {
  sim.free();
}
```

## Quick Start (Node.js)

Build the Node.js target with `wasm-pack` and call it directly from ES modules or CommonJS:

```sh
cd js
wasm-pack build --target nodejs --release --out-dir pkg-node
```

```js
import fs from "node:fs";
import * as lidar from "./pkg-node/lidar_js.js";

const demBytes = fs.readFileSync("../test_data/dem.tif");
const smoothed = lidar.medianFilter(demBytes, "dem.tif", 3);

const sinks = lidar.extractSinks(smoothed, "median.tif", 1000);
try {
  const deps = lidar.delineateDepressions(sinks.sinkTif(), "sink.tif", 1000, 0.3, 0.3);
  try {
    console.log("Regions:", deps.numRegions, "Max level:", deps.maxLevel);
    fs.writeFileSync("depressions.geojson", deps.geojson());
    fs.writeFileSync("depression_level.tif", deps.levelTif());
  } finally {
    deps.free();
  }
} finally {
  sinks.free();
}
```

## API Reference

### Raster Utilities & Filters

| Function | Signature | Returns |
|---|---|---|
| `version` | `()` | `string` — `lidar-rs` crate version |
| `rasterInfo` | `(bytes, filename)` | JSON string with `width`, `height`, `geotransform`, `projection`, `no_data`, `min`, `max` |
| `meanFilter` | `(bytes, filename, kernelSize)` | `Uint8Array` — filtered GeoTIFF bytes |
| `medianFilter` | `(bytes, filename, kernelSize)` | `Uint8Array` — filtered GeoTIFF bytes |
| `gaussianFilter` | `(bytes, filename, sigma)` | `Uint8Array` — filtered GeoTIFF bytes |
| `flipDem` | `(bytes, filename, delta)` | `Uint8Array` — inverted GeoTIFF bytes |
| `hillshadeRgba` | `(bytes, filename, azimuthDeg, elevationDeg, zFactor)` | `Uint8Array` — `width × height × 4` RGBA pixels for `ImageData` |
| `imageToCog` | `(bytes, filename, compression)` | `Uint8Array` — Cloud Optimized GeoTIFF bytes |
| `polygonizeToGeojson` | `(bytes, filename, fieldName)` | `string` — GeoJSON FeatureCollection |

### Hydrology & Delineation Pipelines

| Function | Signature | Return Object |
|---|---|---|
| `extractSinks` | `(bytes, filename, minSize)` | `SinksOutput` (`width`, `height`, `depressionsJson()`, `sinkTif()`, `refinedDemTif()`, `depthTif()`, `regionTif()`, `geojson()`, `depthRgba()`) |
| `delineateDepressions` | `(sinkBytes, filename, minSize, minDepth, interval)` | `DepressionsOutput` (`width`, `height`, `maxLevel`, `numRegions`, `depressionsJson()`, `idTif()`, `levelTif()`, `geojson()`, `levelRgba()`) |
| `delineateMounts` | `(dsmBytes, filename, minSize, minHeight, interval)` | `DepressionsOutput` (same fields as `delineateDepressions`) |
| `delineateCatchments` | `(demBytes, sinkOrRegionBytes, demFilename, sinkFilename)` | `CatchmentsOutput` (`recordsJson()`, `watershedTif()`, `geojson()`) |
| `simulateInundation` | `(demBytes, filename, kernelSize, minSize, minDepth, interval, rainIntensityCmH, rainDurationH, timeStepH)` | `InundationOutput` (`width`, `height`, `projection`, `maxLevel`, `numRegions`, `numSteps`, `depressionsJson()`, `catchmentHierarchyJson()`, `stepsJson()`, `encodeGif(frameDelayMs, pingPong, maxDim)`, `fillTimeTif()`, `catchmentHirTif()`, `sinkTif()`, `levelTif()`, `depressionsGeojson()`, `hillshadeRgba()`, `depthRgba()`, `levelRgba()`, `fillTimeRgba()`, `fillTimeValues()`) |

!!! tip "Memory Management"
    `SinksOutput`, `DepressionsOutput`, `CatchmentsOutput`, and `InundationOutput` hold WebAssembly heap allocations. Call `.free()` (ideally in a `finally` block) once you have copied the buffers or strings you need.

## Building from Source

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

cd js
# Browser ES module
wasm-pack build --target web --release --out-dir pkg

# Node.js module + end-to-end test suite
wasm-pack build --target nodejs --release --out-dir pkg-node
node test.mjs
```

## License Note

The `lidar-rs` source code is licensed under MIT, while the compiled `.wasm` binary statically links [`richdem`](https://github.com/opengeos/richdem-rs) (`GPL-3.0-only`). Accordingly, `lidar-js` declares `GPL-3.0-only` for distributed WebAssembly artifacts; see [NOTICE](https://github.com/opengeos/lidar-rs/blob/main/NOTICE).
