# JavaScript / WebAssembly

`lidar-js` compiles [`lidar-rs`](https://github.com/opengeos/lidar-rs) to WebAssembly via `wasm-bindgen` for browsers and Node.js. All operations execute completely in memory on `Uint8Array` GeoTIFF, GeoJSON, GraphML, PNG, and GIF buffers with zero filesystem, GDAL, or C/C++ dependencies.

Try it right now with zero installation on the interactive [WebAssembly Playground](playground.md).

[![Interactive Depression Inspector and Stage-Volume Rating Curve in the lidar-rs WebAssembly Playground](https://files.opengeos.org/lidar/images/playground_inspector.png)](playground.md)

## Quick Start (Browser)

```js
import init, {
  version,
  rasterInfo,
  simulateInundationAdvanced,
} from "./wasm/lidar_js.js";

await init();
console.log("lidar-rs", version());

const demBytes = new Uint8Array(
  await (await fetch("https://files.opengeos.org/lidar/data/dem.tif")).arrayBuffer(),
);
const info = JSON.parse(rasterInfo(demBytes, "dem.tif"));
console.log(`${info.width}x${info.height}`, info.min, "->", info.max);

// Run the full hydrology, rating-curve, and connectivity-graph pipeline in memory
const sim = simulateInundationAdvanced(
  demBytes,
  "dem.tif",
  3,     // median filter kernel size (<= 1 to skip)
  1000,  // min_size (cells)
  0.3,   // min_depth (m)
  0.3,   // slicing interval (m)
  5.0,   // rain intensity (cm/h)
  50.0,  // rain duration (hours)
  1.0,   // time step (hours)
  0.0,   // optional selective breach max_depth in m (0 / null disables breaching)
  0.85,  // effective runoff coefficient C in (0..=1]
  null,  // optional SCS Type II total precipitation in mm (overrides constant intensity)
);

try {
  // Inspect structured JSON outputs (24-column morphometry, rating curves, connectivity graph)
  const depressions = JSON.parse(sim.depressionsJson());
  const ratingCurves = JSON.parse(sim.ratingCurvesJson());
  const graph = JSON.parse(sim.connectivityGraphJson());
  const graphmlXml = sim.connectivityGraphml();
  const curveCsv = sim.inundationCurveCsv();

  console.log(
    `Delineated ${depressions.length} depressions (${sim.maxLevel} levels), ` +
    `${graph.edges.length} edges, ${ratingCurves.length} rating-curve points`,
  );

  // Render a single publication PNG frame with scale bar, legend, and spill arrows
  const pngBytes = sim.renderFramePng(25, true, true, true);
  const pngUrl = URL.createObjectURL(new Blob([pngBytes], { type: "image/png" }));
  document.querySelector("#snapshot-png").src = pngUrl;

  // Or encode the animated shaded-relief inundation GIF on demand
  const gifBytes = sim.encodeGif(150, true, 400, true);
  const gifUrl = URL.createObjectURL(new Blob([gifBytes], { type: "image/gif" }));
  document.querySelector("#inundation-gif").src = gifUrl;

  // Export GeoJSON / GeoTIFF buffers for MapLibre, Leaflet, or client-side download
  const spillGeojson = JSON.parse(sim.spillPathsGeojson());
  const fillTimeTif = sim.fillTimeTif();
  console.log(
    `Exports ready: ${spillGeojson.features.length} spill paths, ` +
    `fill_time.tif (${fillTimeTif.byteLength} B), ` +
    `GraphML (${graphmlXml.length} chars), curve CSV (${curveCsv.length} chars)`,
  );
} finally {
  sim.free();
}
```

## Quick Start (Node.js)

Build the Node.js target with `wasm-pack` and call it directly from ES modules:

```sh
cd js
wasm-pack build --target nodejs --release --out-dir pkg-node
```

```js
import fs from "node:fs";
import * as lidar from "./pkg-node/lidar_js.js";

const demBytes = fs.readFileSync("../test_data/dem.tif");
const breached = lidar.breachDem(demBytes, "dem.tif", "selective", 1.5, 40, false);

const sim = lidar.simulateInundationAdvanced(
  breached, "breached.tif", 3, 1000, 0.3, 0.3, 5.0, 50.0, 1.0, null, 1.0, null,
);
try {
  fs.writeFileSync("rating_curves.csv", sim.ratingCurvesCsv());
  fs.writeFileSync("connectivity_graph.graphml", sim.connectivityGraphml());
  fs.writeFileSync("spill_paths.geojson", sim.spillPathsGeojson());
  fs.writeFileSync("frame_25.png", sim.renderFramePng(25, true, true, true));
} finally {
  sim.free();
}
```

## API Reference

### Raster Utilities, Filters & Hydro-Conditioning

| Function | Signature | Returns |
| --- | --- | --- |
| `version` | `()` | `string`: `lidar-rs` crate version |
| `rasterInfo` | `(bytes, filename)` | JSON string with `width`, `height`, `geotransform`, `projection`, `no_data`, `min`, `max` |
| `meanFilter` | `(bytes, filename, kernelSize)` | `Uint8Array`: filtered GeoTIFF bytes |
| `medianFilter` | `(bytes, filename, kernelSize)` | `Uint8Array`: filtered GeoTIFF bytes |
| `gaussianFilter` | `(bytes, filename, sigma)` | `Uint8Array`: filtered GeoTIFF bytes |
| `flipDem` | `(bytes, filename, delta)` | `Uint8Array`: inverted GeoTIFF bytes |
| `breachDem` | `(bytes, filename, mode, maxDepth?, maxLength?, fillRemaining)` | `Uint8Array`: Lindsay (2016) breached GeoTIFF bytes (`"complete"`, `"selective"`, or `"constrained"`, resolving mode defaults via `BreachOptions::for_mode`) |
| `hillshadeRgba` | `(bytes, filename, azimuthDeg, elevationDeg, zFactor)` | `Uint8Array`: `width * height * 4` RGBA pixels for `ImageData` |
| `imageToCog` | `(bytes, filename, compression)` | `Uint8Array`: Cloud Optimized GeoTIFF bytes |
| `polygonizeToGeojson` | `(bytes, filename, fieldName)` | `string`: GeoJSON FeatureCollection |

### Hydrology, Rating Curves & Connectivity Pipelines

| Function | Signature | Return Object |
| --- | --- | --- |
| `extractSinks` | `(bytes, filename, minSize)` | `SinksOutput` (`width`, `height`, `depressionsJson()`, `sinkTif()`, `refinedDemTif()`, `depthTif()`, `regionTif()`, `geojson()`, `depthRgba()`) |
| `delineateDepressions` | `(sinkBytes, filename, minSize, minDepth, interval)` | `DepressionsOutput` (`width`, `height`, `maxLevel`, `numRegions`, `depressionsJson()`, `idTif()`, `levelTif()`, `geojson()`, `levelRgba()`) |
| `delineateMounts` | `(dsmBytes, filename, minSize, minHeight, interval)` | `DepressionsOutput` (same fields as `delineateDepressions`) |
| `delineateCatchments` | `(demBytes, sinkOrRegionBytes, demFilename, sinkFilename)` | `CatchmentsOutput` (`recordsJson()`, `watershedTif()`, `geojson()`) |
| `simulateInundation` | `(demBytes, filename, kernelSize, minSize, minDepth, interval, rainIntensityCmH, rainDurationH, timeStepH)` | `InundationOutput` |
| `simulateInundationAdvanced` | `(demBytes, filename, kernelSize, minSize, minDepth, interval, rainIntensityCmH, rainDurationH, timeStepH, breachMaxDepth?, runoffCoeff, scsType2Mm?)` | `InundationOutput` with optional selective breaching, runoff coefficient `C`, and SCS Type II storm hyetograph |

**`InundationOutput` methods:**

- **Metadata & telemetry:** `width`, `height`, `projection`, `maxLevel`, `numRegions`, `numSteps`, `runoffCoeff`
- **Tabular & graph exports:** `depressionsJson()`, `catchmentHierarchyJson()`, `stepsJson()`, `inundationCurveCsv()`, `spillPathsJson()`, `spillPathsGeojson()`, `ratingCurvesJson()`, `ratingCurvesCsv()`, `connectivityGraphJson()`, `connectivityGraphml()`
- **Raster & vector buffers:** `depressionsGeojson()`, `fillTimeTif()`, `catchmentHirTif()`, `sinkTif()`, `levelTif()`, `depressionIdValues()`, `fillTimeValues()`
- **Canvas & publication rendering:** `hillshadeRgba()`, `depthRgba()`, `levelRgba()`, `fillTimeRgba()`, `renderFramePng(stepIndex, showScaleBar, showLegend, showSpillPaths?)`, `encodeGif(frameDelayMs, pingPong, maxDim, showSpillPaths?)`

!!! tip "Memory Management"
    `SinksOutput`, `DepressionsOutput`, `CatchmentsOutput`, and `InundationOutput` hold WebAssembly heap allocations. Call `.free()` (ideally in a `finally` block) once you have copied the buffers or strings you need.

## Building from Source

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

cd js
# Browser ES module
wasm-pack build --target web --release --out-dir pkg -- --locked

# Node.js module + end-to-end test suite
wasm-pack build --target nodejs --release --out-dir pkg-node -- --locked
node test.mjs
```

## License Note

The `lidar-rs` source code is licensed under MIT, while the compiled `.wasm` binary statically links [`richdem`](https://github.com/opengeos/richdem-rs) (`GPL-3.0-only`). Accordingly, `lidar-js` declares `GPL-3.0-only` for distributed WebAssembly artifacts; see [NOTICE](https://github.com/opengeos/lidar-rs/blob/main/NOTICE).
