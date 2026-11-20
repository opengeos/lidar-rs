import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import * as lidar from "./pkg-node/lidar_js.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const testDataDir = path.resolve(__dirname, "..", "test_data");
const demBytes = new Uint8Array(fs.readFileSync(path.join(testDataDir, "dem.tif")));
const dsmBytes = new Uint8Array(fs.readFileSync(path.join(testDataDir, "dsm.tif")));

console.log(`lidar-js version: ${lidar.version()}`);
assert.equal(lidar.version(), "0.3.0");

// 1. Raster info + filters + hillshade + COG
const info = JSON.parse(lidar.rasterInfo(demBytes, "dem.tif"));
assert.equal(info.width, 400);
assert.equal(info.height, 400);
assert.ok(info.max > info.min);

const medBytes = lidar.medianFilter(demBytes, "dem.tif", 3);
assert.ok(medBytes.length > 1000);

const hs = lidar.hillshadeRgba(medBytes, "median.tif", 135, 45, 1.0);
assert.equal(hs.length, 400 * 400 * 4);

const cogBytes = lidar.imageToCog(demBytes, "dem.tif", "DEFLATE");
assert.ok(cogBytes.length > 1000);

// 2. Extract sinks + nested depressions
const sinks = lidar.extractSinks(medBytes, "median.tif", 1000);
const sinkDeps = JSON.parse(sinks.depressionsJson());
assert.ok(sinkDeps.length >= 1, "expected at least 1 sink");
const sinkGeo = JSON.parse(sinks.geojson());
assert.equal(sinkGeo.type, "FeatureCollection");
assert.ok(sinkGeo.features.length >= 1);

const deps = lidar.delineateDepressions(sinks.sinkTif(), "sink.tif", 1000, 0.3, 0.3);
assert.ok(deps.maxLevel >= 2, `expected nested levels >= 2, got ${deps.maxLevel}`);
const nestedDeps = JSON.parse(deps.depressionsJson());
assert.ok(nestedDeps.length >= 2);
sinks.free();
deps.free();

// 3. Delineate mounts from DSM
const mounts = lidar.delineateMounts(dsmBytes, "dsm.tif", 1000, 0.3, 0.3);
assert.ok(mounts.maxLevel >= 2);
assert.ok(JSON.parse(mounts.depressionsJson()).length >= 2);
mounts.free();

// 4. Full in-memory inundation dynamics + animated GIF
const sim = lidar.simulateInundation(
  demBytes,
  "dem.tif",
  3,
  1000,
  0.2,
  0.3,
  5.0,
  10.0,
  2.0,
  150,
  true,
  200,
);
assert.equal(sim.frameCount, 9); // 5 rising + 4 falling
const steps = JSON.parse(sim.stepsJson());
assert.equal(steps.length, 5);
assert.ok(steps[steps.length - 1].inundated_cells > 0);
for (let i = 1; i < steps.length; i++) {
  assert.ok(steps[i].inundated_cells >= steps[i - 1].inundated_cells);
}
const gif = sim.gifBytes();
const header = new TextDecoder().decode(gif.slice(0, 6));
assert.equal(header, "GIF89a");
assert.equal(sim.hillshadeRgba().length, 400 * 400 * 4);
const ftVals = sim.fillTimeValues();
assert.equal(ftVals.length, 400 * 400);
assert.ok(ftVals.some((v) => v >= 0 && Number.isFinite(v)));
assert.ok(sim.fillTimeTif().length > 100);
assert.ok(sim.catchmentHirTif().length > 100);
assert.equal(JSON.parse(sim.depressionsGeojson()).type, "FeatureCollection");
console.log(
  `WASM inundation OK: ${sim.numRegions} region(s), maxLevel=${sim.maxLevel}, ` +
    `${sim.frameCount} GIF frames (${sim.gifWidth}x${sim.gifHeight}, ${gif.length} bytes)`,
);
sim.free();

console.log("ALL lidar-js WASM TESTS PASSED!");
