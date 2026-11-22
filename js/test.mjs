import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import * as lidar from "./pkg-node/lidar_js.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootCargoToml = fs.readFileSync(path.resolve(__dirname, "..", "Cargo.toml"), "utf8");
const jsCargoToml = fs.readFileSync(path.resolve(__dirname, "Cargo.toml"), "utf8");
const wsVersion = rootCargoToml.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
const jsVersion = jsCargoToml.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
assert.ok(wsVersion, "expected workspace version in root Cargo.toml");
assert.equal(jsVersion, wsVersion, "js/Cargo.toml version must match workspace version");

console.log(`lidar-js version: ${lidar.version()}`);
assert.equal(lidar.version(), wsVersion);

const testDataDir = path.resolve(__dirname, "..", "test_data");
const demBytes = new Uint8Array(fs.readFileSync(path.join(testDataDir, "dem.tif")));
const dsmBytes = new Uint8Array(fs.readFileSync(path.join(testDataDir, "dsm.tif")));

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

// 2. Extract sinks + nested depressions + catchments
const sinks = lidar.extractSinks(medBytes, "median.tif", 1000);
const sinkDeps = JSON.parse(sinks.depressionsJson());
assert.ok(sinkDeps.length >= 1, "expected at least 1 sink");
const sinkGeo = JSON.parse(sinks.geojson());
assert.equal(sinkGeo.type, "FeatureCollection");
assert.ok(sinkGeo.features.length >= 1);

const cats = lidar.delineateCatchments(
  sinks.refinedDemTif(),
  sinks.regionTif(),
  "dem.tif",
  "region.tif",
);
const catRecords = JSON.parse(cats.recordsJson());
assert.equal(catRecords.length, sinkDeps.length);
cats.free();

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

// 4. Full in-memory inundation dynamics + lazy animated GIF encoding
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
);
assert.equal(sim.numSteps, 5);
const steps = JSON.parse(sim.stepsJson());
assert.equal(steps.length, 5);
assert.ok(steps[steps.length - 1].inundated_cells > 0);
for (let i = 1; i < steps.length; i++) {
  assert.ok(steps[i].inundated_cells >= steps[i - 1].inundated_cells);
}
assert.equal(sim.hillshadeRgba().length, 400 * 400 * 4);
const ftVals = sim.fillTimeValues();
assert.ok(ftVals instanceof Float64Array, "fillTimeValues must preserve f64 precision");
assert.equal(ftVals.length, 400 * 400);
assert.ok(ftVals.some((v) => v >= 0 && Number.isFinite(v)));
assert.ok(sim.fillTimeTif().length > 100);
assert.ok(sim.catchmentHirTif().length > 100);
assert.equal(JSON.parse(sim.depressionsGeojson()).type, "FeatureCollection");
const spillPaths = JSON.parse(sim.spillPathsJson());
assert.ok(spillPaths.length >= 3, `expected >= 3 spill paths, got ${spillPaths.length}`);
const spillGeo = JSON.parse(sim.spillPathsGeojson());
assert.equal(spillGeo.type, "FeatureCollection");
assert.equal(spillGeo.features.length, spillPaths.length);
const [firstLon, firstLat] = spillGeo.features[0].geometry.coordinates[0];
assert.ok(
  firstLon >= -180 && firstLon <= 180 && firstLat >= -90 && firstLat <= 90,
  `expected WGS 84 lon/lat coordinates, got [${firstLon}, ${firstLat}]`,
);

const gif = sim.encodeGif(150, true, 200, true);
const header = new TextDecoder().decode(gif.slice(0, 6));
assert.equal(header, "GIF89a");

// 5. New v0.5.0 WASM exports: depressionIdValues, rating curves, connectivity graph, PNG frame, breaching
const depIds = sim.depressionIdValues();
assert.ok(depIds instanceof Int32Array);
assert.equal(depIds.length, 400 * 400);
assert.ok(depIds.some((id) => id > 0));
const ratingCurves = JSON.parse(sim.ratingCurvesJson());
assert.ok(ratingCurves.length >= 2);
assert.ok(sim.ratingCurvesCsv().startsWith("dep_id,"));
const graph = JSON.parse(sim.connectivityGraphJson());
assert.ok(graph.nodes.length >= 2);
assert.ok(sim.connectivityGraphml().includes("<graphml"));
assert.ok(sim.inundationCurveCsv().startsWith("step,time_hours,runoff_mm,"));
const pngBytes = sim.renderFramePng(3, true, true, true);
assert.equal(pngBytes[0], 0x89);
assert.equal(pngBytes[1], 0x50);
assert.equal(pngBytes[2], 0x4e);
assert.equal(pngBytes[3], 0x47);
console.log(
  `WASM inundation OK: ${sim.numRegions} region(s), maxLevel=${sim.maxLevel}, ` +
    `${sim.numSteps} steps, ${spillPaths.length} spill path(s), lazy GIF ${gif.length} bytes, PNG ${pngBytes.length} bytes`,
);
sim.free();

// 6. Hydro-conditioning breaching + advanced forced inundation
const breachedBytes = lidar.breachDem(demBytes, "dem.tif", "selective", 0.5, 15, false);
assert.ok(breachedBytes.length > 1000);
const simAdv = lidar.simulateInundationAdvanced(
  demBytes,
  "dem.tif",
  3,
  1000,
  0.3,
  0.3,
  5.0,
  10.0,
  2.0,
  0.4,
  0.75,
  undefined,
);
assert.equal(simAdv.runoffCoeff, 0.75);
assert.equal(simAdv.numSteps, 5);
simAdv.free();

console.log("ALL lidar-js WASM TESTS PASSED!");
