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

const meanBytes = lidar.meanFilter(demBytes, "dem.tif", 3);
assert.ok(meanBytes.length > 1000);
const meanInfo = JSON.parse(lidar.rasterInfo(meanBytes, "mean.tif"));
assert.equal(meanInfo.width, 400);
assert.equal(meanInfo.height, 400);

const gaussBytes = lidar.gaussianFilter(demBytes, "dem.tif", 1.0);
assert.ok(gaussBytes.length > 1000);

const flippedBytes = lidar.flipDem(demBytes, "dem.tif", 100.0);
assert.ok(flippedBytes.length > 1000);
const flippedInfo = JSON.parse(lidar.rasterInfo(flippedBytes, "dem_flip.tif"));
assert.ok(flippedInfo.min >= 99.9, `expected flipped min >= 100, got ${flippedInfo.min}`);

const hs = lidar.hillshadeRgba(medBytes, "median.tif", 135, 45, 1.0);
assert.equal(hs.length, 400 * 400 * 4);

const cogBytes = lidar.imageToCog(demBytes, "dem.tif", "DEFLATE");
assert.ok(cogBytes.length > 1000);

function firstPolygonCoord(geom) {
  if (geom.type === "Polygon") return geom.coordinates[0][0];
  if (geom.type === "MultiPolygon") return geom.coordinates[0][0][0];
  throw new Error(`unexpected geometry type: ${geom.type}`);
}

function assertWgs84Coord([lon, lat], label) {
  assert.ok(
    lon >= -180 && lon <= 180 && lat >= -90 && lat <= 90,
    `${label}: expected WGS 84 lon/lat coordinates, got [${lon}, ${lat}]`,
  );
}

// 2. Extract sinks + nested depressions + catchments + polygonizeToGeojson
const sinks = lidar.extractSinks(medBytes, "median.tif", 1000);
const sinkDeps = JSON.parse(sinks.depressionsJson());
assert.ok(sinkDeps.length >= 1, "expected at least 1 sink");
const sinkGeo = JSON.parse(sinks.geojson());
assert.equal(sinkGeo.type, "FeatureCollection");
assert.equal(sinkGeo.crs, undefined, "RFC 7946 GeoJSON should omit deprecated crs member");
assert.ok(sinkGeo.features.length >= 1);
assertWgs84Coord(firstPolygonCoord(sinkGeo.features[0].geometry), "sinks.geojson");
for (const key of ["region_id", "area", "volume", "avg_depth", "max_depth", "min_elev", "max_elev"]) {
  assert.equal(
    typeof sinkGeo.features[0].properties[key],
    "number",
    `sinks.geojson missing numeric property ${key}`,
  );
}

const polyGeo = JSON.parse(lidar.polygonizeToGeojson(sinks.regionTif(), "region.tif", "region_id"));
assert.equal(polyGeo.type, "FeatureCollection");
assert.equal(polyGeo.crs, undefined, "polygonizeToGeojson should omit deprecated crs member");
assert.ok(polyGeo.features.length >= 1);
assertWgs84Coord(firstPolygonCoord(polyGeo.features[0].geometry), "polygonizeToGeojson");

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
const depsGeoStandalone = JSON.parse(deps.geojson());
assert.equal(depsGeoStandalone.crs, undefined);
assert.ok(depsGeoStandalone.features.length >= 2);
assertWgs84Coord(firstPolygonCoord(depsGeoStandalone.features[0].geometry), "deps.geojson");
assert.equal(typeof depsGeoStandalone.features[0].properties.level, "number");
assert.equal(typeof depsGeoStandalone.features[0].properties.volume, "number");
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
assert.ok(sim.sinkTif().length > 100);
assert.ok(sim.levelTif().length > 100);
const depGeo = JSON.parse(sim.depressionsGeojson());
assert.equal(depGeo.type, "FeatureCollection");
assert.equal(depGeo.crs, undefined, "depressionsGeojson must omit deprecated crs member");
assert.ok(depGeo.features.length >= 2, "expected multiple depression polygon features");
const [depLon, depLat] = firstPolygonCoord(depGeo.features[0].geometry);
assertWgs84Coord([depLon, depLat], "sim.depressionsGeojson");
for (const key of [
  "id",
  "level",
  "parent_id",
  "region_id",
  "area",
  "volume",
  "avg_depth",
  "max_depth",
  "min_elev",
  "max_elev",
  "circularity",
]) {
  assert.equal(
    typeof depGeo.features[0].properties[key],
    "number",
    `depressionsGeojson missing numeric property ${key}`,
  );
}
const spillPaths = JSON.parse(sim.spillPathsJson());
assert.ok(spillPaths.length >= 3, `expected >= 3 spill paths, got ${spillPaths.length}`);
const spillGeo = JSON.parse(sim.spillPathsGeojson());
assert.equal(spillGeo.type, "FeatureCollection");
assert.equal(spillGeo.features.length, spillPaths.length);
const [firstLon, firstLat] = spillGeo.features[0].geometry.coordinates[0];
assertWgs84Coord([firstLon, firstLat], "sim.spillPathsGeojson");
assert.ok(
  Math.abs(depLon - firstLon) < 0.05 && Math.abs(depLat - firstLat) < 0.05,
  `depressionsGeojson [${depLon}, ${depLat}] must align with spillPathsGeojson [${firstLon}, ${firstLat}]`,
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
assert.equal(JSON.parse(simAdv.fsmMassBalanceJson()), null);
simAdv.free();

// 7. Dynamic Fill-Spill-Merge cascade mass-balance routing in WASM
const simFsm = lidar.simulateInundationAdvanced(
  demBytes,
  "dem.tif",
  3,
  1000,
  0.3,
  0.3,
  5.0,
  50.0,
  2.0,
  undefined,
  1.0,
  undefined,
  "fsm",
);
const fsmMb = JSON.parse(simFsm.fsmMassBalanceJson());
assert.ok(fsmMb && fsmMb.total_runoff_m3 > 0, "expected positive FSM total runoff");
assert.ok(
  Math.abs(fsmMb.mass_balance_error_m3) < 1e-6 * fsmMb.total_runoff_m3,
  `FSM mass balance error too large: ${fsmMb.mass_balance_error_m3}`,
);
const fsmBasins = JSON.parse(simFsm.fsmBasinsJson());
assert.ok(fsmBasins.length >= 2, "expected multiple FSM basins");
assert.ok(
  fsmBasins.some((b) => b.spilled_volume_m3 > 0),
  "expected spilling FSM basins",
);
assert.ok(simFsm.fsmBasinsCsv().startsWith("dep_id,parent_id,level,"));
simFsm.free();

// 8. SCS Type II hyetograph + runoff_coeff + GIF encoding & invalid parameter validation
const simScs = lidar.simulateInundationAdvanced(
  demBytes,
  "dem.tif",
  3,
  1000,
  0.3,
  0.3,
  5.0,
  12.0,
  2.0,
  0.0,
  0.7,
  120.0,
  "static",
);
assert.equal(simScs.runoffCoeff, 0.7);
assert.equal(simScs.numSteps, 6);
const scsGif = simScs.encodeGif(120, false, 160, true);
assert.equal(new TextDecoder().decode(scsGif.slice(0, 6)), "GIF89a");
assert.ok(scsGif.length > 500);
const scsPng = simScs.renderFramePng(3, true, true, true);
assert.equal(scsPng[0], 0x89);
simScs.free();

for (const [label, args] of [
  ["negative breach_max_depth", [-0.5, 1.0, undefined, undefined]],
  ["NaN breach_max_depth", [Number.NaN, 1.0, undefined, undefined]],
  ["negative scs_type2_mm", [undefined, 1.0, -25.0, undefined]],
  ["NaN scs_type2_mm", [undefined, 1.0, Number.NaN, undefined]],
  ["zero runoff_coeff", [undefined, 0.0, undefined, undefined]],
  ["runoff_coeff > 1", [undefined, 1.2, undefined, undefined]],
  ["NaN runoff_coeff", [undefined, Number.NaN, undefined, undefined]],
  ["invalid routing mode", [undefined, 1.0, undefined, "not_a_mode"]],
]) {
  assert.throws(
    () =>
      lidar.simulateInundationAdvanced(
        demBytes,
        "dem.tif",
        3,
        1000,
        0.3,
        0.3,
        5.0,
        10.0,
        2.0,
        ...args,
      ),
    `expected simulateInundationAdvanced to reject ${label}`,
  );
}

console.log("ALL lidar-js WASM TESTS PASSED!");
