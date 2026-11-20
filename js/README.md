# `lidar-js` (WebAssembly bindings for `lidar-rs`)

Run nested surface depression delineation, mount delineation, catchment hierarchy, surface inundation dynamics simulation, and animated GIF rendering directly in **browsers and Node.js** via WebAssembly — zero GDAL, zero Python, zero server required.

## Building

```sh
# For browsers (ES module)
wasm-pack build --target web --release --out-dir pkg

# For Node.js
wasm-pack build --target nodejs --release --out-dir pkg-node
node test.mjs
```

## Browser Usage

```js
import init, { simulateInundation, delineateMounts } from "./pkg/lidar_js.js";

await init();
const demBytes = new Uint8Array(await (await fetch("dem.tif")).arrayBuffer());

const sim = simulateInundation(
  demBytes, "dem.tif",
  3,     // median filter kernel size
  1000,  // min_size (pixels)
  0.2,   // min_depth
  0.3,   // slicing interval
  5.0,   // rain intensity (cm/h)
  20.0,  // rain duration (h)
  1.0,   // time step (h)
);

const gifBytes = sim.encodeGif(150, true, 400); // delay ms, ping-pong, max dim px
const gifUrl = URL.createObjectURL(new Blob([gifBytes], { type: "image/gif" }));
document.querySelector("#anim").src = gifUrl;
console.log(JSON.parse(sim.depressionsJson()));
sim.free();
```

## License

`lidar-rs` source code is MIT-licensed. Compiled `.wasm` binaries bundle `richdem` (`GPL-3.0-only`) and are distributed under the GNU General Public License v3.0.
