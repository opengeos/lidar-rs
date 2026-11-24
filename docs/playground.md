# Playground

Run **`lidar-rs`** directly in your browser via WebAssembly: DEM smoothing, selective depression breaching, Priority-Flood sink extraction, Wu et al. (2019) nested depression/mount hierarchy delineation, single-pass D8 catchment hierarchy, surface inundation simulation, stage-storage rating curves, connectivity graphs, and PNG/GIF export all execute locally in browser memory. Files are **never uploaded** to any server.

<div id="lidar-playground" markdown="0">
  <p class="lp-status" id="lp-status">Loading lidar-rs WebAssembly module…</p>

  <div class="lp-drop" id="lp-drop">
    <p><strong>Drop a DEM or DSM GeoTIFF here</strong> or <label class="lp-link">choose a file<input type="file" id="lp-file" accept=".tif,.tiff,.img,.asc" class="lp-visually-hidden"></label>.</p>
    <p class="lp-hint">Single-band GeoTIFF / COG elevation rasters. Everything stays in your browser memory.</p>
    <p class="lp-samples">Or try a sample dataset:
      <button type="button" id="lp-sample-dem">Sample DEM: Depressions &amp; Inundation (400×400)</button>
      <button type="button" id="lp-sample-clsa">Cottonwood Lake Study Area Topobathy (1877×1730, 0.5 m)</button>
      <button type="button" id="lp-sample-dsm">Sample DSM: Mounts / Canopies (400×400)</button>
    </p>
  </div>

  <div class="lp-controls" id="lp-controls">
    <div class="lp-field">
      <label for="lp-mode">Workflow</label>
      <select id="lp-mode">
        <option value="inundation">Nested Depressions + Inundation Simulation</option>
        <option value="mounts">Nested Mounts / Canopies (DSM)</option>
      </select>
    </div>
    <div class="lp-field lp-inundation-only">
      <label for="lp-routing">Routing mode</label>
      <select id="lp-routing">
        <option value="static">Static (Level-Set)</option>
        <option value="fsm">Fill-Spill-Merge (Cascade)</option>
      </select>
    </div>
    <div class="lp-field lp-inundation-only">
      <label for="lp-kernel">Median filter (px)</label>
      <input type="number" id="lp-kernel" value="3" min="1" max="15" step="2">
    </div>
    <div class="lp-field lp-inundation-only">
      <label for="lp-breach">Selective breach (m, 0=off)</label>
      <input type="number" id="lp-breach" value="0" min="0" max="10" step="0.1">
    </div>
    <div class="lp-field">
      <label for="lp-min-size">Min size (cells)</label>
      <input type="number" id="lp-min-size" value="1000" min="10" max="100000" step="100">
    </div>
    <div class="lp-field">
      <label for="lp-min-depth">Min depth / height (m)</label>
      <input type="number" id="lp-min-depth" value="0.3" min="0.05" max="100" step="0.1">
    </div>
    <div class="lp-field">
      <label for="lp-interval">Slicing interval (m)</label>
      <input type="number" id="lp-interval" value="0.3" min="0.05" max="50" step="0.1">
    </div>
    <div class="lp-field lp-inundation-only">
      <label for="lp-rain">Rain intensity (cm/h)</label>
      <input type="number" id="lp-rain" value="5" min="0.5" max="100" step="0.5">
    </div>
    <div class="lp-field lp-inundation-only">
      <label for="lp-runoff-coeff">Runoff coeff C (0..1)</label>
      <input type="number" id="lp-runoff-coeff" value="1.0" min="0.05" max="1.0" step="0.05">
    </div>
    <div class="lp-field lp-inundation-only">
      <label for="lp-duration">Rain duration (h)</label>
      <input type="number" id="lp-duration" value="70" min="1" max="200" step="1">
    </div>
    <div class="lp-field lp-inundation-only">
      <label for="lp-step">Time step (h)</label>
      <input type="number" id="lp-step" value="1" min="0.25" max="24" step="0.5">
    </div>
    <div class="lp-field lp-action">
      <button type="button" id="lp-run" class="md-button md-button--primary" disabled>Run simulation</button>
    </div>
  </div>

  <div id="lp-results" hidden>
    <div class="lp-summary" id="lp-summary"></div>

    <div id="lp-player-section" hidden>
      <h3>Interactive Inundation Simulation</h3>
      <div class="lp-player-grid">
        <div class="lp-player-stage">
          <canvas id="lp-player-canvas" title="Click any depression to inspect hierarchy, morphometry, and stage-volume curve"></canvas>
          <div class="lp-player-telemetry" id="lp-player-telemetry"></div>
          <div class="lp-player-bar">
            <button type="button" id="lp-play-btn" class="md-button md-button--primary">Pause</button>
            <button type="button" id="lp-prev-btn" class="lp-icon-btn" title="Previous time step">&#9664;</button>
            <button type="button" id="lp-next-btn" class="lp-icon-btn" title="Next time step">&#9654;</button>
            <input type="range" id="lp-slider" min="0" max="1" value="0" aria-label="Simulation time step">
            <label class="lp-inline-check"><input type="checkbox" id="lp-spill-paths" checked> Spill paths</label>
            <label class="lp-inline-check"><input type="checkbox" id="lp-id-badges" checked> ID badges</label>
            <label class="lp-inline-check"><input type="checkbox" id="lp-pingpong" checked> Ping-pong</label>
            <select id="lp-speed" aria-label="Playback speed">
              <option value="400">0.5×</option>
              <option value="200" selected>1×</option>
              <option value="100">2×</option>
              <option value="50">4×</option>
            </select>
          </div>
        </div>
        <div class="lp-player-side">
          <h4>Inundation Curve</h4>
          <p class="lp-hint">Click the chart or drag the slider to inspect any time step. Click any depression on the map or table to inspect it.</p>
          <canvas id="lp-chart" width="420" height="200"></canvas>

          <div class="lp-inspector" id="lp-inspector">
            <div class="lp-inspector-head">
              <h4 id="lp-inspector-title">Depression Inspector</h4>
              <button type="button" id="lp-inspector-clear" class="lp-icon-btn" hidden>Clear</button>
            </div>
            <p class="lp-hint" id="lp-inspector-empty">Click any depression on the map or in the feature table below to view its parent/child hierarchy, morphometry, spill threshold, and stage-volume rating curve.</p>
            <div id="lp-inspector-body" hidden>
              <div class="lp-inspector-grid" id="lp-inspector-stats"></div>
              <canvas id="lp-rating-chart" width="420" height="145"></canvas>
            </div>
          </div>
        </div>
      </div>

      <div class="lp-tree-panel" id="lp-tree-panel">
        <div class="lp-tree-head">
          <div>
            <h4>Depression Tree Graph <span class="lp-hint" id="lp-tree-status"></span></h4>
            <p class="lp-hint">Builds progressively bottom-up as rising water layers fill leaf depressions and merge them into enclosing levels. Click any node to inspect.</p>
          </div>
          <div class="lp-tree-controls">
            <label class="lp-tree-region-wrap" id="lp-tree-region-wrap" hidden>
              Region
              <select id="lp-tree-region" aria-label="Sink region hierarchy"></select>
            </label>
            <label class="lp-tree-region-wrap" id="lp-tree-node-wrap">
              Node
              <select id="lp-tree-node-select" aria-label="Inspect depression node"></select>
            </label>
            <label class="lp-inline-check"><input type="checkbox" id="lp-tree-outline" checked> Show unreached outline</label>
            <button type="button" id="lp-tree-gif-btn" class="lp-icon-btn" title="Export animated depression tree GIF">Download GIF</button>
          </div>
        </div>
        <div class="lp-tree-canvas-wrap" id="lp-tree-canvas-wrap">
          <canvas id="lp-tree-canvas" width="960" height="360" tabindex="0" role="application" aria-label="Depression tree graph. Use arrow keys to step through nodes and Enter to inspect." title="Click any node or use arrow keys + Enter to inspect its depression on the map"></canvas>
        </div>
        <div class="lp-tree-legend" id="lp-tree-legend"></div>
      </div>
    </div>

    <h3>Spatial Layers</h3>
    <div class="lp-canvases" id="lp-canvases">
      <figure>
        <canvas id="lp-c-hillshade"></canvas>
        <figcaption>Shaded Relief (Hillshade)</figcaption>
      </figure>
      <figure id="lp-fig-depth">
        <canvas id="lp-c-depth"></canvas>
        <figcaption>Depression Depth (over hillshade)</figcaption>
      </figure>
      <figure>
        <canvas id="lp-c-level"></canvas>
        <figcaption id="lp-cap-level">Nested Hierarchy Levels (over hillshade)</figcaption>
      </figure>
      <figure id="lp-fig-fill">
        <canvas id="lp-c-fill"></canvas>
        <figcaption>Time to Inundation (earliest = brightest)</figcaption>
      </figure>
    </div>

    <h3>Downloads</h3>
    <div class="lp-downloads" id="lp-downloads"></div>
    <p class="lp-gif-link-row" id="lp-gif-link-row" hidden>
      <button type="button" id="lp-gif-preview-link" class="lp-link lp-link-btn" aria-controls="lp-gif-wrap" aria-expanded="false">Preview exported animated GIF</button>
    </p>
    <div class="lp-gif-wrap" id="lp-gif-wrap" hidden>
      <img id="lp-gif" alt="Simulated surface inundation animated GIF">
    </div>

    <h3>Delineated Features <span class="lp-hint" id="lp-table-note"></span></h3>
    <div class="lp-table-wrap">
      <table id="lp-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Level</th>
            <th>Parent</th>
            <th>Region</th>
            <th>Cells</th>
            <th>Area</th>
            <th>Volume</th>
            <th>Max Depth</th>
            <th>Circularity</th>
            <th>HI</th>
            <th>Rim Slope</th>
          </tr>
        </thead>
        <tbody id="lp-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

See [JavaScript / WebAssembly](javascript.md) for the full API reference and Node.js / browser usage examples.
