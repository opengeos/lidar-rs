# Animation & Publication Frames

`lidar-rs` renders publication-grade shaded-relief inundation animations (`.gif`) and lossless individual frame snapshots (`.png`) directly in pure Rust using `gif` and `png` crates with zero Matplotlib, FFmpeg, or ImageMagick dependencies.

| Pure-Rust PNG Snapshot (`render_inundation_frame_png_bytes` / WASM) | Pure-Rust Animated GIF (`animate_inundation_to_bytes_with_spill_paths` / WASM) |
| :---: | :---: |
| ![Publication-ready PNG snapshot with shaded relief, water overlay, cyan spill arrow, metric scale bar, legend, and timestamp HUD](https://files.opengeos.org/lidar/images/publication_frame_spill_paths.png) | ![Animated shaded-relief inundation GIF with overland spill paths, scale bar, and legend](https://files.opengeos.org/lidar/images/inundation_spill_demo.gif) |

## Cartographic Overlays & HUD

Both `export_inundation_frames_png` and `animate_inundation` share the same pure-Rust renderer:

- **Shaded relief background:** Multi-directional hypsometric tint blended with directional hillshade (`azimuth_deg`, `elevation_deg`).
- **Translucent water overlay:** Progressive depth-aware inundation tint (`water_rgb`, `water_alpha`).
- **Active overland spill-path arrows (Rust & WASM):** Cyan flowlines with directional arrowheads drawn once each spill path's `spill_time_h` is reached when `spill_paths` are supplied to `export_inundation_frames_png`, `render_inundation_frame_png_bytes`, `animate_inundation_to_bytes_with_spill_paths` (Rust), or `renderFramePng` / `encodeGif` with `spillPaths = true` (WASM / Playground). Note: the CLI (`export-frames`, `animate-inundation`) and Python file wrappers currently pass an empty spill-path slice, rendering the shaded relief, water overlay, HUD, scale bar, and legend without arrows.
- **Timestamp & cumulative rainfall HUD:** Top dark banner showing elapsed storm time (`T = 32.0H`), cumulative rainfall/runoff depth (`1600 MM`), and a cyan progress bar.
- **Metric scale bar (`--scale-bar`):** Automatically chooses clean metric roundings (`50 M`, `100 M`, `500 M`, `1 KM`, etc.), converts foot-unit projected CRSs to meters, skips geographic degree CRSs automatically, and reserves right-margin clearance so it never overlaps the legend.
- **Legend box (`--legend`):** Bottom-right translucent legend displaying `WATER` and `SPILL` swatches.

## Usage Examples

=== "CLI"

    ```sh
    # 1. Run the all-in-one pipeline to produce fill_time.tif
    lidar pipeline dem.tif out_pipe \
      --min-size 1000 --min-depth 0.3 --interval 0.3 \
      --rain-intensity 5 --duration 50 --time-step 1

    # 2. Export individual publication PNG frames (e.g., for slides or ffmpeg MP4 encoding)
    lidar export-frames dem.tif out_pipe/fill_time.tif out_pipe/frames \
      --num-steps 25 --time-step 2 --rain-intensity 5 \
      --max-dim 600 --scale-bar --legend

    # 3. Export an animated GIF from the step-by-step flood_* raster workflow
    lidar median-filter dem.tif out/median.tif -k 3
    lidar extract-sinks out/median.tif out --min-size 1000
    lidar delineate-depressions out/sink.tif out --min-size 1000 --min-depth 0.2 --interval 0.2 --level-shp
    lidar catchment-hierarchy out/dem.tif out/shp-level out/catchment_hir.tif
    lidar simulate-inundation out/sink.tif out/catchment_hir.tif out/sim \
      --min-size 1000 --min-depth 0.2 --interval 0.2 \
      --rain-intensity 5 --duration 50 --time-step 2
    lidar animate-inundation out/dem.tif out/sim out/inundation.gif \
      --delay-ms 140 --time-step 2 --rain-intensity 5 --max-dim 500 --scale-bar --legend
    ```

    To assemble the 25 exported PNG frames (`frame_01.png` .. `frame_25.png`) into a standard H.264 MP4 video with FFmpeg (lossy `yuv420p` for broad player/browser compatibility; pass `-crf 0 -pix_fmt rgb24` with `libx264rgb` if lossless RGB archival is needed):

    ```sh
    ffmpeg -framerate 8 -i out_pipe/frames/frame_%02d.png -c:v libx264 -pix_fmt yuv420p inundation.mp4
    ```

=== "Python"

    ```python
    import lidar_rs as lidar

    # Export individual PNG frames directly from dem.tif + fill_time.tif
    frames = lidar.export_inundation_frames_png(
        "dem.tif",
        "out_pipe/fill_time.tif",
        "out_pipe/frames",
        num_steps=10,
        time_step_h=5.0,
        rain_intensity_cm_h=5.0,
        max_dimension=600,
        show_scale_bar=True,
        show_legend=True,
    )
    print(f"Exported {len(frames)} PNG frames -> {frames[0]}")

    # Render animated GIF (matching the 2-hour simulation step) and display inline in Jupyter / Colab
    lidar.PlayAnimation(
        "out/dem.tif",
        "out/sim",
        "out/inundation.gif",
        time_step_h=2.0,
        rain_intensity_cm_h=5.0,
        show_scale_bar=True,
        show_legend=True,
        display=True,
    )
    ```

=== "Rust"

    ```rust,no_run
    use lidar::{
        AnimateInundationOptions, InMemoryFullPipelineOptions, SimulateInundationOptions,
        animate_inundation_to_bytes_with_spill_paths, export_inundation_frames_png, load_raster,
        render_inundation_frame_png_bytes, run_full_pipeline_in_memory,
    };

    fn main() -> lidar::Result<()> {
        let dem = load_raster::<f64>("dem.tif", None)?;
        let pipe = run_full_pipeline_in_memory(
            &dem,
            &InMemoryFullPipelineOptions {
                kernel_size: 3,
                breach: None,
                simulation: SimulateInundationOptions {
                    min_size: 1000,
                    min_depth: 0.3,
                    interval: 0.3,
                    rain_intensity_cm_h: 5.0,
                    rain_duration_h: 50.0,
                    time_step_h: 2.0,
                    compress: false,
                },
                forcing: Default::default(),
                scalar_runoff_coeff: 1.0,
                stream_vector: None,
            },
        )?;
        let opts = AnimateInundationOptions {
            max_dimension: Some(600),
            rain_intensity_cm_h: Some(5.0),
            time_step_h: Some(2.0),
            show_scale_bar: true,
            show_legend: true,
            ..Default::default()
        };

        // Write a full sequence frame_01.png .. frame_25.png with active overland spill arrows
        let paths = export_inundation_frames_png(
            &dem, &pipe.fill_time, 25, 2.0, &pipe.spill_paths, "out/frames", &opts,
        )?;
        println!("Wrote {} PNG frames", paths.len());

        // Render a single snapshot with spill arrows directly to PNG bytes in memory
        let png = render_inundation_frame_png_bytes(
            &dem, &pipe.fill_time, 32.0, 0.64, &pipe.spill_paths, &opts,
        )?;
        std::fs::write("out/snapshot_32h.png", png)?;

        // Or render an animated GIF with spill arrows in memory
        let (gif_bytes, _, _, _) = animate_inundation_to_bytes_with_spill_paths(
            &dem, &pipe.fill_time, 25, 2.0, &pipe.spill_paths, &opts,
        )?;
        std::fs::write("out/inundation_spill.gif", gif_bytes)?;
        Ok(())
    }
    ```

=== "JavaScript / WASM"

    ```js
    // Inside a browser or Node.js InundationOutput handle `sim`:
    const pngBytes = sim.renderFramePng(25, true, true, true); // step, scaleBar, legend, spillPaths
    const gifBytes = sim.encodeGif(140, true, 500, true);      // delayMs, pingPong, maxDim, spillPaths
    ```
