# Animation & Publication Frames

`lidar-rs` renders publication-grade shaded-relief inundation animations (`.gif`) and lossless individual frame snapshots (`.png`) directly in pure Rust using `gif` and `png` crates with zero Matplotlib, FFmpeg, or ImageMagick dependencies.

| Pure-Rust PNG Snapshot (`lidar export-frames`) | Pure-Rust Animated GIF (`lidar animate-inundation`) |
| :---: | :---: |
| ![Publication-ready PNG snapshot with shaded relief, water overlay, cyan spill arrow, metric scale bar, legend, and timestamp HUD](https://files.opengeos.org/lidar/images/publication_frame_spill_paths.png) | ![Animated shaded-relief inundation GIF with overland spill paths, scale bar, and legend](https://files.opengeos.org/lidar/images/inundation_spill_demo.gif) |

## Cartographic Overlays & HUD

Both `export_inundation_frames_png` and `animate_inundation` share the same pure-Rust renderer:

- **Shaded relief background:** Multi-directional hypsometric tint blended with directional hillshade (`azimuth_deg`, `elevation_deg`).
- **Translucent water overlay:** Progressive depth-aware inundation tint (`water_rgb`, `water_alpha`).
- **Active overland spill-path arrows:** Cyan flowlines with directional arrowheads drawn once each spill path's `spill_time_h` is reached.
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
      --max-dimension 600 --scale-bar --legend

    # 3. Export an animated GIF from a simulate-inundation flood directory
    lidar simulate-inundation out/sink.tif out/catchment_hir.tif out/sim \
      --min-size 1000 --min-depth 0.2 --interval 0.2 \
      --rain-intensity 5 --duration 50 --time-step 2
    lidar animate-inundation out/dem.tif out/sim out/inundation.gif \
      --delay-ms 140 --rain-intensity 5 --max-dimension 500 --scale-bar --legend
    ```

    To assemble exported PNG frames into a lossless H.264 MP4 video with FFmpeg:

    ```sh
    ffmpeg -framerate 8 -i out_pipe/frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p inundation.mp4
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

    # Render animated GIF and display inline in Jupyter / Google Colab
    lidar.PlayAnimation(
        "out/dem.tif",
        "out/sim",
        "out/inundation.gif",
        rain_intensity_cm_h=5.0,
        show_scale_bar=True,
        show_legend=True,
        display=True,
    )
    ```

=== "Rust"

    ```rust,no_run
    use lidar::{
        AnimateInundationOptions, export_inundation_frames_png, load_raster,
        render_inundation_frame_png_bytes,
    };

    fn main() -> lidar::Result<()> {
        let dem = load_raster::<f64>("dem.tif", None)?;
        let fill_time = load_raster::<f64>("out_pipe/fill_time.tif", None)?;
        let opts = AnimateInundationOptions {
            max_dimension: Some(600),
            rain_intensity_cm_h: Some(5.0),
            time_step_h: Some(2.0),
            show_scale_bar: true,
            show_legend: true,
            ..Default::default()
        };

        // Write a full sequence frame_001.png .. frame_025.png
        let paths = export_inundation_frames_png(&dem, &fill_time, 25, 2.0, &[], "out/frames", &opts)?;
        println!("Wrote {} PNG frames", paths.len());

        // Or render a single frame directly to PNG bytes in memory
        let png = render_inundation_frame_png_bytes(&dem, &fill_time, 32.0, 0.64, &[], &opts)?;
        std::fs::write("out/snapshot_32h.png", png)?;
        Ok(())
    }
    ```

=== "JavaScript / WASM"

    ```js
    // Inside a browser or Node.js InundationOutput handle `sim`:
    const pngBytes = sim.renderFramePng(25, true, true, true); // step, scaleBar, legend, spillPaths
    const gifBytes = sim.encodeGif(140, true, 500, true);      // delayMs, pingPong, maxDim, spillPaths
    ```
