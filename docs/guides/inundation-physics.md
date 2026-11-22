# Inundation Physics & Storm Forcing

Standard level-set inundation simulation assumes spatially uniform rainfall with 100% runoff (`C = 1.0`), zero infiltration losses, and dry initial basins. Real watersheds exhibit spatially varying land cover, intense storm peaks, vertical seepage or karst swallow-hole losses, and pre-existing ponded water.

`lidar-rs` extends fill-and-spill inundation modeling with four composable physical forcing inputs (`InundationForcing`):

1. **Spatially variable runoff coefficients (`runoff_coeff`, `0.0..=1.0`)** plus built-in **NRCS TR-55 Curve Number conversion (`scs_curve_number_runoff`)**.
2. **Time-varying storm hyetographs (`hyetograph`)** plus built-in **SCS Type II 24-hour dimensionless design storm generation (`scs_type2_hyetograph`)**.
3. **Spatially variable vertical infiltration / swallow-hole loss rates (`infiltration_cm_h`)**, allowing basins to drain during dry or low-intensity intervals (capped at zero storage / full slice capacity).
4. **Antecedent ponded water depths (`antecedent_depth`)** that pre-fill depressions prior to storm onset.

![Comparison of SCS Type II design storm hyetograph, spatial SCS-CN runoff coefficient map, and simulated inundation dynamics across forcing scenarios](https://files.opengeos.org/lidar/images/inundation_forcing_comparison.png)

## NRCS Curve Number (SCS-CN) Runoff Conversion

`scs_curve_number_runoff(cn, precip_mm)` converts a Curve Number raster (`CN` in `(0, 100]`) and total storm precipitation `P` (`precip_mm > 0`) into a spatially variable effective runoff-coefficient raster (`C = Q / P` in `[0, 1]`):

- Potential maximum retention: `S = 25400 / CN - 254` (mm)
- Initial abstraction: `Ia = 0.2 * S` (mm)
- Direct runoff: `Q = (P - Ia)^2 / (P - Ia + S)` for `P > Ia`, else `0`
- Effective runoff coefficient: `C = Q / P`

Non-finite and NoData CN cells propagate as NoData (`-9999.0`) in the output raster. During inundation simulation, NoData cells in `runoff_coeff` safely fall back to `1.0`.

## SCS Type II Design Storm Hyetograph

`scs_type2_hyetograph(total_precip_mm, duration_h, time_step_h)` generates piecewise-constant `(step_duration_h, intensity_cm_h)` pulses following the NRCS Type II dimensionless cumulative distribution (with ~45% of storm volume concentrated around the mid-storm peak).

- In `lidar pipeline`, passing `--scs-type2-mm <mm>` automatically defaults `--duration` to `24.0` hours (matching the standard NRCS 24-h design storm and Python default) unless `--duration` is explicitly overridden.
- All exported runoff metrics (`inundation_curve.csv` `runoff_mm`, graph `fill_runoff_mm` / `spill_runoff_mm`, and spill vector attributes) integrate cumulative depth along the hyetograph scaled by the effective runoff coefficient.

## Usage Examples

=== "CLI"

    ```sh
    # Download sample DEM and aligned Curve Number raster
    curl -LO https://files.opengeos.org/lidar/data/dem.tif
    curl -LO https://files.opengeos.org/lidar/data/sample_cn.tif

    # 1. Run a 100 mm 24-hour SCS Type II design storm directly on dem.tif
    lidar pipeline dem.tif out_storm \
      --min-size 1000 --min-depth 0.3 --interval 0.3 \
      --scs-type2-mm 100 --time-step 0.5 \
      --compress

    # 2. With optional aligned forcing GeoTIFFs (same grid dimensions, geotransform, and CRS as dem.tif;
    #    e.g., convert sample_cn.tif with scs_curve_number_runoff and write runoff_coeff.tif,
    #    infil_cm_h.tif, and antecedent_m.tif beforehand)
    lidar pipeline dem.tif out_storm_forced \
      --min-size 1000 --min-depth 0.3 --interval 0.3 \
      --scs-type2-mm 100 --time-step 0.5 \
      --runoff-coeff runoff_coeff.tif \
      --infiltration infil_cm_h.tif \
      --antecedent-depth antecedent_m.tif \
      --compress
    ```

=== "Python"

    ```python
    import numpy as np
    import tifffile  # optional: pip install tifffile (or use rasterio)
    import lidar_rs as lidar

    dem = tifffile.imread("dem.tif").astype(np.float64)
    cn = tifffile.imread("sample_cn.tif").astype(np.float64)

    # 1. Convert NRCS Curve Number raster to effective runoff coefficient C = Q/P
    #    (pass no_data=<val> if the CN GeoTIFF uses a finite NoData sentinel)
    rc = lidar.scs_curve_number_runoff(cn, precip_mm=150.0)
    # Map the -9999.0 output NoData sentinel to NaN so array simulation falls back to C = 1.0
    rc = np.where(rc == -9999.0, np.nan, rc)

    # 2. Generate a 24-hour SCS Type II hyetograph (30-minute pulses)
    hyeto = lidar.scs_type2_hyetograph(total_precip_mm=150.0, duration_h=24.0, time_step_h=0.5)

    # 3. Run forced inundation completely in memory with infiltration + antecedent water
    infil = np.full_like(dem, 0.5)       # 0.5 cm/h vertical loss
    antecedent = np.full_like(dem, 0.1)  # 0.1 m initial ponded depth
    sim = lidar.simulate_inundation_array(
        dem,
        kernel_size=3,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=0.0,  # overridden by hyetograph
        rain_duration_h=24.0,
        time_step_h=0.5,
        runoff_coeff_arr=rc,
        infiltration_arr=infil,
        antecedent_depth_arr=antecedent,
        hyetograph=hyeto,
        cell_size=1.0,
    )
    print(f"Simulated {len(sim['steps'])} storm steps; final inundated fraction: {sim['steps'][-1]['inundated_fraction']:.2%}")
    ```

=== "Rust"

    ```rust,no_run
    use lidar::{
        InMemoryFullPipelineOptions, InundationForcing, SimulateInundationOptions,
        load_raster, run_full_pipeline_in_memory, scs_curve_number_runoff, scs_type2_hyetograph,
    };

    fn main() -> lidar::Result<()> {
        let dem = load_raster::<f64>("dem.tif", None)?;
        let cn = load_raster::<f64>("sample_cn.tif", None)?;
        let rc = scs_curve_number_runoff(&cn, 150.0)?;
        let hyeto = scs_type2_hyetograph(150.0, 24.0, 0.5)?;

        let pipe = run_full_pipeline_in_memory(
            &dem,
            &InMemoryFullPipelineOptions {
                kernel_size: 3,
                breach: None,
                simulation: SimulateInundationOptions {
                    min_size: 1000,
                    min_depth: 0.3,
                    interval: 0.3,
                    rain_intensity_cm_h: 0.0, // overridden by hyetograph
                    rain_duration_h: 24.0,
                    time_step_h: 0.5,
                    compress: false,
                },
                routing: lidar::RoutingMode::FillSpillMerge,
                forcing: InundationForcing {
                    runoff_coeff: Some(&rc),
                    hyetograph: Some(&hyeto),
                    ..Default::default()
                },
                scalar_runoff_coeff: 1.0,
                stream_vector: None,
            },
        )?;
        println!("{}", &pipe.inundation_curve_csv[..pipe.inundation_curve_csv.len().min(300)]);
        Ok(())
    }
    ```

## Dynamic Fill-Spill-Merge Cascade Routing (`routing="fsm"`)

By default (`RoutingMode::Static`), each depression fills independently using its static pre-spill catchment area. Enabling **Fill-Spill-Merge (FSM)** cascade mass-balance routing (`--routing fsm` on the CLI, `routing="fsm"` in Python/WASM, or `RoutingMode::FillSpillMerge` in Rust) models coupled multi-basin fill, spillover, and hierarchical merging:

1. **Catchment-weighted runoff attribution (`O(N)`):** Every valid cell traces its D8 flow path once to its initial receiving depression (or watershed boundary), crediting both geometric area (`cell_area`) and effective runoff area (`cell_area * runoff_coeff`) with zero double-counting between nested levels.
2. **Dynamic spill cascades:** Once an upstream depression fills to its rim capacity at `t_spill`, its net excess discharge `(Q_in - loss).max(0)` routes over its lowest topographic saddle into the downstream receiving depression (or the DEM boundary), accelerating downstream fill times.
3. **Hierarchical parent merging:** When all direct children of a compound parent depression reach full capacity, they merge into the parent at `t_merge`, transferring stored volume and contributing catchments to the parent as water continues rising smoothly to the enclosing rim.
4. **Continuous hypsometric timing & mass conservation:** Per-cell `fill_time` is interpolated continuously from each basin's exact sorted-elevation stage-storage curve, and every run verifies exact watershed-scale conservation:

```text
total_antecedent_m3 + total_runoff_m3
  == total_stored_m3 + total_infiltration_m3 + total_boundary_outflow_m3
```

=== "CLI"

    ```sh
    lidar pipeline dem.tif out_fsm \
      --routing fsm \
      --min-size 1000 --min-depth 0.3 --interval 0.3 \
      --rain-intensity 5 --duration 50 --time-step 1 \
      --compress --json
    ```

=== "Python"

    ```python
    import lidar_rs as lidar

    pipe = lidar.run_pipeline(
        "dem.tif",
        "out_fsm",
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        rain_intensity_cm_h=5.0,
        rain_duration_h=50.0,
        time_step_h=1.0,
        routing="fsm",
    )
    print(pipe["fsm_mass_balance"])
    print("Per-basin CSV:", pipe["fsm_basins_csv"])
    ```

=== "Rust"

    ```rust,no_run
    use lidar::{RoutingMode, RunPipelineOptions, SimulateInundationOptions, run_pipeline};

    fn main() -> lidar::Result<()> {
        let res = run_pipeline(
            "dem.tif",
            "out_fsm",
            &RunPipelineOptions {
                routing: RoutingMode::FillSpillMerge,
                simulation: SimulateInundationOptions {
                    min_size: 1000,
                    min_depth: 0.3,
                    interval: 0.3,
                    rain_intensity_cm_h: 5.0,
                    rain_duration_h: 50.0,
                    time_step_h: 1.0,
                    compress: true,
                },
                ..Default::default()
            },
        )?;
        if let Some(mb) = res.fsm_mass_balance {
            println!("Mass balance residual: {:.2e} m3", mb.mass_balance_error_m3);
        }
        Ok(())
    }
    ```

## Grid Alignment & Validation Rules

`simulate_inundation_with_forcing_in_memory` strictly validates every supplied forcing raster (`runoff_coeff`, `infiltration_cm_h`, `antecedent_depth`) against the sink raster before running:

- **Dimensions:** Exact `(width, height)` match required.
- **Geotransform:** Pixel-relative origin tolerance (`1e-3 * cell_size`) and scale tolerance (`1e-6`), rejecting even 1-pixel shifts at large UTM coordinates.
- **CRS compatibility:** When both rasters carry known CRSs, `SpatialRef::same_as` must succeed.
- **Hyetograph records:** Every `(dt_h, intensity_cm_h)` step must have finite `dt_h > 0` and finite `intensity_cm_h >= 0` (zero-intensity dry intervals are supported).
