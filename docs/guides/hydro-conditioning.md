# Hydro-Conditioning (Breaching & Culvert Burning)

High-resolution 1-m LiDAR DEMs routinely contain artificial impoundments where roads, driveways, railways, and farm lanes cross drainage lines without bare-earth bridge/culvert openings. Pure depression filling turns those road crossings into fake wetlands and blocks realistic downstream routing.

`lidar-rs` provides two complementary pure-Rust hydro-conditioning tools:

1. **Lindsay (2016) least-cost depression breaching (`breach_depressions`):** Carves minimum-elevation-cost breach channels through narrow barriers while preserving genuine deep natural wetlands and karst dolines.
2. **Vector-guided culvert burning (`burn_culverts`):** Uses known culvert polylines or point locations (`.gpkg`, `.geojson`, `.shp`) to carve monotonic channels straight through embankments.

![Comparison of raw DEM with road embankment, selective Lindsay (2016) breaching preserving the natural wetland, and unconstrained complete breaching](https://files.opengeos.org/lidar/images/hydro_conditioning_comparison.png)

## Choosing a Breaching Mode

| Mode | Behavior | Default Limits | Best Use Case |
| --- | --- | --- | --- |
| `selective` (default) | Breaches a depression **only if** a valid breach path exists within both `max_depth` and `max_length`. | `max_depth = 1.5`, `max_length = 50` | Removing narrow road impoundments and micro-pits while preserving true wetlands and karst dolines. |
| `constrained` | Drills a partial breach up to `max_depth` and `max_length` even if full drainage is not reached. | `max_depth = 1.5`, `max_length = 50` | Partial conditioning on gently sloped coastal or agricultural plains. |
| `complete` | Breaches every depression completely to the DEM boundary. | Unconstrained (`None`) | Full hydrologic drainage enforcement before stream network extraction. |

!!! tip "Stage Order Inside `run_pipeline`"
    When both smoothing and hydro-conditioning are enabled in `run_pipeline`, `lidar-rs` applies **median smoothing first**, then **culvert burning**, then **selective breaching**, so a smoothing kernel never erases a narrow 1-cell culvert cut or breach channel.

## Usage Examples

=== "CLI"

    ```sh
    curl -LO https://files.opengeos.org/lidar/data/dem.tif
    curl -LO https://files.opengeos.org/lidar/data/sample_culverts.gpkg

    # 1. Selective Lindsay (2016) breaching (preserves deep wetlands)
    lidar breach-depressions dem.tif out/breached.tif \
      --mode selective --max-depth 1.5 --max-length 40 --compress

    # 2. Complete unconstrained breaching
    lidar breach-depressions dem.tif out/breached_full.tif --mode complete

    # 3. Vector-guided culvert burning across embankments
    lidar burn-culverts dem.tif sample_culverts.gpkg out/burned.tif \
      --half-width 1 --depth-offset 0.05 --compress

    # 4. Run both inside the all-in-one pipeline
    lidar pipeline dem.tif out_conditioned \
      --culverts sample_culverts.gpkg --breach-max-depth 1.5 --breach-max-length 40
    ```

=== "Python"

    ```python
    import lidar_rs as lidar

    # Works on both GeoTIFF paths and 2D NumPy arrays
    lidar.breach_depressions(
        "dem.tif", "out/breached.tif",
        mode="selective", max_depth=1.5, max_length=40, compress=True,
    )

    dem_arr = lidar.MedianFilter("dem.tif", kernel_size=3)
    breached_complete = lidar.breach_depressions(dem_arr, mode="complete")

    # Vector-guided culvert burning (returns carved cell count)
    carved_cells = lidar.burn_culverts(
        "dem.tif", "sample_culverts.gpkg", "out/burned.tif",
        half_width_cells=1, depth_offset=0.05, compress=True,
    )
    print(f"Carved {carved_cells} embankment cells into out/burned.tif")
    ```

=== "Rust"

    ```rust,no_run
    use lidar::{
        BreachMode, BreachOptions, BurnCulvertsOptions,
        breach_depressions_file, burn_culverts_file,
    };

    fn main() -> lidar::Result<()> {
        let breach_opts = BreachOptions::for_mode(BreachMode::Selective, Some(1.5), Some(40));
        breach_depressions_file("dem.tif", "out/breached.tif", &breach_opts)?;

        let (_, carved) = burn_culverts_file(
            "out/breached.tif",
            "sample_culverts.gpkg",
            "out/conditioned.tif",
            &BurnCulvertsOptions {
                half_width_cells: 1,
                depth_offset: 0.05,
                ..Default::default()
            },
        )?;
        println!("Carved {carved} culvert cells");
        Ok(())
    }
    ```

=== "JavaScript / WASM"

    ```js
    import init, { breachDem } from "./wasm/lidar_js.js";

    await init();
    const demBytes = new Uint8Array(await (await fetch("https://files.opengeos.org/lidar/data/dem.tif")).arrayBuffer());

    // Selective breaching (uses default 1.5 m / 50 cell bounds when omitted)
    const breachedTif = breachDem(demBytes, "dem.tif", "selective", 1.5, 40, false);
    ```

## Culvert Vector Geometry & CRS Requirements

- **Polylines (`LineString` / `MultiLineString`):** Rasterized with Liang-Barsky clipped Bresenham traversal; elevations are linearly interpolated between upstream and downstream endpoints (lowered by `depth_offset` in interior cells while anchoring endpoints to the receiving bed so carving never creates an artificial outlet pit).
- **Points (`Point` / `MultiPoint`):** Searches the perimeter of a `(2 * point_radius_cells + 1)` window around each point, selects the lowest opposite-hemisphere pair crossing the embankment, and carves a monotonic channel through the culvert center.
- **CRS Validation:** Culvert vectors must share the DEM's CRS. When both the DEM and culvert layer define known CRSs that differ, `burn_culverts` returns an `InvalidArgument` error instead of silently carving at wrong coordinates.
