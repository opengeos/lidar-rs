# Tiled Regional Delineation & `f32` Memory Mode

Processing regional 1-m USGS 3DEP DEMs (100M+ pixels) as full-grid `f64` arrays requires substantial working memory during median filtering, Priority-Flood filling, and level-set slicing.

`delineate_depressions_tiled` (`lidar tiled-depressions`) splits a large DEM into a regular grid of overlapping square tiles, processes tiles in parallel with Rayon (when the `parallel` feature is enabled), deduplicates features along tile seams, and merges globally renumbered depression hierarchies into a single `depressions_info.csv`.

![Overlapping 2x2 tile windows with seam halos on the sample DEM and f32 vs f64 source DEM RAM footprint comparison](https://files.opengeos.org/lidar/images/tiled_delineation_diagram.png)

## How Seam Deduplication Works

1. **Core + Overlap Windows:** Each tile defines a non-overlapping core (`tile_size x tile_size`) padded on all four sides by `overlap` pixels (clamped to DEM bounds).
2. **Parallel Per-Tile Delineation:** Each padded tile runs optional median smoothing, Priority-Flood sink extraction, and Wu et al. (2019) level-set slicing independently in bounded per-tile working memory.
3. **Core Ownership & Largest-Footprint Ranking:** Each observed top-level sink region records its pixel centroid (to test whether it belongs to the tile's core), its lowest-elevation cell `(min_row, min_col)` as a global deduplication key, and its pixel count. Across all overlapping tiles sharing a `min_cell`, `lidar-rs` requires at least one core-owned observation to emit the group and selects the copy with the **largest footprint** (least clipped by any tile boundary).
4. **Global Renumbering:** All surviving regions and their complete parent-child trees are reassigned globally unique sequential `id`, `parent_id`, `children_ids`, and `region_id` values.

!!! warning "Choosing `--overlap`"
    Core-centroid ownership and `min_cell` deduplication are exact for depressions whose full horizontal extent fits inside the padded overlap halo. For landscapes with very wide compound depressions or karst uvalas, set `--overlap` larger than the expected maximum basin radius (e.g., `--overlap 256` or `--overlap 512` on 1-m DEMs).

## `f32` Storage Helpers

Most 3DEP GeoTIFF DEMs are distributed natively as 32-bit floats (`f32`). Loading them with `use_f32` (default `true` in Rust `TiledDelineationOptions`, Python `delineate_depressions_tiled`, and `lidar tiled-depressions`; pass `--use-f64` in `lidar tiled-depressions` for double-precision loading, or opt in with `--use-f32` in `lidar pipeline` / call `lidar::load_dem_f32` directly) halves the source DEM RAM footprint (from 8 bytes/pixel to 4 bytes/pixel) while preserving exact source NoData sentinels during per-tile conversion to `f64`.

## Usage Examples

=== "CLI"

    ```sh
    curl -LO https://files.opengeos.org/lidar/data/dem.tif

    # Parallel overlapping tiled delineation (f32 source storage by default; pass --use-f64 for f64)
    lidar tiled-depressions dem.tif out_tiled \
      --tile-size 200 --overlap 50 \
      --kernel 3 --min-size 1000 --min-depth 0.3 --interval 0.3 \
      --json

    # Large regional 1-m 3DEP DEM example
    lidar tiled-depressions regional_1m.tif out_regional \
      --tile-size 2048 --overlap 256
    ```

=== "Python"

    ```python
    import lidar_rs as lidar

    res = lidar.delineate_depressions_tiled(
        "dem.tif",
        "out_tiled",
        tile_size=200,
        overlap=50,
        kernel_size=3,
        min_size=1000,
        min_depth=0.3,
        interval=0.3,
        use_f32=True,
    )
    print(
        f"Processed {res['num_tiles']} tiles -> "
        f"{len(res['depressions'])} depressions (max level {res['max_level']})"
    )
    print("Merged CSV:", res["csv_path"])
    ```

=== "Rust"

    ```rust,no_run
    use lidar::{TiledDelineationOptions, delineate_depressions_tiled};

    fn main() -> lidar::Result<()> {
        let res = delineate_depressions_tiled(
            "large_dem.tif",
            "out_tiled",
            &TiledDelineationOptions {
                tile_size: 2048,
                overlap: 256,
                kernel_size: 3,
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                use_f32: true,
                ..Default::default()
            },
        )?;
        println!(
            "Processed {} tiles, {} merged depressions -> {}",
            res.num_tiles,
            res.depressions.len(),
            res.csv_path.display()
        );
        Ok(())
    }
    ```
