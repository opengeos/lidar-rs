# LiDAR Point Clouds (DSM, DTM, CHM)

`lidar-rs` reads, filters, classifies, converts, and grids raw LiDAR point clouds in **100% pure Rust** via `terrio::pointcloud` (supporting **LAS, LAZ, COPC, PLY, and E57**) with zero PDAL, LAStools, or C/C++ dependencies.

![Digital Surface Model, CSF bare-earth Digital Terrain Model, and derived Canopy Height Model generated in pure Rust from sample_forest.laz](https://files.opengeos.org/lidar/images/pointcloud_dsm_dtm_chm.png)

## Supported Operations

| Operation | CLI Subcommand | Python API | Rust API | Description |
| --- | --- | --- | --- | --- |
| **Format conversion** | `lidar convert-lidar` | `convert_lidar` | `convert_lidar` | Convert between LAS, LAZ, COPC, PLY, and E57 with configurable point format (`0..10`). |
| **Filtered DSM gridding** | `lidar lidar-to-dsm` | `lidar_to_dsm` | `lidar_to_dsm` | Grid raw or filtered points (by ASPRS classification codes, pulse return kinds, and `min_z`/`max_z` elevation bounds) into a GeoTIFF DSM. |
| **Ground classification & DTM** | `lidar lidar-to-dtm` | `lidar_to_dtm` | `classify_ground`, `lidar_to_dtm` | Classify bare-earth ground points using pure-Rust **CSF** (Zhang et al., 2016), **PMF** (Zhang et al., 2003), or pre-classified ASPRS ground returns (`class == 2`), then interpolate a continuous bare-earth DTM (`classify_ground` is exposed in Rust; Python runs CSF/PMF via `lidar_to_dtm(..., ground_method="csf"|"pmf")`). |
| **Canopy Height Model** | `lidar lidar-to-chm` | `lidar_to_chm` | `lidar_to_chm` | Grid a first-surface DSM and bare-earth DTM onto one exact shared XY lattice and compute `CHM = max(0, DSM - DTM)` in a single call. |

## Ground Classification Methods

| Method | Algorithm | When to Use |
| --- | --- | --- |
| `csf` (default) | Cloth Simulation Filter (Zhang et al., 2016) | General forested, urban, and mixed terrain; removes both tree canopies and flat-roofed buildings cleanly. |
| `pmf` | Progressive Morphological Filter (Zhang et al., 2003) | Steep terrain or gradual slope transitions where morphological opening windows are preferred. |
| `preclassified` | Filter existing ASPRS class (default `2`) | Pre-classified USGS 3DEP LPC / COPC tiles where vendor bare-earth classification is already verified. |

!!! note "Stale Class-2 Safety"
    When `csf` or `pmf` is selected, `lidar-rs` queries `terrio::pointcloud::filters::csf_classes` / `pmf_classes` directly and selects only points actively classified as ground (`Some(2)`), ensuring stale class-2 labels on non-ground or ignored first-return points never leak into the bare-earth DTM.

## Usage Examples

Try these commands immediately with the sample forested LiDAR cloud [sample_forest.laz](https://files.opengeos.org/lidar/data/sample_forest.laz) (7,821 points, `EPSG:32615`, 22 KB):

=== "CLI"

    ```sh
    curl -LO https://files.opengeos.org/lidar/data/sample_forest.laz

    # 1. Full Digital Surface Model (all returns)
    lidar lidar-to-dsm sample_forest.laz out/dsm.tif --resolution 1.0

    # 2. First-return vegetation & ground filtered DSM
    lidar lidar-to-dsm sample_forest.laz out/dsm_veg.tif \
      --resolution 1.0 --classification 2,5 --return-kind first,only

    # 3. Pure-Rust CSF bare-earth DTM
    lidar lidar-to-dtm sample_forest.laz out/dtm_csf.tif \
      --resolution 1.0 --ground-method csf

    # 4. Pure-Rust PMF bare-earth DTM
    lidar lidar-to-dtm sample_forest.laz out/dtm_pmf.tif \
      --resolution 1.0 --ground-method pmf

    # 5. Canopy Height Model (CHM = max(0, DSM - DTM) on a shared grid)
    lidar lidar-to-chm sample_forest.laz out/chm.tif \
      --resolution 1.0 --ground-method csf

    # 6. Convert LAZ -> uncompressed LAS (point format 6)
    lidar convert-lidar sample_forest.laz out/sample_forest.las --point-format 6
    ```

=== "Python"

    ```python
    import lidar_rs as lidar

    # Filtered DSM (ASPRS classes 2 and 5, first and only returns)
    lidar.lidar_to_dsm(
        "sample_forest.laz",
        "out/dsm.tif",
        resolution=1.0,
        classifications=[2, 5],
        return_kinds=["first", "only"],
    )

    # Bare-earth DTM via pure-Rust Cloth Simulation Filter (CSF)
    cols, rows = lidar.lidar_to_dtm(
        "sample_forest.laz",
        "out/dtm.tif",
        resolution=1.0,
        ground_method="csf",
    )

    # Canopy Height Model (DSM - DTM)
    lidar.lidar_to_chm(
        "sample_forest.laz",
        "out/chm.tif",
        resolution=1.0,
        ground_method="csf",
    )

    # Chain bare-earth DTM directly into depressional hydrology
    pipe = lidar.run_pipeline("out/dtm.tif", "out/hydrology", min_size=10, min_depth=0.2, interval=0.2)
    ```

=== "Rust"

    ```rust,no_run
    use lidar::pointcloud::{
        GroundFilterMethod, LidarToDsmOptions, LidarToDtmOptions, ReturnKind,
        lidar_to_chm, lidar_to_dsm, lidar_to_dtm,
    };

    fn main() -> lidar::Result<()> {
        lidar_to_dsm(
            "sample_forest.laz",
            "out/dsm.tif",
            &LidarToDsmOptions {
                resolution: 1.0,
                classifications: Some(vec![2, 5]),
                return_kinds: Some(vec![ReturnKind::First, ReturnKind::Only]),
                ..Default::default()
            },
        )?;

        let dtm_opts = LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Csf,
            ..Default::default()
        };
        lidar_to_dtm("sample_forest.laz", "out/dtm.tif", &dtm_opts)?;
        lidar_to_chm("sample_forest.laz", "out/chm.tif", &dtm_opts)?;
        Ok(())
    }
    ```

## ASPRS Classification & Return Reference

- **Common ASPRS classification codes:** `1` (Unclassified), `2` (Ground), `3` (Low Vegetation), `4` (Medium Vegetation), `5` (High Vegetation), `6` (Building), `7` (Low Point / Noise), `9` (Water).
- **Return kind names (`--return-kind` / `return_kinds`):** `"first"`, `"last"`, `"intermediate"`, `"only"` (alias `"single"`).
