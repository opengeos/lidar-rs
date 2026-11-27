//! Integration tests for core ExecutionControl progress/cancellation, selectable
//! DelineationMethod::{LevelSet, MergeTree}, RasterOutputOptions { compress, cog },
//! tiled point-cloud DTM/CHM gridding, and remote COPC/EPT AOI streaming helpers.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use lidar::{
    Array2D, CancellationToken, DelineationMethod, Error, ExecutionControl,
    InMemoryFullPipelineOptions, InundationForcing, MergeTreeOptions, ProgressEvent,
    RasterOutputOptions, RatingCurvesFromDemOptions, Resampling, RoutingMode, RunPipelineOptions,
    SimulateInundationOptions, TiledDelineationOptions, TiledOutputOptions, Topology,
    clip_image_with_options, compute_rating_curves_from_dem,
    delineate_catchment_hierarchy_in_memory, delineate_depressions_in_memory_with_control,
    delineate_depressions_merge_tree_in_memory_with_control,
    delineate_depressions_merge_tree_with_dem_in_memory_with_control,
    delineate_depressions_tiled_with_outputs, extract_sinks_in_memory, load_raster,
    mosaic_with_options, reproject_image_with_options, resample_with_options,
    run_full_pipeline_in_memory_with_control, run_pipeline, save_raster,
    simulate_fill_spill_merge_in_memory_with_control, simulate_inundation_in_memory_with_control,
};

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
}

/// Builds a synthetic 24x24 DEM containing two nested bowls connected by a saddle.
fn synthetic_nested_dem() -> Array2D<f64> {
    let (w, h) = (24usize, 24usize);
    let mut data = vec![20.0f64; w * h];
    for r in 2..22 {
        for c in 2..22 {
            // Outer bowl rim at 18.0
            let d_outer =
                (((r as f64 - 12.0).powi(2) + (c as f64 - 12.0).powi(2)).sqrt() / 10.0).min(1.0);
            let mut z = 14.0 + 4.0 * d_outer;
            // Sub-bowl A centered at (9, 9)
            let da = ((r as f64 - 9.0).powi(2) + (c as f64 - 9.0).powi(2)).sqrt();
            if da < 4.5 {
                z = z.min(10.0 + da * 0.8);
            }
            // Sub-bowl B centered at (15, 15)
            let db = ((r as f64 - 15.0).powi(2) + (c as f64 - 15.0).powi(2)).sqrt();
            if db < 4.5 {
                z = z.min(10.5 + db * 0.8);
            }
            data[r * w + c] = z;
        }
    }
    let mut dem = Array2D::from_vec(w, h, data).unwrap();
    dem.geotransform = [500_000.0, 2.0, 0.0, 4_500_000.0, 0.0, -2.0];
    dem.projection = "EPSG:32615".to_string();
    dem.set_no_data(-9999.0);
    dem
}

fn assert_monotonic_progress(events: &[ProgressEvent]) {
    assert!(!events.is_empty(), "expected at least one progress event");
    let mut prev = -1.0f64;
    for ev in events {
        assert!(
            ev.fraction >= prev - 1e-12 && ev.fraction <= 1.0 + 1e-12,
            "non-monotonic progress: prev={prev}, cur={}",
            ev.fraction
        );
        prev = ev.fraction;
    }
    assert!(
        (events.last().unwrap().fraction - 1.0).abs() < 1e-9,
        "final progress event should reach 1.0, got {}",
        events.last().unwrap().fraction
    );
}

#[test]
fn execution_control_progress_and_cancellation_in_core_algorithms() {
    let dem = synthetic_nested_dem();
    let sinks = extract_sinks_in_memory(&dem, 4, Topology::D8).unwrap();

    // 1. Level-set delineation progress + cancellation.
    let evs = Arc::new(Mutex::new(Vec::new()));
    let cap = Arc::clone(&evs);
    let ctrl =
        ExecutionControl::new().with_progress_fn(move |e| cap.lock().unwrap().push(e.clone()));
    let deps =
        delineate_depressions_in_memory_with_control(&sinks.sink, 4, 0.2, 0.5, &ctrl).unwrap();
    assert!(!deps.depressions.is_empty());
    assert_monotonic_progress(&evs.lock().unwrap());

    let tok = CancellationToken::new();
    tok.cancel();
    let cancel_ctrl = ExecutionControl::new().with_cancel(tok);
    assert!(matches!(
        delineate_depressions_in_memory_with_control(&sinks.sink, 4, 0.2, 0.5, &cancel_ctrl),
        Err(Error::Cancelled)
    ));

    // 2. Merge-tree delineation progress + cancellation (both variants).
    let evs = Arc::new(Mutex::new(Vec::new()));
    let cap = Arc::clone(&evs);
    let ctrl =
        ExecutionControl::new().with_progress_fn(move |e| cap.lock().unwrap().push(e.clone()));
    let mt_opts = MergeTreeOptions {
        min_size: 4,
        min_depth: 0.2,
        dem_is_refined: true,
        ..Default::default()
    };
    let mt = delineate_depressions_merge_tree_with_dem_in_memory_with_control(
        &sinks.sink,
        &sinks.dem_refined,
        &mt_opts,
        &ctrl,
    )
    .unwrap();
    assert!(!mt.nodes.is_empty());
    assert_monotonic_progress(&evs.lock().unwrap());
    assert!(matches!(
        delineate_depressions_merge_tree_in_memory_with_control(
            &sinks.sink,
            &mt_opts,
            &cancel_ctrl
        ),
        Err(Error::Cancelled)
    ));

    // 3. Static inundation progress + cancellation.
    let (cat_hir, _) =
        delineate_catchment_hierarchy_in_memory(&sinks.dem_refined, &deps.depression_level, 4)
            .unwrap();
    let sim_opts = SimulateInundationOptions {
        min_size: 4,
        min_depth: 0.2,
        interval: 0.5,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 10.0,
        time_step_h: 2.0,
        compress: false,
    };
    let evs = Arc::new(Mutex::new(Vec::new()));
    let cap = Arc::clone(&evs);
    let ctrl =
        ExecutionControl::new().with_progress_fn(move |e| cap.lock().unwrap().push(e.clone()));
    let (_ft, nr) =
        simulate_inundation_in_memory_with_control(&sinks.sink, &cat_hir, &sim_opts, &ctrl)
            .unwrap();
    assert!(nr >= 1);
    assert_monotonic_progress(&evs.lock().unwrap());
    assert!(matches!(
        simulate_inundation_in_memory_with_control(&sinks.sink, &cat_hir, &sim_opts, &cancel_ctrl),
        Err(Error::Cancelled)
    ));

    // 4. Fill-Spill-Merge progress + cancellation.
    let evs = Arc::new(Mutex::new(Vec::new()));
    let cap = Arc::clone(&evs);
    let ctrl =
        ExecutionControl::new().with_progress_fn(move |e| cap.lock().unwrap().push(e.clone()));
    let fsm = simulate_fill_spill_merge_in_memory_with_control(
        &sinks.dem_refined,
        &sinks.sink,
        &deps,
        &sim_opts,
        &InundationForcing::default(),
        &ctrl,
    )
    .unwrap();
    assert!(!fsm.basins.is_empty());
    assert_monotonic_progress(&evs.lock().unwrap());
    assert!(matches!(
        simulate_fill_spill_merge_in_memory_with_control(
            &sinks.dem_refined,
            &sinks.sink,
            &deps,
            &sim_opts,
            &InundationForcing::default(),
            &cancel_ctrl,
        ),
        Err(Error::Cancelled)
    ));

    // 5. Full in-memory pipeline progress (monotonic) + cancellation + direct sub_range remapping.
    let sub_evs = Arc::new(Mutex::new(Vec::new()));
    let sub_cap = Arc::clone(&sub_evs);
    let root_ctrl =
        ExecutionControl::new().with_progress_fn(move |e| sub_cap.lock().unwrap().push(e.clone()));
    let sub = root_ctrl.sub_range(0.25, 0.75);
    sub.report("phase", 0.0, None);
    sub.report("phase", 0.5, None);
    sub.report("phase", 1.0, None);
    let fracs: Vec<f64> = sub_evs.lock().unwrap().iter().map(|e| e.fraction).collect();
    assert_eq!(fracs, vec![0.25, 0.5, 0.75]);

    let evs = Arc::new(Mutex::new(Vec::new()));
    let cap = Arc::clone(&evs);
    let ctrl =
        ExecutionControl::new().with_progress_fn(move |e| cap.lock().unwrap().push(e.clone()));
    let pipe_opts = InMemoryFullPipelineOptions {
        kernel_size: 1,
        delineation_method: DelineationMethod::MergeTree,
        simulation: sim_opts.clone(),
        routing: RoutingMode::FillSpillMerge,
        ..Default::default()
    };
    let pipe = run_full_pipeline_in_memory_with_control(&dem, &pipe_opts, &ctrl).unwrap();
    assert!(!pipe.depressions.depressions.is_empty());
    assert!(!pipe.rating_curves.is_empty());
    assert_monotonic_progress(&evs.lock().unwrap());
    assert!(matches!(
        run_full_pipeline_in_memory_with_control(&dem, &pipe_opts, &cancel_ctrl),
        Err(Error::Cancelled)
    ));
}

#[test]
fn selectable_delineation_method_merge_tree_across_pipeline_graph_and_tiling() {
    let dem = synthetic_nested_dem();
    let tmp = tempfile::tempdir().unwrap();
    let dem_path = tmp.path().join("nested.tif");
    save_raster(&dem, &dem_path, None, false).unwrap();

    // 1. run_pipeline with MergeTree
    let pipe_res = run_pipeline(
        &dem_path,
        tmp.path().join("pipe_mt"),
        &RunPipelineOptions {
            kernel_size: 1,
            delineation_method: DelineationMethod::MergeTree,
            simulation: SimulateInundationOptions {
                min_size: 4,
                min_depth: 0.2,
                interval: 0.5,
                rain_intensity_cm_h: 5.0,
                rain_duration_h: 8.0,
                time_step_h: 2.0,
                compress: false,
            },
            routing: RoutingMode::FillSpillMerge,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(pipe_res.num_depressions >= 2);
    assert!(pipe_res.max_level >= 2);
    assert!(pipe_res.rating_curves_csv.as_ref().unwrap().exists());

    // 2. compute_rating_curves_from_dem with MergeTree
    let (n_deps, curves) = compute_rating_curves_from_dem(
        &dem_path,
        &RatingCurvesFromDemOptions {
            kernel_size: 1,
            min_size: 4,
            min_depth: 0.2,
            delineation_method: DelineationMethod::MergeTree,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(n_deps >= 2);
    assert!(!curves.is_empty());

    // 3. delineate_depressions_tiled_with_outputs with MergeTree
    let tiled_res = delineate_depressions_tiled_with_outputs(
        &dem_path,
        tmp.path().join("tiled_mt"),
        &TiledDelineationOptions {
            tile_size: 16,
            overlap: 6,
            kernel_size: 1,
            min_size: 4,
            min_depth: 0.2,
            interval: 0.5,
            topology: Topology::D8,
            use_f32: false,
            delineation_method: DelineationMethod::MergeTree,
        },
        &TiledOutputOptions {
            write_rasters: true,
            write_gpkg: false,
            breach: None,
            compress: true,
        },
    )
    .unwrap();
    assert!(!tiled_res.depressions.is_empty());
    assert!(tiled_res.depression_id_tif.as_ref().unwrap().exists());
}

#[test]
fn raster_ops_with_options_compress_and_cog() {
    let dem_path = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();

    // Resample to COG
    let cog_opts = RasterOutputOptions {
        compress: true,
        cog: true,
    };
    let resampled_cog = tmp.path().join("resampled_cog.tif");
    resample_with_options(
        &dem_path,
        &resampled_cog,
        4.0,
        Resampling::Bilinear,
        &cog_opts,
    )
    .unwrap();
    assert!(resampled_cog.exists());
    let loaded = load_raster::<f64>(&resampled_cog, None).unwrap();
    let (rw, rh) = (loaded.width(), loaded.height());
    assert!(rw > 0 && rh > 0);

    // Mosaic compressed GeoTIFF
    let comp_opts = RasterOutputOptions {
        compress: true,
        cog: false,
    };
    let mosaicked = tmp.path().join("mosaic_comp.tif");
    mosaic_with_options(
        &[&resampled_cog],
        &mosaicked,
        Resampling::Nearest,
        &comp_opts,
    )
    .unwrap();
    let m_loaded = load_raster::<f64>(&mosaicked, None).unwrap();
    assert_eq!((m_loaded.width(), m_loaded.height()), (rw, rh));

    // Reproject with COG output
    let reproj_cog = tmp.path().join("reproj_cog.tif");
    reproject_image_with_options(
        &resampled_cog,
        &reproj_cog,
        "EPSG:4326",
        Resampling::Bilinear,
        None,
        &cog_opts,
    )
    .unwrap();
    assert!(reproj_cog.exists());
    let p_loaded = load_raster::<f64>(&reproj_cog, None).unwrap();
    assert!(p_loaded.width() > 0 && p_loaded.height() > 0);

    // Clip with COG output using a WGS 84 GeoJSON mask covering the center of `reproj_cog`
    let (pw, ph) = (p_loaded.width(), p_loaded.height());
    let gt = p_loaded.geotransform;
    let x0 = gt[0] + (pw as f64 * 0.25) * gt[1];
    let x1 = gt[0] + (pw as f64 * 0.75) * gt[1];
    let y1 = gt[3] + (ph as f64 * 0.25) * gt[5];
    let y0 = gt[3] + (ph as f64 * 0.75) * gt[5];
    let mask_path = tmp.path().join("mask.geojson");
    let geojson = format!(
        r#"{{"type":"FeatureCollection","features":[{{"type":"Feature","properties":{{}},"geometry":{{"type":"Polygon","coordinates":[[[{x0},{y0}],[{x1},{y0}],[{x1},{y1}],[{x0},{y1}],[{x0},{y0}]]]}}}}]}}"#
    );
    std::fs::write(&mask_path, geojson).unwrap();
    let clipped_cog = tmp.path().join("clipped_cog.tif");
    clip_image_with_options(&reproj_cog, &mask_path, &clipped_cog, &cog_opts).unwrap();
    let c_loaded = load_raster::<f64>(&clipped_cog, None).unwrap();
    let (cw, ch) = (c_loaded.width(), c_loaded.height());
    assert!(cw > 0 && ch > 0 && cw <= pw && ch <= ph);
}

#[cfg(feature = "pointcloud")]
#[test]
fn tiled_pointcloud_dtm_and_chm_match_and_support_cog_and_cancellation() {
    use lidar::pointcloud::{
        Dimension, GroundFilterMethod, LidarToDtmOptions, PointCloud, PointWriteOptions,
        TiledPointCloudOptions, Values, lidar_to_chm_tiled, lidar_to_dtm_tiled_with_control,
        write_lidar,
    };
    use terrio::crs::SpatialRef;

    let tmp = tempfile::tempdir().unwrap();
    let las_path = tmp.path().join("synth_forest.las");

    // 20m x 20m sloped terrain with preclassified ground (class 2) and canopy (class 5, +10m).
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    for iy in 0..20 {
        for ix in 0..20 {
            let x = 500_000.0 + ix as f64 + 0.5;
            let y = 4_500_000.0 + iy as f64 + 0.5;
            let gz = 100.0 + 0.2 * ix as f64 + 0.1 * iy as f64;
            xs.push(x);
            ys.push(y);
            zs.push(gz);
            cls.push(2u8);
            // Canopy points in the middle patch
            if (6..14).contains(&ix) && (6..14).contains(&iy) {
                xs.push(x);
                ys.push(y);
                zs.push(gz + 10.0);
                cls.push(5u8);
            }
        }
    }
    let cloud = PointCloud::new(
        vec![
            Dimension::new("x", Values::F64(xs)),
            Dimension::new("y", Values::F64(ys)),
            Dimension::new("z", Values::F64(zs)),
            Dimension::new("classification", Values::U8(cls)),
        ],
        SpatialRef::from_epsg(32615).unwrap(),
    );
    write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();

    let tiled_opts = TiledPointCloudOptions {
        tile_size: 10.0,
        overlap: 4.0,
        dtm: LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            compress: true,
            cog: false,
            ..Default::default()
        },
        read_resolution: None,
        max_level: None,
    };

    let dtm_out = tmp.path().join("tiled_dtm.tif");
    let evs = Arc::new(Mutex::new(Vec::new()));
    let cap = Arc::clone(&evs);
    let ctrl =
        ExecutionControl::new().with_progress_fn(move |e| cap.lock().unwrap().push(e.clone()));
    let (dw, dh) =
        lidar_to_dtm_tiled_with_control(&las_path, &dtm_out, &tiled_opts, &ctrl).unwrap();
    assert!(dw >= 19 && dh >= 19);
    assert_monotonic_progress(&evs.lock().unwrap());
    let dtm = load_raster::<f64>(&dtm_out, None).unwrap();
    let center_z = dtm.get(dw / 2, dh / 2);
    assert!(
        (center_z - 103.0).abs() < 2.0,
        "unexpected tiled DTM elevation at center: {center_z}"
    );

    // Cancellation works cleanly
    let tok = CancellationToken::new();
    tok.cancel();
    let cancel_ctrl = ExecutionControl::new().with_cancel(tok);
    assert!(matches!(
        lidar_to_dtm_tiled_with_control(
            &las_path,
            tmp.path().join("cancelled.tif"),
            &tiled_opts,
            &cancel_ctrl
        ),
        Err(Error::Cancelled)
    ));

    // Tiled CHM with COG output
    let mut chm_opts = tiled_opts;
    chm_opts.dtm.cog = true;
    let chm_out = tmp.path().join("tiled_chm_cog.tif");
    let (cw, ch) = lidar_to_chm_tiled(&las_path, &chm_out, &chm_opts).unwrap();
    assert_eq!((cw, ch), (dw, dh));
    let chm = load_raster::<f64>(&chm_out, None).unwrap();
    let canopy_h = chm.get(cw / 2, ch / 2);
    assert!(
        (canopy_h - 10.0).abs() < 1.5,
        "expected ~10m canopy height at center, got {canopy_h}"
    );
}

#[cfg(all(feature = "download", feature = "pointcloud"))]
#[test]
fn remote_copc_ept_url_validation_and_wgs84_bbox_projection() {
    use lidar::{
        Bbox, EptStreamOptions, stream_remote_pointcloud_aoi, transform_wgs84_bbox_to_crs,
    };
    use terrio::crs::SpatialRef;

    // 1. Unsafe / non-HTTPS / WHATWG-obfuscated URLs must be rejected before any network call.
    for bad in [
        "http://example.org/test.copc.laz",
        "https://localhost/test.copc.laz",
        "https://127.0.0.1/test.copc.laz",
        "https://169.254.169.254/latest/meta-data",
        "https://127.0.0.1\\.example.com/test.copc.laz",
        "https://%31%32%37.0.0.1/test.copc.laz",
        "https://127.0.0.\t1/test.copc.laz",
        "file:///etc/passwd",
    ] {
        let err = stream_remote_pointcloud_aoi(&EptStreamOptions {
            source_url: bad.to_string(),
            ..Default::default()
        });
        assert!(
            matches!(err, Err(Error::InvalidArgument(_))),
            "expected InvalidArgument for unsafe URL {bad}, got {err:?}"
        );
    }

    // 2. WGS 84 bbox projects cleanly into UTM Zone 15N (EPSG:32615).
    let bb = Bbox::new(-93.1, 44.9, -93.0, 45.0).unwrap();
    let utm = SpatialRef::from_epsg(32615).unwrap();
    let proj = transform_wgs84_bbox_to_crs(&bb, &utm).unwrap();
    assert!(proj[0] < proj[2] && proj[1] < proj[3]);
    assert!(proj[0] > 400_000.0 && proj[2] < 600_000.0);
    assert!(proj[1] > 4_900_000.0 && proj[3] < 5_100_000.0);
}
