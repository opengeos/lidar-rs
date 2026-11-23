//! End-to-end integration tests exercising `lidar-rs` against the real sample
//! datasets from `opengeos/lidar` (`dem.tif`, `sink.tif`, `dsm.tif`).

use std::path::PathBuf;

use lidar::{
    DelineateOptions, ExtractSinksOptions, Resampling, add_crs, clip_image, csv_points_to_vector,
    delineate_depressions, delineate_mounts, extract_sinks, flip_dem_file, gaussian_filter_file,
    image_to_cog, join_csv_to_vector, load_raster, mean_filter_file, median_filter_file, mosaic,
    polygonize, reproject_image, resample,
};

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("test_data")
}

#[test]
fn test_mean_median_gaussian_filters_on_sample_dem() {
    let dem_path = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();

    let mean_out = tmp.path().join("mean.tif");
    let med_out = tmp.path().join("median.tif");
    let gau_out = tmp.path().join("gaussian.tif");

    let mean = mean_filter_file(&dem_path, 3, Some(&mean_out)).unwrap();
    let med = median_filter_file(&dem_path, 3, Some(&med_out)).unwrap();
    let gau = gaussian_filter_file(&dem_path, 1.0, Some(&gau_out)).unwrap();

    assert!(mean_out.exists());
    assert!(med_out.exists());
    assert!(gau_out.exists());
    assert_eq!((mean.width(), mean.height()), (400, 400));
    assert_eq!((med.width(), med.height()), (400, 400));
    assert_eq!((gau.width(), gau.height()), (400, 400));

    // Reload from GeoTIFF and verify dimensions & CRS survived.
    let reloaded = load_raster::<f64>(&med_out, None).unwrap();
    assert_eq!((reloaded.width(), reloaded.height()), (400, 400));
    assert!(!reloaded.projection.is_empty());
}

#[test]
fn test_extract_sinks_on_sample_dem() {
    let dem_path = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();

    let opts = ExtractSinksOptions {
        min_size: 1000,
        keep_files: true,
        ..Default::default()
    };
    let res = extract_sinks(&dem_path, tmp.path(), &opts).unwrap();

    assert!(res.sink_path.as_ref().unwrap().exists());
    assert!(res.refined_dem_path.as_ref().unwrap().exists());
    assert!(res.filled_dem_path.as_ref().unwrap().exists());
    assert!(res.dem_diff_path.as_ref().unwrap().exists());
    assert!(res.depth_path.as_ref().unwrap().exists());
    assert!(res.region_path.as_ref().unwrap().exists());
    assert!(res.csv_path.as_ref().unwrap().exists());
    assert!(res.shapefile_path.as_ref().unwrap().exists());
    assert!(res.gpkg_path.exists());
    assert!(
        !res.depressions.is_empty(),
        "expected at least one sink > 1000 pixels in sample dem.tif"
    );

    // Check GeoPackage has joined attributes.
    let vds = terrio::open_vector(res.gpkg_path.to_str().unwrap(), &Default::default()).unwrap();
    assert_eq!(vds.layers.len(), 1);
    let layer = &vds.layers[0];
    assert_eq!(layer.features.len(), res.depressions.len());
    let field_names: Vec<&str> = layer
        .schema
        .fields()
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert!(field_names.contains(&"region_id"));
    assert!(field_names.contains(&"volume"));
    assert!(field_names.contains(&"avg_depth"));
}

#[test]
fn test_extract_sinks_drop_intermediates() {
    let dem_path = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();

    let opts = ExtractSinksOptions {
        min_size: 1000,
        keep_files: false,
        ..Default::default()
    };
    let res = extract_sinks(&dem_path, tmp.path(), &opts).unwrap();
    assert!(res.gpkg_path.exists());
    assert!(res.sink_path.is_none());
    assert!(res.shapefile_path.is_none());
    assert!(!tmp.path().join("sink.tif").exists());
    assert!(!tmp.path().join("regions.shp").exists());
}

#[test]
fn test_delineate_depressions_on_sample_sink() {
    let sink_path = test_data_dir().join("sink.tif");
    let tmp = tempfile::tempdir().unwrap();

    let opts = DelineateOptions {
        min_size: 1000,
        min_depth: 0.3,
        interval: 0.3,
        bool_level_shp: true,
        bool_comb_tif: false,
        export_gpkg: true,
        compress: false,
    };
    let res = delineate_depressions(&sink_path, tmp.path(), &opts).unwrap();

    assert!(res.depression_id_path.exists());
    assert!(res.depression_level_path.exists());
    assert!(res.shapefile_path.exists());
    assert!(res.gpkg_path.as_ref().unwrap().exists());
    assert!(res.csv_path.exists());
    assert!(res.num_regions >= 1);
    assert!(res.max_level >= 1);
    assert_eq!(res.level_shapefiles.len(), res.max_level as usize);
    for shp in &res.level_shapefiles {
        assert!(shp.exists(), "missing level shapefile: {}", shp.display());
    }

    // Verify CSV has expected 24-column header and row count.
    let csv = std::fs::read_to_string(&res.csv_path).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines[0],
        "id,level,count,area,volume,avg_depth,max_depth,min_elev,max_elev,children_id,region_id,perimeter,major_axis,minor_axis,elongatedness,eccentricity,orientation,area_bbox_ratio,parent_id,circularity,depth_diam_ratio,hypsometric_integral,rim_slope_deg,artifact_flag"
    );
    assert_eq!(lines.len(), res.depressions.len() + 1);
}

#[test]
fn test_full_pipeline_median_extract_delineate() {
    let dem_path = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();

    let med_path = tmp.path().join("median.tif");
    median_filter_file(&dem_path, 3, Some(&med_path)).unwrap();

    let sink_res = extract_sinks(
        &med_path,
        tmp.path(),
        &ExtractSinksOptions {
            min_size: 1000,
            ..Default::default()
        },
    )
    .unwrap();

    let dep_res = delineate_depressions(
        sink_res.sink_path.as_ref().unwrap(),
        tmp.path(),
        &DelineateOptions {
            min_size: 1000,
            min_depth: 0.5,
            interval: 0.3,
            bool_level_shp: false,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(!dep_res.depressions.is_empty());
    assert!(dep_res.depression_id_path.exists());
    assert!(dep_res.depression_level_path.exists());
}

#[test]
fn test_delineate_mounts_on_sample_dsm() {
    let dsm_path = test_data_dir().join("dsm.tif");
    let tmp = tempfile::tempdir().unwrap();

    let opts = DelineateOptions {
        min_size: 1000,
        min_depth: 0.3,
        interval: 0.3,
        bool_level_shp: false,
        ..Default::default()
    };
    let res = delineate_mounts(&dsm_path, tmp.path(), &opts).unwrap();

    assert!(res.flipped_dem_path.exists());
    assert!(res.mounts.depression_id_path.exists());
    assert!(res.mounts.depression_level_path.exists());
    assert!(!res.mounts.depressions.is_empty());
}

#[test]
fn test_raster_and_vector_utilities() {
    let dem_path = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();

    // Flip DEM
    let flipped = tmp.path().join("flipped.tif");
    flip_dem_file(&dem_path, 100.0, Some(&flipped)).unwrap();
    assert!(flipped.exists());

    // Resample to 2m
    let resampled = tmp.path().join("resampled_2m.tif");
    resample(&dem_path, &resampled, 2.0, Resampling::Bilinear).unwrap();
    let r_arr = load_raster::<f64>(&resampled, None).unwrap();
    assert_eq!((r_arr.width(), r_arr.height()), (200, 200));

    // Reproject to EPSG:4326
    let reproj = tmp.path().join("reproj_4326.tif");
    reproject_image(&resampled, &reproj, "EPSG:4326", Resampling::Bilinear, None).unwrap();
    assert!(reproj.exists());

    // Convert to COG
    let cog = tmp.path().join("dem_cog.tif");
    image_to_cog(&resampled, &cog, Some("DEFLATE")).unwrap();
    assert!(cog.exists());
    let cog_validation = terrio::validate_cog(cog.to_str().unwrap(), false).unwrap();
    assert!(
        cog_validation.is_valid,
        "expected valid COG: {cog_validation:?}"
    );

    // Add CRS
    let with_crs = tmp.path().join("with_crs.tif");
    add_crs(&resampled, "26915", Some(&with_crs)).unwrap();
    assert!(with_crs.exists());

    // Mosaic single and duplicate inputs
    let mosaicked = tmp.path().join("mosaic.tif");
    mosaic(&[&resampled, &with_crs], &mosaicked).unwrap();
    assert!(mosaicked.exists());

    // Clip by polygon GeoPackage in the DEM's own CRS covering the inner 200x200 pixels.
    let mask_path = tmp.path().join("mask.gpkg");
    let dem_ds = terrio::open_raster(dem_path.to_str().unwrap()).unwrap();
    let gt = dem_ds.geo_transform();
    let x0 = gt[0] + 100.0 * gt[1];
    let x1 = gt[0] + 300.0 * gt[1];
    let y1 = gt[3] + 100.0 * gt[5];
    let y0 = gt[3] + 300.0 * gt[5];
    let mut mask_layer =
        terrio::Layer::new("mask".to_string()).with_geom_type(terrio::GeometryType::Polygon);
    dem_ds.crs().apply_to_layer(&mut mask_layer);
    mask_layer.features.push(terrio::Feature {
        fid: 1,
        geometry: Some(terrio::Geometry::Polygon {
            exterior: terrio::vector::Ring(vec![
                terrio::vector::Coord::xy(x0, y0),
                terrio::vector::Coord::xy(x1, y0),
                terrio::vector::Coord::xy(x1, y1),
                terrio::vector::Coord::xy(x0, y1),
                terrio::vector::Coord::xy(x0, y0),
            ]),
            interiors: Vec::new(),
        }),
        attributes: Vec::new(),
    });
    terrio::write_vector(
        &[mask_layer],
        mask_path.to_str().unwrap(),
        &terrio::VectorWriteOptions::default(),
    )
    .unwrap();
    let clipped = tmp.path().join("clipped.tif");
    clip_image(&dem_path, &mask_path, &clipped).unwrap();
    let c_arr = load_raster::<f64>(&clipped, None).unwrap();
    assert_eq!((c_arr.width(), c_arr.height()), (200, 200));

    // CSV points to Shapefile & GeoPackage
    let pts_csv = tmp.path().join("cities.csv");
    std::fs::write(
        &pts_csv,
        "name,latitude,longitude,elev\nKnoxville,35.9606,-83.9207,270.0\nDenver,39.7392,-104.9903,1609.0\n",
    )
    .unwrap();
    let pts_shp = tmp.path().join("cities.shp");
    let n_pts = csv_points_to_vector(&pts_csv, &pts_shp, "latitude", "longitude").unwrap();
    assert_eq!(n_pts, 2);
    assert!(pts_shp.exists());

    // Polygonize + Join tables round-trip
    let sink_res = extract_sinks(
        &dem_path,
        tmp.path().join("sinks"),
        &ExtractSinksOptions {
            min_size: 1000,
            ..Default::default()
        },
    )
    .unwrap();
    let poly_geojson = tmp.path().join("regions.geojson");
    let n_poly = polygonize(sink_res.region_path.as_ref().unwrap(), &poly_geojson, "id").unwrap();
    assert_eq!(n_poly, sink_res.depressions.len());
    let joined_shp = tmp.path().join("joined.shp");
    let n_joined = join_csv_to_vector(
        &poly_geojson,
        sink_res.csv_path.as_ref().unwrap(),
        &joined_shp,
        "id",
        "region_id",
    )
    .unwrap();
    assert_eq!(n_joined, n_poly);

    // Safety guards: reject overwriting input DEM, mosaic/cog input == output, invalid lat/lon.
    assert!(extract_sinks(&dem_path, test_data_dir(), &ExtractSinksOptions::default()).is_err());
    assert!(mosaic(&[&resampled], &resampled).is_err());
    assert!(image_to_cog(&resampled, &resampled, None).is_err());

    let bad_csv = tmp.path().join("bad_pts.csv");
    std::fs::write(&bad_csv, "name,latitude,longitude\nBad,95.0,-83.0\n").unwrap();
    assert!(
        csv_points_to_vector(
            &bad_csv,
            tmp.path().join("bad.shp"),
            "latitude",
            "longitude"
        )
        .is_err()
    );

    // UTF-8 BOM + quoted multiline CSV record should parse cleanly into 1 point.
    let multi_csv = tmp.path().join("multi_pts.csv");
    std::fs::write(
        &multi_csv,
        "\u{feff}latitude,longitude,name,note\n35.96,-83.92,A,\"first\nsecond\"\n",
    )
    .unwrap();
    let multi_gpkg = tmp.path().join("multi.gpkg");
    assert_eq!(
        csv_points_to_vector(&multi_csv, &multi_gpkg, "latitude", "longitude").unwrap(),
        1
    );
}

#[cfg(feature = "pointcloud")]
#[test]
fn test_pointcloud_convert_and_lidar_to_dsm() {
    use lidar::pointcloud::{
        Dimension, LidarToDsmOptions, PointCloud, PointWriteOptions, Values, convert_lidar,
        lidar_to_dsm, read_lidar, write_lidar,
    };

    let tmp = tempfile::tempdir().unwrap();
    let las_path = tmp.path().join("sample.las");
    let laz_path = tmp.path().join("sample.laz");
    let dsm_path = tmp.path().join("sample_dsm.tif");

    // Build a synthetic 20x20 point cloud with a mound in the center.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    for r in 0..20 {
        for c in 0..20 {
            xs.push(500_000.0 + c as f64);
            ys.push(4_000_000.0 + r as f64);
            let dist = ((r as f64 - 10.0).powi(2) + (c as f64 - 10.0).powi(2)).sqrt();
            zs.push(100.0 + (10.0 - dist).max(0.0));
        }
    }
    let cloud = PointCloud::new(
        vec![
            Dimension::new("x", Values::F64(xs)),
            Dimension::new("y", Values::F64(ys)),
            Dimension::new("z", Values::F64(zs)),
        ],
        terrio::crs::SpatialRef::parse("EPSG:32615").unwrap(),
    );

    write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();
    let n = convert_lidar(&las_path, &laz_path, Some(6), Some(4)).unwrap();
    assert_eq!(n, 400);

    let reloaded = read_lidar(&laz_path).unwrap();
    assert_eq!(reloaded.len(), 400);

    let (cols, rows) = lidar_to_dsm(
        &laz_path,
        &dsm_path,
        &LidarToDsmOptions {
            resolution: 1.0,
            radius: Some(1.5),
            min_z: Some(90.0),
            max_z: Some(120.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(cols >= 19 && rows >= 19);
    assert!(dsm_path.exists());
}

#[test]
fn test_full_inundation_hydrology_pipeline_on_sample_dem() {
    use lidar::{
        AnimateInundationOptions, CatchmentHierarchyOptions, DelineateCatchmentsOptions,
        DelineateFlowPathsOptions, SimulateInundationOptions, animate_inundation,
        delineate_catchment_hierarchy, delineate_catchments, delineate_flow_paths,
        simulate_inundation,
    };

    let tmp = tempfile::tempdir().unwrap();
    let dem_path = test_data_dir().join("dem.tif");

    // Step 1: Median filter + Extract sinks
    let med_path = tmp.path().join("median.tif");
    median_filter_file(&dem_path, 3, Some(&med_path)).unwrap();
    let sinks = extract_sinks(
        &med_path,
        tmp.path(),
        &ExtractSinksOptions {
            min_size: 1000,
            ..Default::default()
        },
    )
    .unwrap();
    let refined_dem = sinks.refined_dem_path.as_ref().unwrap();
    let filled_dem = sinks.filled_dem_path.as_ref().unwrap();
    let region_tif = sinks.region_path.as_ref().unwrap();
    let sink_tif = sinks.sink_path.as_ref().unwrap();

    // Step 2: Delineate catchments (2_Delineate_Catchment.py)
    let cat_dir = tmp.path().join("catchments");
    let cats = delineate_catchments(
        refined_dem,
        region_tif,
        &cat_dir,
        &DelineateCatchmentsOptions::default(),
    )
    .unwrap();
    assert!(!cats.records.is_empty());
    assert!(cats.watershed_path.exists());
    assert!(cats.vector_path.exists());
    assert!(cats.csv_path.exists());
    for rec in &cats.records {
        assert!(rec.catchment_area >= rec.sink_area);
        assert!(rec.dep_to_cat_ratio > 0.0 && rec.dep_to_cat_ratio <= 1.0);
    }

    // Step 3: Delineate overland flow paths (3_Flow_Path.py)
    let flowpath_shp = tmp.path().join("flowpath.shp");
    let fp = delineate_flow_paths(
        filled_dem,
        region_tif,
        &flowpath_shp,
        &DelineateFlowPathsOptions {
            rain_intensity_cm_h: 5.0,
            refined_dem: Some(refined_dem.clone()),
        },
    )
    .unwrap();
    assert!(fp.flowpath_vector_path.exists());
    assert!(fp.csv_path.exists());

    // Step 4: Delineate depression hierarchy with per-level shapefiles
    let deps = delineate_depressions(
        sink_tif,
        tmp.path(),
        &DelineateOptions {
            min_size: 1000,
            min_depth: 0.3,
            interval: 0.3,
            bool_level_shp: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(deps.max_level >= 2);
    let level_dir = tmp.path().join("shp-level");

    // Step 5: Delineate catchment hierarchy (5_Catchment_Hierarchy.py)
    let hir_dir = tmp.path().join("catchment_hir");
    let hir_tif = hir_dir.join("catchment_hir.tif");
    let hir = delineate_catchment_hierarchy(
        refined_dem,
        &level_dir,
        &hir_tif,
        &CatchmentHierarchyOptions::default(),
    )
    .unwrap();
    assert!(hir.catchment_raster_path.exists());
    assert!(hir.merged_shapefile_path.exists());
    assert!(hir.csv_path.exists());
    assert_eq!(hir.level_shapefiles.len(), deps.max_level as usize);
    assert!(!hir.records.is_empty());

    // Verify in-memory compressed-forest catchment hierarchy matches per-level reference
    let dem = load_raster::<f64>(refined_dem, None).unwrap();
    let dep_lvl = load_raster::<i32>(&deps.depression_level_path, None).unwrap();
    let (mem_hir, mem_recs) =
        lidar::delineate_catchment_hierarchy_in_memory(&dem, &dep_lvl, 1000).unwrap();
    let file_hir = load_raster::<f64>(&hir.catchment_raster_path, None).unwrap();
    assert_eq!(mem_hir.data().len(), file_hir.data().len());
    assert_eq!(mem_recs.len(), hir.records.len());
    for (a, b) in mem_hir.data().iter().zip(file_hir.data().iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "catchment hierarchy raster mismatch: in-memory={a} vs file={b}"
        );
    }

    // Step 6: Simulate inundation dynamics (6_Simulate Inundation.py)
    let sim_dir = tmp.path().join("simulation");
    let sim = simulate_inundation(
        sink_tif,
        &hir_tif,
        &sim_dir,
        &SimulateInundationOptions {
            min_size: 1000,
            min_depth: 0.2,
            interval: 0.3,
            rain_intensity_cm_h: 5.0,
            rain_duration_h: 10.0,
            time_step_h: 2.0,
            compress: false,
        },
    )
    .unwrap();
    assert_eq!(sim.steps.len(), 5);
    assert!(sim.fill_time_path.exists());
    assert!(sim.summary_csv_path.exists());
    for pair in sim.steps.windows(2) {
        assert!(pair[1].inundated_cells >= pair[0].inundated_cells);
    }
    assert!(sim.steps.last().unwrap().inundated_cells > 0);

    // Step 7: Animate inundation to GIF (7_Play_Animation.py)
    let gif_path = tmp.path().join("inundation.gif");
    let anim = animate_inundation(
        refined_dem,
        &sim_dir,
        &gif_path,
        &AnimateInundationOptions {
            frame_delay_ms: 150,
            ping_pong: true,
            max_dimension: Some(200),
            time_step_h: Some(2.0),
            rain_intensity_cm_h: Some(5.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(anim.frame_count, 9); // 5 rising + 4 falling
    assert!(anim.gif_path.exists());
    let bytes = std::fs::read(&gif_path).unwrap();
    assert!(bytes.starts_with(b"GIF89a"));
}

#[test]
fn test_spill_flow_paths_on_sample_dem() {
    use lidar::{
        InundationPipelineOptions, SimulateInundationOptions,
        simulate_inundation_pipeline_in_memory,
    };

    let dem_path = test_data_dir().join("dem.tif");
    let dem = load_raster::<f64>(&dem_path, None).unwrap();
    let pipe = simulate_inundation_pipeline_in_memory(
        &dem,
        &InundationPipelineOptions {
            kernel_size: 3,
            simulation: SimulateInundationOptions {
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                rain_intensity_cm_h: 5.0,
                rain_duration_h: 50.0,
                time_step_h: 1.0,
                compress: false,
            },
        },
    )
    .unwrap();

    eprintln!(
        "Total depressions: {}, spill paths: {}",
        pipe.depressions.depressions.len(),
        pipe.spill_paths.len()
    );
    for p in &pipe.spill_paths {
        eprintln!(
            "  dep {} (L{}) -> dep {} | spill_t={:.2}h merge_t={:?} elev={:.2} len={} start={:?} end={:?}",
            p.from_dep_id,
            p.from_level,
            p.to_dep_id,
            p.spill_time_h,
            p.merge_time_h.map(|v| format!("{v:.2}h")),
            p.spill_elev,
            p.cells.len(),
            p.cells.first(),
            p.cells.last(),
        );
    }
    assert!(
        pipe.spill_paths.len() >= 3,
        "expected at least 3 inter-depression spill flow paths on sample DEM, got {}",
        pipe.spill_paths.len()
    );
    for p in &pipe.spill_paths {
        assert!(p.cells.len() >= 2);
        assert!(p.spill_time_h.is_finite() && p.spill_time_h >= 0.0);
        if let Some(mt) = p.merge_time_h {
            assert!(mt > p.spill_time_h);
        }
        if p.to_dep_id != 0 {
            let &(end_c, end_r) = p.cells.last().unwrap();
            assert_eq!(
                pipe.depressions.depression_id.get(end_c, end_r) as u32,
                p.to_dep_id,
                "spill path {} -> {} must terminate inside target depression {}",
                p.from_dep_id,
                p.to_dep_id,
                p.to_dep_id
            );
        }
    }
}

#[test]
fn test_hydro_conditioning_breach_and_culverts() {
    use lidar::{
        BreachMode, BreachOptions, BurnCulvertsOptions, breach_depressions_file, burn_culverts_file,
    };
    use terrio::vector::Coord;
    use terrio::{Feature, Geometry, GeometryType, Layer, VectorWriteOptions};

    let dem_path = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();

    // Selective breaching preserves deep wetlands while carving shallow barriers
    let breached_path = tmp.path().join("breached.tif");
    let breached = breach_depressions_file(
        &dem_path,
        &breached_path,
        &BreachOptions {
            mode: BreachMode::Selective,
            max_depth: Some(0.5),
            max_length: Some(15),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(breached_path.exists());
    assert_eq!((breached.width(), breached.height()), (400, 400));

    // Culvert burning across a polyline inside DEM bounds
    let gt = breached.geotransform;
    let culvert_path = tmp.path().join("culverts.gpkg");
    let mut layer = Layer::new("culverts".to_string()).with_geom_type(GeometryType::LineString);
    if !breached.projection.is_empty()
        && let Ok(srs) = terrio::crs::SpatialRef::parse(&breached.projection)
    {
        srs.apply_to_layer(&mut layer);
    }
    layer.features.push(Feature {
        fid: 1,
        geometry: Some(Geometry::LineString(vec![
            Coord::xy(gt[0] + 100.0 * gt[1], gt[3] + 200.0 * gt[5]),
            Coord::xy(gt[0] + 120.0 * gt[1], gt[3] + 200.0 * gt[5]),
        ])),
        attributes: vec![],
    });
    terrio::write_vector(
        &[layer],
        culvert_path.to_str().unwrap(),
        &VectorWriteOptions::default(),
    )
    .unwrap();

    let burned_path = tmp.path().join("burned.tif");
    let (_burned, carved) = burn_culverts_file(
        &dem_path,
        &culvert_path,
        &burned_path,
        &BurnCulvertsOptions::default(),
    )
    .unwrap();
    assert!(burned_path.exists());
    assert!(carved >= 1);
}

#[test]
fn test_inundation_forcing_scs_and_run_pipeline() {
    use lidar::{
        InundationForcing, RunPipelineOptions, SimulateInundationOptions,
        delineate_catchment_hierarchy_in_memory, delineate_depressions_in_memory,
        extract_sinks_in_memory, median_filter, run_pipeline, scs_curve_number_runoff,
        scs_type2_hyetograph, simulate_inundation_with_forcing_in_memory,
    };

    // Verify SCS-CN runoff equation and Type II hyetograph
    let cn_grid = lidar::Array2D::from_vec(2, 1, vec![70.0, 90.0]).unwrap();
    let q0 = scs_curve_number_runoff(&cn_grid, 5.0).unwrap();
    let q100 = scs_curve_number_runoff(&cn_grid, 100.0).unwrap();
    assert_eq!(q0.get(0, 0), 0.0);
    assert!(q100.get(0, 0) > 0.2 && q100.get(0, 0) < 0.6);
    assert!(q100.get(1, 0) > q100.get(0, 0));

    let hyeto = scs_type2_hyetograph(100.0, 24.0, 1.0).unwrap();
    assert_eq!(hyeto.len(), 24);
    let sum_mm: f64 = hyeto.iter().map(|&(dt, cm_h)| dt * cm_h * 10.0).sum();
    assert!((sum_mm - 100.0).abs() < 1e-3);

    // Forced inundation with runoff_coeff < 1 should delay fill times relative to C = 1
    let dem = load_raster::<f64>(test_data_dir().join("dem.tif"), None).unwrap();
    let med = median_filter(&dem, 3).unwrap();
    let sinks = extract_sinks_in_memory(&med, 1000, lidar::Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 1000, 0.3, 0.3).unwrap();
    let (hir, _) =
        delineate_catchment_hierarchy_in_memory(&sinks.dem_refined, &deps.depression_level, 1000)
            .unwrap();
    let sim_opts = SimulateInundationOptions {
        min_size: 1000,
        min_depth: 0.3,
        interval: 0.3,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 20.0,
        time_step_h: 2.0,
        compress: false,
    };
    let (ft_full, _) = simulate_inundation_with_forcing_in_memory(
        &sinks.sink,
        &hir,
        &sim_opts,
        &InundationForcing::default(),
    )
    .unwrap();
    let mut rc_half =
        lidar::Array2D::from_vec(dem.width(), dem.height(), vec![0.5; dem.data().len()]).unwrap();
    rc_half.copy_metadata_from(&dem);
    let mut infil =
        lidar::Array2D::from_vec(dem.width(), dem.height(), vec![0.5; dem.data().len()]).unwrap();
    infil.copy_metadata_from(&dem);
    let (ft_half, _) = simulate_inundation_with_forcing_in_memory(
        &sinks.sink,
        &hir,
        &sim_opts,
        &InundationForcing {
            runoff_coeff: Some(&rc_half),
            infiltration_cm_h: Some(&infil),
            antecedent_depth: None,
            hyetograph: Some(&hyeto),
        },
    )
    .unwrap();
    assert!(ft_full.data().iter().any(|&v| v >= 0.0));
    assert!(ft_half.data().iter().any(|&v| v >= 0.0));

    // Full all-in-one run_pipeline end-to-end
    let tmp = tempfile::tempdir().unwrap();
    let res = run_pipeline(
        test_data_dir().join("dem.tif"),
        tmp.path(),
        &RunPipelineOptions {
            kernel_size: 3,
            simulation: SimulateInundationOptions {
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                rain_intensity_cm_h: 5.0,
                rain_duration_h: 20.0,
                time_step_h: 2.0,
                compress: false,
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert!(res.num_depressions >= 2);
    assert!(res.max_level >= 2);
    assert!(res.num_spill_paths >= 1);
    assert!(res.depressions_csv.exists());
    assert!(res.depressions_gpkg.as_ref().unwrap().exists());
    assert!(res.spill_paths_gpkg.as_ref().unwrap().exists());
    assert!(res.rating_curves_csv.as_ref().unwrap().exists());
    #[cfg(feature = "serde")]
    assert!(res.connectivity_graph_json.as_ref().unwrap().exists());
    assert!(res.connectivity_graphml.as_ref().unwrap().exists());
    assert!(res.inundation_csv.exists());
    assert!(tmp.path().join("fill_time.tif").exists());
    assert!(res.summary_json.exists());

    // Verify hyetograph step-limit error & cumulative_runoff_mm integration
    assert!(scs_type2_hyetograph(100.0, 24.0, 0.001).is_err());
    let cum_24 = lidar::cumulative_runoff_mm(24.0, 999.0, Some(&hyeto));
    assert!((cum_24 - 100.0).abs() < 1e-3);

    // Verify export_inundation_frames_png rejects mismatched fill_time dimensions
    let bad_ft = lidar::Array2D::from_vec(10, 10, vec![0.0; 100]).unwrap();
    assert!(
        lidar::export_inundation_frames_png(
            &dem,
            &bad_ft,
            2,
            1.0,
            &[],
            tmp.path().join("bad_frames"),
            &lidar::AnimateInundationOptions::default(),
        )
        .is_err()
    );

    // Verify GraphML has valid XML graph tags and morphometry fields exist in GeoPackage
    let graphml = std::fs::read_to_string(res.connectivity_graphml.as_ref().unwrap()).unwrap();
    assert!(graphml.contains("<graphml") && graphml.contains("<node id="));
    let vds = terrio::open_vector(
        res.depressions_gpkg.as_ref().unwrap().to_str().unwrap(),
        &Default::default(),
    )
    .unwrap();
    let fnames: Vec<&str> = vds.layers[0]
        .schema
        .fields()
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert!(fnames.contains(&"parent_id"));
    assert!(fnames.contains(&"circularity"));
    assert!(fnames.contains(&"hypsometric_integral"));
    assert!(fnames.contains(&"rim_slope_deg"));
    assert!(fnames.contains(&"artifact_flag"));
}

#[test]
fn test_tiled_delineation_and_png_frame_export() {
    use lidar::{
        AnimateInundationOptions, InundationPipelineOptions, SimulateInundationOptions,
        TiledDelineationOptions, delineate_depressions_tiled, export_inundation_frames_png,
        simulate_inundation_pipeline_in_memory,
    };

    let tmp = tempfile::tempdir().unwrap();
    let tiled_dir = tmp.path().join("tiled");
    let tiled = delineate_depressions_tiled(
        test_data_dir().join("dem.tif"),
        &tiled_dir,
        &TiledDelineationOptions {
            tile_size: 200,
            overlap: 50,
            kernel_size: 3,
            min_size: 1000,
            min_depth: 0.3,
            interval: 0.3,
            use_f32: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(tiled.num_tiles, 4);
    assert!(!tiled.depressions.is_empty());
    assert!(tiled.csv_path.exists());

    // PNG frame sequence export with scale bar and legend
    let dem = load_raster::<f64>(test_data_dir().join("dem.tif"), None).unwrap();
    let pipe = simulate_inundation_pipeline_in_memory(
        &dem,
        &InundationPipelineOptions {
            kernel_size: 3,
            simulation: SimulateInundationOptions {
                min_size: 1000,
                min_depth: 0.3,
                interval: 0.3,
                rain_intensity_cm_h: 5.0,
                rain_duration_h: 6.0,
                time_step_h: 2.0,
                compress: false,
            },
        },
    )
    .unwrap();
    let frames_dir = tmp.path().join("frames");
    let pngs = export_inundation_frames_png(
        &pipe.sinks.dem_refined,
        &pipe.fill_time,
        pipe.steps.len(),
        2.0,
        &pipe.spill_paths,
        &frames_dir,
        &AnimateInundationOptions {
            max_dimension: Some(200),
            rain_intensity_cm_h: Some(5.0),
            show_scale_bar: true,
            show_legend: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(pngs.len(), 3);
    for p in &pngs {
        assert!(p.exists());
        let b = std::fs::read(p).unwrap();
        assert_eq!(&b[..8], b"\x89PNG\r\n\x1a\n");
    }
}

#[cfg(feature = "pointcloud")]
#[test]
fn test_pointcloud_filtering_dtm_and_chm() {
    use lidar::pointcloud::{
        Dimension, GroundFilterMethod, LidarToDsmOptions, LidarToDtmOptions, PointCloud,
        PointWriteOptions, ReturnKind, Values, lidar_to_chm, lidar_to_dsm, lidar_to_dtm,
        write_lidar,
    };

    let tmp = tempfile::tempdir().unwrap();
    let las_path = tmp.path().join("forest.las");

    // 25x25 ground plane at z=100 m (class 2, return 2/2) + elevated canopy points at z=115 m (class 5, return 1/2)
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();
    for r in 0..25 {
        for c in 0..25 {
            let x = 500_000.0 + c as f64;
            let y = 4_000_000.0 + r as f64;
            // Ground point
            xs.push(x);
            ys.push(y);
            zs.push(100.0 + 0.05 * c as f64);
            cls.push(2u8);
            rn.push(2u8);
            nr.push(2u8);
            // Canopy point in central patch
            if (8..17).contains(&r) && (8..17).contains(&c) {
                xs.push(x);
                ys.push(y);
                zs.push(115.0 + 0.05 * c as f64);
                cls.push(5u8);
                rn.push(1u8);
                nr.push(2u8);
            }
        }
    }
    let cloud = PointCloud::new(
        vec![
            Dimension::new("x", Values::F64(xs)),
            Dimension::new("y", Values::F64(ys)),
            Dimension::new("z", Values::F64(zs)),
            Dimension::new("classification", Values::U8(cls)),
            Dimension::new("return_number", Values::U8(rn)),
            Dimension::new("number_of_returns", Values::U8(nr)),
        ],
        terrio::crs::SpatialRef::parse("EPSG:32615").unwrap(),
    );
    write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();

    // DSM filtered to class 2 + Last returns should match bare ground (~100 m)
    let dsm_ground = tmp.path().join("dsm_ground.tif");
    lidar_to_dsm(
        &las_path,
        &dsm_ground,
        &LidarToDsmOptions {
            resolution: 1.0,
            classifications: Some(vec![2]),
            return_kinds: Some(vec![ReturnKind::Last]),
            ..Default::default()
        },
    )
    .unwrap();
    let dsm_arr = load_raster::<f64>(&dsm_ground, None).unwrap();
    let max_g = dsm_arr
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_g.is_finite() && max_g > 99.0 && max_g < 103.0,
        "ground-filtered DSM max should be in (99, 103), got {max_g}"
    );

    // DTM using preclassified and CSF
    let dtm_pre = tmp.path().join("dtm_pre.tif");
    lidar_to_dtm(
        &las_path,
        &dtm_pre,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(dtm_pre.exists());

    let dtm_csf = tmp.path().join("dtm_csf.tif");
    lidar_to_dtm(
        &las_path,
        &dtm_csf,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Csf,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(dtm_csf.exists());

    // CHM = DSM - DTM should capture ~15 m canopy height in the center and ~0 m outside
    let chm_path = tmp.path().join("chm.tif");
    lidar_to_chm(
        &las_path,
        &chm_path,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            ..Default::default()
        },
    )
    .unwrap();
    let chm_arr = load_raster::<f64>(&chm_path, None).unwrap();
    let max_chm = chm_arr
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .fold(0.0f64, f64::max);
    assert!(
        max_chm > 12.0 && max_chm < 18.0,
        "expected ~15 m CHM peak, got {max_chm}"
    );
}

#[test]
fn test_fill_spill_merge_end_to_end_on_sample_dem() {
    use lidar::{
        InMemoryFullPipelineOptions, InundationForcing, RoutingMode, RunPipelineOptions,
        SimulateInundationOptions, run_full_pipeline_in_memory, run_pipeline,
    };

    let dem_path = test_data_dir().join("dem.tif");
    let dem = load_raster::<f64>(&dem_path, None).unwrap();

    let sim_opts = SimulateInundationOptions {
        min_size: 1000,
        min_depth: 0.3,
        interval: 0.3,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 50.0,
        time_step_h: 2.0,
        compress: false,
    };

    let pipe_fsm = run_full_pipeline_in_memory(
        &dem,
        &InMemoryFullPipelineOptions {
            kernel_size: 3,
            breach: None,
            simulation: sim_opts.clone(),
            routing: RoutingMode::FillSpillMerge,
            forcing: InundationForcing::default(),
            scalar_runoff_coeff: 1.0,
            stream_vector: None,
        },
    )
    .unwrap();

    let mb = pipe_fsm
        .fsm_mass_balance
        .as_ref()
        .expect("FSM mass balance must be populated");
    assert!(mb.total_runoff_m3 > 0.0);
    assert!(
        mb.mass_balance_error_m3.abs() < 1e-6 * mb.total_runoff_m3.max(1.0),
        "sample DEM FSM mass balance error too large: {mb:?}"
    );

    let basins = pipe_fsm
        .fsm_basins
        .as_ref()
        .expect("FSM basin summaries must be populated");
    assert_eq!(basins.len(), pipe_fsm.depressions.depressions.len());
    assert!(
        basins.iter().any(|b| b.spilled_volume_m3 > 0.0),
        "expected at least one spilling depression on sample DEM"
    );
    assert!(
        basins.iter().any(|b| b.merge_time_h.is_some()),
        "expected hierarchical merges on nested sample DEM"
    );

    // With constant rain and no infiltration no basin drains after filling, so here
    // `fill_time_h`/`spill_to_dep_id` are bounded by the storm and agree with the storm-end `final_volume_m3`:
    // the top-level basin is still filling at 50 h, so it must not report the extrapolated
    // sustained-rainfall fill time (~65 h) that `fill_time.tif` keeps for its cells.
    for b in basins {
        let full = b.final_volume_m3 >= b.capacity_m3 - 1e-6 * b.capacity_m3.max(1.0);
        match b.fill_time_h {
            Some(t) => {
                assert!(t <= sim_opts.rain_duration_h + 1e-9, "{b:?}");
                assert!(full, "basin reports fill_time_h but is not full: {b:?}");
            }
            None => assert!(!full, "full basin is missing fill_time_h: {b:?}"),
        }
    }
    for b in basins.iter().filter(|b| b.fill_time_h.is_none()) {
        // A basin still filling at storm end cannot have spilled during the storm.
        assert_eq!(b.spill_to_dep_id, None, "{b:?}");
    }
    for b in basins {
        if let Some(t) = b.merge_time_h {
            assert!(t <= sim_opts.rain_duration_h + 1e-9, "{b:?}");
            assert_eq!(b.sustained_merge_time_h, Some(t), "{b:?}");
        }
    }
    // Graph `merge` edges keep the sustained-rainfall merge times, matching its spill edges.
    for e in pipe_fsm
        .connectivity_graph
        .edges
        .iter()
        .filter(|e| e.edge_type == "merge")
    {
        let b = basins.iter().find(|b| b.dep_id == e.from_id).unwrap();
        assert_eq!(e.merge_time_h, b.sustained_merge_time_h, "{e:?}");
    }
    assert!(
        basins.iter().any(|b| b.fill_time_h.is_none()),
        "expected the top-level sample basin to still be filling at storm end"
    );
    let max_ft = pipe_fsm
        .fill_time
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .fold(0.0f64, f64::max);
    assert!(
        max_ft > sim_opts.rain_duration_h,
        "fill_time raster should keep sustained-rainfall timing beyond the storm ({max_ft})"
    );

    // Verify monotonic inundation curve and consistent spill paths.
    assert_eq!(pipe_fsm.steps.len(), 25);
    for pair in pipe_fsm.steps.windows(2) {
        assert!(pair[1].inundated_cells >= pair[0].inundated_cells);
    }
    assert!(pipe_fsm.steps.last().unwrap().inundated_cells > 0);
    assert!(
        !pipe_fsm.spill_paths.is_empty(),
        "expected dynamically realized spill paths on sample DEM"
    );
    for sp in &pipe_fsm.spill_paths {
        assert!(sp.cells.len() >= 2);
        assert!(sp.spill_time_h.is_finite() && sp.spill_time_h >= 0.0);
        if let Some(mt) = sp.merge_time_h {
            assert!(
                mt > sp.spill_time_h,
                "merge_time ({mt}) must strictly exceed spill_time ({})",
                sp.spill_time_h
            );
        }
    }

    // Verify end-to-end file pipeline with RoutingMode::FillSpillMerge writes all artifacts.
    let tmp = tempfile::tempdir().unwrap();
    let res = run_pipeline(
        &dem_path,
        tmp.path(),
        &RunPipelineOptions {
            kernel_size: 3,
            simulation: sim_opts,
            routing: RoutingMode::FillSpillMerge,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res.routing, RoutingMode::FillSpillMerge);
    let fsm_csv = res
        .fsm_basins_csv
        .as_ref()
        .expect("fsm_basins.csv must be written");
    assert!(fsm_csv.exists());
    let csv_text = std::fs::read_to_string(fsm_csv).unwrap();
    let csv_lines: Vec<&str> = csv_text.lines().collect();
    assert!(csv_lines[0].starts_with("dep_id,parent_id,level,capacity_m3,"));
    assert_eq!(csv_lines.len(), res.num_depressions + 1);
    assert!(res.fsm_mass_balance.is_some());
    assert!(res.depressions_gpkg.as_ref().unwrap().exists());
    assert!(res.spill_paths_gpkg.as_ref().unwrap().exists());
    assert!(tmp.path().join("fill_time.tif").exists());
}
