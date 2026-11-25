//! Integration tests for Batch 4c: optional pure-Rust `terrio` formats (GeoParquet,
//! FlatGeobuf, LERC, tiles), PDAL-equivalent point-cloud filters (SMRF, ELM, SOR, HAG),
//! and COPC/LAS spatial & point-cap windowing.

use lidar::load_raster;
use lidar::pointcloud::{
    Dimension, GroundFilterMethod, LidarReadOptions, LidarToDtmOptions, PointCloud,
    PointWriteOptions, ReturnKind, Values, classify_ground, filter_outliers_elm,
    filter_outliers_statistical, lidar_to_chm, lidar_to_dtm, normalize_hag, read_lidar_windowed,
    write_lidar,
};

fn make_forest_cloud() -> PointCloud {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();

    // 20x20 sloped bare-earth ground lattice at z = 100 + 0.05 * x_local.
    for r in 0..20 {
        for c in 0..20 {
            xs.push(500_000.0 + c as f64);
            ys.push(4_000_000.0 + r as f64);
            zs.push(100.0 + 0.05 * c as f64);
            cls.push(1u8);
            rn.push(1u8);
            nr.push(1u8);
        }
    }

    // Elevated tree canopy cluster (~15 m above ground) in the center.
    for r in 8..12 {
        for c in 8..12 {
            xs.push(500_000.0 + c as f64 + 0.25);
            ys.push(4_000_000.0 + r as f64 + 0.25);
            zs.push(115.0 + 0.05 * c as f64);
            cls.push(1u8);
            rn.push(1u8);
            nr.push(1u8);
        }
    }

    PointCloud::new(
        vec![
            Dimension::new("x", Values::F64(xs)),
            Dimension::new("y", Values::F64(ys)),
            Dimension::new("z", Values::F64(zs)),
            Dimension::new("classification", Values::U8(cls)),
            Dimension::new("return_number", Values::U8(rn)),
            Dimension::new("number_of_returns", Values::U8(nr)),
        ],
        terrio::crs::SpatialRef::parse("EPSG:32615").unwrap(),
    )
}

#[test]
fn smrf_ground_classification_and_dtm_chm_separate_ground_from_canopy() {
    assert_eq!(
        "smrf".parse::<GroundFilterMethod>().unwrap(),
        GroundFilterMethod::Smrf
    );

    let cloud = make_forest_cloud();
    let classified = classify_ground(&cloud, GroundFilterMethod::Smrf).unwrap();
    let Values::U8(classes) = &classified.dim("classification").unwrap().values else {
        panic!("expected U8 classification");
    };
    let Values::F64(zs) = &classified.dim("z").unwrap().values else {
        panic!("expected F64 z");
    };

    let ground_zs: Vec<f64> = classes
        .iter()
        .zip(zs.iter())
        .filter(|&(c, _)| *c == 2)
        .map(|(_, &z)| z)
        .collect();
    assert!(
        ground_zs.len() >= 300,
        "SMRF should classify most bare-earth returns as ground (got {})",
        ground_zs.len()
    );
    assert!(
        ground_zs.iter().all(|&z| z < 105.0),
        "SMRF must not classify 15 m elevated canopy points as ground"
    );

    // End-to-end lidar_to_dtm and lidar_to_chm using SMRF.
    let tmp = tempfile::tempdir().unwrap();
    let las_path = tmp.path().join("forest.las");
    let dtm_path = tmp.path().join("dtm_smrf.tif");
    let chm_path = tmp.path().join("chm_smrf.tif");
    write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();

    let (dc, dr) = lidar_to_dtm(
        &las_path,
        &dtm_path,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Smrf,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(dc > 0 && dr > 0);
    let dtm = load_raster::<f64>(&dtm_path, None).unwrap();
    let max_dtm = dtm
        .data()
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && v != dtm.no_data())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_dtm < 105.0,
        "SMRF DTM should stay near bare-earth elevation, got {max_dtm}"
    );

    let (cc, cr) = lidar_to_chm(
        &las_path,
        &chm_path,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Smrf,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(cc > 0 && cr > 0);
    let chm = load_raster::<f64>(&chm_path, None).unwrap();
    let max_chm = chm
        .data()
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && v != chm.no_data())
        .fold(0.0f64, f64::max);
    assert!(
        max_chm > 12.0 && max_chm < 18.0,
        "SMRF CHM should capture ~15 m canopy peak, got {max_chm}"
    );
}

#[test]
fn elm_and_statistical_outlier_filters_strip_low_and_high_noise_and_clean_dtm() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();

    for r in 0..16 {
        for c in 0..16 {
            xs.push(500_000.0 + c as f64);
            ys.push(4_000_000.0 + r as f64);
            zs.push(100.0);
            cls.push(2u8);
            rn.push(1u8);
            nr.push(1u8);
        }
    }
    // Inject a severe low-noise pit outlier (z = 50.0, 50 m below terrain) and an ASPRS class-18 high-noise spike.
    xs.push(500_008.0);
    ys.push(4_000_008.0);
    zs.push(50.0);
    cls.push(2u8);
    rn.push(1u8);
    nr.push(1u8);

    xs.push(500_004.0);
    ys.push(4_000_004.0);
    zs.push(250.0);
    cls.push(18u8);
    rn.push(1u8);
    nr.push(1u8);

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

    // ELM with remove_noise=false tags the low spike as ASPRS class 7 without dropping points.
    let tagged = filter_outliers_elm(&cloud, 10.0, 1.0, false).unwrap();
    assert_eq!(tagged.len(), cloud.len());
    let Values::U8(tcls) = &tagged.dim("classification").unwrap().values else {
        panic!("expected U8 classification");
    };
    assert_eq!(
        tcls[256], 7,
        "low spike should be classified as ASPRS class 7"
    );

    // ELM with remove_noise=true removes both the ELM-flagged low noise (class 7) and pre-tagged high noise (class 18).
    let cleaned = filter_outliers_elm(&cloud, 10.0, 1.0, true).unwrap();
    assert_eq!(cleaned.len(), 256);
    let Values::F64(cz) = &cleaned.dim("z").unwrap().values else {
        panic!("expected F64 z");
    };
    assert!(cz.iter().all(|&z| (z - 100.0).abs() < 1e-9));

    // Statistical outlier removal strips both extreme Z spikes as well.
    let sor_cleaned = filter_outliers_statistical(&cloud, 8, 2.0, true).unwrap();
    assert!(sor_cleaned.len() <= 256);
    let Values::F64(sz) = &sor_cleaned.dim("z").unwrap().values else {
        panic!("expected F64 z");
    };
    assert!(sz.iter().all(|&z| z > 90.0 && z < 110.0));

    // lidar_to_dtm with remove_outliers=true avoids the 50 m low spike pulling down the bare-earth DTM.
    let tmp = tempfile::tempdir().unwrap();
    let las_path = tmp.path().join("noisy.las");
    let dtm_path = tmp.path().join("dtm_clean.tif");
    write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();
    lidar_to_dtm(
        &las_path,
        &dtm_path,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            remove_outliers: true,
            ..Default::default()
        },
    )
    .unwrap();
    let dtm = load_raster::<f64>(&dtm_path, None).unwrap();
    let min_dtm = dtm
        .data()
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && v != dtm.no_data())
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_dtm > 99.0,
        "remove_outliers=true must prevent low-noise pit from distorting DTM (min={min_dtm})"
    );
}

#[test]
fn normalize_hag_delaunay_and_nn_compute_accurate_heights_above_ground() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();

    // Tilted ground plane z = 100 + 0.5 * x_local.
    for r in 0..10 {
        for c in 0..10 {
            xs.push(500_000.0 + c as f64);
            ys.push(4_000_000.0 + r as f64);
            zs.push(100.0 + 0.5 * c as f64);
            cls.push(2u8);
        }
    }
    // Add a non-ground canopy point at (500_004.5, 4_000_004.5) exactly 12.0 m above the tilted plane.
    let expected_ground = 100.0 + 0.5 * 4.5;
    xs.push(500_004.5);
    ys.push(4_000_004.5);
    zs.push(expected_ground + 12.0);
    cls.push(5u8);

    let cloud = PointCloud::new(
        vec![
            Dimension::new("x", Values::F64(xs)),
            Dimension::new("y", Values::F64(ys)),
            Dimension::new("z", Values::F64(zs)),
            Dimension::new("classification", Values::U8(cls)),
        ],
        terrio::crs::SpatialRef::parse("EPSG:32615").unwrap(),
    );

    let hag_del = normalize_hag(&cloud, true).unwrap();
    let Values::F64(h_del) = &hag_del.dim("height_above_ground").unwrap().values else {
        panic!("expected F64 height_above_ground");
    };
    assert!(h_del[..100].iter().all(|&h| h.abs() < 1e-9));
    assert!(
        (h_del[100] - 12.0).abs() < 1e-6,
        "Delaunay HAG on planar slope should be exact 12.0, got {}",
        h_del[100]
    );

    let hag_near = normalize_hag(&cloud, false).unwrap();
    let Values::F64(h_nn) = &hag_near.dim("height_above_ground").unwrap().values else {
        panic!("expected F64 height_above_ground");
    };
    assert!(h_nn[..100].iter().all(|&h| h.abs() < 1e-9));
    assert!(
        (h_nn[100] - 12.0).abs() < 0.5,
        "NN HAG should be within 0.5 m of 12.0, got {}",
        h_nn[100]
    );
}

#[test]
fn read_lidar_windowed_filters_spatial_bounds_max_points_and_z_range() {
    let cloud = make_forest_cloud();
    let tmp = tempfile::tempdir().unwrap();
    let copc_path = tmp.path().join("forest.copc.laz");
    write_lidar(&cloud, &copc_path, &PointWriteOptions::default()).unwrap();

    // Spatial bounding-box windowing.
    let sub = read_lidar_windowed(
        &copc_path,
        &LidarReadOptions {
            bounds: Some([500_005.0, 4_000_005.0, 500_010.0, 4_000_010.0]),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!sub.is_empty());
    assert!(sub.len() < cloud.len());
    let b = sub.bounds().unwrap();
    assert!(b[0] >= 500_005.0 - 1e-6 && b[3] <= 500_010.0 + 1e-6);
    assert!(b[1] >= 4_000_005.0 - 1e-6 && b[4] <= 4_000_010.0 + 1e-6);

    // Z-range + return kind filtering.
    let canopy = read_lidar_windowed(
        &copc_path,
        &LidarReadOptions {
            min_z: Some(110.0),
            return_kinds: vec![ReturnKind::Only],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(canopy.len(), 16);

    // max_points thinning on LAS.
    let las_path = tmp.path().join("forest.las");
    write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();
    let thinned = read_lidar_windowed(
        &las_path,
        &LidarReadOptions {
            max_points: Some(50),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!thinned.is_empty());
    assert!(
        thinned.len() <= 60,
        "expected ~50 points, got {}",
        thinned.len()
    );

    // Invalid bounds and resolution must return errors.
    assert!(
        read_lidar_windowed(
            &las_path,
            &LidarReadOptions {
                bounds: Some([10.0, 0.0, 5.0, 10.0]),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        read_lidar_windowed(
            &las_path,
            &LidarReadOptions {
                resolution: Some(0.0),
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[cfg(feature = "parquet")]
#[test]
fn optional_geoparquet_vector_io_works_through_lidar_pipeline() {
    let tmp = tempfile::tempdir().unwrap();
    let dem_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/dem.tif");
    let sink_dir = tmp.path().join("sinks");
    let res = lidar::extract_sinks(
        &dem_path,
        &sink_dir,
        &lidar::ExtractSinksOptions {
            min_size: 1000,
            ..Default::default()
        },
    )
    .unwrap();

    let parquet_path = tmp.path().join("sinks.parquet");
    let count = lidar::polygonize(
        res.region_path.as_ref().unwrap(),
        &parquet_path,
        "region_id",
    )
    .unwrap();
    assert!(count > 0);
    assert!(parquet_path.exists());

    let ds = terrio::open_vector(
        parquet_path.to_str().unwrap(),
        &terrio::VectorOpenOptions::default(),
    )
    .unwrap();
    assert_eq!(ds.layers.len(), 1);
    assert_eq!(ds.layers[0].features.len(), count);
}

#[cfg(feature = "flatgeobuf")]
#[test]
fn optional_flatgeobuf_vector_io_works_through_lidar_pipeline() {
    let tmp = tempfile::tempdir().unwrap();
    let dem_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/dem.tif");
    let sink_dir = tmp.path().join("sinks");
    let res = lidar::extract_sinks(
        &dem_path,
        &sink_dir,
        &lidar::ExtractSinksOptions {
            min_size: 1000,
            ..Default::default()
        },
    )
    .unwrap();

    let fgb_path = tmp.path().join("sinks.fgb");
    let count =
        lidar::polygonize(res.region_path.as_ref().unwrap(), &fgb_path, "region_id").unwrap();
    assert!(count > 0);
    assert!(fgb_path.exists());

    let ds = terrio::open_vector(
        fgb_path.to_str().unwrap(),
        &terrio::VectorOpenOptions::default(),
    )
    .unwrap();
    assert_eq!(ds.layers.len(), 1);
    assert_eq!(ds.layers[0].features.len(), count);
}
