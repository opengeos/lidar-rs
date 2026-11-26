#![cfg(feature = "pointcloud")]
//! Regression tests for v0.6.0 point cloud, animation, and raster_ops audit findings.

use lidar::{
    AnimateInundationOptions, Array2D, GroundClassifyOptions, GroundFilterMethod,
    LidarToDtmOptions, PitFreeChmOptions, classify_ground_with_options, clip_image,
    csv_points_to_vector, encode_rgb_to_png_bytes, export_inundation_frames_png,
    join_csv_to_vector, lidar_to_dtm, normalize_hag, points_to_chm_pit_free, points_to_dtm,
    polygonize, render_dem_preview_png_bytes, save_raster,
};
use richdem::DemNum;
use terrio::crs::SpatialRef;
use terrio::pointcloud::{Dimension, PointCloud, Values};
use terrio::vector::{Coord, Ring};
use terrio::{
    Feature, FieldDef, FieldType, FieldValue, Geometry, GeometryType, Layer, VectorWriteOptions,
};

fn make_geo_canopy_cloud() -> PointCloud {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let step = 2e-6;
    for r in 0..7 {
        for c in 0..7 {
            let x = -90.0 + (c as f64) * step;
            let y = 45.0 + (r as f64) * step;
            xs.push(x);
            ys.push(y);
            zs.push(100.0);
            cls.push(2u8);
            if (1..=5).contains(&r) && (1..=5).contains(&c) {
                let canopy_z = if r == 3 && c == 3 { 101.0 } else { 115.0 };
                xs.push(x);
                ys.push(y);
                zs.push(canopy_z);
                cls.push(5u8);
            }
        }
    }
    PointCloud::new(
        vec![
            Dimension::new("x", Values::F64(xs)),
            Dimension::new("y", Values::F64(ys)),
            Dimension::new("z", Values::F64(zs)),
            Dimension::new("classification", Values::U8(cls)),
        ],
        SpatialRef::from_epsg(4326).unwrap(),
    )
}

#[test]
fn c1_pit_free_chm_bridges_crown_pit_in_geographic_crs() {
    let cloud = make_geo_canopy_cloud();
    let opts = PitFreeChmOptions {
        resolution: 2e-6,
        height_thresholds: vec![2.0, 5.0, 10.0],
        max_triangle_edge: Some(6e-6),
        dtm: LidarToDtmOptions {
            resolution: 2e-6,
            ground_method: GroundFilterMethod::Preclassified,
            ..Default::default()
        },
        ..Default::default()
    };
    let chm = points_to_chm_pit_free(&cloud, &opts).unwrap();
    let center = chm.get(3, 3);
    assert!(
        center.is_finite() && !center.rd_eq(chm.no_data()) && center >= 10.0,
        "geographic-CRS pit-free CHM should bridge interior crown pit to >= 10 m, got {center}"
    );
}

#[test]
fn c6_normalize_hag_and_classify_ground_handle_capitalized_classification_dim() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    for r in 0..5 {
        for c in 0..5 {
            xs.push(c as f64 * 2.0);
            ys.push(r as f64 * 2.0);
            zs.push(10.0);
            cls.push(2u8);
        }
    }
    xs.push(4.0);
    ys.push(4.0);
    zs.push(22.0);
    cls.push(1u8);

    let cloud = PointCloud::new(
        vec![
            Dimension::new("x", Values::F64(xs)),
            Dimension::new("y", Values::F64(ys)),
            Dimension::new("z", Values::F64(zs)),
            Dimension::new("Classification", Values::U8(cls)),
        ],
        SpatialRef::from_epsg(32615).unwrap(),
    );

    let hag = normalize_hag(&cloud, true)
        .expect("normalize_hag must recognize capitalized Classification dimension");
    assert!(hag.dim("height_above_ground").is_some());

    let classified = classify_ground_with_options(
        &cloud,
        &GroundClassifyOptions {
            method: GroundFilterMethod::Csf,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        classified.dim("Classification").is_none(),
        "classify_ground_with_options must not leave stale capitalized Classification dimension"
    );
    assert!(classified.dim("classification").is_some());
}

#[test]
fn c7_lidar_to_dtm_validates_outlier_options_before_creating_output_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let missing_in = tmp.path().join("does_not_exist.laz");
    let nested_out_dir = tmp.path().join("should_not_be_created");
    let out_dtm = nested_out_dir.join("dtm.tif");
    let opts = LidarToDtmOptions {
        remove_outliers: true,
        outlier_cell: -1.0,
        ..Default::default()
    };
    let err = lidar_to_dtm(&missing_in, &out_dtm, &opts).unwrap_err();
    assert!(
        err.to_string().contains("outlier_cell"),
        "expected invalid outlier_cell error, got: {err}"
    );
    assert!(
        !nested_out_dir.exists(),
        "output directory must not be created when option validation fails"
    );
}

#[test]
fn c8_c9_c10_animation_validation_and_scale_bar_guard() {
    assert!(encode_rgb_to_png_bytes(&[], 0, 0).is_err());

    let mut dem = Array2D::from_vec(64, 64, vec![10.0; 64 * 64]).unwrap();
    dem.set_no_data(-9999.0);
    dem.geotransform = [0.0; 6];
    let opts_no_bar = AnimateInundationOptions {
        show_scale_bar: false,
        show_legend: false,
        ..Default::default()
    };
    let opts_with_bar = AnimateInundationOptions {
        show_scale_bar: true,
        show_legend: false,
        ..Default::default()
    };
    let png_no_bar = render_dem_preview_png_bytes(&dem, None, &opts_no_bar).unwrap();
    let png_with_bar = render_dem_preview_png_bytes(&dem, None, &opts_with_bar).unwrap();
    assert_eq!(
        png_no_bar, png_with_bar,
        "zero geotransform must suppress scale bar rather than fabricating one at 1e-6 m/px"
    );

    let tmp = tempfile::tempdir().unwrap();
    let mut dem2 = Array2D::from_vec(8, 8, vec![10.0; 64]).unwrap();
    dem2.set_no_data(-9999.0);
    dem2.geotransform = [0.0, 1.0, 0.0, 8.0, 0.0, -1.0];
    let mut ft_bad = Array2D::from_vec(8, 8, vec![1.0; 64]).unwrap();
    ft_bad.set_no_data(-1.0);
    ft_bad.geotransform = [100.0, 1.0, 0.0, 8.0, 0.0, -1.0];
    let bad_dir = tmp.path().join("align_fail_dir");
    assert!(
        export_inundation_frames_png(
            &dem2,
            &ft_bad,
            2,
            1.0,
            &[],
            &bad_dir,
            &AnimateInundationOptions::default(),
        )
        .is_err()
    );
    assert!(!bad_dir.exists());

    let mut ft_ok = Array2D::from_vec(8, 8, vec![1.0; 64]).unwrap();
    ft_ok.set_no_data(-1.0);
    ft_ok.geotransform = dem2.geotransform;
    let ok_dir = tmp.path().join("frames_ok");
    let frames = export_inundation_frames_png(
        &dem2,
        &ft_ok,
        2,
        1.0,
        &[],
        &ok_dir,
        &AnimateInundationOptions::default(),
    )
    .unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(
        frames[0].file_name().and_then(|n| n.to_str()),
        Some("frame_001.png")
    );
}

#[test]
fn c11_c12_raster_ops_path_collision_and_csv_edge_cases() {
    let tmp = tempfile::tempdir().unwrap();

    let dem_path = tmp.path().join("dem.tif");
    let mut dem = Array2D::from_vec(4, 4, vec![10.0; 16]).unwrap();
    dem.set_no_data(-9999.0);
    dem.geotransform = [0.0, 1.0, 0.0, 4.0, 0.0, -1.0];
    save_raster(&dem, &dem_path, None, false).unwrap();
    assert!(polygonize(&dem_path, &dem_path, "val").is_err());

    let mask_path = tmp.path().join("mask.geojson");
    let mut layer = Layer::new("mask".to_string()).with_geom_type(GeometryType::Polygon);
    layer
        .schema
        .add_field(FieldDef::new("id", FieldType::Integer));
    layer.features.push(Feature {
        fid: 1,
        geometry: Some(Geometry::Polygon {
            exterior: Ring::new(vec![
                Coord::xy(0.0, 0.0),
                Coord::xy(4.0, 0.0),
                Coord::xy(4.0, 4.0),
                Coord::xy(0.0, 4.0),
                Coord::xy(0.0, 0.0),
            ]),
            interiors: vec![],
        }),
        attributes: vec![FieldValue::Integer(1)],
    });
    terrio::write_vector(
        &[layer],
        mask_path.to_str().unwrap(),
        &VectorWriteOptions::default(),
    )
    .unwrap();
    assert!(
        clip_image(&dem_path, &mask_path, &mask_path).is_err(),
        "clip_image must reject mask == output collision"
    );
    let traversal_out = tmp
        .path()
        .join("nonexistent_dir")
        .join("..")
        .join("dem.tif");
    assert!(
        clip_image(&dem_path, &mask_path, &traversal_out).is_err(),
        "clip_image must reject nonexistent_dir/../dem.tif collision with input"
    );

    // Preclassified DTM must accept capitalized `Classification` dimension names.
    let pre_cloud = PointCloud::new(
        vec![
            Dimension::new(
                "x",
                Values::F64(vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            ),
            Dimension::new(
                "y",
                Values::F64(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            ),
            Dimension::new("z", Values::F64(vec![10.0; 9])),
            Dimension::new("Classification", Values::U8(vec![2u8; 9])),
        ],
        SpatialRef::default(),
    );
    let pre_dtm = points_to_dtm(
        &pre_cloud,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            ground_class: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((pre_dtm.get(1, 1) - 10.0).abs() < 1e-6);

    let csv_ok = tmp.path().join("pts.csv");
    std::fs::write(
        &csv_ok,
        "label,lat,lon\n12\" pipe,40.0,-90.0\n14\" pipe,41.0,-91.0\n",
    )
    .unwrap();
    assert!(csv_points_to_vector(&csv_ok, &csv_ok, "lat", "lon").is_err());
    let pts_out = tmp.path().join("pts.geojson");
    let n_pts = csv_points_to_vector(&csv_ok, &pts_out, "lat", "lon").unwrap();
    assert_eq!(
        n_pts, 2,
        "mid-field quote in unquoted CSV field must not swallow subsequent rows"
    );

    let csv_unclosed = tmp.path().join("unclosed.csv");
    std::fs::write(&csv_unclosed, "label,lat,lon\n\"unclosed,40.0,-90.0\n").unwrap();
    assert!(
        csv_points_to_vector(&csv_unclosed, tmp.path().join("u.geojson"), "lat", "lon").is_err(),
        "unclosed quote at EOF must return Parse error"
    );

    let empty_csv = tmp.path().join("empty.csv");
    std::fs::write(&empty_csv, "").unwrap();
    assert!(
        join_csv_to_vector(
            &mask_path,
            &empty_csv,
            tmp.path().join("joined.geojson"),
            "id",
            "id"
        )
        .is_err(),
        "join_csv_to_vector must reject 0-byte CSV"
    );
}
