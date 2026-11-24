//! Integration regression tests for Batch 2 (I/O, raster/vector ops, point cloud, and visualization).

use std::path::PathBuf;

use lidar::{Array2D, Resampling, add_crs, join_csv_to_vector, reproject_image, save_raster};
use terrio::crs::SpatialRef;
use terrio::vector::Coord;
use terrio::{
    Feature, FieldDef, FieldType, FieldValue, Geometry, GeometryType, Layer, VectorOpenOptions,
    VectorWriteOptions,
};

fn write_tiny_dem(path: &std::path::Path) {
    let mut arr = Array2D::from_vec(8, 8, vec![100.0; 64]).unwrap();
    arr.geotransform = [500_000.0, 1.0, 0.0, 4_500_000.0, 0.0, -1.0];
    arr.projection = "EPSG:26915".to_string();
    save_raster(&arr, path, None, false).unwrap();
}

#[test]
fn add_crs_rejects_empty_crs_and_handles_aliased_in_place_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let dem_path = tmp.path().join("dem.tif");
    write_tiny_dem(&dem_path);

    assert!(add_crs(&dem_path, "", None).is_err());
    assert!(add_crs(&dem_path, "   ", None).is_err());

    // Passing an aliased path (`dir/./dem.tif` vs `dir/dem.tif`) must be treated as in-place atomic
    // replacement without corrupting the file.
    let aliased = tmp.path().join(".").join("dem.tif");
    add_crs(&dem_path, "32615", Some(&aliased)).unwrap();
    let ds = terrio::open_raster(dem_path.to_str().unwrap()).unwrap();
    assert_eq!(ds.raster.crs.epsg, Some(32615));
}

#[test]
fn reproject_image_validates_resolution_upfront() {
    let tmp = tempfile::tempdir().unwrap();
    let dem_path = tmp.path().join("dem.tif");
    write_tiny_dem(&dem_path);

    for bad_res in [
        (0.0, 1.0),
        (-2.0, 1.0),
        (1.0, 0.0),
        (f64::NAN, 1.0),
        (1.0, f64::INFINITY),
    ] {
        let out_dir = tmp.path().join("should_not_create");
        let out = out_dir.join("out.tif");
        assert!(
            reproject_image(
                &dem_path,
                &out,
                "EPSG:4326",
                Resampling::Bilinear,
                Some(bad_res)
            )
            .is_err()
        );
        assert!(!out_dir.exists());
    }
}

#[test]
fn join_csv_to_vector_hardening_nan_leading_zeros_empty_rows_collision_and_shp_dbf() {
    let tmp = tempfile::tempdir().unwrap();
    let vec_path = tmp.path().join("regions.geojson");

    let mut layer = Layer::new("regions".to_string()).with_geom_type(GeometryType::Point);
    SpatialRef::from_epsg(4326)
        .unwrap()
        .apply_to_layer(&mut layer);
    layer
        .schema
        .add_field(FieldDef::new("code".to_string(), FieldType::Text));
    layer.schema.add_field(FieldDef::new(
        "long_attribute_alpha".to_string(),
        FieldType::Text,
    ));
    layer.schema.add_field(FieldDef::new(
        "long_attribute_beta".to_string(),
        FieldType::Text,
    ));

    layer.features.push(Feature {
        fid: 1,
        geometry: Some(Geometry::Point(Coord::xy(-99.0, 47.0))),
        attributes: vec![
            FieldValue::Text("01010001".to_string()),
            FieldValue::Text("a1".to_string()),
            FieldValue::Text("b1".to_string()),
        ],
    });
    layer.features.push(Feature {
        fid: 2,
        geometry: Some(Geometry::Point(Coord::xy(-99.1, 47.1))),
        attributes: vec![
            FieldValue::Text("1010001".to_string()),
            FieldValue::Text("a2".to_string()),
            FieldValue::Text("b2".to_string()),
        ],
    });
    layer.features.push(Feature {
        fid: 3,
        geometry: Some(Geometry::Point(Coord::xy(-99.2, 47.2))),
        attributes: vec![
            FieldValue::Text("123".to_string()),
            FieldValue::Text("a3".to_string()),
            FieldValue::Text("b3".to_string()),
        ],
    });
    terrio::write_vector(
        &[layer],
        vec_path.to_str().unwrap(),
        &VectorWriteOptions::default(),
    )
    .unwrap();

    // CSV containing leading-zero key, numeric "123.0" key, and non-finite float strings.
    let csv_path = tmp.path().join("attrs.csv");
    std::fs::write(
        &csv_path,
        "code,label,special_num\n01010001,leading_zero,NaN\n1010001,no_leading_zero,inf\n123.0,numeric_match,Infinity\n",
    )
    .unwrap();

    // Same-path collision must be rejected.
    assert!(join_csv_to_vector(&vec_path, &csv_path, &vec_path, "code", "code").is_err());

    // Join into GeoPackage and verify leading-zero distinction, numeric matching, and text type for NaN/inf.
    let out_gpkg = tmp.path().join("joined.gpkg");
    let count = join_csv_to_vector(&vec_path, &csv_path, &out_gpkg, "code", "code").unwrap();
    assert_eq!(count, 3);

    let ds =
        terrio::open_vector(out_gpkg.to_str().unwrap(), &VectorOpenOptions::default()).unwrap();
    let lyr = &ds.layers[0];
    let special_idx = lyr
        .schema
        .fields()
        .iter()
        .position(|f| f.name == "special_num")
        .unwrap();
    assert_eq!(lyr.schema.fields()[special_idx].field_type, FieldType::Text);

    let label_idx = lyr
        .schema
        .fields()
        .iter()
        .position(|f| f.name == "label")
        .unwrap();
    let labels: Vec<Option<&str>> = lyr
        .features
        .iter()
        .map(|f| match f.attributes.get(label_idx) {
            Some(FieldValue::Text(s)) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        labels,
        vec![
            Some("leading_zero"),
            Some("no_leading_zero"),
            Some("numeric_match")
        ]
    );

    // Join into Shapefile: existing colliding >10-byte fields must be deduplicated uniquely.
    let out_shp = tmp.path().join("joined.shp");
    let shp_count = join_csv_to_vector(&vec_path, &csv_path, &out_shp, "code", "code").unwrap();
    assert_eq!(shp_count, 3);
    let shp_ds =
        terrio::open_vector(out_shp.to_str().unwrap(), &VectorOpenOptions::default()).unwrap();
    let names: std::collections::HashSet<String> = shp_ds.layers[0]
        .schema
        .fields()
        .iter()
        .map(|f| f.name.to_ascii_lowercase())
        .collect();
    assert_eq!(names.len(), shp_ds.layers[0].schema.fields().len());

    // 0-row CSV with headers must still validate join column names.
    let header_only_csv = tmp.path().join("header_only.csv");
    std::fs::write(&header_only_csv, "other_col,val\n").unwrap();
    assert!(
        join_csv_to_vector(
            &vec_path,
            &header_only_csv,
            tmp.path().join("bad_join.gpkg"),
            "code",
            "code"
        )
        .is_err()
    );
    assert!(
        join_csv_to_vector(
            &vec_path,
            &header_only_csv,
            tmp.path().join("bad_vec_col.gpkg"),
            "missing_vec_col",
            "other_col"
        )
        .is_err()
    );

    // 0-row CSV with valid join column succeeds and attaches Null values.
    let valid_empty_csv = tmp.path().join("valid_empty.csv");
    std::fs::write(&valid_empty_csv, "code,extra\n").unwrap();
    let out_empty = tmp.path().join("empty_join.gpkg");
    assert_eq!(
        join_csv_to_vector(&vec_path, &valid_empty_csv, &out_empty, "code", "code").unwrap(),
        3
    );
}

#[cfg(feature = "pointcloud")]
#[test]
fn convert_lidar_rejects_same_path_and_lidar_to_dsm_validates_bounds() {
    use lidar::pointcloud::{
        Dimension, LidarToDsmOptions, PointCloud, PointWriteOptions, Values, convert_lidar,
        lidar_to_dsm, parse_return_kinds, write_lidar,
    };

    let tmp = tempfile::tempdir().unwrap();
    let las_path = tmp.path().join("sample.las");
    let cloud = PointCloud::new(
        vec![
            Dimension::new("x", Values::F64(vec![500_000.0, 500_001.0])),
            Dimension::new("y", Values::F64(vec![4_500_000.0, 4_500_001.0])),
            Dimension::new("z", Values::F64(vec![100.0, 102.0])),
        ],
        SpatialRef::parse("EPSG:32615").unwrap(),
    );
    write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();
    let orig_bytes = std::fs::read(&las_path).unwrap();

    // Reject same path and aliased path without modifying/truncating the source file.
    assert!(convert_lidar(&las_path, &las_path, None, None).is_err());
    let aliased = tmp.path().join(".").join("sample.las");
    assert!(convert_lidar(&las_path, &aliased, None, None).is_err());
    assert_eq!(std::fs::read(&las_path).unwrap(), orig_bytes);

    let out_dsm: PathBuf = tmp.path().join("dsm.tif");
    assert!(
        lidar_to_dsm(
            &las_path,
            &out_dsm,
            &LidarToDsmOptions {
                radius: Some(0.0),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        lidar_to_dsm(
            &las_path,
            &out_dsm,
            &LidarToDsmOptions {
                min_z: Some(200.0),
                max_z: Some(100.0),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        lidar_to_dsm(
            &las_path,
            &out_dsm,
            &LidarToDsmOptions {
                min_z: Some(f64::NAN),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(parse_return_kinds("").is_err());
    assert!(parse_return_kinds(" , ").is_err());
}
