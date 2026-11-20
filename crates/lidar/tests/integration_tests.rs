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
        compress: false,
    };
    let res = delineate_depressions(&sink_path, tmp.path(), &opts).unwrap();

    assert!(res.depression_id_path.exists());
    assert!(res.depression_level_path.exists());
    assert!(res.shapefile_path.exists());
    assert!(res.csv_path.exists());
    assert!(res.num_regions >= 1);
    assert!(res.max_level >= 1);
    assert_eq!(res.level_shapefiles.len(), res.max_level as usize);
    for shp in &res.level_shapefiles {
        assert!(shp.exists(), "missing level shapefile: {}", shp.display());
    }

    // Verify CSV has expected 18-column header and row count.
    let csv = std::fs::read_to_string(&res.csv_path).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines[0],
        "id,level,count,area,volume,avg_depth,max_depth,min_elev,max_elev,children_id,region_id,perimeter,major_axis,minor_axis,elongatedness,eccentricity,orientation,area_bbox_ratio"
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

    // Safety guards: reject overwriting input DEM, mosaic input == output, invalid lat/lon.
    assert!(extract_sinks(&dem_path, test_data_dir(), &ExtractSinksOptions::default()).is_err());
    assert!(mosaic(&[&resampled], &resampled).is_err());

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

    // Quoted multiline CSV record should parse cleanly into 1 point.
    let multi_csv = tmp.path().join("multi_pts.csv");
    std::fs::write(
        &multi_csv,
        "name,note,latitude,longitude\nA,\"first\nsecond\",35.96,-83.92\n",
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
