//! Integration tests for Roadmap Issue #14 Section 4:
//! LiDAR Point-Cloud Pipeline Enhancements:
//! 1. Tunable CSF / PMF / SMRF parameters and Radius Outlier Removal (ROR)
//! 2. Khosravipour et al. (2014) pit-free CHM
//! 3. Polygon spatial query on COPC / point clouds (exterior + interior holes)
//! 4. In-memory `Array2D<f64>` gridding variants and DEFLATE / COG outputs

#![cfg(feature = "pointcloud")]

use lidar::load_raster;
use lidar::pointcloud::{
    CsfOptions, Dimension, GroundClassifyOptions, GroundFilterMethod, LidarReadOptions,
    LidarToDsmOptions, LidarToDtmOptions, PitFreeChmOptions, PmfOptions, PointCloud,
    PointWriteOptions, SmrfOptions, Values, classify_ground_with_options, filter_outliers_ror,
    lidar_to_chm, lidar_to_chm_pit_free, lidar_to_dsm, lidar_to_dtm, points_to_chm,
    points_to_chm_pit_free, points_to_dsm, points_to_dtm, read_lidar_in_polygon,
    read_lidar_windowed, write_lidar,
};

fn build_cloud(
    xs: Vec<f64>,
    ys: Vec<f64>,
    zs: Vec<f64>,
    cls: Vec<u8>,
    rn: Vec<u8>,
    nr: Vec<u8>,
) -> PointCloud {
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
fn tunable_csf_pmf_smrf_classification_and_dtm_options() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();

    // 24x24 sloped bare-earth surface with a 1.2 m low mound/step in the center.
    for r in 0..24 {
        for c in 0..24 {
            let x = 500_000.0 + c as f64;
            let y = 4_000_000.0 + r as f64;
            let base = 100.0 + 0.08 * c as f64;
            xs.push(x);
            ys.push(y);
            zs.push(base);
            cls.push(1);
            rn.push(1);
            nr.push(1);
        }
    }
    // Add a 1.2 m low shrub/step over a small patch and a 12 m tall building/canopy patch.
    for r in 6..10 {
        for c in 6..10 {
            xs.push(500_000.0 + c as f64 + 0.25);
            ys.push(4_000_000.0 + r as f64 + 0.25);
            zs.push(101.2 + 0.08 * c as f64);
            cls.push(1);
            rn.push(1);
            nr.push(1);
        }
    }
    for r in 14..18 {
        for c in 14..18 {
            xs.push(500_000.0 + c as f64 + 0.25);
            ys.push(4_000_000.0 + r as f64 + 0.25);
            zs.push(112.0 + 0.08 * c as f64);
            cls.push(1);
            rn.push(1);
            nr.push(1);
        }
    }
    let cloud = build_cloud(xs, ys, zs, cls, rn, nr);

    // Tight CSF threshold (0.3 m) excludes the 1.2 m patch; loose CSF threshold (2.0 m) includes it.
    let tight_csf = classify_ground_with_options(
        &cloud,
        &GroundClassifyOptions {
            method: GroundFilterMethod::Csf,
            csf: CsfOptions {
                resolution: 1.0,
                rigidness: 3,
                threshold: 0.3,
                smooth: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let loose_csf = classify_ground_with_options(
        &cloud,
        &GroundClassifyOptions {
            method: GroundFilterMethod::Csf,
            csf: CsfOptions {
                resolution: 1.0,
                rigidness: 2,
                threshold: 2.0,
                smooth: false,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let count_ground = |pc: &PointCloud| -> usize {
        let Values::U8(c) = &pc.dim("classification").unwrap().values else {
            panic!("expected u8 classification");
        };
        c.iter().filter(|&&v| v == 2).count()
    };
    let tight_n = count_ground(&tight_csf);
    let loose_n = count_ground(&loose_csf);
    assert!(
        loose_n > tight_n,
        "looser CSF threshold should classify more points as ground (tight={tight_n}, loose={loose_n})"
    );

    // Tunable PMF and SMRF options execute cleanly and reject the 12 m structure.
    let pmf_out = classify_ground_with_options(
        &cloud,
        &GroundClassifyOptions {
            method: GroundFilterMethod::Pmf,
            pmf: PmfOptions {
                cell: 1.0,
                max_window: 17.0,
                slope: 0.5,
                initial_distance: 0.2,
                max_distance: 1.5,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let smrf_out = classify_ground_with_options(
        &cloud,
        &GroundClassifyOptions {
            method: GroundFilterMethod::Smrf,
            smrf: SmrfOptions {
                cell: 1.0,
                slope: 0.2,
                threshold: 0.4,
                scalar: 1.2,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert!(count_ground(&pmf_out) >= 400);
    assert!(count_ground(&smrf_out) >= 400);

    // Verify LidarToDtmOptions respects tuned CSF parameters in points_to_dtm.
    let dtm = points_to_dtm(
        &cloud,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Csf,
            csf: CsfOptions {
                resolution: 1.0,
                rigidness: 3,
                threshold: 0.3,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let max_dtm = dtm
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_dtm < 104.0,
        "DTM with tight CSF must reject 12 m elevated canopy (got max={max_dtm})"
    );
}

#[test]
fn radius_outlier_removal_strips_isolated_noise_points_and_integrates_with_dtm() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();

    // Dense 15x15 ground lattice at z = 100.0 (1 m spacing; interior points have 4 neighbors within 1.1 m).
    for r in 0..15 {
        for c in 0..15 {
            xs.push(500_000.0 + c as f64);
            ys.push(4_000_000.0 + r as f64);
            zs.push(100.0);
            cls.push(2);
            rn.push(1);
            nr.push(1);
        }
    }
    // Add 3 isolated stray noise points far above and below the ground surface.
    let stray = [
        (500_007.0, 4_000_007.0, 145.0),
        (500_003.0, 4_000_005.0, 40.0),
        (500_010.0, 4_000_010.0, 210.0),
    ];
    for (sx, sy, sz) in stray {
        xs.push(sx);
        ys.push(sy);
        zs.push(sz);
        cls.push(2);
        rn.push(1);
        nr.push(1);
    }

    let cloud = build_cloud(xs, ys, zs, cls, rn, nr);
    assert_eq!(cloud.len(), 228);

    // Validation errors for invalid arguments.
    assert!(filter_outliers_ror(&cloud, 0.0, 2).is_err());
    assert!(filter_outliers_ror(&cloud, -1.0, 2).is_err());
    assert!(filter_outliers_ror(&cloud, f64::NAN, 2).is_err());
    assert!(filter_outliers_ror(&cloud, 1.5, 0).is_err());

    // With radius = 1.5 m and min_neighbors = 2, every lattice point has >= 3 neighbors
    // (corner points have 3 within sqrt(2) = 1.414 m), while all 3 stray points have 0 neighbors.
    let cleaned = filter_outliers_ror(&cloud, 1.5, 2).unwrap();
    assert_eq!(
        cleaned.len(),
        225,
        "ROR must remove exactly the 3 isolated stray points"
    );
    let Values::F64(cz) = &cleaned.dim("z").unwrap().values else {
        panic!("expected f64 z");
    };
    assert!(cz.iter().all(|&z| (z - 100.0).abs() < 1e-9));

    // Integrate ROR directly inside LidarToDtmOptions so the z=40 m low spike does not pull down the DTM.
    let dtm_ror = points_to_dtm(
        &cloud,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            ror_radius: Some(1.5),
            ror_min_neighbors: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let min_z = dtm_ror
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > -9000.0)
        .fold(f64::INFINITY, f64::min);
    let max_z = dtm_ror
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > -9000.0)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (min_z - 100.0).abs() < 1e-6 && (max_z - 100.0).abs() < 1e-6,
        "DTM with ROR should stay flat at 100.0 m (got min={min_z}, max={max_z})"
    );

    // Also test SOR option inside LidarToDtmOptions.
    let dtm_sor = points_to_dtm(
        &cloud,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            sor_k: Some(6),
            sor_sigma: 1.5,
            ..Default::default()
        },
    )
    .unwrap();
    let min_sor = dtm_sor
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > -9000.0)
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_sor > 95.0,
        "DTM with SOR should reject the 40 m low spike (got min={min_sor})"
    );
}

#[test]
fn pit_free_chm_fills_interior_crown_pit_while_preserving_inter_tree_gap() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();

    // 30x20 flat bare-earth ground lattice at z = 100.0 (class 2).
    for r in 0..20 {
        for c in 0..30 {
            xs.push(500_000.0 + c as f64 + 0.5);
            ys.push(4_000_000.0 + r as f64 + 0.5);
            zs.push(100.0);
            cls.push(2);
            rn.push(2);
            nr.push(2);
        }
    }

    // Tree 1 crown: x in [4..11], y in [6..13] at canopy height 16 m (z = 116.0),
    // EXCEPT a deep interior canopy pit right at the crown center (x=7.5, y=9.5) where
    // the laser penetrated down to height 0.5 m (z = 100.5)!
    for r in 6..=13 {
        for c in 4..=11 {
            let x = 500_000.0 + c as f64 + 0.5;
            let y = 4_000_000.0 + r as f64 + 0.5;
            let is_pit = c == 7 && r == 9;
            xs.push(x);
            ys.push(y);
            zs.push(if is_pit { 100.5 } else { 116.0 });
            cls.push(5); // high vegetation
            rn.push(1);
            nr.push(2);
        }
    }

    // Tree 2 crown: x in [18..25], y in [6..13] at canopy height 14 m (z = 114.0),
    // separated from Tree 1 by a 6 m open meadow gap (c in 12..=17).
    for r in 6..=13 {
        for c in 18..=25 {
            let x = 500_000.0 + c as f64 + 0.5;
            let y = 4_000_000.0 + r as f64 + 0.5;
            xs.push(x);
            ys.push(y);
            zs.push(114.0);
            cls.push(5);
            rn.push(1);
            nr.push(2);
        }
    }

    let cloud = build_cloud(xs, ys, zs, cls, rn, nr);

    // Helper to look up raster value at world coordinate (wx, wy).
    let sample_at = |arr: &lidar::Array2D<f64>, wx: f64, wy: f64| -> f64 {
        let gt = arr.geotransform;
        let c = ((wx - gt[0]) / gt[1]).floor() as usize;
        let r = ((wy - gt[3]) / gt[5]).floor() as usize;
        arr.get(c, r)
    };

    // Standard CHM (DSM - DTM) with bin/small radius exhibits the pit inside Tree 1.
    let std_chm = points_to_chm(
        &cloud,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            radius: Some(0.5),
            ..Default::default()
        },
    )
    .unwrap();
    let std_pit_val = sample_at(&std_chm, 500_007.5, 4_000_009.5);
    assert!(
        std_pit_val < 2.0,
        "standard CHM should show the deep interior crown pit (got {std_pit_val})"
    );

    // Khosravipour et al. (2014) pit-free CHM fills the interior pit while preserving the 6 m gap!
    let pf_opts = PitFreeChmOptions {
        resolution: 1.0,
        height_thresholds: vec![0.0, 2.0, 5.0, 10.0, 15.0],
        max_triangle_edge: Some(3.5),
        dtm: LidarToDtmOptions {
            ground_method: GroundFilterMethod::Preclassified,
            ..Default::default()
        },
        compress: true,
        cog: false,
    };
    let pf_chm = points_to_chm_pit_free(&cloud, &pf_opts).unwrap();

    let pf_pit_val = sample_at(&pf_chm, 500_007.5, 4_000_009.5);
    let pf_tree1_val = sample_at(&pf_chm, 500_006.5, 4_000_009.5);
    let pf_tree2_val = sample_at(&pf_chm, 500_021.5, 4_000_009.5);
    let pf_gap_val = sample_at(&pf_chm, 500_014.5, 4_000_009.5);

    assert!(
        pf_pit_val > 14.5 && pf_pit_val <= 16.5,
        "pit-free CHM must bridge the interior crown pit to ~16 m (got {pf_pit_val})"
    );
    assert!(
        (pf_tree1_val - 16.0).abs() < 0.5,
        "Tree 1 canopy height should remain ~16 m (got {pf_tree1_val})"
    );
    assert!(
        (pf_tree2_val - 14.0).abs() < 0.5,
        "Tree 2 canopy height should remain ~14 m (got {pf_tree2_val})"
    );
    assert!(
        pf_gap_val < 0.5,
        "pit-free CHM must preserve the 6 m open gap between trees at ~0 m (got {pf_gap_val})"
    );

    // Also test file-based `lidar_to_chm_pit_free` round-trip.
    let tmp = tempfile::tempdir().unwrap();
    let las_path = tmp.path().join("canopy.las");
    let chm_path = tmp.path().join("chm_pit_free.tif");
    write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();
    let returned = lidar_to_chm_pit_free(&las_path, &chm_path, &pf_opts).unwrap();
    assert!(chm_path.exists());
    let reloaded = load_raster::<f64>(&chm_path, None).unwrap();
    assert_eq!(
        (returned.width(), returned.height()),
        (reloaded.width(), reloaded.height())
    );
    for (a, b) in returned.data().iter().zip(reloaded.data().iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn polygon_spatial_query_on_copc_respects_exterior_and_interior_hole() {
    let tmp = tempfile::tempdir().unwrap();
    let copc_path = tmp.path().join("scene.copc.laz");
    let poly_path = tmp.path().join("aoi_with_hole.geojson");

    // 40x40 point grid spanning x in [500000..500039], y in [4000000..400039].
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();
    for r in 0..40 {
        for c in 0..40 {
            xs.push(500_000.0 + c as f64 + 0.5);
            ys.push(4_000_000.0 + r as f64 + 0.5);
            zs.push(100.0 + 0.1 * c as f64);
            cls.push(2);
            rn.push(1);
            nr.push(1);
        }
    }
    let cloud = build_cloud(xs, ys, zs, cls, rn, nr);
    write_lidar(&cloud, &copc_path, &PointWriteOptions::default()).unwrap();

    // GeoJSON Polygon with exterior [500010, 4000010]..[500030, 4000030] (20x20 = 400 points)
    // and interior hole [500016, 4000016]..[500024, 4000024] (8x8 = 64 points excluded).
    // Expected retained points = 400 - 64 = 336.
    let geojson = r#"{
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32615"}},
        "features": [{
            "type": "Feature",
            "properties": {"name": "donut"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [500010.0, 4000010.0],
                        [500030.0, 4000010.0],
                        [500030.0, 4000030.0],
                        [500010.0, 4000030.0],
                        [500010.0, 4000010.0]
                    ],
                    [
                        [500016.0, 4000016.0],
                        [500024.0, 4000016.0],
                        [500024.0, 4000024.0],
                        [500016.0, 4000024.0],
                        [500016.0, 4000016.0]
                    ]
                ]
            }
        }]
    }"#;
    std::fs::write(&poly_path, geojson).unwrap();

    let clipped =
        read_lidar_in_polygon(&copc_path, &poly_path, &LidarReadOptions::default()).unwrap();
    assert_eq!(
        clipped.len(),
        336,
        "polygon with 8x8 hole inside 20x20 exterior should retain exactly 336 points"
    );

    let cx = clipped.coords("x").unwrap();
    let cy = clipped.coords("y").unwrap();
    for (&x, &y) in cx.iter().zip(cy.iter()) {
        assert!(
            (500_010.0..=500_030.0).contains(&x) && (4_000_010.0..=4_000_030.0).contains(&y),
            "retained point ({x}, {y}) fell outside exterior ring"
        );
        let in_hole =
            (500_016.0..500_024.0).contains(&x) && (4_000_016.0..4_000_024.0).contains(&y);
        assert!(
            !in_hole,
            "retained point ({x}, {y}) fell inside the excluded interior hole"
        );
    }

    // Also verify `LidarReadOptions::polygon_path` works via `read_lidar_windowed` and intersects
    // cleanly with `bounds` (narrowing to x <= 500020 removes the right half).
    let half = read_lidar_windowed(
        &copc_path,
        &LidarReadOptions {
            bounds: Some([500_000.0, 4_000_000.0, 500_020.0, 4_000_040.0]),
            polygon_path: Some(poly_path.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    // Left half of exterior [10..20] x [10..30] = 10*20 = 200 minus left half of hole [16..20] x [16..24] = 4*8 = 32 -> 168.
    assert_eq!(half.len(), 168);

    // Disjoint bounds returns an empty cloud preserving x/y/z schema without error.
    let empty = read_lidar_in_polygon(
        &copc_path,
        &poly_path,
        &LidarReadOptions {
            bounds: Some([600_000.0, 4_000_000.0, 600_100.0, 4_000_100.0]),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(empty.is_empty());
    assert!(empty.dim("x").is_some() && empty.dim("y").is_some() && empty.dim("z").is_some());
}

#[test]
fn in_memory_gridding_matches_file_outputs_and_writes_deflate_and_cog() {
    let tmp = tempfile::tempdir().unwrap();
    let las_path = tmp.path().join("terrain.las");

    // 20x20 ground + center canopy cluster.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();
    for r in 0..20 {
        for c in 0..20 {
            xs.push(500_000.0 + c as f64);
            ys.push(4_000_000.0 + r as f64);
            zs.push(100.0 + 0.1 * c as f64);
            cls.push(2);
            rn.push(2);
            nr.push(2);
        }
    }
    for r in 7..13 {
        for c in 7..13 {
            xs.push(500_000.0 + c as f64 + 0.2);
            ys.push(4_000_000.0 + r as f64 + 0.2);
            zs.push(115.0 + 0.1 * c as f64);
            cls.push(5);
            rn.push(1);
            nr.push(2);
        }
    }
    let cloud = build_cloud(xs, ys, zs, cls, rn, nr);
    write_lidar(&cloud, &las_path, &PointWriteOptions::default()).unwrap();

    // 1. In-memory points_to_dsm / points_to_dtm / points_to_chm match disk outputs bit-for-bit.
    let dsm_opts = LidarToDsmOptions {
        resolution: 1.0,
        ..Default::default()
    };
    let dtm_opts = LidarToDtmOptions {
        resolution: 1.0,
        ground_method: GroundFilterMethod::Preclassified,
        ..Default::default()
    };

    let dsm_mem = points_to_dsm(&cloud, &dsm_opts).unwrap();
    let dtm_mem = points_to_dtm(&cloud, &dtm_opts).unwrap();
    let chm_mem = points_to_chm(&cloud, &dtm_opts).unwrap();

    let dsm_uncomp = tmp.path().join("dsm_uncomp.tif");
    let dsm_deflate = tmp.path().join("dsm_deflate.tif");
    let dtm_deflate = tmp.path().join("dtm_deflate.tif");
    let chm_deflate = tmp.path().join("chm_deflate.tif");

    lidar_to_dsm(&las_path, &dsm_uncomp, &dsm_opts).unwrap();
    lidar_to_dsm(
        &las_path,
        &dsm_deflate,
        &LidarToDsmOptions {
            compress: true,
            ..dsm_opts.clone()
        },
    )
    .unwrap();
    lidar_to_dtm(
        &las_path,
        &dtm_deflate,
        &LidarToDtmOptions {
            compress: true,
            ..dtm_opts.clone()
        },
    )
    .unwrap();
    lidar_to_chm(
        &las_path,
        &chm_deflate,
        &LidarToDtmOptions {
            compress: true,
            ..dtm_opts.clone()
        },
    )
    .unwrap();

    let dsm_disk = load_raster::<f64>(&dsm_deflate, None).unwrap();
    let dtm_disk = load_raster::<f64>(&dtm_deflate, None).unwrap();
    let chm_disk = load_raster::<f64>(&chm_deflate, None).unwrap();

    // Unquantized in-memory cloud vs LAS-quantized disk output match within 1e-9 (LAS scale/offset).
    assert_eq!(dsm_mem.data().len(), dsm_disk.data().len());
    for (a, b) in dsm_mem.data().iter().zip(dsm_disk.data().iter()) {
        assert!((a - b).abs() < 1e-9, "DSM mismatch: {a} vs {b}");
    }
    for (a, b) in dtm_mem.data().iter().zip(dtm_disk.data().iter()) {
        assert!((a - b).abs() < 1e-9, "DTM mismatch: {a} vs {b}");
    }
    for (a, b) in chm_mem.data().iter().zip(chm_disk.data().iter()) {
        assert!((a - b).abs() < 1e-9, "CHM mismatch: {a} vs {b}");
    }

    // Re-read from LAS and verify bit-for-bit equality between in-memory and disk gridding.
    let cloud_las = lidar::pointcloud::read_lidar(&las_path).unwrap();
    assert_eq!(
        points_to_dsm(&cloud_las, &dsm_opts).unwrap().data(),
        dsm_disk.data()
    );
    assert_eq!(
        points_to_dtm(&cloud_las, &dtm_opts).unwrap().data(),
        dtm_disk.data()
    );
    assert_eq!(
        points_to_chm(&cloud_las, &dtm_opts).unwrap().data(),
        chm_disk.data()
    );

    // Verify CRS (EPSG:32615) and geotransform survive GeoTIFF round-trip in lidar_to_chm.
    assert_eq!(chm_disk.geotransform, chm_mem.geotransform);
    let chm_lazy = terrio::open_raster_lazy(chm_deflate.to_str().unwrap()).unwrap();
    let expected_crs = terrio::crs::SpatialRef::parse("EPSG:32615").unwrap();
    assert!(
        chm_lazy.crs().same_as(&expected_crs),
        "lidar_to_chm GeoTIFF must preserve EPSG:32615 CRS (got {:?})",
        chm_lazy.crs()
    );
    assert_eq!(chm_lazy.geo_transform(), chm_mem.geotransform);

    // DEFLATE-compressed GeoTIFF should be smaller than uncompressed GeoTIFF for smooth terrain.
    let uncomp_bytes = std::fs::metadata(&dsm_uncomp).unwrap().len();
    let deflate_bytes = std::fs::metadata(&dsm_deflate).unwrap().len();
    assert!(
        deflate_bytes < uncomp_bytes,
        "DEFLATE GeoTIFF ({deflate_bytes} B) should be smaller than uncompressed ({uncomp_bytes} B)"
    );

    // 2. COG output on a raster larger than the 512 px COG tile size produces internal overviews
    //    readable via `terrio::open_raster_lazy`.
    let dsm_cog = tmp.path().join("dsm_cog.tif");
    let dtm_cog = tmp.path().join("dtm_cog.tif");
    let chm_cog = tmp.path().join("chm_cog.tif");

    // Gridding the 19 m x 19 m cloud at 0.032 m resolution yields a ~594x594 raster (> 512 tile).
    let (cw, ch) = lidar_to_dsm(
        &las_path,
        &dsm_cog,
        &LidarToDsmOptions {
            resolution: 0.032,
            method: lidar::pointcloud::GridMethod::Linear,
            cog: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(cw > 512 && ch > 512);

    let lazy_dsm = terrio::open_raster_lazy(dsm_cog.to_str().unwrap()).unwrap();
    assert_eq!((lazy_dsm.width(), lazy_dsm.height()), (cw, ch));
    assert!(
        !lazy_dsm.overviews().is_empty(),
        "COG DSM should contain internal overview pyramid levels"
    );
    assert!(lazy_dsm.block_size().is_some(), "COG DSM should be tiled");

    // Verify COG output also works cleanly for lidar_to_dtm and lidar_to_chm.
    lidar_to_dtm(
        &las_path,
        &dtm_cog,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            cog: true,
            ..Default::default()
        },
    )
    .unwrap();
    lidar_to_chm(
        &las_path,
        &chm_cog,
        &LidarToDtmOptions {
            resolution: 1.0,
            ground_method: GroundFilterMethod::Preclassified,
            cog: true,
            ..Default::default()
        },
    )
    .unwrap();

    let lazy_dtm = terrio::open_raster_lazy(dtm_cog.to_str().unwrap()).unwrap();
    let lazy_chm = terrio::open_raster_lazy(chm_cog.to_str().unwrap()).unwrap();
    assert!(lazy_dtm.block_size().is_some());
    assert!(lazy_chm.block_size().is_some());
}

#[test]
fn multipolygon_query_and_pit_free_chm_with_csf_and_cog() {
    let tmp = tempfile::tempdir().unwrap();
    let copc_path = tmp.path().join("multi.copc.laz");
    let mp_path = tmp.path().join("multipoly.geojson");
    let chm_cog_path = tmp.path().join("pit_free_cog.tif");

    // 25x25 ground + two distinct trees.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();
    for r in 0..25 {
        for c in 0..25 {
            xs.push(500_000.0 + c as f64 + 0.5);
            ys.push(4_000_000.0 + r as f64 + 0.5);
            zs.push(50.0);
            cls.push(1);
            rn.push(1);
            nr.push(1);
        }
    }
    // Elevated tree crown in bottom-left quadrant.
    for r in 3..8 {
        for c in 3..8 {
            if c == 5 && r == 5 {
                continue;
            }
            xs.push(500_000.0 + c as f64 + 0.5);
            ys.push(4_000_000.0 + r as f64 + 0.5);
            zs.push(65.0);
            cls.push(1);
            rn.push(1);
            nr.push(1);
        }
    }
    let cloud = build_cloud(xs, ys, zs, cls, rn, nr);
    write_lidar(&cloud, &copc_path, &PointWriteOptions::default()).unwrap();

    // MultiPolygon with two disjoint 4x4 squares: [1..5]x[1..5] and [18..22]x[18..22].
    let geojson = r#"{
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32615"}},
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[
                        [500001.0, 4000001.0],
                        [500005.0, 4000001.0],
                        [500005.0, 4000005.0],
                        [500001.0, 4000005.0],
                        [500001.0, 4000001.0]
                    ]],
                    [[
                        [500018.0, 4000018.0],
                        [500022.0, 4000018.0],
                        [500022.0, 4000022.0],
                        [500018.0, 4000022.0],
                        [500018.0, 4000018.0]
                    ]]
                ]
            }
        }]
    }"#;
    std::fs::write(&mp_path, geojson).unwrap();

    let subset = read_lidar_in_polygon(&copc_path, &mp_path, &LidarReadOptions::default()).unwrap();
    // Square 1 has 16 ground + 4 canopy (c in 3..5, r in 3..5) = 20 points; Square 2 has 16 ground points = 36 total.
    assert_eq!(subset.len(), 36);

    // Pit-free CHM with CSF ground classification and COG output.
    let pf_chm = lidar_to_chm_pit_free(
        &copc_path,
        &chm_cog_path,
        &PitFreeChmOptions {
            resolution: 1.0,
            height_thresholds: vec![0.0, 2.0, 5.0, 10.0],
            max_triangle_edge: Some(3.5),
            dtm: LidarToDtmOptions {
                ground_method: GroundFilterMethod::Csf,
                csf: CsfOptions {
                    resolution: 1.0,
                    rigidness: 3,
                    threshold: 0.5,
                    ..Default::default()
                },
                ..Default::default()
            },
            compress: false,
            cog: true,
        },
    )
    .unwrap();
    assert!(chm_cog_path.exists());
    let lazy = terrio::open_raster_lazy(chm_cog_path.to_str().unwrap()).unwrap();
    assert!(lazy.block_size().is_some());

    // Center of tree crown (500005.5, 4000005.5) had no canopy return (pit), should be filled to ~15 m.
    let gt = pf_chm.geotransform;
    let c = ((500_005.5 - gt[0]) / gt[1]).floor() as usize;
    let r = ((4_000_005.5 - gt[3]) / gt[5]).floor() as usize;
    let pit_h = pf_chm.get(c, r);
    assert!(
        pit_h > 13.0 && pit_h < 17.0,
        "CSF pit-free CHM should bridge interior crown pit to ~15 m (got {pit_h})"
    );
}

#[test]
fn polygon_query_reprojects_wgs84_geojson_to_utm_and_applies_post_clip_max_points_and_grid_index() {
    let tmp = tempfile::tempdir().unwrap();
    let copc_path = tmp.path().join("utm_scene.copc.laz");
    let wgs84_poly_path = tmp.path().join("aoi_wgs84.geojson");
    let many_polys_path = tmp.path().join("many_polys.geojson");

    // 40x40 UTM Zone 15N (EPSG:32615) cloud spanning [500000..500039] x [4000000..4000039].
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();
    for r in 0..40 {
        for c in 0..40 {
            xs.push(500_000.0 + c as f64 + 0.5);
            ys.push(4_000_000.0 + r as f64 + 0.5);
            zs.push(100.0);
            cls.push(2);
            rn.push(1);
            nr.push(1);
        }
    }
    let cloud = build_cloud(xs, ys, zs, cls, rn, nr);
    write_lidar(&cloud, &copc_path, &PointWriteOptions::default()).unwrap();

    // 1. Build an RFC 7946 EPSG:4326 GeoJSON polygon covering [500010, 4000010]..[500030, 4000030]
    //    by transforming the 4 UTM corners to WGS 84 lon/lat.
    let utm = terrio::crs::SpatialRef::parse("EPSG:32615").unwrap();
    let wgs84 = terrio::crs::SpatialRef::parse("EPSG:4326").unwrap();
    let to_wgs84 = terrio::crs::Transformer::new(&utm, &wgs84).unwrap();
    let corners_utm = [
        (500_010.0, 4_000_010.0),
        (500_030.0, 4_000_010.0),
        (500_030.0, 4_000_030.0),
        (500_010.0, 4_000_030.0),
        (500_010.0, 4_000_010.0),
    ];
    let ring_coords: Vec<String> = corners_utm
        .iter()
        .map(|&(x, y)| {
            let (lon, lat) = to_wgs84.transform(x, y).unwrap();
            format!("[{lon:.12}, {lat:.12}]")
        })
        .collect();
    // Standard RFC 7946 GeoJSON (no "crs" member -> EPSG:4326).
    let wgs84_geojson = format!(
        r#"{{"type":"FeatureCollection","features":[{{"type":"Feature","properties":{{}},"geometry":{{"type":"Polygon","coordinates":[[{}]]}}}}]}}"#,
        ring_coords.join(",")
    );
    std::fs::write(&wgs84_poly_path, wgs84_geojson).unwrap();

    let clipped_wgs84 =
        read_lidar_in_polygon(&copc_path, &wgs84_poly_path, &LidarReadOptions::default()).unwrap();
    assert_eq!(
        clipped_wgs84.len(),
        400,
        "EPSG:4326 GeoJSON polygon must automatically reproject to EPSG:32615 and match 20x20=400 points"
    );

    // 2. Post-clip `max_points`: even with a diagonal-thin L-shaped/small polygon or cap of 50,
    //    returns exactly 50 interior points without starving the polygon.
    let capped = read_lidar_in_polygon(
        &copc_path,
        &wgs84_poly_path,
        &LidarReadOptions {
            max_points: Some(50),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(capped.len(), 50);
    let cx = capped.coords("x").unwrap();
    let cy = capped.coords("y").unwrap();
    for (&x, &y) in cx.iter().zip(cy.iter()) {
        assert!(
            (500_010.0..=500_030.0).contains(&x) && (4_000_010.0..=4_000_030.0).contains(&y),
            "capped point ({x}, {y}) must remain inside reprojected polygon"
        );
    }

    // 3. Spatial grid index path (> 8 polygons): 16 disjoint 2x2 squares (4x4 grid of polygons),
    //    each capturing 2x2 = 4 points -> 64 total points.
    let mut features = Vec::new();
    for py in 0..4 {
        for px in 0..4 {
            let x0 = 500_001.0 + px as f64 * 9.0;
            let y0 = 4_000_001.0 + py as f64 * 9.0;
            let x1 = x0 + 2.0;
            let y1 = y0 + 2.0;
            features.push(format!(
                r#"{{"type":"Feature","properties":{{}},"geometry":{{"type":"Polygon","coordinates":[[[{x0},{y0}],[{x1},{y0}],[{x1},{y1}],[{x0},{y1}],[{x0},{y0}]]]}}}}"#
            ));
        }
    }
    let many_geojson = format!(
        r#"{{"type":"FeatureCollection","crs":{{"type":"name","properties":{{"name":"urn:ogc:def:crs:EPSG::32615"}}}},"features":[{}]}}"#,
        features.join(",")
    );
    std::fs::write(&many_polys_path, many_geojson).unwrap();

    let indexed_clipped =
        read_lidar_in_polygon(&copc_path, &many_polys_path, &LidarReadOptions::default()).unwrap();
    assert_eq!(
        indexed_clipped.len(),
        64,
        "16 disjoint 2x2 polygons should retain exactly 64 points via PolygonGridIndex"
    );
}

#[test]
fn pit_free_chm_on_perfectly_regular_integer_grid_with_canopy_pit() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut cls = Vec::new();
    let mut rn = Vec::new();
    let mut nr = Vec::new();

    // Perfectly regular integer lattice (exact integer x, y coordinates, no fractional offsets).
    for r in 0..20 {
        for c in 0..20 {
            let x = 500_000.0 + c as f64;
            let y = 4_000_000.0 + r as f64;
            xs.push(x);
            ys.push(y);
            zs.push(100.0);
            cls.push(2);
            rn.push(2);
            nr.push(2);

            if (5..=14).contains(&c) && (5..=14).contains(&r) {
                let is_pit = c == 9 && r == 9;
                xs.push(x);
                ys.push(y);
                zs.push(if is_pit { 100.2 } else { 118.0 });
                cls.push(5);
                rn.push(1);
                nr.push(2);
            }
        }
    }
    let cloud = build_cloud(xs, ys, zs, cls, rn, nr);
    let chm = points_to_chm_pit_free(
        &cloud,
        &PitFreeChmOptions {
            resolution: 1.0,
            height_thresholds: vec![0.0, 2.0, 5.0, 10.0, 15.0],
            max_triangle_edge: Some(3.5),
            dtm: LidarToDtmOptions {
                ground_method: GroundFilterMethod::Preclassified,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();

    // Pit at integer coordinate (500009, 4000009) should be bridged to ~18 m without dropping vertices.
    let gt = chm.geotransform;
    let c = ((500_009.0 - gt[0]) / gt[1]).floor() as usize;
    let r = ((4_000_009.0 - gt[3]) / gt[5]).floor() as usize;
    let pit_h = chm.get(c, r);
    assert!(
        pit_h > 16.5 && pit_h <= 18.5,
        "regular integer grid pit-free CHM should bridge pit to ~18 m (got {pit_h})"
    );
}
