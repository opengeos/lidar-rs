//! Regression tests for v0.6.0 core algorithms and hydrology audit findings.

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use lidar::{
    Array2D, BurnStreamsOptions, FsmHydrographPoint, InMemoryDepressions, MergeTreeOptions,
    NestedDepression, SimulateInundationOptions, SpillAttenuationOptions, Topology,
    attenuate_fsm_hydrograph_with_options, build_connectivity_graph, burn_streams,
    chicago_hyetograph, compute_rating_curves, compute_twi, crs_metric_scale,
    delineate_depressions_in_memory, delineate_depressions_merge_tree_with_dem_in_memory,
    dinf_flow_direction, extract_sinks_in_memory, extract_sinks_with_filled_in_memory,
    mfd_flow_accumulation, parse_hyetograph_csv, region_props_xy,
    simulate_fill_spill_merge_in_memory, simulate_inundation_in_memory,
};
use richdem::DemNum;
use terrio::vector::Coord;
use terrio::{Feature, Geometry, GeometryType, Layer, VectorWriteOptions};

fn make_dem(w: usize, h: usize, data: Vec<f64>, nd: f64, gt: [f64; 6]) -> Array2D<f64> {
    let mut arr = Array2D::from_vec(w, h, data).unwrap();
    arr.set_no_data(nd);
    arr.geotransform = gt;
    arr
}

fn make_dep(
    id: u32,
    level: u32,
    parent_id: u32,
    children_ids: Vec<u32>,
    count: usize,
) -> NestedDepression {
    NestedDepression {
        id,
        level,
        count,
        area: count as f64,
        volume: count as f64,
        avg_depth: 1.0,
        max_depth: 1.0,
        min_elev: 10.0,
        max_elev: 11.0,
        children_ids,
        region_id: 1,
        perimeter: 4.0,
        major_axis: 1.0,
        minor_axis: 1.0,
        elongatedness: 1.0,
        eccentricity: 0.0,
        orientation: 0.0,
        area_bbox_ratio: 1.0,
        parent_id,
        circularity: 0.78,
        depth_diam_ratio: 0.5,
        hypsometric_integral: 0.5,
        rim_slope_deg: 45.0,
        artifact_flag: false,
    }
}

#[test]
fn flat_high_elevation_sink_with_zero_min_depth_terminates_promptly() {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut flat = Array2D::from_vec(3, 3, vec![1e30; 9]).unwrap();
        flat.set_no_data(-9999.0);
        flat.geotransform = [0.0, 1.0, 0.0, 3.0, 0.0, -1.0];
        let res = delineate_depressions_in_memory(&flat, 1, 0.0, 1.0);
        let _ = tx.send(res.map(|d| d.depressions.len()));
    });
    let count = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("flat high-elevation sink with min_depth=0.0 hung")
        .expect("delineate_depressions_in_memory should succeed");
    assert_eq!(count, 1);
}

#[test]
fn merge_tree_root_outlet_not_hijacked_by_pruned_exterior_pit() {
    // 7x7 DEM with a 3x3 depression (z in [5.0, 8.0]), true rim pour point at (1, 3) = 10.0,
    // and a pruned 1-cell pit at (1, 1) diagonally adjacent to (2, 2). Test both a pit
    // below max_z (2.0) and a pit between max_z and the true rim (9.0).
    for pit_elev in [2.0f64, 9.0f64] {
        let (w, h) = (7usize, 7usize);
        let mut z = vec![15.0f64; w * h];
        for r in 2..=4 {
            for c in 2..=4 {
                z[r * w + c] = if r == 3 && c == 3 { 5.0 } else { 8.0 };
            }
        }
        z[w + 3] = 10.0;
        z[3] = 9.5;
        z[w + 1] = pit_elev;

        let dem = make_dem(w, h, z, -9999.0, [0.0, 1.0, 0.0, 7.0, 0.0, -1.0]);
        let sinks = extract_sinks_in_memory(&dem, 2, Topology::D8).unwrap();
        let mt = delineate_depressions_merge_tree_with_dem_in_memory(
            &sinks.sink,
            &dem,
            &MergeTreeOptions {
                min_size: 2,
                min_depth: 0.5,
                topology: Topology::D8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(mt.nodes.len(), 1);
        let root = &mt.nodes[0];
        assert!(
            (root.saddle_elev - 10.0).abs() < 1e-9,
            "pit_elev={pit_elev}: root saddle_elev should be true rim 10.0, got {} at ({}, {})",
            root.saddle_elev,
            root.pour_point_row,
            root.pour_point_col
        );
        assert_eq!((root.pour_point_row, root.pour_point_col), (1, 3));
    }
}

#[test]
fn derivatives_use_crs_metric_scale_on_geographic_dem() {
    let (w, h) = (11usize, 11usize);
    let ddeg = 0.0001;
    let mut z = vec![0.0f64; w * h];
    for r in 0..h {
        for c in 0..w {
            z[r * w + c] = 100.0 - (c as f64) * 0.78847;
        }
    }
    let mut dem = make_dem(w, h, z, -9999.0, [-90.0, ddeg, 0.0, 45.0005, 0.0, -ddeg]);
    dem.projection = "EPSG:4326".to_string();

    let twi = compute_twi(&dem, None).unwrap();
    let v = twi.get(5, 5);
    assert!(
        v > 4.0 && v < 8.5,
        "EPSG:4326 TWI at 45N should be realistic (~6.3), got {v}"
    );

    let mut z_diag = vec![0.0f64; w * h];
    for r in 0..h {
        for c in 0..w {
            z_diag[r * w + c] = 100.0 - (c as f64) + (r as f64);
        }
    }
    let gt60 = [-90.0, ddeg, 0.0, 60.0005, 0.0, -ddeg];
    let mut dem60 = make_dem(w, h, z_diag, -9999.0, gt60);
    dem60.projection = "EPSG:4326".to_string();
    let scale60 = crs_metric_scale("EPSG:4326", &gt60, w, h);
    let expected_ang = (1.0 / scale60.cell_height_m).atan2(1.0 / scale60.cell_width_m);
    let dir = dinf_flow_direction(&dem60).unwrap();
    let ang = dir.get(5, 5);
    assert!(
        (ang - expected_ang).abs() < 0.02,
        "EPSG:4326 D-inf at 60N expected geodesic angle {expected_ang:.4}, got {ang:.4}"
    );

    // Non-finite geographic geotransform must return Err rather than silently using degrees.
    let mut dem_nan = dem60.clone();
    dem_nan.geotransform[3] = f64::NAN;
    assert!(dinf_flow_direction(&dem_nan).is_err());
    assert!(compute_twi(&dem_nan, None).is_err());
}

#[test]
fn attenuate_hydrograph_auto_tail_converges_on_coarse_dt_and_multi_reservoir() {
    let hydro = vec![
        FsmHydrographPoint {
            time_h: 0.0,
            q_boundary_m3_s: 2.0,
            total_stored_m3: 0.0,
            connected_area_fraction: 1.0,
            spilling_basin_count: 1,
        },
        FsmHydrographPoint {
            time_h: 10.0,
            q_boundary_m3_s: 0.0,
            total_stored_m3: 0.0,
            connected_area_fraction: 1.0,
            spilling_basin_count: 1,
        },
    ];
    let opts_coarse = SpillAttenuationOptions {
        mean_travel_time_h: 1.0,
        num_reservoirs: 2,
        dt_h: 40.0,
        tail_duration_h: None,
    };
    let out_coarse = attenuate_fsm_hydrograph_with_options(&hydro, &opts_coarse)
        .expect("coarse dt_h with auto tail should converge cleanly");
    assert!(out_coarse.len() >= 3);

    let opts_multi = SpillAttenuationOptions {
        mean_travel_time_h: 1.0,
        num_reservoirs: 4,
        dt_h: 10.0,
        tail_duration_h: None,
    };
    let out_multi = attenuate_fsm_hydrograph_with_options(&hydro, &opts_multi)
        .expect("multi-reservoir cascade with auto tail should converge cleanly");
    assert!(out_multi.len() >= 5);
}

#[test]
fn mfd_flow_accumulation_conserves_flow_at_extreme_exponents_and_slope_spreads() {
    // 1. High exponent where raw slope^p is finite (~1e308) but naive sum_w overflows to +inf.
    let mut z = vec![0.0f64; 9];
    z[4] = 1e6;
    let dem = make_dem(3, 3, z, -9999.0, [0.0, 1.0, 0.0, 3.0, 0.0, -1.0]);
    let acc = mfd_flow_accumulation(&dem, 51.35).unwrap();
    let rim_sum: f64 = (0..9).filter(|&i| i != 4).map(|i| acc.get_i(i)).sum();
    assert!(
        (rim_sum - 9.0).abs() < 1e-6,
        "MFD must conserve flow without overflow; expected rim_sum=9.0, got {rim_sum}"
    );

    // 2. Extreme slope spread with small positive exponent must not underflow valid outflow edge.
    let z_spread = vec![
        1e200, 1e200, 1e200, 1e200, 1e-200, 0.0, 1e200, -1e200, 1e200,
    ];
    let dem_spread = make_dem(3, 3, z_spread, -9999.0, [0.0, 1.0, 0.0, 3.0, 0.0, -1.0]);
    let acc_spread = mfd_flow_accumulation(&dem_spread, 0.001).unwrap();
    // Center (1,1) at 1e-200 has East neighbor at 0.0 (slope 1e-200) and South at -1e200 (slope ~1e200).
    // With exponent = 0.001, (1e-400)^0.001 = 10^-0.4 ~ 0.3981, so East must receive > 1.2 cells.
    assert!(
        acc_spread.get(2, 1) > 1.2,
        "East branch must retain representable log-space weight, got {}",
        acc_spread.get(2, 1)
    );
}

#[test]
fn region_props_xy_clamps_minor_axis_to_min_res() {
    let (w, h) = (101usize, 3usize);
    let mut labels = vec![0u32; w * h];
    labels[..100].fill(1);
    labels[w + 50] = 1;
    let elev = vec![10.0f64; w * h];

    let all_props = region_props_xy(&labels, 1, &elev, w, h, 1.0, 1.0, 1.0);
    assert!(
        all_props[0].minor_axis >= 1.0 - 1e-12,
        "region_props_xy minor_axis must be >= min_res (1.0), got {}",
        all_props[0].minor_axis
    );
}

#[test]
fn alignment_checks_reject_nan_geotransforms() {
    let mut dem = make_dem(
        3,
        3,
        vec![10.0; 9],
        -9999.0,
        [0.0, 1.0, 0.0, 3.0, 0.0, -1.0],
    );
    let mut bad_cat = make_dem(
        3,
        3,
        vec![1.0; 9],
        -9999.0,
        [0.0, 1.0, 0.0, f64::NAN, 0.0, -1.0],
    );
    let err = simulate_inundation_in_memory(&dem, &bad_cat, &SimulateInundationOptions::default());
    assert!(
        err.is_err(),
        "validate_grid_alignment must reject NaN geotransform entries"
    );

    dem.geotransform[3] = f64::NAN;
    bad_cat.geotransform[3] = f64::NAN;
    let err2 = extract_sinks_with_filled_in_memory(&dem, Some(&bad_cat), 1, Topology::D8);
    assert!(
        err2.is_err(),
        "extract_sinks_with_filled_in_memory must reject NaN geotransform entries"
    );
}

#[test]
fn graph_and_fsm_handle_sparse_large_depression_ids_without_oom() {
    let gt = [0.0, 1.0, 0.0, 3.0, 0.0, -1.0];
    let big_id: u32 = 1_000_000_000;
    let mut dep_id = Array2D::from_vec(3, 3, vec![0i32; 9]).unwrap();
    dep_id.set_no_data(0);
    dep_id.geotransform = gt;
    dep_id.set(1, 1, big_id as i32);

    let mut dep_lvl = Array2D::from_vec(3, 3, vec![0i32; 9]).unwrap();
    dep_lvl.set_no_data(0);
    dep_lvl.geotransform = gt;
    dep_lvl.set(1, 1, 1);

    let mut sink = make_dem(3, 3, vec![-9999.0; 9], -9999.0, gt);
    sink.set(1, 1, 10.0);

    let mut dem = make_dem(3, 3, vec![12.0; 9], -9999.0, gt);
    dem.set(1, 1, 10.0);

    let mut fill_time = make_dem(3, 3, vec![-1.0; 9], -1.0, gt);
    fill_time.set(1, 1, 1.0);

    let deps = InMemoryDepressions {
        depression_id: dep_id,
        depression_level: dep_lvl,
        depressions: vec![make_dep(big_id, 1, 0, vec![], 1)],
        max_level: 1,
        num_regions: 1,
    };

    let rc = compute_rating_curves(&sink, &deps, 0.5).unwrap();
    assert!(!rc.is_empty());

    let graph = build_connectivity_graph(&deps, &[], &fill_time, &[], 5.0, None, None).unwrap();
    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(graph.nodes[0].id, big_id);

    let fsm = simulate_fill_spill_merge_in_memory(
        &dem,
        &sink,
        &deps,
        &SimulateInundationOptions {
            min_size: 1,
            min_depth: 0.0,
            interval: 0.5,
            rain_intensity_cm_h: 5.0,
            rain_duration_h: 2.0,
            time_step_h: 1.0,
            compress: false,
        },
        &Default::default(),
    )
    .unwrap();
    assert_eq!(fsm.basins.len(), 1);
    assert_eq!(fsm.basins[0].dep_id, big_id);
}

#[test]
fn burn_streams_avoids_nodata_collision() {
    let dir = tempfile::tempdir().unwrap();
    let stream_path = dir.path().join("stream.geojson");
    let mut layer = Layer::new("streams".to_string()).with_geom_type(GeometryType::LineString);
    layer.features.push(Feature {
        fid: 1,
        geometry: Some(Geometry::LineString(vec![
            Coord::xy(0.5, 2.5),
            Coord::xy(4.5, 2.5),
        ])),
        attributes: Vec::new(),
    });
    terrio::write_vector(
        &[layer],
        stream_path.to_str().unwrap(),
        &VectorWriteOptions::default(),
    )
    .unwrap();

    let dem = make_dem(5, 5, vec![1.0; 25], 0.0, [0.0, 1.0, 0.0, 5.0, 0.0, -1.0]);
    let (burned, modified) = burn_streams(
        &dem,
        Path::new(&stream_path),
        &BurnStreamsOptions {
            burn_depth: 1.0,
            stream_width_cells: 0,
            buffer_cells: 0,
            smooth_drop: 0.0,
            compress: false,
        },
    )
    .unwrap();
    assert!(modified >= 5);
    let nd = burned.no_data();
    for c in 0..5 {
        let v = burned.get(c, 2);
        assert!(
            v.is_finite() && !v.rd_eq(nd),
            "burned stream cell ({c}, 2) collided with NoData ({nd}): got {v}"
        );
    }
}

#[test]
fn hyetograph_constructors_reject_overflowing_intensities_and_depths() {
    let csv = "time_h,precip_mm\n0.0,0.0\n1e-310,100.0\n";
    assert!(
        parse_hyetograph_csv(csv).is_err(),
        "subnormal dt_h overflowing intensity_cm_h to Inf must be rejected"
    );
    assert!(
        chicago_hyetograph(1e308, 0.0, 0.5, 0.5, 24.0, 1.0).is_err(),
        "overflowing Chicago IDF total depth must be rejected rather than producing zeros"
    );
}

#[test]
fn projected_feet_and_meters_dems_match_dinf_angles_and_slopes() {
    // Tilted plane with both horizontal coordinates and elevations in feet (EPSG:2263) vs meters
    // (EPSG:32618, scaled by 0.3048): D-inf flow angles and dimensionless slopes must match.
    let (w, h) = (7, 7);
    let mut z_ft = vec![0.0f64; w * h];
    let mut z_m = vec![0.0f64; w * h];
    let ft_to_m = 0.3048;
    for r in 0..h {
        for c in 0..w {
            let v_ft = 100.0 - 2.0 * (c as f64) - 1.0 * (r as f64);
            z_ft[r * w + c] = v_ft;
            z_m[r * w + c] = v_ft * ft_to_m;
        }
    }
    let mut dem_ft = make_dem(
        w,
        h,
        z_ft,
        -9999.0,
        [1_000_000.0, 10.0, 0.0, 200_000.0, 0.0, -10.0],
    );
    dem_ft.projection = "EPSG:2263".to_string();

    let mut dem_m = make_dem(
        w,
        h,
        z_m,
        -9999.0,
        [
            500_000.0,
            10.0 * ft_to_m,
            0.0,
            4_500_000.0,
            0.0,
            -10.0 * ft_to_m,
        ],
    );
    dem_m.projection = "EPSG:32618".to_string();

    let ang_ft = dinf_flow_direction(&dem_ft).unwrap();
    let ang_m = dinf_flow_direction(&dem_m).unwrap();
    assert!(
        (ang_ft.get(3, 3) - ang_m.get(3, 3)).abs() < 1e-9,
        "D-inf angle must match between feet and meters DEMs: ft={}, m={}",
        ang_ft.get(3, 3),
        ang_m.get(3, 3)
    );

    let acc_ft = mfd_flow_accumulation(&dem_ft, 1.1).unwrap();
    let acc_m = mfd_flow_accumulation(&dem_m, 1.1).unwrap();
    assert!(
        (acc_ft.get(3, 3) - acc_m.get(3, 3)).abs() < 1e-9,
        "MFD accumulation must match between feet and meters DEMs: ft={}, m={}",
        acc_ft.get(3, 3),
        acc_m.get(3, 3)
    );

    let twi_ft = compute_twi(&dem_ft, None).unwrap();
    let twi_m = compute_twi(&dem_m, None).unwrap();
    // With identical dimensionless tan(beta) and native-unit SCA (ft vs m),
    // twi_ft - twi_m = ln(1 / 0.3048) rather than -ln(1 / 0.3048).
    let expected_twi_diff = (1.0 / ft_to_m).ln();
    assert!(
        ((twi_ft.get(3, 3) - twi_m.get(3, 3)) - expected_twi_diff).abs() < 1e-6,
        "TWI difference must reflect native SCA units only (identical slope): got {}, expected {}",
        twi_ft.get(3, 3) - twi_m.get(3, 3),
        expected_twi_diff
    );
}
