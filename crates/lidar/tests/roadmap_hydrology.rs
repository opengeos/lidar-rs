//! Analytical unit and integration tests for Roadmap Issue #14 Wave 1 PR A:
//! - Catchment-weighted runoff coefficients and compressed D8+DSU file catchment hierarchy
//! - Expanded design storms (SCS I/IA/II/III, alternating-block, Chicago, CSV loader, spatial rainfall scaling)
//! - Post-storm drainage and recession modeling
//! - Terrain-hydrology derivatives (D-Infinity, MFD, TWI, HAND, AGREE stream burning)

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
use std::path::PathBuf;

use lidar::{
    Array2D, BurnStreamsOptions, CatchmentHierarchyOptions, DelineateOptions, ExtractSinksOptions,
    InMemoryFullPipelineOptions, InundationForcing, RecessionOptions, RoutingMode, ScsStormType,
    SimulateInundationOptions, Topology, alternating_block_hyetograph, burn_streams,
    burn_streams_file, chicago_hyetograph, compute_hand, compute_hand_from_threshold, compute_twi,
    d8_flow_directions_resolved, d8_watersheds, delineate_catchment_hierarchy,
    delineate_catchment_hierarchy_in_memory, delineate_catchment_hierarchy_weighted_in_memory,
    delineate_depressions, delineate_depressions_in_memory, dinf_flow_accumulation,
    dinf_flow_direction, extract_sinks, extract_sinks_in_memory, load_hyetograph_csv, load_raster,
    mfd_flow_accumulation, parse_hyetograph_csv, run_full_pipeline_in_memory, save_raster,
    scale_forcing_by_rainfall_grid, scs_hyetograph, scs_type2_hyetograph,
    simulate_fill_spill_merge_in_memory, simulate_inundation_with_weighted_catchment_in_memory,
    simulate_post_storm_recession,
};
use richdem::DemNum;
use terrio::vector::Coord;
use terrio::{Feature, Geometry, GeometryType, Layer, VectorWriteOptions};

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
}

fn make_dem(w: usize, h: usize, data: Vec<f64>, nd: f64, gt: [f64; 6]) -> Array2D<f64> {
    let mut arr = Array2D::from_vec(w, h, data).unwrap();
    arr.set_no_data(nd);
    arr.geotransform = gt;
    arr
}

/// Builds a 21x21 watershed with a central 5x5 depression (floor 10.0, spill rim 12.0)
/// surrounded by sloping upland hillslopes (12.2..20.0) that drain into the depression,
/// plus a narrow south saddle spillway at 12.0 so only the central 5x5 bowl is a sink.
fn build_funnel_dem(cell_size: f64) -> Array2D<f64> {
    let (w, h) = (21usize, 21usize);
    let mut z = vec![0.0f64; w * h];
    for r in 0..h {
        for c in 0..w {
            let dr = (r as isize - 10).abs();
            let dc = (c as isize - 10).abs();
            let d = dr.max(dc) as f64;
            if d <= 2.0 {
                // Central 5x5 depression bowl sloping from 10.0 at center up to 11.4 at inner rim
                z[r * w + c] = 10.0 + 0.7 * d;
            } else {
                // Sloping upland walls rising from 12.2 outward to 19.2
                z[r * w + c] = 12.2 + (d - 3.0) * 1.0;
            }
        }
    }
    // Narrow saddle spillway exiting south from the rim at elevation 12.0 down to 11.5 at the edge
    // so Priority-Flood fills only the central 5x5 bowl to 12.0 while the surrounding upland drains in.
    for r in 13..h {
        let t = (r - 13) as f64 / (h - 1 - 13) as f64;
        z[r * w + 10] = 12.0 - 0.5 * t;
    }
    make_dem(
        w,
        h,
        z,
        -9999.0,
        [0.0, cell_size, 0.0, h as f64 * cell_size, 0.0, -cell_size],
    )
}

// ---------------------------------------------------------------------------
// 1. Catchment-weighted runoff coefficients & compressed D8+DSU port
// ---------------------------------------------------------------------------

#[test]
fn weighted_catchment_hierarchy_controls_static_fill_time_by_upland_c() {
    let dem = build_funnel_dem(2.0);
    let w = dem.width();
    let h = dem.height();
    let cell_area = 4.0;

    let sinks = extract_sinks_in_memory(&dem, 10, Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 10, 0.0, 0.2).unwrap();
    assert_eq!(deps.depressions.len(), 1);

    // Scenario A: high upland C (0.9) vs low depression-floor C (0.1).
    // Scenario B: low upland C (0.1) vs high depression-floor C (0.9).
    let mut rc_a_data = vec![0.9f64; w * h];
    let mut rc_b_data = vec![0.1f64; w * h];
    for r in 0..h {
        for c in 0..w {
            if deps.depression_level.get(c, r) > 0 {
                rc_a_data[r * w + c] = 0.1;
                rc_b_data[r * w + c] = 0.9;
            }
        }
    }
    let rc_a = make_dem(w, h, rc_a_data, -9999.0, dem.geotransform);
    let rc_b = make_dem(w, h, rc_b_data, -9999.0, dem.geotransform);

    // Reference exact weighted area via explicit D8 watershed sum.
    let flowdirs = d8_flow_directions_resolved(&sinks.dem_refined).unwrap();
    let pour: Vec<u32> = deps
        .depression_level
        .data()
        .iter()
        .map(|&lv| u32::from(lv > 0))
        .collect();
    let ws = d8_watersheds(&flowdirs, &pour).unwrap();
    let mut expected_geom = 0.0f64;
    let mut expected_eff_a = 0.0f64;
    let mut expected_eff_b = 0.0f64;
    for (idx, &lbl) in ws.data().iter().enumerate() {
        if lbl > 0 {
            expected_geom += cell_area;
            expected_eff_a += cell_area * rc_a.get_i(idx);
            expected_eff_b += cell_area * rc_b.get_i(idx);
        }
    }

    let w_none = delineate_catchment_hierarchy_weighted_in_memory(
        &sinks.dem_refined,
        &deps.depression_level,
        10,
        None,
    )
    .unwrap();
    let w_a = delineate_catchment_hierarchy_weighted_in_memory(
        &sinks.dem_refined,
        &deps.depression_level,
        10,
        Some(&rc_a),
    )
    .unwrap();
    let w_b = delineate_catchment_hierarchy_weighted_in_memory(
        &sinks.dem_refined,
        &deps.depression_level,
        10,
        Some(&rc_b),
    )
    .unwrap();

    assert!((w_none.catchment_hierarchy.get(10, 10) - expected_geom).abs() < 1e-6);
    assert!((w_a.unweighted_catchment_hierarchy.get(10, 10) - expected_geom).abs() < 1e-6);
    assert!(
        (w_a.catchment_hierarchy.get(10, 10) - expected_eff_a).abs() < 1e-6,
        "expected upland-weighted area {expected_eff_a}, got {}",
        w_a.catchment_hierarchy.get(10, 10)
    );
    assert!(
        (w_b.catchment_hierarchy.get(10, 10) - expected_eff_b).abs() < 1e-6,
        "expected upland-weighted area {expected_eff_b}, got {}",
        w_b.catchment_hierarchy.get(10, 10)
    );
    assert!(
        expected_eff_a > expected_eff_b * 3.0,
        "upland area dominates floor area: eff_a={expected_eff_a}, eff_b={expected_eff_b}"
    );

    // Run full pipeline in Static mode for both scenarios: high upland C must fill MUCH faster
    // than low upland C even though its depression floor has lower local C!
    let sim = SimulateInundationOptions {
        min_size: 10,
        min_depth: 0.0,
        interval: 0.2,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 20.0,
        time_step_h: 1.0,
        compress: false,
    };
    let pipe_a = run_full_pipeline_in_memory(
        &dem,
        &InMemoryFullPipelineOptions {
            kernel_size: 1,
            breach: None,
            simulation: sim.clone(),
            routing: RoutingMode::Static,
            forcing: InundationForcing {
                runoff_coeff: Some(&rc_a),
                ..Default::default()
            },
            scalar_runoff_coeff: 1.0,
            stream_vector: None,
        },
    )
    .unwrap();
    let pipe_b = run_full_pipeline_in_memory(
        &dem,
        &InMemoryFullPipelineOptions {
            kernel_size: 1,
            breach: None,
            simulation: sim.clone(),
            routing: RoutingMode::Static,
            forcing: InundationForcing {
                runoff_coeff: Some(&rc_b),
                ..Default::default()
            },
            scalar_runoff_coeff: 1.0,
            stream_vector: None,
        },
    )
    .unwrap();

    let ft_a = pipe_a.fill_time.get(10, 10);
    let ft_b = pipe_b.fill_time.get(10, 10);
    assert!(ft_a > 0.0 && ft_b > 0.0);
    assert!(
        ft_a < ft_b / 3.0,
        "high-upland-C static fill_time ({ft_a:.4} h) must be much faster than low-upland-C ({ft_b:.4} h)"
    );
    // Ratio of fill times must match inverse ratio of true upland-weighted areas.
    let expected_ratio = expected_eff_a / expected_eff_b;
    let actual_ratio = ft_b / ft_a;
    assert!(
        (actual_ratio - expected_ratio).abs() / expected_ratio < 1e-6,
        "fill_time ratio {actual_ratio:.6} must match inverse effective area ratio {expected_ratio:.6}"
    );

    // Zero runoff coefficient everywhere must NOT fall back to obj.area: depression stays dry.
    let rc_zero = make_dem(w, h, vec![0.0; w * h], -9999.0, dem.geotransform);
    let w_zero = delineate_catchment_hierarchy_weighted_in_memory(
        &sinks.dem_refined,
        &deps.depression_level,
        10,
        Some(&rc_zero),
    )
    .unwrap();
    let (ft_zero, _) = simulate_inundation_with_weighted_catchment_in_memory(
        &sinks.sink,
        &w_zero.unweighted_catchment_hierarchy,
        Some(&w_zero.catchment_hierarchy),
        &sim,
        &InundationForcing {
            runoff_coeff: Some(&rc_zero),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(ft_zero.get(10, 10), -1.0, "zero runoff must keep sink dry");
}

#[test]
fn file_based_catchment_hierarchy_compressed_engine_matches_in_memory_on_sample_dem() {
    let dem_path = test_data_dir().join("dem.tif");
    let tmp = tempfile::tempdir().unwrap();

    let sinks_dir = tmp.path().join("sinks");
    let sinks = extract_sinks(
        &dem_path,
        &sinks_dir,
        &ExtractSinksOptions {
            min_size: 1000,
            ..Default::default()
        },
    )
    .unwrap();

    let deps_dir = tmp.path().join("deps");
    let deps = delineate_depressions(
        sinks.sink_path.as_ref().unwrap(),
        &deps_dir,
        &DelineateOptions {
            min_size: 1000,
            min_depth: 0.2,
            interval: 0.3,
            bool_level_shp: true,
            ..Default::default()
        },
    )
    .unwrap();

    let refined_dem_path = sinks.refined_dem_path.as_ref().unwrap();
    let hir_out = tmp.path().join("hir").join("catchment_hir.tif");
    let file_res = delineate_catchment_hierarchy(
        refined_dem_path,
        deps_dir.join("shp-level"),
        &hir_out,
        &CatchmentHierarchyOptions::default(),
    )
    .unwrap();

    let dem_ref = load_raster::<f64>(refined_dem_path, None).unwrap();
    let dep_lvl = load_raster::<i32>(&deps.depression_level_path, None).unwrap();
    let (mem_hir, mem_recs) =
        delineate_catchment_hierarchy_in_memory(&dem_ref, &dep_lvl, 1000).unwrap();

    let file_hir = load_raster::<f64>(&file_res.catchment_raster_path, None).unwrap();
    assert_eq!(file_hir.data().len(), mem_hir.data().len());
    for (i, (&a, &b)) in file_hir
        .data()
        .iter()
        .zip(mem_hir.data().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-6,
            "pixel {i} mismatch between file-based ({a}) and in-memory ({b})"
        );
    }

    // Compare sorted CSV records (area, cat_area, dep2catR, elevation stats, volume).
    assert_eq!(file_res.records.len(), mem_recs.len());
    let mut f_sorted = file_res.records.clone();
    let mut m_sorted = mem_recs.clone();
    let sort_key = |r: &lidar::CatchmentHierarchyRecord| {
        (
            r.level,
            r.count,
            r.min_elev.to_bits(),
            r.max_elev.to_bits(),
            r.cat_area.to_bits(),
        )
    };
    f_sorted.sort_by_key(sort_key);
    m_sorted.sort_by_key(sort_key);
    for (fr, mr) in f_sorted.iter().zip(m_sorted.iter()) {
        assert_eq!(fr.level, mr.level);
        assert_eq!(fr.count, mr.count);
        assert!((fr.area - mr.area).abs() < 1e-4);
        assert!((fr.cat_area - mr.cat_area).abs() < 1e-4);
        assert!((fr.dep_to_cat_ratio - mr.dep_to_cat_ratio).abs() < 1e-6);
        assert!((fr.min_elev - mr.min_elev).abs() < 1e-6);
        assert!((fr.max_elev - mr.max_elev).abs() < 1e-6);
        assert!((fr.mean_elev - mr.mean_elev).abs() < 1e-6);
        assert!((fr.std_elev - mr.std_elev).abs() < 1e-5);
        assert!((fr.volume - mr.volume).abs() < 1e-3);
    }
    assert!(file_res.csv_path.exists());
    assert!(file_res.merged_shapefile_path.exists());
    assert_eq!(file_res.catchment_shapefiles.len(), deps.max_level as usize);
}

// ---------------------------------------------------------------------------
// 2. Expanded design storms, CSV loader, spatial rainfall scaling, recession
// ---------------------------------------------------------------------------

#[test]
fn scs_all_four_types_conserve_volume_and_type2_matches_bit_for_bit() {
    let total_mm = 125.0;
    let duration_h = 24.0;
    let dt_h = 0.25;

    let mut peaks = Vec::new();
    for st in [
        ScsStormType::TypeIA,
        ScsStormType::TypeI,
        ScsStormType::TypeIII,
        ScsStormType::TypeII,
    ] {
        let h = scs_hyetograph(st, total_mm, duration_h, dt_h).unwrap();
        let sum_mm: f64 = h.iter().map(|&(dt, inten)| dt * inten * 10.0).sum();
        assert!(
            (sum_mm - total_mm).abs() < 1e-6,
            "{st:?} total precip mismatch: got {sum_mm}, expected {total_mm}"
        );
        let (peak_idx, &(_, peak_inten)) = h
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.1.total_cmp(&b.1.1))
            .unwrap();
        let peak_time_h = (peak_idx as f64 + 0.5) * dt_h;
        peaks.push((st, peak_time_h, peak_inten));
    }

    // Peak timing: TypeIA (~8h) < TypeI (~10h) < TypeIII & TypeII (~12h)
    assert!(peaks[0].1 < peaks[1].1, "TypeIA should peak before TypeI");
    assert!(peaks[1].1 < peaks[2].1, "TypeI should peak before TypeIII");
    // Peak intensity: TypeIA < TypeI < TypeIII < TypeII
    assert!(peaks[0].2 < peaks[1].2);
    assert!(peaks[1].2 < peaks[2].2);
    assert!(peaks[2].2 < peaks[3].2);

    // Bit-for-bit parity between scs_type2_hyetograph and scs_hyetograph(TypeII)
    let legacy = scs_type2_hyetograph(total_mm, duration_h, dt_h).unwrap();
    let unified = scs_hyetograph(ScsStormType::TypeII, total_mm, duration_h, dt_h).unwrap();
    assert_eq!(legacy.len(), unified.len());
    for (a, b) in legacy.iter().zip(unified.iter()) {
        assert_eq!(a.0.to_bits(), b.0.to_bits());
        assert_eq!(a.1.to_bits(), b.1.to_bits());
    }

    // String parsing
    assert_eq!(
        "type_ia".parse::<ScsStormType>().unwrap(),
        ScsStormType::TypeIA
    );
    assert_eq!(
        "III".parse::<ScsStormType>().unwrap(),
        ScsStormType::TypeIII
    );
    assert!("invalid".parse::<ScsStormType>().is_err());
}

#[test]
fn alternating_block_and_chicago_storms_peak_and_conserve_volume() {
    // Alternating block from Sherman IDF intensities i(d_min) = 50 / (d_min + 10)^0.75
    let dt_h = 0.25; // 15-min steps
    let n = 8usize;
    let idf: Vec<f64> = (1..=n)
        .map(|k| {
            let d_min = k as f64 * dt_h * 60.0;
            50.0 / (d_min + 10.0).powf(0.75)
        })
        .collect();
    let ab = alternating_block_hyetograph(&idf, dt_h).unwrap();
    assert_eq!(ab.len(), n);
    let expected_total_cm = idf[n - 1] * (n as f64 * dt_h);
    let actual_total_cm: f64 = ab.iter().map(|&(dt, i)| dt * i).sum();
    assert!(
        (actual_total_cm - expected_total_cm).abs() < 1e-12,
        "alternating block must conserve total rainfall: {actual_total_cm} vs {expected_total_cm}"
    );
    let peak_idx = ab
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.1.total_cmp(&b.1.1))
        .unwrap()
        .0;
    assert_eq!(peak_idx, (n - 1) / 2, "peak must be at storm center");
    assert!(alternating_block_hyetograph(&[], dt_h).is_err());
    assert!(alternating_block_hyetograph(&[10.0, 2.0], dt_h).is_err()); // cumulative decreases: 10*dt > 2*2*dt

    // Chicago design storm with r = 0.35 and b = 10.0, plus edge case b = 0.0
    for (b_val, r_val) in [(10.0, 0.35), (0.0, 0.4), (5.0, 0.65)] {
        let a = 60.0;
        let c = 0.72;
        let dur_h = 2.0;
        let step_h = 0.05;
        let ch = chicago_hyetograph(a, b_val, c, r_val, dur_h, step_h).unwrap();
        let expected_cm = dur_h * a / (dur_h * 60.0 + b_val).powf(c);
        let got_cm: f64 = ch.iter().map(|&(dt, i)| dt * i).sum();
        assert!(
            (got_cm - expected_cm).abs() < 1e-10,
            "Chicago storm (b={b_val}, r={r_val}) volume mismatch: {got_cm} vs {expected_cm}"
        );
        let peak_i = ch
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.1.total_cmp(&y.1.1))
            .unwrap()
            .0;
        let peak_t = (peak_i as f64 + 0.5) * step_h;
        assert!(
            (peak_t - r_val * dur_h).abs() <= step_h + 1e-9,
            "Chicago peak time {peak_t} should be near r*T = {}",
            r_val * dur_h
        );
    }
    assert!(chicago_hyetograph(50.0, 10.0, 1.0, 0.4, 2.0, 0.1).is_err());
    assert!(chicago_hyetograph(50.0, 10.0, 0.7, 0.0, 2.0, 0.1).is_err());
}

#[test]
fn hyetograph_csv_parser_and_loader_support_both_formats_and_reject_malformed() {
    let csv_dt =
        "\u{feff}# Rain gauge record\ndt_h, intensity_cm_h\n0.5, 2.0\n0.5, 6.0\n1.0, 0.0\n";
    let h1 = parse_hyetograph_csv(csv_dt).unwrap();
    assert_eq!(h1, vec![(0.5, 2.0), (0.5, 6.0), (1.0, 0.0)]);

    let csv_tp = "time_h,precip_mm\n0.0,0.0\n0.5,10.0\n1.0,30.0\n2.0,0.0\n";
    let h2 = parse_hyetograph_csv(csv_tp).unwrap();
    assert_eq!(h2.len(), 3);
    assert!((h2[0].0 - 0.5).abs() < 1e-12 && (h2[0].1 - 2.0).abs() < 1e-12);
    assert!((h2[1].0 - 0.5).abs() < 1e-12 && (h2[1].1 - 6.0).abs() < 1e-12);
    assert!((h2[2].0 - 1.0).abs() < 1e-12 && (h2[2].1 - 0.0).abs() < 1e-12);

    // Without leading 0.0,0.0 anchor row
    let csv_tp_no_zero = "time_h,precip_mm\n0.5,10.0\n1.5,20.0\n";
    let h3 = parse_hyetograph_csv(csv_tp_no_zero).unwrap();
    assert_eq!(h3.len(), 2);
    assert!((h3[0].0 - 0.5).abs() < 1e-12 && (h3[0].1 - 2.0).abs() < 1e-12);
    assert!((h3[1].0 - 1.0).abs() < 1e-12 && (h3[1].1 - 2.0).abs() < 1e-12);

    // File loader round-trip
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("gauge.csv");
    std::fs::write(&p, csv_tp).unwrap();
    assert_eq!(load_hyetograph_csv(&p).unwrap(), h2);

    // Malformed cases rejected
    assert!(parse_hyetograph_csv("").is_err());
    assert!(parse_hyetograph_csv("dt_h,intensity_cm_h\n").is_err());
    assert!(parse_hyetograph_csv("foo,bar\n1,2\n").is_err());
    assert!(parse_hyetograph_csv("dt_h,intensity_cm_h\n-0.5,2.0\n").is_err());
    assert!(parse_hyetograph_csv("dt_h,intensity_cm_h\n0.5,-1.0\n").is_err());
    assert!(parse_hyetograph_csv("time_h,precip_mm\n0.0,5.0\n1.0,2.0\n").is_err());
    assert!(parse_hyetograph_csv("time_h,precip_mm\n1.0,5.0\n0.5,2.0\n").is_err());
    assert!(parse_hyetograph_csv("dt_h,intensity_cm_h\n0.5,2.0,9.0\n").is_err());
}

#[test]
fn spatial_rainfall_grid_scaling_composes_with_static_and_fsm() {
    let dem = build_funnel_dem(2.0);
    let (w, h) = (dem.width(), dem.height());
    let base_rc = make_dem(w, h, vec![0.8; w * h], -9999.0, dem.geotransform);
    let rain_hi = make_dem(w, h, vec![100.0; w * h], -9999.0, dem.geotransform);
    let rain_lo = make_dem(w, h, vec![50.0; w * h], -9999.0, dem.geotransform);

    let scaled_hi = scale_forcing_by_rainfall_grid(Some(&base_rc), &rain_hi, 100.0).unwrap();
    let scaled_lo = scale_forcing_by_rainfall_grid(Some(&base_rc), &rain_lo, 100.0).unwrap();
    assert!((scaled_hi.get(10, 10) - 0.8).abs() < 1e-12);
    assert!((scaled_lo.get(10, 10) - 0.4).abs() < 1e-12);

    // Negative rainfall and non-positive reference precip rejected
    let rain_neg = make_dem(w, h, vec![-1.0; w * h], -9999.0, dem.geotransform);
    assert!(scale_forcing_by_rainfall_grid(None, &rain_neg, 100.0).is_err());
    assert!(scale_forcing_by_rainfall_grid(None, &rain_hi, 0.0).is_err());

    // Compose with FSM: 2x rainfall multiplier fills in half the time.
    let sinks = extract_sinks_in_memory(&dem, 10, Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 10, 0.0, 0.2).unwrap();
    let sim = SimulateInundationOptions {
        min_size: 10,
        min_depth: 0.0,
        interval: 0.2,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 50.0,
        time_step_h: 1.0,
        compress: false,
    };
    let fsm_hi = simulate_fill_spill_merge_in_memory(
        &sinks.dem_refined,
        &sinks.sink,
        &deps,
        &sim,
        &InundationForcing {
            runoff_coeff: Some(&scaled_hi),
            ..Default::default()
        },
    )
    .unwrap();
    let fsm_lo = simulate_fill_spill_merge_in_memory(
        &sinks.dem_refined,
        &sinks.sink,
        &deps,
        &sim,
        &InundationForcing {
            runoff_coeff: Some(&scaled_lo),
            ..Default::default()
        },
    )
    .unwrap();
    let t_hi = fsm_hi.basins[0].fill_time_h.unwrap();
    let t_lo = fsm_lo.basins[0].fill_time_h.unwrap();
    assert!(
        (t_lo / t_hi - 2.0).abs() < 1e-6,
        "half rainfall grid must double fill time: t_hi={t_hi}, t_lo={t_lo}"
    );
}

#[test]
fn post_storm_recession_drains_analytical_cylinder_and_conserves_volume() {
    // 11x11 DEM with flat-bottomed 5x5 cylindrical depression:
    // floor z = 10.0, vertical rim wall z = 12.0 (depth = 2.0 m, cell_area = 1.0 m^2, cap = 50 m^3).
    let (w, h) = (11usize, 11usize);
    let mut z = vec![12.0f64; w * h];
    for r in 3..8 {
        for c in 3..8 {
            z[r * w + c] = 10.0;
        }
    }
    // Outer ring at 11.9 so boundary drains outward and only the 5x5 cylinder is a depression.
    for r in 0..h {
        for c in 0..w {
            if r == 0 || r + 1 == h || c == 0 || c + 1 == w {
                z[r * w + c] = 11.9;
            }
        }
    }
    let dem = make_dem(w, h, z, -9999.0, [0.0, 1.0, 0.0, h as f64, 0.0, -1.0]);
    let sinks = extract_sinks_in_memory(&dem, 10, Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 10, 0.0, 0.2).unwrap();
    assert_eq!(deps.depressions.len(), 1);

    // Fill completely to capacity (depth = 2.0 m) during a long storm.
    let sim = SimulateInundationOptions {
        min_size: 10,
        min_depth: 0.0,
        interval: 0.2,
        rain_intensity_cm_h: 50.0,
        rain_duration_h: 20.0,
        time_step_h: 1.0,
        compress: false,
    };
    let fsm = simulate_fill_spill_merge_in_memory(
        &sinks.dem_refined,
        &sinks.sink,
        &deps,
        &sim,
        &InundationForcing::default(),
    )
    .unwrap();
    let cap = fsm.basins[0].capacity_m3;
    let init_stored = fsm.mass_balance.total_stored_m3;
    assert!((init_stored - cap).abs() < 1e-6);
    let depth_m = cap / 25.0; // exact cylinder depth (2.0 m)

    // Exact full drainage: et = 2.0 cm/h, seepage = 8.0 cm/h => total rate = 10.0 cm/h = 0.10 m/h.
    // Exact drain time = depth_m / 0.10 = 20.0 hours.
    let rec_full = simulate_post_storm_recession(
        &sinks.dem_refined,
        &deps,
        &fsm,
        &RecessionOptions {
            et_cm_h: 2.0,
            seepage_cm_h: 8.0,
            max_recession_h: 50.0,
        },
    )
    .unwrap();
    let expected_t = depth_m / 0.10;
    assert_eq!(rec_full.drained_basin_count, 1);
    assert!(rec_full.remaining_volume_m3.abs() < 1e-9);
    assert!((rec_full.total_drained_m3 - init_stored).abs() < 1e-6);
    for r in 3..8 {
        for c in 3..8 {
            let dt = rec_full.drain_time_h.get(c, r);
            assert!(
                (dt - expected_t).abs() < 1e-6,
                "cell ({c},{r}) drain_time={dt}, expected {expected_t}"
            );
        }
    }
    assert_eq!(rec_full.drain_time_h.get(0, 0), -1.0);
    assert!((rec_full.curve.first().unwrap().stored_vol_m3 - init_stored).abs() < 1e-6);
    assert!(rec_full.curve.last().unwrap().stored_vol_m3.abs() < 1e-9);
    assert_eq!(rec_full.curve.last().unwrap().wet_basin_count, 0);

    // Independent physical flux integration: integrate `rate_m_h * A_wet(t) dt` across the
    // sampled recession curve (using the active wet area during each interval `[t_k, t_{k+1}]`)
    // and verify it matches both analytical cylinder volume `init_stored` and `total_drained_m3`.
    let rate_m_h = (2.0 + 8.0) / 100.0;
    let integrated_full_drain_m3: f64 = rec_full
        .curve
        .windows(2)
        .map(|w| rate_m_h * w[0].inundated_area_m2 * (w[1].time_h - w[0].time_h))
        .sum();
    assert!(
        (integrated_full_drain_m3 - init_stored).abs() < 1e-6
            && (integrated_full_drain_m3 - rec_full.total_drained_m3).abs() < 1e-6,
        "integrated vertical flux ({integrated_full_drain_m3:.6} m^3) must match initial volume ({init_stored:.6} m^3) and total_drained_m3 ({:.6} m^3)",
        rec_full.total_drained_m3
    );

    // Partial drainage (10 h = half of 20 h): exact half volume remains and cells stay INFINITY.
    let rec_half = simulate_post_storm_recession(
        &sinks.dem_refined,
        &deps,
        &fsm,
        &RecessionOptions {
            et_cm_h: 2.0,
            seepage_cm_h: 8.0,
            max_recession_h: expected_t * 0.5,
        },
    )
    .unwrap();
    assert_eq!(rec_half.drained_basin_count, 0);
    assert!((rec_half.remaining_volume_m3 - 0.5 * init_stored).abs() < 1e-6);
    assert!((rec_half.total_drained_m3 - 0.5 * init_stored).abs() < 1e-6);
    assert!(rec_half.drain_time_h.get(5, 5).is_infinite());
    let integrated_half_drain_m3: f64 = rec_half
        .curve
        .windows(2)
        .map(|w| rate_m_h * w[0].inundated_area_m2 * (w[1].time_h - w[0].time_h))
        .sum();
    assert!(
        (integrated_half_drain_m3 - 0.5 * init_stored).abs() < 1e-6
            && (integrated_half_drain_m3 - rec_half.total_drained_m3).abs() < 1e-6,
        "integrated half-recession flux ({integrated_half_drain_m3:.6} m^3) must match half initial volume ({:.6} m^3)",
        0.5 * init_stored
    );

    // Zero loss rate conserves full volume.
    let rec_zero = simulate_post_storm_recession(
        &sinks.dem_refined,
        &deps,
        &fsm,
        &RecessionOptions {
            et_cm_h: 0.0,
            seepage_cm_h: 0.0,
            max_recession_h: 24.0,
        },
    )
    .unwrap();
    assert_eq!(rec_zero.total_drained_m3, 0.0);
    assert!((rec_zero.remaining_volume_m3 - init_stored).abs() < 1e-6);

    // Real sample DEM recession mass conservation and monotonic curve.
    let real_dem = load_raster::<f64>(test_data_dir().join("dem.tif"), None).unwrap();
    let r_sinks = extract_sinks_in_memory(&real_dem, 1000, Topology::D8).unwrap();
    let r_deps = delineate_depressions_in_memory(&r_sinks.sink, 1000, 0.2, 0.3).unwrap();
    let r_sim = SimulateInundationOptions {
        min_size: 1000,
        min_depth: 0.2,
        interval: 0.3,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 6.0,
        time_step_h: 1.0,
        compress: false,
    };
    let r_fsm = simulate_fill_spill_merge_in_memory(
        &r_sinks.dem_refined,
        &r_sinks.sink,
        &r_deps,
        &r_sim,
        &InundationForcing::default(),
    )
    .unwrap();
    let r_rec = simulate_post_storm_recession(
        &r_sinks.dem_refined,
        &r_deps,
        &r_fsm,
        &RecessionOptions::default(),
    )
    .unwrap();
    assert!(
        (r_rec.total_drained_m3 + r_rec.remaining_volume_m3 - r_fsm.mass_balance.total_stored_m3)
            .abs()
            < 1e-4,
        "recession must conserve storm-end stored volume on dem.tif"
    );
    for w in r_rec.curve.windows(2) {
        assert!(w[1].time_h > w[0].time_h);
        assert!(w[1].stored_vol_m3 <= w[0].stored_vol_m3 + 1e-6);
    }
}

// ---------------------------------------------------------------------------
// 3. Terrain-hydrology derivatives: D-Inf, MFD, TWI, HAND, burn_streams
// ---------------------------------------------------------------------------

#[test]
fn dinf_direction_exact_on_inclined_planes_and_conserves_flow() {
    let (w, h) = (15usize, 15usize);
    let gt = [0.0, 2.0, 0.0, h as f64 * 2.0, 0.0, -2.0];

    // Plane sloping due East: z = 100 - 0.5 * c -> angle = 0.0
    let z_east: Vec<f64> = (0..h)
        .flat_map(|_| (0..w).map(|c| 100.0 - 0.5 * c as f64))
        .collect();
    let dem_e = make_dem(w, h, z_east, -9999.0, gt);
    let dir_e = dinf_flow_direction(&dem_e).unwrap();
    assert!(dir_e.get(5, 5).abs() < 1e-12);
    // East edge cells have no lower East neighbor -> flat/pit (-1.0)
    assert_eq!(dir_e.get(w - 1, 5), -1.0);

    // Plane sloping due North (decreasing row): z = 100 + 0.5 * r -> angle = pi/2
    let z_north: Vec<f64> = (0..h)
        .flat_map(|r| (0..w).map(move |_| 100.0 + 0.5 * r as f64))
        .collect();
    let dem_n = make_dem(w, h, z_north, -9999.0, gt);
    let dir_n = dinf_flow_direction(&dem_n).unwrap();
    assert!((dir_n.get(5, 5) - FRAC_PI_2).abs() < 1e-12);

    // Plane sloping due West: angle = pi
    let z_west: Vec<f64> = (0..h)
        .flat_map(|_| (0..w).map(|c| 100.0 + 0.5 * c as f64))
        .collect();
    let dem_w = make_dem(w, h, z_west, -9999.0, gt);
    let dir_w = dinf_flow_direction(&dem_w).unwrap();
    assert!((dir_w.get(5, 5) - PI).abs() < 1e-12);

    // Plane sloping due South: angle = 3*pi/2
    let z_south: Vec<f64> = (0..h)
        .flat_map(|r| (0..w).map(move |_| 100.0 - 0.5 * r as f64))
        .collect();
    let dem_s = make_dem(w, h, z_south, -9999.0, gt);
    let dir_s = dinf_flow_direction(&dem_s).unwrap();
    assert!((dir_s.get(5, 5) - 3.0 * FRAC_PI_2).abs() < 1e-12);

    // Diagonal NE plane: z = 100 - c + r -> drops equally East (+c) and North (-r) -> angle = pi/4
    let z_ne: Vec<f64> = (0..h)
        .flat_map(|r| (0..w).map(move |c| 100.0 - c as f64 + r as f64))
        .collect();
    let dem_ne = make_dem(w, h, z_ne, -9999.0, gt);
    let dir_ne = dinf_flow_direction(&dem_ne).unwrap();
    assert!((dir_ne.get(5, 5) - FRAC_PI_4).abs() < 1e-12);

    // Oblique plane: z = 100 - 2*c + 1*r -> gradient vector (+2 East, +1 North) -> atan2(1, 2)
    let z_oblique: Vec<f64> = (0..h)
        .flat_map(|r| (0..w).map(move |c| 100.0 - 2.0 * c as f64 + 1.0 * r as f64))
        .collect();
    let dem_ob = make_dem(w, h, z_oblique, -9999.0, gt);
    let dir_ob = dinf_flow_direction(&dem_ob).unwrap();
    assert!((dir_ob.get(5, 5) - 1.0f64.atan2(2.0)).abs() < 1e-12);

    // Flow conservation on East-sloping plane: sum of outflow column == w * h
    let acc_e = dinf_flow_accumulation(&dem_e).unwrap();
    let outflow_e: f64 = (0..h).map(|r| acc_e.get(w - 1, r)).sum();
    assert!((outflow_e - (w * h) as f64).abs() < 1e-9);
    // Every cell along row 5 increases linearly 1, 2, ..., w
    for c in 0..w {
        assert!((acc_e.get(c, 5) - (c + 1) as f64).abs() < 1e-9);
    }

    // Flow conservation on oblique plane: sum along North + East boundary exits == w * h
    let acc_ob = dinf_flow_accumulation(&dem_ob).unwrap();
    let mut outflow_ob = 0.0f64;
    for r in 0..h {
        for c in 0..w {
            if dir_ob.get(c, r) < 0.0 {
                outflow_ob += acc_ob.get(c, r);
            }
        }
    }
    // Note: on oblique plane, top row (r=0) still flows East along the edge, terminating at (w-1, 0),
    // and right column (c=w-1) still flows North along the edge, also terminating at (w-1, 0)!
    assert!((outflow_ob - (w * h) as f64).abs() < 1e-8);
}

#[test]
fn mfd_and_dinf_conserve_flow_on_valley_and_real_dem() {
    // 21x21 V-shaped valley sloping gently South along col 10, walls rising East & West.
    let (w, h) = (21usize, 21usize);
    let mut z = vec![0.0f64; w * h];
    for r in 0..h {
        for c in 0..w {
            let wall = (c as isize - 10).abs() as f64 * 1.0;
            let valley = (h - 1 - r) as f64 * 0.1;
            z[r * w + c] = 10.0 + wall + valley;
        }
    }
    let dem = make_dem(w, h, z, -9999.0, [0.0, 2.0, 0.0, h as f64 * 2.0, 0.0, -2.0]);

    let dinf_acc = dinf_flow_accumulation(&dem).unwrap();
    let mfd_acc = mfd_flow_accumulation(&dem, 1.1).unwrap();
    let mfd_steep = mfd_flow_accumulation(&dem, 4.0).unwrap();

    // Every single cell drains to the single outlet at (col=10, row=h-1)!
    assert!(
        (dinf_acc.get(10, h - 1) - (w * h) as f64).abs() < 1e-8,
        "D-inf must conserve all {} cells at valley outlet, got {}",
        w * h,
        dinf_acc.get(10, h - 1)
    );
    assert!(
        (mfd_acc.get(10, h - 1) - (w * h) as f64).abs() < 1e-8,
        "MFD must conserve all {} cells at valley outlet, got {}",
        w * h,
        mfd_acc.get(10, h - 1)
    );
    // Higher MFD exponent concentrates more flow into the steeper competing neighbor.
    // East (2, 1) = 8.0 (slope 2.0) and West (0, 1) = 9.0 (slope 1.0) are non-adjacent outlets.
    let div_dem = make_dem(
        3,
        3,
        vec![
            12.0, 12.0, 12.0, //
            9.0, 10.0, 8.0, // West=9.0 (slope 1.0), center=10.0, East=8.0 (slope 2.0)
            12.0, 12.0, 12.0, //
        ],
        -9999.0,
        [0.0, 1.0, 0.0, 3.0, 0.0, -1.0],
    );
    let div_lo = mfd_flow_accumulation(&div_dem, 1.1).unwrap();
    let div_hi = mfd_flow_accumulation(&div_dem, 4.0).unwrap();
    assert!(
        div_hi.get(2, 1) > div_lo.get(2, 1) && div_hi.get(0, 1) < div_lo.get(0, 1),
        "higher MFD exponent must concentrate flow toward steeper East neighbor: hi_E={}, lo_E={}, hi_W={}, lo_W={}",
        div_hi.get(2, 1),
        div_lo.get(2, 1),
        div_hi.get(0, 1),
        div_lo.get(0, 1)
    );
    assert!((div_hi.get(2, 1) + div_hi.get(0, 1) - 9.0).abs() < 1e-9);
    assert!((div_lo.get(2, 1) + div_lo.get(0, 1) - 9.0).abs() < 1e-9);
    let _ = mfd_steep;
    assert!(mfd_flow_accumulation(&dem, 0.0).is_err());
    assert!(mfd_flow_accumulation(&dem, -1.0).is_err());

    // Verify finite non-negative outputs and total conservation on real test_data/dem.tif.
    let real_dem = load_raster::<f64>(test_data_dir().join("dem.tif"), None).unwrap();
    let r_dinf = dinf_flow_accumulation(&real_dem).unwrap();
    let r_mfd = mfd_flow_accumulation(&real_dem, 1.1).unwrap();
    let nd = real_dem.no_data();
    let valid_cnt = real_dem
        .data()
        .iter()
        .filter(|&&v| v.is_finite() && !v.rd_eq(nd))
        .count();
    let dinf_max = r_dinf
        .data()
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && !v.rd_eq(r_dinf.no_data()))
        .fold(0.0f64, f64::max);
    let mfd_max = r_mfd
        .data()
        .iter()
        .copied()
        .filter(|&v| v.is_finite() && !v.rd_eq(r_mfd.no_data()))
        .fold(0.0f64, f64::max);
    assert!(dinf_max > 100.0 && dinf_max <= valid_cnt as f64 + 1e-6);
    assert!(mfd_max > 100.0 && mfd_max <= valid_cnt as f64 + 1e-6);
}

#[test]
fn twi_higher_in_valley_bottom_than_ridge_and_hand_exact() {
    // 21x21 terrain with ridge crests at c=0 and c=20, sloping into a valley at c=10.
    let (w, h) = (21usize, 21usize);
    let mut z = vec![0.0f64; w * h];
    for r in 0..h {
        for c in 0..w {
            let wall = (c as isize - 10).abs() as f64 * 1.5;
            let valley = (h - 1 - r) as f64 * 0.05;
            z[r * w + c] = 10.0 + wall + valley;
        }
    }
    let dem = make_dem(w, h, z, -9999.0, [0.0, 1.0, 0.0, h as f64, 0.0, -1.0]);

    let twi = compute_twi(&dem, None).unwrap();
    let twi_valley = twi.get(10, 15);
    let twi_slope = twi.get(5, 15);
    let twi_ridge = twi.get(0, 15);
    assert!(
        twi_valley > twi_slope && twi_slope > twi_ridge,
        "TWI must decrease from valley ({twi_valley:.3}) to slope ({twi_slope:.3}) to ridge ({twi_ridge:.3})"
    );

    // Explicit stream mask along valley floor (col 10)
    let mut mask_data = vec![0u8; w * h];
    for r in 0..h {
        mask_data[r * w + 10] = 1;
    }
    let mut mask = Array2D::from_vec(w, h, mask_data).unwrap();
    mask.copy_metadata_from(&dem);
    mask.set_no_data(0);

    let hand = compute_hand(&dem, &mask).unwrap();
    for r in 0..h {
        assert_eq!(hand.get(10, r), 0.0, "stream cell must have HAND == 0");
    }
    // Because wall slope (1.5/1) >> valley slope (0.05/1), D8 from interior hillslope cells
    // routes East into the valley channel; verify HAND increases linearly moving away from col 10.
    let h_near = hand.get(8, 10);
    let h_mid = hand.get(5, 10);
    let h_far = hand.get(1, 10);
    assert!(
        h_near > 0.0 && h_mid > h_near && h_far > h_mid,
        "expected h_far ({h_far}) > h_mid ({h_mid}) > h_near ({h_near}) > 0"
    );
    // Wall rises 1.5 m per cell; for pure East flow (5 cells away from col 10 -> 7.5 m drop).
    assert!(
        (h_mid - 7.5).abs() <= 0.35,
        "expected ~7.5 m HAND, got {h_mid}"
    );

    // Threshold-based HAND on V-shaped valley and real DEM
    let hand_thr = compute_hand_from_threshold(&dem, 30.0).unwrap();
    assert_eq!(hand_thr.get(10, h - 1), 0.0);
    assert!(hand_thr.get(1, 10) > 10.0);
    assert!(compute_hand_from_threshold(&dem, 0.0).is_err());

    let real_dem = load_raster::<f64>(test_data_dir().join("dem.tif"), None).unwrap();
    let real_twi = compute_twi(&real_dem, None).unwrap();
    let real_hand = compute_hand_from_threshold(&real_dem, 500.0).unwrap();
    let twi_valid = real_twi
        .data()
        .iter()
        .filter(|&&v| v.is_finite() && !v.rd_eq(real_twi.no_data()))
        .count();
    let hand_zeros = real_hand.data().iter().filter(|&&v| v == 0.0).count();
    assert!(twi_valid > 1000);
    assert!(hand_zeros > 10);
}

#[test]
fn burn_streams_lowers_channel_and_agree_buffer_without_touching_far_cells() {
    let (w, h) = (21usize, 21usize);
    let gt = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];
    let mut dem = make_dem(w, h, vec![100.0; w * h], -9999.0, gt);
    dem.projection = "EPSG:32616".to_string();

    let tmp = tempfile::tempdir().unwrap();
    let stream_path = tmp.path().join("streams.gpkg");
    let mut layer = Layer::new("streams".to_string()).with_geom_type(GeometryType::LineString);
    let srs = terrio::crs::SpatialRef::from_epsg(32616).unwrap();
    srs.apply_to_layer(&mut layer);
    // Horizontal stream along row 10 (y = 10.5) from col 2 (x = 2.5) to col 18 (x = 18.5).
    layer.features.push(Feature {
        fid: 1,
        geometry: Some(Geometry::LineString(vec![
            Coord::xy(2.5, 10.5),
            Coord::xy(18.5, 10.5),
        ])),
        attributes: Vec::new(),
    });
    terrio::write_vector(
        &[layer],
        stream_path.to_str().unwrap(),
        &VectorWriteOptions::default(),
    )
    .unwrap();

    // 1. Sharp centerline burn (burn_depth = 2.5, width = 0, buffer = 0)
    let (burned, mod_cnt) = burn_streams(
        &dem,
        &stream_path,
        &BurnStreamsOptions {
            burn_depth: 2.5,
            stream_width_cells: 0,
            buffer_cells: 0,
            smooth_drop: 0.0,
            compress: false,
        },
    )
    .unwrap();
    assert_eq!(mod_cnt, 17); // cols 2..=18 on row 10
    assert!((burned.get(10, 10) - 97.5).abs() < 1e-12);
    assert_eq!(burned.get(10, 9), 100.0);
    assert_eq!(burned.get(0, 0), 100.0);

    // 2. AGREE-style burn (burn_depth = 1.0, width = 0, buffer = 4, smooth_drop = 2.0)
    let (agree, _) = burn_streams(
        &dem,
        &stream_path,
        &BurnStreamsOptions {
            burn_depth: 1.0,
            stream_width_cells: 0,
            buffer_cells: 4,
            smooth_drop: 2.0,
            compress: false,
        },
    )
    .unwrap();
    // Centerline drops by burn_depth + smooth_drop = 3.0 -> 97.0
    assert!((agree.get(10, 10) - 97.0).abs() < 1e-12);
    // 1 cell off stream: smooth_drop * (1 - 1/4) = 1.5 -> 98.5
    assert!((agree.get(10, 9) - 98.5).abs() < 1e-12);
    // 2 cells off stream: smooth_drop * (1 - 2/4) = 1.0 -> 99.0
    assert!((agree.get(10, 8) - 99.0).abs() < 1e-12);
    // 4 cells off stream (edge of buffer): drop = 0.0 -> untouched 100.0
    assert_eq!(agree.get(10, 6), 100.0);
    assert_eq!(agree.get(0, 0), 100.0);

    // 3. File wrapper & path collision guard
    let in_dem_path = tmp.path().join("in_dem.tif");
    let out_dem_path = tmp.path().join("out_dem.tif");
    save_raster(&dem, &in_dem_path, None, false).unwrap();
    assert!(
        burn_streams_file(
            &in_dem_path,
            &stream_path,
            &in_dem_path,
            &BurnStreamsOptions::default()
        )
        .is_err()
    );
    let (file_out, file_cnt) = burn_streams_file(
        &in_dem_path,
        &stream_path,
        &out_dem_path,
        &BurnStreamsOptions::default(),
    )
    .unwrap();
    assert_eq!(file_cnt, 17);
    assert!((file_out.get(10, 10) - 99.0).abs() < 1e-12);
    assert!(out_dem_path.exists());
}
