use std::path::PathBuf;

use lidar::{
    Array2D, ArtifactClassifierOptions, BreachMode, BreachOptions, FsmForcing, GreenAmptParams,
    InMemoryDepressions, InMemoryFullPipelineOptions, InundationForcing, RoutingMode,
    RunPipelineOptions, SimulateInundationOptions, SpillAttenuationOptions, Topology,
    attenuate_fsm_hydrograph, attenuate_fsm_hydrograph_with_options,
    breach_depressions_file_with_no_data, classify_embankment_artifacts,
    compute_fsm_hazard_rasters, crs_metric_scale, delineate_depressions_in_memory,
    estimate_mean_travel_time_h, extract_sinks_in_memory, fsm_hydrograph_to_csv_string,
    green_ampt_hyetograph_losses, green_ampt_infiltration_rate_cm_h, integrate_fsm_hydrograph_m3,
    load_raster, run_full_pipeline_in_memory, run_pipeline, save_raster,
    scs_cn_incremental_hyetograph, scs_type2_hyetograph, simulate_fill_spill_merge_in_memory,
    simulate_fill_spill_merge_with_green_ampt_in_memory, write_fsm_hazard_rasters,
    write_fsm_hydrograph_csv,
};

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data")
}

/// Builds a 34x20 DEM with a compound depression (two asymmetric sibling bowls A and B)
/// plus a sloping outer perimeter draining to the DEM boundary so both internal cascades
/// and watershed boundary outflow occur within a single storm.
fn build_compound_watershed_dem() -> (Array2D<f64>, Array2D<f64>, InMemoryDepressions) {
    let (w, h) = (34usize, 20usize);
    let mut z = vec![20.0f64; w * h];
    for r in 0..h {
        for c in 0..w {
            if r == 0 || r + 1 == h || c == 0 || c + 1 == w {
                // Lower outer boundary so exterior cells drain off-grid directly.
                z[r * w + c] = 17.0;
            } else if r == 1 || r + 2 == h || c == 1 || c + 2 == w {
                z[r * w + c] = 18.0;
            } else {
                let d1 = ((r as f64 - 10.0).powi(2) + (c as f64 - 9.0).powi(2)).sqrt();
                let d2 = ((r as f64 - 10.0).powi(2) + (c as f64 - 23.0).powi(2)).sqrt();
                let z1 = 13.0 + 0.45 * d1;
                let z2 = 9.5 + 0.80 * d2;
                z[r * w + c] = z1.min(z2).min(18.5);
            }
        }
    }
    // Lower spill saddle on the east edge so the compound parent drains to boundary once full.
    z[10 * w + (w - 2)] = 17.8;

    let mut dem = Array2D::from_vec(w, h, z).unwrap();
    dem.set_no_data(-9999.0);
    dem.geotransform = [500_000.0, 2.0, 0.0, 4_500_040.0, 0.0, -2.0];
    dem.projection = "EPSG:26914".to_string();

    let sinks = extract_sinks_in_memory(&dem, 10, Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 10, 0.3, 0.3).unwrap();
    (sinks.dem_refined, sinks.sink, deps)
}

#[test]
fn fsm_hydrograph_connectivity_and_integral_match_mass_balance() {
    let (dem_refined, sink, deps) = build_compound_watershed_dem();
    assert!(deps.max_level >= 2, "expected nested depression hierarchy");

    let opts = SimulateInundationOptions {
        min_size: 10,
        min_depth: 0.3,
        interval: 0.3,
        rain_intensity_cm_h: 8.0,
        rain_duration_h: 120.0,
        time_step_h: 2.0,
        compress: false,
    };

    let res = simulate_fill_spill_merge_in_memory(
        &dem_refined,
        &sink,
        &deps,
        &opts,
        &InundationForcing::default(),
    )
    .unwrap();

    assert!(
        res.hydrograph.len() >= 3,
        "expected initial, intermediate event, and storm-end hydrograph points, got {}",
        res.hydrograph.len()
    );
    assert!((res.hydrograph[0].time_h - 0.0).abs() < 1e-12);
    assert!((res.hydrograph.last().unwrap().time_h - opts.rain_duration_h).abs() < 1e-12);

    // Timestamps must be strictly increasing (cleanly deduplicated).
    for pair in res.hydrograph.windows(2) {
        assert!(
            pair[1].time_h > pair[0].time_h + 1e-12,
            "duplicate or non-monotonic hydrograph timestamps: {} -> {}",
            pair[0].time_h,
            pair[1].time_h
        );
    }

    // Connected area fraction must remain within [0, 1] and increase as basins connect to outlet.
    for pt in &res.hydrograph {
        assert!((0.0..=1.0).contains(&pt.connected_area_fraction));
        assert!(pt.q_boundary_m3_s >= 0.0);
        assert!(pt.total_stored_m3 >= 0.0);
    }
    let initial_conn = res.hydrograph.first().unwrap().connected_area_fraction;
    let final_conn = res.hydrograph.last().unwrap().connected_area_fraction;
    assert!(
        final_conn > initial_conn,
        "dynamic threshold connectivity must rise as compound basin spills to outlet ({initial_conn} -> {final_conn})"
    );
    assert!(
        (final_conn - 1.0).abs() < 1e-6,
        "once all basins spill to boundary, connected_area_fraction should approach 1.0, got {final_conn}"
    );

    // Hydrograph integral must equal mass_balance.total_boundary_outflow_m3 to machine precision.
    let hydro_vol_m3 = integrate_fsm_hydrograph_m3(&res.hydrograph);
    let mb_vol_m3 = res.mass_balance.total_boundary_outflow_m3;
    let rel_err = (hydro_vol_m3 - mb_vol_m3).abs() / mb_vol_m3.max(1.0);
    assert!(
        rel_err < 1e-11,
        "hydrograph integral ({hydro_vol_m3:.9}) must match boundary outflow ({mb_vol_m3:.9}), rel_err={rel_err:.3e}"
    );

    // CSV formatting & file export check.
    let csv = fsm_hydrograph_to_csv_string(&res.hydrograph);
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines[0],
        "time_h,q_boundary_m3_s,total_stored_m3,connected_area_fraction,spilling_basin_count"
    );
    assert_eq!(lines.len(), res.hydrograph.len() + 1);

    let tmp = tempfile::tempdir().unwrap();
    let csv_path = tmp.path().join("fsm_hydrograph.csv");
    write_fsm_hydrograph_csv(&res.hydrograph, &csv_path).unwrap();
    assert_eq!(std::fs::read_to_string(&csv_path).unwrap(), csv);
}

#[test]
fn fsm_hydrograph_empty_depressions_fast_path_conserves_volume() {
    let mut dem = Array2D::from_vec(8, 8, vec![10.0f64; 64]).unwrap();
    dem.set_no_data(-9999.0);
    dem.geotransform = [0.0, 2.0, 0.0, 16.0, 0.0, -2.0];
    let mut sink = Array2D::from_vec(8, 8, vec![0.0f64; 64]).unwrap();
    sink.copy_metadata_from(&dem);
    sink.set_no_data(0.0);
    let mut id_arr = Array2D::from_vec(8, 8, vec![0i32; 64]).unwrap();
    id_arr.copy_metadata_from(&dem);
    id_arr.set_no_data(0);
    let deps = InMemoryDepressions {
        depression_id: id_arr.clone(),
        depression_level: id_arr,
        depressions: Vec::new(),
        max_level: 0,
        num_regions: 0,
    };
    let hyeto = [(1.0, 2.0), (2.0, 6.0), (1.0, 0.0)];
    let opts = SimulateInundationOptions {
        min_size: 4,
        min_depth: 0.2,
        interval: 0.2,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 4.0,
        time_step_h: 1.0,
        compress: false,
    };
    let res = simulate_fill_spill_merge_in_memory(
        &dem,
        &sink,
        &deps,
        &opts,
        &InundationForcing {
            hyetograph: Some(&hyeto),
            ..Default::default()
        },
    )
    .unwrap();
    let vol = integrate_fsm_hydrograph_m3(&res.hydrograph);
    assert!((vol - res.mass_balance.total_boundary_outflow_m3).abs() < 1e-10);
    assert!((res.hydrograph[0].connected_area_fraction - 1.0).abs() < 1e-12);
}

#[test]
fn fsm_spill_attenuation_conserves_volume_and_attenuates_lags_peak() {
    let (dem_refined, sink, deps) = build_compound_watershed_dem();
    // Pulsed hyetograph: quiet -> intense burst -> dry tail.
    let hyeto = vec![(1.0, 2.0), (2.0, 40.0), (2.0, 60.0), (5.0, 0.0)];
    let opts = SimulateInundationOptions {
        min_size: 10,
        min_depth: 0.3,
        interval: 0.3,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 10.0,
        time_step_h: 1.0,
        compress: false,
    };

    let res = simulate_fill_spill_merge_in_memory(
        &dem_refined,
        &sink,
        &deps,
        &opts,
        &InundationForcing {
            hyetograph: Some(&hyeto),
            ..Default::default()
        },
    )
    .unwrap();

    let tt = estimate_mean_travel_time_h(&res.spill_paths, &dem_refined.geotransform, 0.2).unwrap();
    assert!(tt >= 0.0 && tt.is_finite());

    let raw_vol = integrate_fsm_hydrograph_m3(&res.hydrograph);
    assert!(raw_vol > 0.0);

    let att = attenuate_fsm_hydrograph(&res.hydrograph, 1.5, 0.25).unwrap();
    let att_vol = integrate_fsm_hydrograph_m3(&att);
    let rel_err = (att_vol - raw_vol).abs() / raw_vol;
    assert!(
        rel_err < 1e-9,
        "attenuated hydrograph must conserve total outflow volume (raw={raw_vol:.6}, att={att_vol:.6}, rel_err={rel_err:.3e})"
    );

    let raw_peak = res
        .hydrograph
        .iter()
        .max_by(|a, b| a.q_boundary_m3_s.total_cmp(&b.q_boundary_m3_s))
        .unwrap();
    let att_peak = att
        .iter()
        .max_by(|a, b| a.q_boundary_m3_s.total_cmp(&b.q_boundary_m3_s))
        .unwrap();

    assert!(
        att_peak.q_boundary_m3_s < raw_peak.q_boundary_m3_s,
        "attenuated peak ({}) must be strictly lower than raw peak ({})",
        att_peak.q_boundary_m3_s,
        raw_peak.q_boundary_m3_s
    );
    assert!(
        att_peak.time_h > raw_peak.time_h,
        "attenuated peak time ({}) must lag raw peak time ({})",
        att_peak.time_h,
        raw_peak.time_h
    );

    // Verify configurable options validation.
    let single_res = attenuate_fsm_hydrograph_with_options(
        &res.hydrograph,
        &SpillAttenuationOptions {
            mean_travel_time_h: 1.0,
            dt_h: 0.25,
            num_reservoirs: 1,
            tail_duration_h: None,
        },
    )
    .unwrap();
    assert!((integrate_fsm_hydrograph_m3(&single_res) - raw_vol).abs() / raw_vol < 1e-9);
    assert!(attenuate_fsm_hydrograph(&res.hydrograph, -1.0, 0.25).is_err());
    assert!(attenuate_fsm_hydrograph(&res.hydrograph, 1.0, 0.0).is_err());
}

#[test]
fn fsm_hazard_rasters_verify_geometry_and_draining_duration() {
    let (dem_refined, sink, deps) = build_compound_watershed_dem();

    // Case 1: monotonic storm where depressions fill and merge.
    let opts = SimulateInundationOptions {
        min_size: 10,
        min_depth: 0.3,
        interval: 0.3,
        rain_intensity_cm_h: 8.0,
        rain_duration_h: 80.0,
        time_step_h: 2.0,
        compress: false,
    };
    let res = simulate_fill_spill_merge_in_memory(
        &dem_refined,
        &sink,
        &deps,
        &opts,
        &InundationForcing::default(),
    )
    .unwrap();

    let hazards = compute_fsm_hazard_rasters(
        &dem_refined,
        &res.fill_time,
        &deps,
        &res,
        opts.rain_duration_h,
    )
    .unwrap();

    let z_nd = dem_refined.no_data();
    let ft_nd = res.fill_time.no_data();
    let wse_nd = hazards.wse.no_data();
    let mut inundated_count = 0usize;
    let mut max_observed_depth = 0.0f64;

    for idx in 0..dem_refined.data().len() {
        let z = dem_refined.data()[idx];
        if !z.is_finite() || (z - z_nd).abs() < 1e-12 {
            continue;
        }
        let ft = res.fill_time.data()[idx];
        let depth = hazards.max_depth.data()[idx];
        let wse = hazards.wse.data()[idx];
        let dur = hazards.duration_h.data()[idx];

        assert!(depth >= 0.0, "max_depth must be >= 0, got {depth}");
        assert!(
            (0.0..=opts.rain_duration_h + 1e-9).contains(&dur),
            "duration_h must be in [0, storm_duration_h], got {dur}"
        );

        let is_wet = ft.is_finite()
            && (ft - ft_nd).abs() > 1e-12
            && ft >= 0.0
            && ft <= opts.rain_duration_h + 1e-9;
        if is_wet {
            inundated_count += 1;
            assert!(
                wse.is_finite() && (wse - wse_nd).abs() > 1e-12,
                "inundated cell must have valid WSE"
            );
            assert!(
                wse >= z - 1e-9,
                "WSE ({wse}) must be >= DEM elevation ({z}) on inundated cell"
            );
            assert!((depth - (wse - z).max(0.0)).abs() < 1e-9);
            assert!((dur - (opts.rain_duration_h - ft)).abs() < 1e-6);
            if depth > max_observed_depth {
                max_observed_depth = depth;
            }
        } else {
            assert_eq!(depth, 0.0);
            assert_eq!(dur, 0.0);
            assert!((wse - wse_nd).abs() < 1e-12);
        }
    }
    assert!(inundated_count > 20, "expected inundated depression cells");
    assert!(max_observed_depth > 2.0, "expected deep basin ponding");

    // Verify file writing of all three GeoTIFFs.
    let tmp = tempfile::tempdir().unwrap();
    let (p_depth, p_wse, p_dur) = write_fsm_hazard_rasters(&hazards, tmp.path(), false).unwrap();
    assert!(p_depth.exists() && p_wse.exists() && p_dur.exists());
    let reloaded_depth = load_raster::<f64>(&p_depth, None).unwrap();
    assert_eq!(reloaded_depth.width(), dem_refined.width());
    assert_eq!(reloaded_depth.height(), dem_refined.height());

    // Case 2: storm with a burst followed by a dry tail + infiltration so basins drain!
    let mut inf = Array2D::from_vec(
        dem_refined.width(),
        dem_refined.height(),
        vec![25.0f64; dem_refined.data().len()],
    )
    .unwrap();
    inf.copy_metadata_from(&dem_refined);
    inf.set_no_data(-9999.0);
    let hyeto = [(2.0, 80.0), (18.0, 0.0)];
    let opts_drain = SimulateInundationOptions {
        min_size: 10,
        min_depth: 0.3,
        interval: 0.3,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 20.0,
        time_step_h: 1.0,
        compress: false,
    };
    let res_drain = simulate_fill_spill_merge_in_memory(
        &dem_refined,
        &sink,
        &deps,
        &opts_drain,
        &InundationForcing {
            infiltration_cm_h: Some(&inf),
            hyetograph: Some(&hyeto),
            ..Default::default()
        },
    )
    .unwrap();
    let haz_drain = compute_fsm_hazard_rasters(
        &dem_refined,
        &res_drain.fill_time,
        &deps,
        &res_drain,
        opts_drain.rain_duration_h,
    )
    .unwrap();

    // At least some inundated cells must have drained before t=20h (duration_h < 20 - fill_time)
    // while still reporting positive peak max_depth > 0 from the storm peak!
    let mut saw_drained_cell = false;
    for idx in 0..dem_refined.data().len() {
        let ft = res_drain.fill_time.data()[idx];
        if ft.is_finite() && ft >= 0.0 && ft <= opts_drain.rain_duration_h {
            let dur = haz_drain.duration_h.data()[idx];
            let depth = haz_drain.max_depth.data()[idx];
            assert!((0.0..=opts_drain.rain_duration_h + 1e-9).contains(&dur));
            assert!(depth >= 0.0);
            if depth > 0.1 && dur < (opts_drain.rain_duration_h - ft) - 0.5 {
                saw_drained_cell = true;
            }
        }
    }
    assert!(
        saw_drained_cell,
        "draining storm must produce duration_h strictly shorter than (storm_duration_h - fill_time) while preserving peak depth"
    );
}

#[test]
fn scs_cn_incremental_hyetograph_initial_abstraction_and_exact_conservation() {
    let raw = scs_type2_hyetograph(100.0, 24.0, 0.5).unwrap();
    for (cn, ia_ratio) in [(75.0, 0.2), (80.0, 0.05), (90.0, 0.2), (100.0, 0.2)] {
        let excess = scs_cn_incremental_hyetograph(&raw, cn, ia_ratio).unwrap();
        assert_eq!(excess.len(), raw.len());

        let s_mm = if (cn - 100.0).abs() < 1e-12 {
            0.0
        } else {
            25_400.0 / cn - 254.0
        };
        let ia_mm = ia_ratio * s_mm;

        // Check initial abstraction delay: every prefix step with cum_p <= ia_mm must emit 0.0.
        let mut cum_p = 0.0f64;
        let mut total_excess_mm = 0.0f64;
        for (&(dt, raw_i), &(e_dt, exc_i)) in raw.iter().zip(excess.iter()) {
            assert!((dt - e_dt).abs() < 1e-12);
            assert!(exc_i >= 0.0);
            assert!(exc_i <= raw_i + 1e-12);
            cum_p += dt * raw_i * 10.0;
            if cum_p <= ia_mm {
                assert_eq!(
                    exc_i, 0.0,
                    "step with cumulative P={cum_p:.4} <= Ia={ia_mm:.4} must emit 0.0"
                );
            }
            total_excess_mm += e_dt * exc_i * 10.0;
        }

        // Exact storm-total SCS-CN runoff Q(P_total) conservation.
        let expected_q = if cum_p <= ia_mm {
            0.0
        } else if s_mm <= 0.0 {
            cum_p
        } else {
            (cum_p - ia_mm).powi(2) / (cum_p - ia_mm + s_mm)
        };
        assert!(
            (total_excess_mm - expected_q).abs() < 1e-11,
            "CN={cn}, ia_ratio={ia_ratio}: sum(excess)={total_excess_mm:.12} != Q(P_total)={expected_q:.12}"
        );
    }

    // Validation checks.
    assert!(scs_cn_incremental_hyetograph(&[], 80.0, 0.2).is_err());
    assert!(scs_cn_incremental_hyetograph(&raw, 0.0, 0.2).is_err());
    assert!(scs_cn_incremental_hyetograph(&raw, 101.0, 0.2).is_err());
    assert!(scs_cn_incremental_hyetograph(&raw, 80.0, -0.1).is_err());
}

#[test]
fn green_ampt_infiltration_physics_hyetograph_and_fsm_mass_conservation() {
    let ga = GreenAmptParams {
        ks_cm_h: 1.09, // sandy loam / loam
        psi_cm: 11.01,
        delta_theta: 0.25,
    };
    ga.validate().unwrap();

    // (a) Monotonic decrease with cumulative F and convergence to Ks as F -> infinity.
    let f_vals = [0.1, 0.5, 1.0, 5.0, 20.0, 100.0, 1e6];
    let rates: Vec<f64> = f_vals
        .iter()
        .map(|&f| green_ampt_infiltration_rate_cm_h(ga, f))
        .collect();
    for pair in rates.windows(2) {
        assert!(
            pair[0] > pair[1],
            "Green-Ampt rate must decrease monotonically with F: {} -> {}",
            pair[0],
            pair[1]
        );
    }
    assert!(
        (rates.last().unwrap() - ga.ks_cm_h).abs() < 1e-4,
        "Green-Ampt rate must converge to Ks as F -> inf"
    );
    assert!(green_ampt_infiltration_rate_cm_h(ga, 0.0).is_infinite());

    // (b) Step-integrated hyetograph losses across all 3 Mein-Larson regimes.
    // Step 1: i = 0.5 <= Ks -> 0 excess
    // Step 2: i = 4.0 > Ks, crosses ponding Fp = 1.09 * (11.01 * 0.25) / (4.0 - 1.09) ~= 1.031 cm
    // Step 3: i = 5.0 > Ks, already ponded -> positive excess
    let raw = [(1.0, 0.5), (1.0, 4.0), (1.0, 5.0)];
    let excess = green_ampt_hyetograph_losses(&raw, ga).unwrap();
    assert_eq!(excess[0].1, 0.0, "i <= Ks must produce zero excess");
    assert!(
        excess[1].1 > 0.0 && excess[1].1 < raw[1].1,
        "mid-step ponding must produce 0 < excess < i, got {}",
        excess[1].1
    );
    assert!(
        excess[2].1 > 0.0 && excess[2].1 < raw[2].1 - ga.ks_cm_h + 1e-9,
        "ponded step excess must be bounded by (i - Ks), got {}",
        excess[2].1
    );

    // (c) Direct integration of GreenAmptParams inside FSM preserves exact mass conservation.
    let (dem_refined, sink, deps) = build_compound_watershed_dem();
    let opts = SimulateInundationOptions {
        min_size: 10,
        min_depth: 0.3,
        interval: 0.3,
        rain_intensity_cm_h: 8.0,
        rain_duration_h: 40.0,
        time_step_h: 1.0,
        compress: false,
    };

    let res_no_ga = simulate_fill_spill_merge_in_memory(
        &dem_refined,
        &sink,
        &deps,
        &opts,
        &InundationForcing::default(),
    )
    .unwrap();
    let res_ga = simulate_fill_spill_merge_with_green_ampt_in_memory(
        &dem_refined,
        &sink,
        &deps,
        &opts,
        &FsmForcing {
            base: InundationForcing::default(),
            green_ampt: Some(ga),
            green_ampt_substep_h: Some(0.5),
        },
    )
    .unwrap();

    assert!(
        res_ga.mass_balance.mass_balance_error_m3.abs() < 1e-6,
        "FSM with Green-Ampt must conserve mass to < 1e-6 m^3, got {:?}",
        res_ga.mass_balance
    );
    assert!(
        res_ga.mass_balance.total_infiltration_m3 > 0.0,
        "Green-Ampt must record positive vertical infiltration volume"
    );
    assert!(
        res_ga.mass_balance.total_boundary_outflow_m3
            < res_no_ga.mass_balance.total_boundary_outflow_m3
            || res_ga.mass_balance.total_stored_m3 < res_no_ga.mass_balance.total_stored_m3,
        "Green-Ampt infiltration must reduce stored/exported surface water"
    );

    let hydro_vol = integrate_fsm_hydrograph_m3(&res_ga.hydrograph);
    assert!(
        (hydro_vol - res_ga.mass_balance.total_boundary_outflow_m3).abs() < 1e-6,
        "FSM+Green-Ampt hydrograph integral must match boundary outflow"
    );
}

#[test]
fn crs_metric_scale_handles_utm_state_plane_feet_and_geographic_at_45n() {
    // 1. Projected UTM (meters): EPSG:26914, 2.5 m x 2.5 m cells.
    let gt_utm = [500_000.0, 2.5, 0.0, 5_000_000.0, 0.0, -2.5];
    let s_utm = crs_metric_scale("EPSG:26914", &gt_utm, 100);
    assert!(!s_utm.is_geographic);
    assert!((s_utm.to_meter - 1.0).abs() < 1e-12);
    assert!((s_utm.cell_width_m - 2.5).abs() < 1e-12);
    assert!((s_utm.cell_height_m - 2.5).abs() < 1e-12);
    assert!((s_utm.cell_area_m2 - 6.25).abs() < 1e-12);

    // 2. Projected State Plane (US survey feet): EPSG:2263 (NAD83 / NY Long Island ftUS), 10 ft cells.
    let gt_ft = [1_000_000.0, 10.0, 0.0, 200_000.0, 0.0, -10.0];
    let s_ft = crs_metric_scale("EPSG:2263", &gt_ft, 100);
    assert!(!s_ft.is_geographic);
    let us_ft = 1200.0 / 3937.0;
    assert!(
        (s_ft.to_meter - us_ft).abs() < 1e-7,
        "expected US survey foot to_meter ~= {us_ft}, got {}",
        s_ft.to_meter
    );
    assert!((s_ft.cell_width_m - 10.0 * us_ft).abs() < 1e-5);
    assert!((s_ft.cell_height_m - 10.0 * us_ft).abs() < 1e-5);
    assert!((s_ft.cell_area_m2 - (10.0 * us_ft).powi(2)).abs() < 1e-4);

    // 3. Geographic WGS 84 (EPSG:4326) and NAD83 (EPSG:4269) centered at 45°N.
    // 100 rows at 0.001 deg/cell with top = 45.05 -> center latitude = 45.00°N.
    let gt_geo = [-95.0, 0.001, 0.0, 45.05, 0.0, -0.001];
    for epsg in ["EPSG:4326", "EPSG:4269"] {
        let s_geo = crs_metric_scale(epsg, &gt_geo, 100);
        assert!(s_geo.is_geographic, "{epsg} must be detected as geographic");
        // At 45°N on WGS 84: ~78.847 m per 0.001° lon, ~111.132 m per 0.001° lat.
        assert!(
            (s_geo.cell_width_m - 78.847).abs() < 0.25,
            "{epsg} cell_width_m at 45N expected ~78.85 m, got {}",
            s_geo.cell_width_m
        );
        assert!(
            (s_geo.cell_height_m - 111.132).abs() < 0.25,
            "{epsg} cell_height_m at 45N expected ~111.13 m, got {}",
            s_geo.cell_height_m
        );
        assert!((s_geo.cell_area_m2 - s_geo.cell_width_m * s_geo.cell_height_m).abs() < 1e-6);
    }
}

#[test]
fn embankment_artifact_classifier_separates_natural_pothole_from_road_ditch() {
    // 60x40 1-m UTM DEM containing TWO distinct depressions:
    //   1. Natural circular pothole centered at (col=14, row=20), radius 5 cells, surrounded
    //      by a wide flat upland plateau at 20.0 m (no nearby drop).
    //   2. Road-embankment-impounded linear ditch at cols 36..40, rows 8..32 (parallel to a
    //      narrow 2-cell-wide raised road embankment at cols 40..42, elev 22.0 m), with a
    //      lower downstream valley floor at cols 42..60 (elev 12.0 m).
    let (w, h) = (60usize, 40usize);
    let mut z = vec![20.0f64; w * h];

    // Natural circular pothole at (14, 20): floor 15.0 m rising smoothly to 20.0 m at r=6.
    for r in 0..h {
        for c in 0..w {
            let d = ((r as f64 - 20.0).powi(2) + (c as f64 - 14.0).powi(2)).sqrt();
            if d < 6.0 {
                z[r * w + c] = 15.0 + 5.0 * (d / 6.0);
            }
        }
    }

    // Road-impounded linear ditch at cols 35..40, rows 6..34 (elongated N-S, floor 14.0 m)
    // directly upstream of a 2-cell raised road berm at cols 40..42 (z = 22.0 m) and
    // lower downstream terrain at cols 42..60 (z = 12.0 m).
    for r in 0..h {
        for c in 42..w {
            z[r * w + c] = 12.0;
        }
        z[r * w + 40] = 22.0;
        z[r * w + 41] = 22.0;
    }
    for r in 6..34 {
        for c in 35..40 {
            let dc = (c as f64 - 37.0).abs();
            z[r * w + c] = 14.0 + 0.8 * dc;
        }
    }

    let mut dem = Array2D::from_vec(w, h, z).unwrap();
    dem.set_no_data(-9999.0);
    dem.geotransform = [500_000.0, 1.0, 0.0, 4_500_040.0, 0.0, -1.0];
    dem.projection = "EPSG:26914".to_string();

    let sinks = extract_sinks_in_memory(&dem, 10, Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 10, 0.3, 0.3).unwrap();

    let candidates =
        classify_embankment_artifacts(&dem, &deps, &ArtifactClassifierOptions::default()).unwrap();
    assert_eq!(candidates.len(), 2, "expected 2 leaf depressions");

    // Identify which depression ID lies at pothole center (14, 20) vs ditch center (37, 20).
    let pothole_id = deps.depression_id.get(14, 20) as u32;
    let ditch_id = deps.depression_id.get(37, 20) as u32;
    assert!(pothole_id > 0 && ditch_id > 0 && pothole_id != ditch_id);

    let pothole = candidates.iter().find(|c| c.dep_id == pothole_id).unwrap();
    let ditch = candidates.iter().find(|c| c.dep_id == ditch_id).unwrap();

    assert!(
        !pothole.is_likely_artifact,
        "natural circular pothole must NOT be flagged as an embankment artifact: {pothole:?}"
    );
    assert!(
        pothole.artifact_score < 0.35,
        "natural pothole should have low artifact score, got {}",
        pothole.artifact_score
    );

    assert!(
        ditch.is_likely_artifact,
        "road-impounded linear ditch MUST be flagged as an embankment artifact: {ditch:?}"
    );
    assert!(
        ditch.artifact_score > 0.65 && ditch.artifact_score > pothole.artifact_score + 0.35,
        "ditch score ({}) should cleanly exceed pothole score ({})",
        ditch.artifact_score,
        pothole.artifact_score
    );
    assert!(
        ditch.rim_linearity > 0.7,
        "road embankment rim should have high linearity, got {}",
        ditch.rim_linearity
    );
    let bw = ditch
        .barrier_width_m
        .expect("ditch must have measured barrier width");
    assert!(
        bw > 1.0 && bw < 10.0,
        "2-m embankment breach should have narrow barrier_width_m, got {bw}"
    );

    // Suggested breach segment must cross the road embankment at x = 500_040..500_042.
    let ((x1, _y1), (x2, _y2)) = ditch
        .suggested_breach
        .expect("ditch must provide suggested culvert breach coordinates");
    let min_x = x1.min(x2);
    let max_x = x1.max(x2);
    assert!(
        min_x < 500_040.0 && max_x > 500_042.0,
        "suggested breach ({x1}, {x2}) must span across the road embankment [500040, 500042]"
    );
}

#[test]
fn breach_depressions_file_with_no_data_honors_override_and_collision_guard() {
    let tmp = tempfile::tempdir().unwrap();
    let in_path = tmp.path().join("in_dem.tif");
    let out_path = tmp.path().join("out_dem.tif");

    // 11x11 surface at 20.0 m with a lower edge at col 0 (19.0 m), a shallow pit (19.5 m)
    // at cols 3..5, rows 4..6, and a custom NoData sentinel cell (-777.0) at (9, 9).
    let (w, h) = (11usize, 11usize);
    let mut z = vec![20.0f64; w * h];
    for r in 0..h {
        z[r * w] = 19.0;
    }
    for r in 4..=6 {
        for c in 3..=5 {
            z[r * w + c] = 19.5;
        }
    }
    z[9 * w + 9] = -777.0;
    let mut dem = Array2D::from_vec(w, h, z).unwrap();
    dem.set_no_data(-9999.0);
    save_raster(&dem, &in_path, None, false).unwrap();

    let opts = BreachOptions {
        mode: BreachMode::Selective,
        max_depth: Some(1.0),
        max_length: Some(5),
        eps_gradients: false,
        fill_remaining: false,
        ..Default::default()
    };

    // Collision guard must reject overwriting input DEM.
    let coll = breach_depressions_file_with_no_data(&in_path, &in_path, &opts, Some(-777.0));
    assert!(coll.is_err());

    // Custom NoData override must keep cell (9, 9) untouched at -777.0 while breaching the pit.
    let out =
        breach_depressions_file_with_no_data(&in_path, &out_path, &opts, Some(-777.0)).unwrap();
    assert!(out_path.exists());
    assert!((out.no_data() - (-777.0)).abs() < 1e-12);
    assert!((out.get(9, 9) - (-777.0)).abs() < 1e-12);
    let carved = (3..=7)
        .flat_map(|r| (1..=2).map(move |c| (c, r)))
        .any(|(c, r)| out.get(c, r) < 20.0);
    assert!(carved, "expected shallow pit to breach toward col 0");
}

#[test]
fn run_pipeline_fsm_exports_hydrograph_csv_and_in_memory_pipeline_populates_hydrograph() {
    let dem_path = test_data_dir().join("dem.tif");
    let dem = load_raster::<f64>(&dem_path, None).unwrap();

    let sim_opts = SimulateInundationOptions {
        min_size: 1000,
        min_depth: 0.3,
        interval: 0.3,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 20.0,
        time_step_h: 2.0,
        compress: false,
    };

    let mem = run_full_pipeline_in_memory(
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
    let hydro = mem
        .fsm_hydrograph
        .as_ref()
        .expect("FSM in-memory pipeline must populate fsm_hydrograph");
    let mb = mem.fsm_mass_balance.as_ref().unwrap();
    assert!(hydro.len() >= 2);
    let rel_err = (integrate_fsm_hydrograph_m3(hydro) - mb.total_boundary_outflow_m3).abs()
        / mb.total_boundary_outflow_m3.max(1.0);
    assert!(rel_err < 1e-10);

    let tmp = tempfile::tempdir().unwrap();
    let res = run_pipeline(
        &dem_path,
        tmp.path(),
        &RunPipelineOptions {
            kernel_size: 3,
            simulation: sim_opts,
            routing: RoutingMode::FillSpillMerge,
            export_vectors: false,
            export_graph: false,
            save_rasters: false,
            ..Default::default()
        },
    )
    .unwrap();
    let hydro_csv = res
        .fsm_hydrograph_csv
        .as_ref()
        .expect("run_pipeline in FSM mode must export fsm_hydrograph.csv");
    assert!(hydro_csv.exists());
    let text = std::fs::read_to_string(hydro_csv).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(
        lines[0],
        "time_h,q_boundary_m3_s,total_stored_m3,connected_area_fraction,spilling_basin_count"
    );
    assert!(lines.len() >= 3);
}
