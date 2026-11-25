use std::collections::HashSet;

use lidar::{
    Array2D, BurnCulvertsOptions, CatchmentHierarchyOptions, ConnectivityGraph, GraphEdge,
    GraphNode, InMemoryDepressions, InundationForcing, NestedDepression, SimulateInundationOptions,
    TiledDelineationOptions, Topology, build_connectivity_graph, burn_culverts,
    compute_rating_curves, connectivity_graph_to_graphml_string, delineate_catchment_hierarchy,
    delineate_catchment_hierarchy_in_memory, delineate_catchments_in_memory,
    delineate_depressions_in_memory, delineate_depressions_tiled,
    delineate_spill_flow_paths_in_memory, extract_sink_labels_from_raster, extract_sinks_in_memory,
    gaussian_filter, region_group, region_props_xy, scs_type2_hyetograph,
    simulate_fill_spill_merge_in_memory, simulate_inundation_in_memory,
    simulate_inundation_with_forcing_in_memory, write_depressions_vector,
};
use richdem::{load_raster, save_raster};
use terrio::vector::Coord;
use terrio::{
    Feature, FieldValue, Geometry, GeometryType, Layer, VectorOpenOptions, VectorWriteOptions,
};

fn make_dem(w: usize, h: usize, data: Vec<f64>, nd: f64, gt: [f64; 6]) -> Array2D<f64> {
    let mut arr = Array2D::from_vec(w, h, data).unwrap();
    arr.set_no_data(nd);
    arr.geotransform = gt;
    arr
}

fn dummy_dep(
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
        max_depth: 2.0,
        min_elev: 10.0,
        max_elev: 12.0,
        children_ids,
        region_id: 1,
        perimeter: 10.0,
        major_axis: 3.0,
        minor_axis: 2.0,
        elongatedness: 1.5,
        eccentricity: 0.7,
        orientation: 0.0,
        area_bbox_ratio: 0.8,
        parent_id,
        circularity: 0.8,
        depth_diam_ratio: 0.5,
        hypsometric_integral: 0.5,
        rim_slope_deg: 20.0,
        artifact_flag: false,
    }
}

/// Item 1: Full basin under infiltration > inflow with a tiny `dt_next` (caused by a second
/// basin filling almost immediately) drains cleanly without snapping back, oscillating, or
/// creating phantom mass.
#[test]
fn fsm_draining_basin_with_tiny_dt_no_snapback_or_phantom_mass() {
    // 13x13 plateau at 20.0 with two separated 3x3 graded bowls (ring=19.0, center=18.0):
    // Each bowl has capacity = 8*(20-19) + 1*(20-18) = 10.0 m^3.
    // Basin A (centered at c=3, r=6): starts full via antecedent depth, high infiltration.
    // Basin B (centered at c=9, r=6): starts nearly full (10.0 - 1e-10 m^3) so it fills
    // after a tiny dt_next during segment 1 while Basin A is draining under infiltration > inflow.
    let w = 13usize;
    let h = 13usize;
    let mut z = vec![20.0f64; w * h];
    for r in 5..=7 {
        for c in 2..=4 {
            z[r * w + c] = 19.0;
        }
        for c in 8..=10 {
            z[r * w + c] = 19.0;
        }
    }
    z[6 * w + 3] = 18.0;
    z[6 * w + 9] = 18.0;
    let dem = make_dem(w, h, z, -9999.0, [0.0, 1.0, 0.0, h as f64, 0.0, -1.0]);

    let sinks = extract_sinks_in_memory(&dem, 1, Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 1, 0.1, 0.5).unwrap();
    assert_eq!(deps.depressions.len(), 2);

    // High infiltration on Basin A (500 cm/h = 5.0 m/h > rain inflow); zero on Basin B.
    let mut infil = vec![0.0f64; w * h];
    for r in 5..=7 {
        for c in 2..=4 {
            infil[r * w + c] = 500.0;
        }
    }
    let mut infil_arr = Array2D::from_vec(w, h, infil).unwrap();
    infil_arr.copy_metadata_from(&dem);

    // Pre-fill Basin A completely (2.0 m everywhere, clamped to cell depth => 10.0 m^3)
    // and Basin B to 10.0 - 1e-10 m^3 so Basin B hits capacity after a tiny `dt_next`.
    let mut ant = vec![0.0f64; w * h];
    for r in 5..=7 {
        for c in 2..=4 {
            ant[r * w + c] = 2.0;
        }
        for c in 8..=10 {
            ant[r * w + c] = 1.0;
        }
    }
    ant[6 * w + 9] = 2.0 - 1e-10;
    let mut ant_arr = Array2D::from_vec(w, h, ant).unwrap();
    ant_arr.copy_metadata_from(&dem);

    let hyeto = [(1.0, 1.0), (1.0, 0.0)]; // 1 cm/h for 1 h, then dry for 1 h
    let opts = SimulateInundationOptions {
        min_size: 1,
        min_depth: 0.1,
        interval: 0.5,
        rain_intensity_cm_h: 1.0,
        rain_duration_h: 2.0,
        time_step_h: 0.5,
        compress: false,
    };
    let forcing = InundationForcing {
        runoff_coeff: None,
        infiltration_cm_h: Some(&infil_arr),
        antecedent_depth: Some(&ant_arr),
        hyetograph: Some(&hyeto),
    };

    let res = simulate_fill_spill_merge_in_memory(
        &sinks.dem_refined,
        &sinks.sink,
        &deps,
        &opts,
        &forcing,
    )
    .expect("FSM must not hit iteration guard or oscillate");

    assert!(
        res.mass_balance.mass_balance_error_m3.abs() < 1e-6,
        "phantom mass detected: error = {} m^3",
        res.mass_balance.mass_balance_error_m3
    );

    // Identify Basin A (the one with non-zero infiltration) and confirm it drained cleanly.
    let basin_a = res
        .basins
        .iter()
        .find(|b| b.infiltrated_volume_m3 > 0.1)
        .expect("expected infiltrating basin A");
    assert!(
        basin_a.final_volume_m3 < basin_a.capacity_m3 - 0.5,
        "basin A should drain below capacity, got final={} capacity={}",
        basin_a.final_volume_m3,
        basin_a.capacity_m3
    );
}

/// Item 2: Clipped irregular watershed surrounded by NoData padding inside the raster bbox
/// discovers its boundary spill exit cleanly in both static spill delineation and FSM.
#[test]
fn clipped_watershed_nodata_border_discovers_spill_exit() {
    // 17x17 raster with a 3-cell NoData halo around an interior 11x11 valid watershed (rows 3..14, cols 3..14).
    // Two separate graded depressions inside the clipped watershed so static spill paths are emitted
    // (num_regions == 2), with the eastern depression spilling downhill across valid upland to the
    // NoData border at col 13.
    let w = 17usize;
    let h = 17usize;
    let nd = -9999.0f64;
    let mut z = vec![nd; w * h];
    for r in 3..14 {
        for c in 3..14 {
            // Slopes gently east toward the clipped NoData boundary at col 13.
            z[r * w + c] = 20.0 - (c - 3) as f64 * 0.4;
        }
    }
    // Graded Depression 1 on the west side: 3x3 ring=15.0, center=13.0 at (r=8, c=5)
    for r in 7..=9 {
        for c in 4..=6 {
            z[r * w + c] = 15.0;
        }
    }
    z[8 * w + 5] = 13.0;
    // Graded Depression 2 on the east side: 3x3 ring=13.0, center=11.0 at (r=8, c=10),
    // spilling east across cols 12..13 to the NoData border at col 14.
    for r in 7..=9 {
        for c in 9..=11 {
            z[r * w + c] = 13.0;
        }
    }
    z[8 * w + 10] = 11.0;

    let dem = make_dem(w, h, z, nd, [0.0, 1.0, 0.0, h as f64, 0.0, -1.0]);
    let sinks = extract_sinks_in_memory(&dem, 1, Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 1, 0.2, 0.5).unwrap();
    assert!(
        deps.num_regions >= 2 && deps.depressions.len() >= 2,
        "expected >= 2 depressions inside clipped watershed, got regions={} deps={}",
        deps.num_regions,
        deps.depressions.len()
    );

    let (hir, _) =
        delineate_catchment_hierarchy_in_memory(&sinks.dem_refined, &deps.depression_level, 1)
            .unwrap();
    let opts = SimulateInundationOptions {
        min_size: 1,
        min_depth: 0.2,
        interval: 0.5,
        rain_intensity_cm_h: 20.0,
        rain_duration_h: 50.0,
        time_step_h: 5.0,
        compress: false,
    };
    let (ft, _) = simulate_inundation_in_memory(&sinks.sink, &hir, &opts).unwrap();

    // Static spill path discovery must find a boundary exit (`to_dep_id == 0`) terminating on a
    // valid DEM cell (never a NoData cell).
    let static_paths =
        delineate_spill_flow_paths_in_memory(&sinks.dem_refined, &deps, &ft).unwrap();
    let boundary_path = static_paths
        .iter()
        .find(|p| p.to_dep_id == 0)
        .expect("clipped watershed must discover a boundary spill exit (to_dep_id == 0)");
    for &(c, r) in &boundary_path.cells {
        let zv = sinks.dem_refined.get(c, r);
        assert!(
            zv.is_finite() && zv != nd,
            "spill path cell ({c}, {r}) must be a valid DEM cell, got {zv}"
        );
    }

    // Dynamic FSM must also spill to boundary (to_dep_id == 0) with exact mass conservation.
    let fsm = simulate_fill_spill_merge_in_memory(
        &sinks.dem_refined,
        &sinks.sink,
        &deps,
        &opts,
        &InundationForcing::default(),
    )
    .unwrap();
    assert!(
        fsm.basins.iter().any(|b| b.spill_to_dep_id == Some(0)),
        "FSM must discover boundary exit on clipped watershed, basins={:?}",
        fsm.basins
    );
    assert!(
        fsm.mass_balance.total_boundary_outflow_m3 > 0.0,
        "FSM must discharge water out of clipped watershed boundary"
    );
    assert!(fsm.mass_balance.mass_balance_error_m3.abs() < 1e-6);
}

/// Item 2b: An interior NoData void (such as a lake dropout surrounded by valid DEM cells and
/// disconnected from the raster border) must NOT create a fake domain boundary exit (`to_dep_id == 0`);
/// instead, the upstream basin must spill over its valid saddle into the downstream basin.
#[test]
fn interior_nodata_void_does_not_create_fake_boundary_exit() {
    let w = 13usize;
    let h = 13usize;
    let nd = -9999.0f64;
    // Gently east-sloping valid terrain (20.0 -> 19.4) with zero exterior NoData.
    let mut z = vec![0.0f64; w * h];
    for r in 0..h {
        for c in 0..w {
            z[r * w + c] = 20.0 - c as f64 * 0.05;
        }
    }
    // Upstream Basin A (west, cols 2..=4, rows 5..=7): graded 16.0 -> 14.0.
    for r in 5..=7 {
        for c in 2..=4 {
            z[r * w + c] = 16.0;
        }
    }
    z[6 * w + 3] = 14.0;
    // Downstream Basin B (east, cols 8..=10, rows 4..=8): wide enough to catch all eastbound
    // flow around the interior NoData void at (5..=6, 6).
    for r in 4..=8 {
        for c in 8..=10 {
            z[r * w + c] = 14.0;
        }
    }
    z[6 * w + 9] = 12.0;
    // Place an interior NoData void at (c=6, r=6) with a lower valid pit cell (c=5, r=6) = 19.0
    // directly adjacent to both Basin A and the interior NoData void. Because the void is
    // disconnected from the raster border, neither static spill discovery nor FSM may treat
    // (c=5, r=6) -> (c=6, r=6) as a domain exit (`to_dep_id == 0`).
    z[6 * w + 5] = 19.0;
    z[6 * w + 6] = nd;

    let dem = make_dem(w, h, z, nd, [0.0, 1.0, 0.0, h as f64, 0.0, -1.0]);
    let sinks = extract_sinks_in_memory(&dem, 2, Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 2, 0.2, 0.5).unwrap();
    assert_eq!(deps.depressions.len(), 2, "expected Basin A and Basin B");

    let dep_a = deps.depression_id.get(3, 6) as u32;
    let dep_b = deps.depression_id.get(9, 6) as u32;
    assert!(dep_a > 0 && dep_b > 0 && dep_a != dep_b);

    let (hir, _) =
        delineate_catchment_hierarchy_in_memory(&sinks.dem_refined, &deps.depression_level, 2)
            .unwrap();
    let opts = SimulateInundationOptions {
        min_size: 2,
        min_depth: 0.2,
        interval: 0.5,
        rain_intensity_cm_h: 50.0,
        rain_duration_h: 5.0,
        time_step_h: 1.0,
        compress: false,
    };
    let (ft, _) = simulate_inundation_in_memory(&sinks.sink, &hir, &opts).unwrap();
    let static_paths =
        delineate_spill_flow_paths_in_memory(&sinks.dem_refined, &deps, &ft).unwrap();
    let path_a = static_paths
        .iter()
        .find(|p| p.from_dep_id == dep_a)
        .expect("expected static spill path from Basin A");
    assert_eq!(
        path_a.to_dep_id, dep_b,
        "interior NoData void must NOT become a fake boundary exit; Basin A must spill to Basin B"
    );

    let fsm = simulate_fill_spill_merge_in_memory(
        &sinks.dem_refined,
        &sinks.sink,
        &deps,
        &opts,
        &InundationForcing::default(),
    )
    .unwrap();
    let basin_a = fsm.basins.iter().find(|b| b.dep_id == dep_a).unwrap();
    assert_eq!(
        basin_a.spill_to_dep_id,
        Some(dep_b),
        "FSM Basin A must spill into Basin B, not treat interior NoData void as boundary exit"
    );
    assert!(fsm.mass_balance.mass_balance_error_m3.abs() < 1e-6);

    // Also verify the `in_source` direct-touch branch in `find_topographic_outlet`:
    // when Basin A's own cell directly touches an interior NoData void (at (c=5, r=6)) while the
    // raster border is completely valid, FSM must still route Basin A into Basin B, not to 0.
    let mut dem_direct = sinks.dem_refined.clone();
    dem_direct.set(5, 6, nd);
    let fsm_direct = simulate_fill_spill_merge_in_memory(
        &dem_direct,
        &sinks.sink,
        &deps,
        &opts,
        &InundationForcing::default(),
    )
    .unwrap();
    let basin_a_direct = fsm_direct
        .basins
        .iter()
        .find(|b| b.dep_id == dep_a)
        .unwrap();
    assert_eq!(
        basin_a_direct.spill_to_dep_id,
        Some(dep_b),
        "direct touch of an interior NoData void must not trigger a fake boundary exit"
    );
}

/// Item 3: Low-gradient polyline culvert (with NoData tails and `half_width_cells = 1`) and point
/// culvert carve monotonically without creating an interior depression below the outlet endpoint
/// or overwriting endpoints.
#[test]
fn culvert_carving_monotonic_no_interior_pit_polyline_and_point() {
    let w = 15usize;
    let h = 11usize;
    let nd = -9999.0f64;
    let mut z = vec![10.02f64; w * h];
    for r in 0..h {
        // Left side at 10.02, road embankment at cols 6..=8 (15.0), right side at 10.00.
        for c in 6..=8 {
            z[r * w + c] = 15.0;
        }
        for c in 9..w {
            z[r * w + c] = 10.00;
        }
        // Leading & trailing NoData columns to verify interpolation uses `i_start..=i_end`.
        z[r * w] = nd;
        z[r * w + 1] = nd;
        z[r * w + 13] = nd;
        z[r * w + 14] = nd;
    }
    let dem = make_dem(w, h, z, nd, [0.0, 1.0, 0.0, h as f64, 0.0, -1.0]);

    let tmp = tempfile::tempdir().unwrap();
    let line_path = tmp.path().join("culvert_line.geojson");
    let mut layer = Layer::new("culverts".to_string()).with_geom_type(GeometryType::LineString);
    layer.features.push(Feature {
        fid: 1,
        geometry: Some(Geometry::LineString(vec![
            Coord::xy(0.5, 5.5),
            Coord::xy(14.5, 5.5),
        ])),
        attributes: vec![],
    });
    terrio::write_vector(
        &[layer],
        line_path.to_str().unwrap(),
        &VectorWriteOptions::default(),
    )
    .unwrap();

    let opts = BurnCulvertsOptions {
        depth_offset: 0.5, // Larger than the 0.02 m drop across the embankment
        half_width_cells: 1,
        point_radius_cells: 3,
        compress: false,
    };
    let (burned, carved) = burn_culverts(&dem, &line_path, &opts).unwrap();
    assert!(carved >= 3);

    // Valid endpoints at col 2 (10.02) and col 12 (10.00) must not be overwritten.
    assert!((burned.get(2, 5) - 10.02).abs() < 1e-12);
    assert!((burned.get(12, 5) - 10.00).abs() < 1e-12);

    // Every cell along the carved channel (cols 2..=12, rows 4..=6) must be >= 10.00 (no pit)
    // and centerline cols 2..=12 must be monotonically non-increasing.
    let mut prev = f64::INFINITY;
    for c in 2..=12 {
        let v = burned.get(c, 5);
        assert!(
            v >= 10.00 - 1e-12,
            "interior culvert cell ({c}, 5) undercut outlet 10.00: got {v}"
        );
        assert!(
            v <= prev + 1e-12,
            "culvert centerline must be monotonic: col {c} has {v} > prev {prev}"
        );
        prev = v;
        for r in 4..=6 {
            assert!(
                burned.get(c, r) >= 10.00 - 1e-12,
                "half-width cell ({c}, {r}) undercut outlet 10.00: got {}",
                burned.get(c, r)
            );
        }
    }
    // Embankment at col 7 must be breached down to valley bed.
    assert!(burned.get(7, 5) <= 10.02);

    // Point culvert test on the same embankment.
    let pt_path = tmp.path().join("culvert_point.geojson");
    let mut pt_layer = Layer::new("pts".to_string()).with_geom_type(GeometryType::Point);
    pt_layer.features.push(Feature {
        fid: 1,
        geometry: Some(Geometry::Point(Coord::xy(7.5, 5.5))),
        attributes: vec![],
    });
    terrio::write_vector(
        &[pt_layer],
        pt_path.to_str().unwrap(),
        &VectorWriteOptions::default(),
    )
    .unwrap();
    let (pt_burned, pt_carved) = burn_culverts(&dem, &pt_path, &opts).unwrap();
    assert!(pt_carved >= 1);
    assert!(pt_burned.get(7, 5) < 15.0);
    for r in 2..=8 {
        for c in 4..=10 {
            let v = pt_burned.get(c, r);
            assert!(
                v >= 10.00 - 1e-12,
                "point culvert cell ({c}, {r}) undercut outlet 10.00: got {v}"
            );
        }
    }
}

/// Item 4: Coastal DEM with valid sink cells at `z == 0.0` retains those cells across
/// `extract_sinks_in_memory`, `delineate_depressions_in_memory`, and `delineate_catchments_in_memory`,
/// while positive-elevation DEMs preserve `0.0` as `sink` NoData.
#[test]
fn coastal_zero_elevation_sink_preserved_end_to_end() {
    let w = 9usize;
    let h = 9usize;
    let mut z = vec![5.0f64; w * h];
    for r in 2..=6 {
        for c in 2..=6 {
            z[r * w + c] = 2.0;
        }
    }
    // Inner 3x3 depression floor right at sea level (z == 0.0).
    for r in 3..=5 {
        for c in 3..=5 {
            z[r * w + c] = 0.0;
        }
    }
    let dem = make_dem(w, h, z, -9999.0, [0.0, 1.0, 0.0, 9.0, 0.0, -1.0]);

    let sinks = extract_sinks_in_memory(&dem, 1, Topology::D8).unwrap();
    assert_ne!(
        sinks.sink.no_data(),
        0.0,
        "sink NoData must switch to a collision-free sentinel when z == 0.0 is a valid sink cell"
    );
    assert_eq!(sinks.sink.get(4, 4), 0.0);
    assert_eq!(sinks.sink.get(0, 0), sinks.sink.no_data());

    let deps = delineate_depressions_in_memory(&sinks.sink, 1, 0.1, 0.5).unwrap();
    assert!(!deps.depressions.is_empty());
    let min_z = deps
        .depressions
        .iter()
        .map(|d| d.min_elev)
        .fold(f64::INFINITY, f64::min);
    assert_eq!(
        min_z, 0.0,
        "coastal depression must preserve z == 0.0 floor"
    );
    assert!(deps.depression_id.get(4, 4) > 0);

    let labels = extract_sink_labels_from_raster(&sinks.dem_refined, &sinks.sink).unwrap();
    assert!(
        labels[4 * w + 4] > 0,
        "z == 0.0 sink cell must remain foreground in extract_sink_labels_from_raster"
    );
    let cats = delineate_catchments_in_memory(&sinks.dem_refined, &labels).unwrap();
    assert_eq!(cats.records.len(), 1);
    assert_eq!(cats.records[0].sink_cells, 25);

    // Backward compatibility: positive-only DEM keeps 0.0 as sink NoData.
    let mut pos_z = vec![15.0f64; w * h];
    for r in 3..=5 {
        for c in 3..=5 {
            pos_z[r * w + c] = 10.0;
        }
    }
    let pos_dem = make_dem(w, h, pos_z, -9999.0, [0.0, 1.0, 0.0, 9.0, 0.0, -1.0]);
    let pos_sinks = extract_sinks_in_memory(&pos_dem, 1, Topology::D8).unwrap();
    assert_eq!(pos_sinks.sink.no_data(), 0.0);
}

/// Item 5: Out-of-range ID in `depression_id` raster does not panic in FSM, and a 3-level
/// hierarchy where the intermediate node has both `parent_id > 0` and `children_ids` (while
/// the leaf only relies on parent's `children_ids`) populates `parent_map` and merges cleanly.
#[test]
fn fsm_out_of_range_dep_id_no_panic_and_3level_hierarchy() {
    let w = 9usize;
    let h = 9usize;
    let mut z = vec![20.0f64; w * h];
    let mut dep_id = vec![0i32; w * h];
    let mut dep_lvl = vec![0i32; w * h];
    // Level 3 grandparent (id=1): rows 1..8, cols 1..8, z=16.0
    for r in 1..8 {
        for c in 1..8 {
            z[r * w + c] = 16.0;
            dep_id[r * w + c] = 1;
            dep_lvl[r * w + c] = 3;
        }
    }
    // Level 2 intermediate parent (id=2): rows 2..7, cols 2..7, z=14.0
    for r in 2..7 {
        for c in 2..7 {
            z[r * w + c] = 14.0;
            dep_id[r * w + c] = 2;
            dep_lvl[r * w + c] = 2;
        }
    }
    // Level 1 leaf (id=3): rows 3..6, cols 3..6, z=10.0
    for r in 3..6 {
        for c in 3..6 {
            z[r * w + c] = 10.0;
            dep_id[r * w + c] = 3;
            dep_lvl[r * w + c] = 1;
        }
    }
    // Inject an out-of-range bogus ID on one border pixel to verify bounds safety.
    dep_id[0] = 9999;

    let gt = [0.0, 1.0, 0.0, 9.0, 0.0, -1.0];
    let dem = make_dem(w, h, z.clone(), -9999.0, gt);
    let mut sink = make_dem(w, h, z, 0.0, gt);
    sink.set(0, 0, 0.0);
    let mut id_arr = Array2D::from_vec(w, h, dep_id).unwrap();
    id_arr.copy_metadata_from(&dem);
    id_arr.set_no_data(0);
    let mut lvl_arr = Array2D::from_vec(w, h, dep_lvl).unwrap();
    lvl_arr.copy_metadata_from(&dem);
    lvl_arr.set_no_data(0);

    // Note: intermediate node 2 has BOTH `parent_id = 1` AND `children_ids = [3]`, while leaf 3
    // intentionally has `parent_id = 0` so it tests unconditional `children_ids` indexing.
    let d1 = dummy_dep(1, 3, 0, vec![2], 49);
    let d2 = dummy_dep(2, 2, 1, vec![3], 25);
    let mut d3 = dummy_dep(3, 1, 0, vec![], 9);
    d3.min_elev = 10.0;
    d3.max_elev = 14.0;
    let deps = InMemoryDepressions {
        depression_id: id_arr,
        depression_level: lvl_arr,
        depressions: vec![d1, d2, d3],
        num_regions: 1,
        max_level: 3,
    };

    // Multi-step fractional hyetograph (10 steps of 0.1 h) with high intensity so leaf 3 fills
    // and merges into parent 2 -> grandparent 1, exercising both `parent_map` and explicit
    // `snapshot_after_segment` tagging.
    let hyeto = vec![(0.1, 1000.0); 10];
    let opts = SimulateInundationOptions {
        min_size: 1,
        min_depth: 0.1,
        interval: 0.5,
        rain_intensity_cm_h: 1000.0,
        rain_duration_h: 1.0,
        time_step_h: 0.1,
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
    .expect("out-of-range dep_id must not panic");

    let b3 = res.basins.iter().find(|b| b.dep_id == 3).unwrap();
    assert_eq!(
        b3.parent_id, 2,
        "leaf 3 must discover parent_id=2 via intermediate node 2's children_ids"
    );
    assert!(res.num_merge_events >= 1);
    assert!(res.mass_balance.mass_balance_error_m3.abs() < 1e-6);
}

/// Item 6: Two same-level depressions with identical pixel counts both receive their exact
/// distinct catchment areas in `build_connectivity_graph` instead of falling back to `d.area`.
#[test]
fn connectivity_graph_matches_equal_count_same_level_catchments() {
    // 11x21 grid with two identical 3x3 graded sinks (count=9 each, ring=12.0, center=10.0),
    // both spilling over a 15.0 m lip on row 0:
    // - Left sink (r=5, c=5) has a large upland hillslope (z=16..22) draining into it (cols 1..=9).
    // - Right sink (r=5, c=15) sits on a 15.0 m shelf surrounded by outward-sloping terrain so
    //   almost no extra upland drains into it.
    let w = 21usize;
    let h = 11usize;
    let mut z = vec![15.0f64; w * h];
    for r in 0..h {
        for c in 0..w {
            if c <= 10 {
                // Upland above 15.0 sloping inward toward left sink (r=5, c=5), with a 15.0 outlet
                // channel at col 5, rows 0..=3 so fill elevation is capped at 15.0!
                let dist = ((r as i32 - 5).abs().max((c as i32 - 5).abs())) as f64;
                if dist > 1.0 {
                    z[r * w + c] = 15.0 + dist * 0.5;
                }
            } else {
                // Outward-sloping dome around right sink (r=5, c=15) so cells beyond dist 1 drain off-grid.
                let dist = ((r as i32 - 5).abs().max((c as i32 - 15).abs())) as f64;
                if dist > 1.0 {
                    z[r * w + c] = (15.0 - (dist - 2.0) * 0.3).min(15.0);
                }
            }
        }
    }
    // Carve a 15.0 spill lip from left sink (r=3, c=5) north to border (r=0, c=5) so left sink
    // fills only to 15.0 (keeping its sink footprint strictly 3x3 = 9 cells!).
    for r in 0..=3 {
        z[r * w + 5] = 15.0;
    }
    // Identical 3x3 graded bowls (ring=12.0, center=10.0) inside both sinks.
    for r in 4..=6 {
        for c in 4..=6 {
            z[r * w + c] = 12.0;
        }
        for c in 14..=16 {
            z[r * w + c] = 12.0;
        }
    }
    z[5 * w + 5] = 10.0;
    z[5 * w + 15] = 10.0;

    let dem = make_dem(w, h, z, -9999.0, [0.0, 1.0, 0.0, h as f64, 0.0, -1.0]);
    let sinks = extract_sinks_in_memory(&dem, 1, Topology::D8).unwrap();
    let deps = delineate_depressions_in_memory(&sinks.sink, 1, 0.2, 1.0).unwrap();
    assert_eq!(deps.depressions.len(), 2);
    assert_eq!(
        deps.depressions[0].count, deps.depressions[1].count,
        "test setup requires equal pixel counts"
    );

    let (hir, cat_recs) =
        delineate_catchment_hierarchy_in_memory(&sinks.dem_refined, &deps.depression_level, 1)
            .unwrap();
    assert_eq!(cat_recs.len(), 2);
    assert!(
        (cat_recs[0].cat_area - cat_recs[1].cat_area).abs() > 1.0,
        "test setup requires distinct catchment areas: {:?}",
        cat_recs
    );

    let opts = SimulateInundationOptions {
        min_size: 1,
        min_depth: 0.2,
        interval: 1.0,
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 10.0,
        time_step_h: 1.0,
        compress: false,
    };
    let (ft, _) = simulate_inundation_in_memory(&sinks.sink, &hir, &opts).unwrap();
    let graph = build_connectivity_graph(&deps, &cat_recs, &ft, &[], 5.0, None, None).unwrap();

    assert_eq!(graph.nodes.len(), 2);
    let mut node_cats: Vec<f64> = graph.nodes.iter().map(|n| n.cat_area).collect();
    let mut rec_cats: Vec<f64> = cat_recs.iter().map(|r| r.cat_area).collect();
    node_cats.sort_by(f64::total_cmp);
    rec_cats.sort_by(f64::total_cmp);
    for (nc, rc) in node_cats.iter().zip(rec_cats.iter()) {
        assert!(
            (nc - rc).abs() < 1e-9,
            "expected exact catchment match {rc}, got {nc} (fallback would be area={})",
            deps.depressions[0].area
        );
    }
}

/// Item 7: Compound slice whose `first_pixel` sits inside a child depression gets the enclosing
/// catchment area in both `simulate_inundation_in_memory` and
/// `simulate_inundation_with_forcing_in_memory`, and a 5 m UTM origin shift + CRS mismatch are
/// consistently rejected across all grid-pair functions.
#[test]
fn static_inundation_compound_slice_sampling_and_grid_alignment() {
    // 5x5 sink where the top-left pixel (0, 0) has child catchment area = 20.0 while the outer
    // rim pixels have parent catchment area = 200.0.
    let w = 5usize;
    let h = 5usize;
    let mut sink_data = vec![12.0f64; w * h];
    sink_data[0] = 10.0; // top-left pixel is the child floor -> `first_pixel` of top slice!
    sink_data[1] = 10.5;
    let gt = [500_000.0, 1.0, 0.0, 4_500_005.0, 0.0, -1.0];
    let mut sink = make_dem(w, h, sink_data, 0.0, gt);
    sink.projection = "EPSG:26916".to_string();

    let mut cat_data = vec![200.0f64; w * h];
    cat_data[0] = 20.0; // child catchment at `first_pixel`
    cat_data[1] = 20.0;
    let mut cat = make_dem(w, h, cat_data, 0.0, gt);
    cat.projection = "EPSG:26916".to_string();

    let opts = SimulateInundationOptions {
        min_size: 1,
        min_depth: 0.1,
        interval: 0.5,
        rain_intensity_cm_h: 10.0, // 0.1 m/h
        rain_duration_h: 5.0,
        time_step_h: 1.0,
        compress: false,
    };

    // Total volume of top slice (elev=12.0): (12-10) + (12-10.5) = 3.5 m^3.
    // With max catchment 200.0 m^2: t_fill = 3.5 / (200 * 0.1) = 0.175 h.
    // Buggy `first_pixel` sampling (20.0 m^2) gave t_fill = 1.75 h!
    let (ft_static, _) = simulate_inundation_in_memory(&sink, &cat, &opts).unwrap();
    assert!(
        (ft_static.get(4, 4) - 0.175).abs() < 1e-6,
        "static inundation must sample max catchment (200) across slice, got t_fill={}",
        ft_static.get(4, 4)
    );

    let mut rc = make_dem(w, h, vec![1.0f64; w * h], -9999.0, gt);
    rc.projection = "EPSG:26916".to_string();
    let (ft_forced, _) = simulate_inundation_with_forcing_in_memory(
        &sink,
        &cat,
        &opts,
        &InundationForcing {
            runoff_coeff: Some(&rc),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        (ft_forced.get(4, 4) - 0.175).abs() < 1e-6,
        "forced inundation must sample max catchment (200) across slice, got t_fill={}",
        ft_forced.get(4, 4)
    );

    // 5 m UTM origin shift (would have passed the old 1e-6 * 5_000_000 = 5 m tolerance!).
    let mut shifted_cat = cat.clone();
    shifted_cat.geotransform[3] += 5.0;
    assert!(simulate_inundation_in_memory(&sink, &shifted_cat, &opts).is_err());
    assert!(
        simulate_inundation_with_forcing_in_memory(
            &sink,
            &shifted_cat,
            &opts,
            &InundationForcing::default(),
        )
        .is_err(),
        "empty InundationForcing must still reject 5 m UTM origin shift"
    );
    assert!(extract_sink_labels_from_raster(&sink, &shifted_cat).is_err());

    // CRS mismatch must also be rejected across all standardized validators.
    let mut wrong_crs_cat = cat.clone();
    wrong_crs_cat.projection = "EPSG:4326".to_string();
    assert!(simulate_inundation_in_memory(&sink, &wrong_crs_cat, &opts).is_err());
    assert!(
        simulate_inundation_with_forcing_in_memory(
            &sink,
            &wrong_crs_cat,
            &opts,
            &InundationForcing::default(),
        )
        .is_err()
    );
    assert!(extract_sink_labels_from_raster(&sink, &wrong_crs_cat).is_err());

    let mut dep_lvl = Array2D::from_vec(w, h, vec![1i32; w * h]).unwrap();
    dep_lvl.copy_metadata_from(&sink);
    dep_lvl.geotransform[0] += 5.0;
    assert!(delineate_catchment_hierarchy_in_memory(&sink, &dep_lvl, 1).is_err());

    let mut dep_id = Array2D::from_vec(w, h, vec![1i32; w * h]).unwrap();
    dep_id.copy_metadata_from(&sink);
    dep_id.geotransform[0] += 5.0;
    let deps_shifted = InMemoryDepressions {
        depression_id: dep_id,
        depression_level: dep_lvl,
        depressions: vec![dummy_dep(1, 1, 0, vec![], 25)],
        num_regions: 1,
        max_level: 1,
    };
    assert!(compute_rating_curves(&sink, &deps_shifted, 0.5).is_err());
    assert!(
        build_connectivity_graph(&deps_shifted, &[], &ft_static, &[], 5.0, None, None).is_err()
    );
}

/// Item 8: Core numerical stability and edge-case regressions across filtering, hyetograph,
/// catchment, tiling, slicing, region, and graph modules.
#[test]
fn core_numerical_and_edge_case_regressions() {
    // 8a. Tiny Gaussian sigma (radius == 0, 2*sigma^2 underflows to 0) must not produce NaN.
    let dem_tiny = make_dem(
        4,
        4,
        vec![42.0; 16],
        -9999.0,
        [0.0, 1.0, 0.0, 4.0, 0.0, -1.0],
    );
    let g_out = gaussian_filter(&dem_tiny, 1e-200).unwrap();
    for &v in g_out.data() {
        assert!(v.is_finite() && (v - 42.0).abs() < 1e-12);
    }

    // 8b. SCS Type II hyetograph must not emit an extra 1e-9 overshoot step when duration/dt
    // slightly exceeds an integer in IEEE 754 (0.7 / 0.1 = 7.000000000000001), and must reject
    // sub-epsilon durations rather than returning an empty vector.
    let hyeto = scs_type2_hyetograph(50.0, 0.7, 0.1).unwrap();
    assert_eq!(hyeto.len(), 7, "got hyeto={hyeto:?}");
    let total_t: f64 = hyeto.iter().map(|&(dt, _)| dt).sum();
    let total_mm: f64 = hyeto.iter().map(|&(dt, cm_h)| dt * cm_h * 10.0).sum();
    assert!((total_t - 0.7).abs() < 1e-9);
    assert!((total_mm - 50.0).abs() < 1e-6);
    assert!(
        scs_type2_hyetograph(50.0, 1e-13, 1e-13).is_err(),
        "sub-epsilon duration_h must return InvalidArgument instead of empty vec"
    );

    // 8c. High-elevation numerical stability (Chan variance & z_min-shifted volumes).
    let base_z = 1e8f64;
    let w = 5usize;
    let h = 5usize;
    let mut hi_z = vec![base_z + 2.0; w * h];
    for r in 1..4 {
        for c in 1..4 {
            hi_z[r * w + c] = base_z + ((r + c) % 2) as f64; // 5 cells at +0, 4 cells at +1 (or vice versa)
        }
    }
    let hi_dem = make_dem(w, h, hi_z.clone(), -9999.0, [0.0, 1.0, 0.0, 5.0, 0.0, -1.0]);
    let mut hi_lvl = Array2D::from_vec(w, h, vec![0i32; w * h]).unwrap();
    hi_lvl.copy_metadata_from(&hi_dem);
    for r in 1..4 {
        for c in 1..4 {
            hi_lvl.set(c, r, 1);
        }
    }
    let (_, hi_recs) = delineate_catchment_hierarchy_in_memory(&hi_dem, &hi_lvl, 0).unwrap();
    assert_eq!(hi_recs.len(), 1);
    // 5 values of 0.0 and 4 values of 1.0 -> mean=4/9, pop variance = (5*(4/9)^2 + 4*(5/9)^2)/9 = 20/81
    let expected_std = (20.0f64 / 81.0).sqrt();
    assert!(
        (hi_recs[0].std_elev - expected_std).abs() < 1e-6,
        "Chan variance failed at base 1e8: got {} expected {}",
        hi_recs[0].std_elev,
        expected_std
    );
    assert!(
        (hi_recs[0].volume - 5.0).abs() < 1e-6,
        "z_min-shifted volume failed at base 1e8: got {}",
        hi_recs[0].volume
    );

    // 8d. 1-pixel sink preservation, coastal integer elevation bands {1.0, 2.0, 3.0}, and
    // 0-background dense label rasters whose declared NoData is not 0.0 (e.g. -9999.0).
    let mut coastal_dem_data = vec![5.0f64; 25];
    coastal_dem_data[6] = 1.0;
    coastal_dem_data[7] = 2.0;
    coastal_dem_data[8] = 3.0;
    coastal_dem_data[18] = 1.0; // isolated 1-pixel sink
    let coastal_dem = make_dem(
        5,
        5,
        coastal_dem_data,
        -9999.0,
        [0.0, 1.0, 0.0, 5.0, 0.0, -1.0],
    );
    let mut coastal_sink_data = vec![0.0f64; 25];
    coastal_sink_data[6] = 1.0;
    coastal_sink_data[7] = 2.0;
    coastal_sink_data[8] = 3.0;
    coastal_sink_data[18] = 1.0;
    let coastal_sink = make_dem(
        5,
        5,
        coastal_sink_data,
        0.0,
        [0.0, 1.0, 0.0, 5.0, 0.0, -1.0],
    );
    let clabels = extract_sink_labels_from_raster(&coastal_dem, &coastal_sink).unwrap();
    // Connected {1.0, 2.0, 3.0} must be ONE region (not split into 3 label IDs), and the isolated
    // 1-pixel sink at index 18 must be preserved (label 2).
    assert_eq!(clabels[6], clabels[7]);
    assert_eq!(clabels[7], clabels[8]);
    assert!(clabels[6] > 0);
    assert!(clabels[18] > 0 && clabels[18] != clabels[6]);

    // 0-background dense label raster (0/1/2) with declared NoData = -9999.0 must keep 0 as
    // background and preserve dense labels 1 and 2 instead of merging 0 cells into foreground.
    let label_dem = make_dem(
        5,
        5,
        vec![20.0; 25],
        -9999.0,
        [0.0, 1.0, 0.0, 5.0, 0.0, -1.0],
    );
    let mut label_mask_data = vec![0.0f64; 25];
    label_mask_data[6] = 1.0;
    label_mask_data[7] = 1.0;
    label_mask_data[18] = 2.0;
    let label_mask = make_dem(
        5,
        5,
        label_mask_data,
        -9999.0,
        [0.0, 1.0, 0.0, 5.0, 0.0, -1.0],
    );
    let mlabels = extract_sink_labels_from_raster(&label_dem, &label_mask).unwrap();
    assert_eq!(mlabels[0], 0, "0.0 background cell must remain label 0");
    assert_eq!(mlabels[6], 1);
    assert_eq!(mlabels[7], 1);
    assert_eq!(mlabels[18], 2);
    assert_eq!(
        mlabels.iter().filter(|&&l| l > 0).count(),
        3,
        "only the 3 labeled foreground pixels should be non-zero"
    );

    // 8e. File-based `delineate_catchment_hierarchy` saves `f64` catchment_hir.tif preserving
    // fractional areas below 0.5 m^2 (1-cell sink at 0.5x0.5 m resolution -> exact 0.25 m^2).
    let tmp = tempfile::tempdir().unwrap();
    let frac_dem_path = tmp.path().join("frac_dem.tif");
    let lvl_dir = tmp.path().join("levels");
    std::fs::create_dir_all(&lvl_dir).unwrap();
    let lvl1_path = lvl_dir.join("Single_level_1.tif");
    let hir_out_path = tmp.path().join("hir").join("catchment_hir.tif");
    // Ring around center at 10.0 drains strictly outward to 9.0 border; center at 9.5 is an
    // isolated 1-cell sink whose exact catchment area is 1 * 0.25 = 0.25 m^2 (< 0.5 m^2).
    let mut frac_z = vec![9.0f64; 25];
    for r in 1..4 {
        for c in 1..4 {
            frac_z[r * 5 + c] = 10.0;
        }
    }
    frac_z[12] = 9.5;
    let frac_dem = make_dem(5, 5, frac_z, -9999.0, [0.0, 0.5, 0.0, 2.5, 0.0, -0.5]);
    let mut lvl1_data = vec![0.0f64; 25];
    lvl1_data[12] = 1.0;
    let lvl1_arr = make_dem(5, 5, lvl1_data, 0.0, frac_dem.geotransform);
    save_raster(&frac_dem, &frac_dem_path, None, false).unwrap();
    save_raster(&lvl1_arr, &lvl1_path, None, false).unwrap();
    let hir_res = delineate_catchment_hierarchy(
        &frac_dem_path,
        &lvl_dir,
        &hir_out_path,
        &CatchmentHierarchyOptions::default(),
    )
    .unwrap();
    let hir_loaded = load_raster::<f64>(&hir_res.catchment_raster_path, None).unwrap();
    assert!(
        (hir_res.records[0].cat_area - 0.25).abs() < 1e-6,
        "expected exact 0.25 m^2 catchment area, got {}",
        hir_res.records[0].cat_area
    );
    assert!(
        (hir_loaded.get(2, 2) - 0.25).abs() < 1e-6,
        "f64 catchment_hir.tif must preserve exact fractional area 0.25 m^2 (< 0.5), got {}",
        hir_loaded.get(2, 2)
    );

    // 8f. `delineate_depressions_tiled` upfront validation and seam deduplication.
    let bad_ov = TiledDelineationOptions {
        tile_size: 16,
        overlap: 1,
        kernel_size: 5, // needs overlap >= 2
        ..Default::default()
    };
    assert!(delineate_depressions_tiled(&frac_dem_path, tmp.path().join("bad1"), &bad_ov).is_err());
    let bad_depth = TiledDelineationOptions {
        tile_size: 16,
        min_depth: -0.1,
        ..Default::default()
    };
    assert!(
        delineate_depressions_tiled(&frac_dem_path, tmp.path().join("bad2"), &bad_depth).is_err()
    );
    let bad_int = TiledDelineationOptions {
        tile_size: 16,
        interval: 0.0,
        ..Default::default()
    };
    assert!(
        delineate_depressions_tiled(&frac_dem_path, tmp.path().join("bad3"), &bad_int).is_err()
    );

    // Build a 32x32 DEM (2x2 tiles of size 16 with overlap 6) containing:
    // 1. A basin straddling all 4 tiles right at the corner seam (rows 13..=18, cols 13..=18),
    // 2. An interior-only basin strictly inside tile (0, 0) (rows 3..=6, cols 3..=6).
    let tw = 32usize;
    let th = 32usize;
    let mut seam_z = vec![20.0f64; tw * th];
    for r in 13..=18 {
        for c in 13..=18 {
            seam_z[r * tw + c] =
                15.0 + ((r as i32 - 16).abs() + (c as i32 - 16).abs()) as f64 * 0.2;
        }
    }
    for r in 3..=6 {
        for c in 3..=6 {
            seam_z[r * tw + c] = 15.0 + ((r as i32 - 4).abs() + (c as i32 - 4).abs()) as f64 * 0.2;
        }
    }
    let seam_dem = make_dem(
        tw,
        th,
        seam_z,
        -9999.0,
        [0.0, 1.0, 0.0, th as f64, 0.0, -1.0],
    );
    let seam_path = tmp.path().join("seam_dem.tif");
    save_raster(&seam_dem, &seam_path, None, false).unwrap();
    let tiled_res = delineate_depressions_tiled(
        &seam_path,
        tmp.path().join("tiled_out"),
        &TiledDelineationOptions {
            tile_size: 16,
            overlap: 6,
            kernel_size: 1,
            min_size: 5,
            min_depth: 0.2,
            interval: 0.5,
            topology: Topology::D8,
            use_f32: false,
        },
    )
    .unwrap();
    assert_eq!(tiled_res.num_tiles, 4);
    assert_eq!(
        tiled_res.depressions.len(),
        2,
        "2x2 corner-straddling basin + interior basin must appear exactly once each, got {:?}",
        tiled_res.depressions
    );

    // 8g. `slicing.rs`: multi-part parent rim dissolved into a single MultiPolygon feature in
    // `write_depressions_vector`, plus `min_depth == 0.0` floor slice and 1D valid_count.
    let mut mp_id = vec![0i32; 25];
    // Disconnected two-part rim with id=1 separated by child id=2.
    mp_id[5 + 1] = 1;
    mp_id[5 + 2] = 2;
    mp_id[5 + 3] = 1;
    let mut mp_id_arr = Array2D::from_vec(5, 5, mp_id).unwrap();
    mp_id_arr.geotransform = [0.0, 1.0, 0.0, 5.0, 0.0, -1.0];
    mp_id_arr.set_no_data(0);
    let mp_gpkg = tmp.path().join("multipart.gpkg");
    write_depressions_vector(
        &mp_id_arr,
        &[
            dummy_dep(1, 2, 0, vec![2], 3),
            dummy_dep(2, 1, 1, vec![], 1),
        ],
        &mp_gpkg,
    )
    .unwrap();
    let ds = terrio::open_vector(mp_gpkg.to_str().unwrap(), &VectorOpenOptions::default()).unwrap();
    assert_eq!(
        ds.layers[0].features.len(),
        2,
        "multi-part id=1 must be dissolved into 1 feature, not duplicated"
    );
    let ids: HashSet<i64> = ds.layers[0]
        .features
        .iter()
        .filter_map(|f| match f.attributes.first() {
            Some(FieldValue::Integer(v)) => Some(*v),
            _ => None,
        })
        .collect();
    assert_eq!(ids, HashSet::from([1, 2]));

    // 1-row depression with a NaN gap -> valid_count (not bw*bh) used.
    let row_sink = make_dem(
        4,
        1,
        vec![10.0, 11.0, 12.0, f64::NAN],
        0.0,
        [0.0, 1.0, 0.0, 1.0, 0.0, -1.0],
    );
    let row_deps = delineate_depressions_in_memory(&row_sink, 0, 0.0, 0.5).unwrap();
    assert_eq!(row_deps.depressions.len(), 1);
    assert_eq!(row_deps.depressions[0].count, 3);

    // `min_depth == 0.0` at `elev == min_elev`:
    // (a) Single flat-bottom bowl must NOT create a spurious zero-depth child.
    let mut single_bowl = vec![12.0f64; 25];
    for r in 1..4 {
        for c in 1..4 {
            single_bowl[r * 5 + c] = 11.0;
        }
    }
    single_bowl[2 * 5 + 2] = 10.0;
    let single_sink = make_dem(5, 5, single_bowl, 0.0, [0.0, 1.0, 0.0, 5.0, 0.0, -1.0]);
    let single_deps = delineate_depressions_in_memory(&single_sink, 0, 0.0, 0.5).unwrap();
    assert_eq!(
        single_deps.depressions.len(),
        1,
        "single flat-bottom bowl with min_depth=0.0 must not spawn a spurious zero-depth child"
    );

    // (b) Compound basin with two disconnected flat-bottom sub-bowls at z == min_elev (10.0)
    // separated by a 10.5 saddle inside an 11.0 rim MUST split into 1 parent + 2 children.
    let mut twin_flat = vec![11.0f64; 3 * 7];
    // Left 1x2 flat floor at cols 1..=2, row 1; saddle 10.5 at col 3; right 1x2 flat floor at cols 4..=5.
    twin_flat[7 + 1] = 10.0;
    twin_flat[7 + 2] = 10.0;
    twin_flat[7 + 3] = 10.5;
    twin_flat[7 + 4] = 10.0;
    twin_flat[7 + 5] = 10.0;
    let twin_sink = make_dem(7, 3, twin_flat, 0.0, [0.0, 1.0, 0.0, 3.0, 0.0, -1.0]);
    let twin_deps = delineate_depressions_in_memory(&twin_sink, 1, 0.0, 0.5).unwrap();
    assert_eq!(
        twin_deps.depressions.len(),
        3,
        "two disconnected flat-bottom sub-bowls at z == min_elev must split when min_depth == 0.0, got {:?}",
        twin_deps.depressions
    );
    assert_eq!(twin_deps.max_level, 2);

    // 8h. `region.rs`: NaN elevation guarded + anisotropic perimeter exactness.
    let labels_4x4 = vec![1u32; 16];
    let mut elev_nan = vec![10.0f64; 16];
    elev_nan[0] = f64::NAN;
    let props_aniso = region_props_xy(&labels_4x4, 1, &elev_nan, 4, 4, 2.0, 10.0, 20.0);
    assert_eq!(props_aniso.len(), 1);
    assert!(props_aniso[0].volume.is_finite());
    assert!(
        (props_aniso[0].perimeter - 72.0).abs() < 1e-9,
        "expected exact anisotropic 4x4 perimeter 72.0, got {}",
        props_aniso[0].perimeter
    );
    let _ = region_group(&[1.0, 0.0, 1.0, 1.0], 2, 2, 0, |v| v > 0.0);

    // 8i. `graph.rs`: flat basin single row, cyclic children_ids safe, GraphML timing attrs.
    let flat_sink = make_dem(3, 3, vec![100.0; 9], 0.0, [0.0, 1.0, 0.0, 3.0, 0.0, -1.0]);
    let mut flat_id = Array2D::from_vec(3, 3, vec![1i32; 9]).unwrap();
    flat_id.copy_metadata_from(&flat_sink);
    let mut flat_lvl = Array2D::from_vec(3, 3, vec![1i32; 9]).unwrap();
    flat_lvl.copy_metadata_from(&flat_sink);
    let mut cyclic_dep = dummy_dep(1, 1, 0, vec![1], 9); // self-cycle in children_ids!
    cyclic_dep.min_elev = 100.0;
    cyclic_dep.max_elev = 100.0;
    let flat_deps = InMemoryDepressions {
        depression_id: flat_id,
        depression_level: flat_lvl,
        depressions: vec![cyclic_dep],
        num_regions: 1,
        max_level: 1,
    };
    let rc = compute_rating_curves(&flat_sink, &flat_deps, 0.5).unwrap();
    assert_eq!(
        rc.len(),
        1,
        "flat basin (span == 0.0) must emit exactly 1 rating curve row, got {}",
        rc.len()
    );

    let g = ConnectivityGraph {
        rain_intensity_cm_h: 5.0,
        nodes: vec![GraphNode {
            id: 1,
            parent_id: 0,
            level: 1,
            region_id: 1,
            area: 9.0,
            volume: 9.0,
            avg_depth: 1.0,
            max_depth: 2.0,
            min_elev: 10.0,
            max_elev: 12.0,
            circularity: 0.8,
            hypsometric_integral: 0.5,
            cat_area: 25.0,
            fill_time_h: Some(1.25),
            fill_runoff_mm: Some(62.5),
        }],
        edges: vec![GraphEdge {
            from_id: 1,
            to_id: 0,
            edge_type: "spill".to_string(),
            spill_time_h: Some(1.25),
            spill_runoff_mm: Some(62.5),
            merge_time_h: Some(2.5),
            spill_elev: Some(12.0),
            length_m: 10.0,
            start_xy: Some([0.5, 0.5]),
            end_xy: Some([10.5, 0.5]),
            dist_to_stream: Some(f64::NAN), // non-finite must be omitted from GraphML!
        }],
    };
    let gml = connectivity_graph_to_graphml_string(&g);
    assert!(gml.contains(r#"attr.name="fill_time_h""#));
    assert!(gml.contains(r#"<data key="n_fill_h">1.2500</data>"#));
    assert!(gml.contains(r#"attr.name="spill_time_h""#));
    assert!(gml.contains(r#"<data key="e_spill_h">1.2500</data>"#));
    assert!(gml.contains(r#"attr.name="merge_time_h""#));
    assert!(gml.contains(r#"<data key="e_merge_h">2.5000</data>"#));
    assert!(gml.contains(r#"attr.name="spill_elev""#));
    assert!(gml.contains(r#"<data key="e_spill_elev">12.0000</data>"#));
    assert!(
        !gml.contains("NaN"),
        "GraphML must omit non-finite attribute values"
    );
}
