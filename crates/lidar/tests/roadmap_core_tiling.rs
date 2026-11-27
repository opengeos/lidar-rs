//! Integration tests for Roadmap Issue #14 Wave 1 PR B:
//! 1. Exact bottom-up merge-tree / contour-tree depression hierarchy (`merge_tree.rs`)
//! 2. 5x5 median network, O(N) sliding-window mean, bilateral, and Lindsay et al. (2019)
//!    feature-preserving DEM smoothing (`filtering.rs`)
//! 3. True out-of-core windowed tiled processing, per-tile breaching, and cross-tile
//!    stitched raster + GeoPackage outputs (`tiling.rs`)
//! 4. Complete child-hole-free enclosing compound polygons in GeoPackage (`slicing.rs`)

use std::collections::HashSet;
use std::path::PathBuf;

use lidar::{
    Array2D, BreachMode, BreachOptions, FeaturePreservingOptions, InundationForcing,
    MergeTreeOptions, SimulateInundationOptions, TiledDelineationOptions, TiledOutputOptions,
    Topology, bilateral_filter, bilateral_filter_file, build_connectivity_graph,
    build_enclosing_depressions_layer, delineate_depressions_in_memory,
    delineate_depressions_merge_tree_in_memory, delineate_depressions_tiled_with_outputs,
    extract_sinks_in_memory, feature_preserving_smoothing, feature_preserving_smoothing_file,
    gaussian_filter, load_raster, mean_filter, median_filter, save_raster,
    simulate_fill_spill_merge_in_memory, write_depressions_vector_with_enclosing,
};
use terrio::vector::Ring;
use terrio::{FieldValue, Geometry, VectorOpenOptions};

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data")
}

fn ring_area_abs(ring: &Ring) -> f64 {
    let pts = ring.coords();
    if pts.len() < 3 {
        return 0.0;
    }
    let n = pts.len();
    let mut sum = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    0.5 * sum.abs()
}

fn geom_area(geom: &Geometry) -> (f64, usize) {
    match geom {
        Geometry::Polygon {
            exterior,
            interiors,
        } => {
            let ext = ring_area_abs(exterior);
            let holes: f64 = interiors.iter().map(ring_area_abs).sum();
            ((ext - holes).max(0.0), interiors.len())
        }
        Geometry::MultiPolygon(parts) => {
            let mut total = 0.0;
            let mut hole_cnt = 0usize;
            for (ext, ints) in parts {
                let e = ring_area_abs(ext);
                let h: f64 = ints.iter().map(ring_area_abs).sum();
                total += (e - h).max(0.0);
                hole_cnt += ints.len();
            }
            (total, hole_cnt)
        }
        _ => (0.0, 0),
    }
}

// =============================================================================
// 1. Exact Merge-Tree / Contour-Tree Hierarchy Tests
// =============================================================================

#[test]
fn merge_tree_recovers_exact_sub_interval_saddle_and_pour_point() {
    // Two bowls separated by a single saddle cell at (4, 8) with z = 10.37,
    // surrounded by a rim at z = 12.0. Coarse slicing interval = 0.5 cannot
    // resolve z = 10.37 exactly, whereas the merge tree recovers it to machine precision.
    let (w, h) = (17, 9);
    let mut data = vec![0.0f64; w * h];
    for r in 1..8 {
        for c in 1..16 {
            data[r * w + c] = 12.0;
        }
    }
    // Left bowl: rows 2..7, cols 2..7 (25 cells, z in [7.0, 9.0]).
    for r in 2..7 {
        for c in 2..7 {
            let d = ((r as i32 - 4).abs() + (c as i32 - 4).abs()) as f64;
            data[r * w + c] = 7.0 + d * 0.5;
        }
    }
    // Right bowl: rows 2..7, cols 10..15 (25 cells, z in [7.5, 9.5]).
    for r in 2..7 {
        for c in 10..15 {
            let d = ((r as i32 - 4).abs() + (c as i32 - 12).abs()) as f64;
            data[r * w + c] = 7.5 + d * 0.5;
        }
    }
    // Connecting channel at row 4: cols 7 and 9 at 10.10, exact saddle at col 8 = 10.37.
    data[4 * w + 7] = 10.10;
    data[4 * w + 8] = 10.37;
    data[4 * w + 9] = 10.10;

    let mut sink = Array2D::from_vec(w, h, data).unwrap();
    sink.set_no_data(0.0);
    sink.geotransform = [500.0, 2.0, 0.0, 1000.0, 0.0, -2.0];

    let coarse_slicing = delineate_depressions_in_memory(&sink, 10, 0.5, 0.5).unwrap();
    for d in &coarse_slicing.depressions {
        if d.level == 1 {
            // Slicing at interval=0.5 steps on 12.0, 11.5, 11.0, 10.5, 10.0, ...
            // so it cannot land on 10.37.
            assert!(
                (d.max_elev - 10.37).abs() > 0.05,
                "expected coarse slicing to miss exact 10.37 saddle, got {}",
                d.max_elev
            );
        }
    }

    let mt = delineate_depressions_merge_tree_in_memory(
        &sink,
        &MergeTreeOptions {
            min_size: 10,
            min_depth: 0.5,
            topology: Topology::D4,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(mt.depressions.max_level, 2);
    assert_eq!(mt.nodes.len(), 3);
    assert_eq!(mt.depressions.depressions.len(), 3);

    let root = &mt.nodes[0];
    assert_eq!(root.id, 1);
    assert_eq!(root.parent_id, 0);
    assert_eq!(root.level, 2);
    assert!(!root.is_leaf);
    assert_eq!(root.children_ids.len(), 2);
    assert!((root.saddle_elev - 12.0).abs() < 1e-12);

    for &cid in &root.children_ids {
        let child = &mt.nodes[(cid - 1) as usize];
        assert!(child.is_leaf);
        assert_eq!(child.level, 1);
        assert_eq!(child.parent_id, root.id);
        assert!(
            (child.saddle_elev - 10.37).abs() < 1e-12,
            "expected exact saddle_elev == 10.37, got {}",
            child.saddle_elev
        );
        assert_eq!((child.pour_point_row, child.pour_point_col), (4, 8));
        assert!((child.pour_point_x - (500.0 + 8.5 * 2.0)).abs() < 1e-12);
        assert!((child.pour_point_y - (1000.0 - 4.5 * 2.0)).abs() < 1e-12);
        assert_eq!(child.count, 26); // 25 bowl cells + 1 channel cell (10.10)
    }

    // Exact volume conservation at root.
    let expected_vol: f64 = sink
        .data()
        .iter()
        .filter(|&&z| z > 0.0)
        .map(|&z| (12.0 - z) * 4.0)
        .sum();
    assert!((root.volume - expected_vol).abs() < 1e-9);
}

#[test]
fn merge_tree_three_level_nested_hierarchy_and_persistence_pruning() {
    // 3-level hierarchy:
    //   Sub-bowl A (floor 5.0) + Sub-bowl B (floor 5.5) meet at saddle_1 = 7.42
    //     -> Mid-parent M (level 2)
    //   Mid-parent M + Sub-bowl C (floor 6.0) meet at saddle_2 = 9.83
    //     -> Root R (level 3, rim 12.0)
    // Plus a sub-threshold 2-cell micro-pit (depth 0.15 < min_depth 0.5) inside the rim
    // that must be absorbed without spawning a spurious branch!
    let (w, h) = (25, 9);
    let mut data = vec![0.0f64; w * h];
    for r in 1..8 {
        for c in 1..24 {
            data[r * w + c] = 12.0;
        }
    }
    // Mid-basin terrace covering A and B (cols 2..14, rows 2..7) at z = 9.0.
    for r in 2..7 {
        for c in 2..14 {
            data[r * w + c] = 9.0;
        }
    }
    // Bowl A: rows 3..6, cols 3..6 (9 cells, z in [5.0, 6.0]).
    for r in 3..6 {
        for c in 3..6 {
            let d = ((r as i32 - 4).abs() + (c as i32 - 4).abs()) as f64;
            data[r * w + c] = 5.0 + d * 0.5;
        }
    }
    // Bowl B: rows 3..6, cols 9..12 (9 cells, z in [5.5, 6.5]).
    for r in 3..6 {
        for c in 9..12 {
            let d = ((r as i32 - 4).abs() + (c as i32 - 10).abs()) as f64;
            data[r * w + c] = 5.5 + d * 0.5;
        }
    }
    // Exact saddle 1 between A and B at (4, 7) = 7.42.
    data[4 * w + 6] = 7.10;
    data[4 * w + 7] = 7.42;
    data[4 * w + 8] = 7.10;

    // Bowl C: rows 3..6, cols 18..21 (9 cells, z in [6.0, 7.0]).
    for r in 3..6 {
        for c in 18..21 {
            let d = ((r as i32 - 4).abs() + (c as i32 - 19).abs()) as f64;
            data[r * w + c] = 6.0 + d * 0.5;
        }
    }
    // Exact saddle 2 between Mid-parent (col 13 at 9.0) and Bowl C at (4, 15) = 9.83.
    data[4 * w + 14] = 9.50;
    data[4 * w + 15] = 9.83;
    data[4 * w + 16] = 9.50;
    data[4 * w + 17] = 9.20;

    // Sub-threshold noise micro-pit on the outer rim (only 2 cells, depth 0.15).
    data[w + 22] = 11.85;
    data[w + 23] = 11.90;

    let mut sink = Array2D::from_vec(w, h, data).unwrap();
    sink.set_no_data(0.0);
    sink.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

    let mt = delineate_depressions_merge_tree_in_memory(
        &sink,
        &MergeTreeOptions {
            min_size: 8,
            min_depth: 0.5,
            topology: Topology::D4,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(mt.depressions.max_level, 3);
    assert_eq!(
        mt.nodes.len(),
        5,
        "expected exactly 5 nodes (root + mid + 3 leaves; noise pit pruned), got {:?}",
        mt.nodes
    );

    let root = &mt.nodes[0];
    assert_eq!(root.level, 3);
    assert_eq!(root.parent_id, 0);
    assert_eq!(root.children_ids.len(), 2);

    let level2_nodes: Vec<_> = mt.nodes.iter().filter(|n| n.level == 2).collect();
    let level1_nodes: Vec<_> = mt.nodes.iter().filter(|n| n.level == 1).collect();
    assert_eq!(level2_nodes.len(), 1);
    assert_eq!(level1_nodes.len(), 3);

    let mid = level2_nodes[0];
    assert_eq!(mid.parent_id, root.id);
    assert_eq!(mid.children_ids.len(), 2);
    assert!((mid.saddle_elev - 9.83).abs() < 1e-12);
    assert_eq!((mid.pour_point_row, mid.pour_point_col), (4, 15));

    for &cid in &mid.children_ids {
        let leaf = &mt.nodes[(cid - 1) as usize];
        assert_eq!(leaf.level, 1);
        assert_eq!(leaf.parent_id, mid.id);
        assert!((leaf.saddle_elev - 7.42).abs() < 1e-12);
        assert_eq!((leaf.pour_point_row, leaf.pour_point_col), (4, 7));
    }

    // Root rating curve must contain BOTH exact saddles (7.42 and 9.83) and rim (12.0).
    let root_rc: Vec<_> = mt
        .rating_curves
        .iter()
        .filter(|p| p.dep_id == root.id)
        .collect();
    assert!(root_rc.iter().any(|p| (p.stage - 7.42).abs() < 1e-12));
    assert!(root_rc.iter().any(|p| (p.stage - 9.83).abs() < 1e-12));
    assert!((root_rc.last().unwrap().stage - 12.0).abs() < 1e-12);
    assert!((root_rc.last().unwrap().volume - root.volume).abs() < 1e-9);
}

#[test]
fn merge_tree_on_sample_sink_conserves_volume_and_runs_fsm() {
    let dem = load_raster::<f64>(test_data_dir().join("dem.tif"), None).unwrap();
    let smoothed = median_filter(&dem, 3).unwrap();
    let sinks = extract_sinks_in_memory(&smoothed, 500, Topology::D8).unwrap();

    let mt = delineate_depressions_merge_tree_in_memory(
        &sinks.sink,
        &MergeTreeOptions {
            min_size: 500,
            min_depth: 0.3,
            topology: Topology::D8,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(!mt.nodes.is_empty());
    assert_eq!(mt.nodes.len(), mt.depressions.depressions.len());
    assert!(mt.depressions.max_level >= 1);

    // Total labeled pixel count in depression_id must equal sum of root node counts.
    let labeled_cells = mt
        .depressions
        .depression_id
        .data()
        .iter()
        .filter(|&&v| v > 0)
        .count();
    let root_count_sum: usize = mt
        .nodes
        .iter()
        .filter(|n| n.parent_id == 0)
        .map(|n| n.count)
        .sum();
    assert_eq!(labeled_cells, root_count_sum);

    // Every root node's volume must match exact direct cell summation below its saddle_elev.
    let gt = sinks.sink.geotransform;
    let cell_area = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    let id_data = mt.depressions.depression_id.data();
    let sdata = sinks.sink.data();
    for root in mt.nodes.iter().filter(|n| n.parent_id == 0) {
        // Collect all descendant IDs of this root.
        let mut subtree = HashSet::new();
        let mut stack = vec![root.id];
        while let Some(u) = stack.pop() {
            if subtree.insert(u) {
                for &c in &mt.nodes[(u - 1) as usize].children_ids {
                    stack.push(c);
                }
            }
        }
        let mut direct_cnt = 0usize;
        let mut direct_vol = 0.0f64;
        for (&did, &z) in id_data.iter().zip(sdata.iter()) {
            if did > 0 && subtree.contains(&(did as u32)) {
                direct_cnt += 1;
                direct_vol += (root.saddle_elev - z).max(0.0) * cell_area;
            }
        }
        assert_eq!(direct_cnt, root.count);
        assert!(
            (direct_vol - root.volume).abs() < 1e-6 * root.volume.max(1.0),
            "root {} volume mismatch: direct={direct_vol}, node={}",
            root.id,
            root.volume
        );
    }

    // Verify full compatibility with FSM routing and connectivity graph builder.
    let fsm_opts = SimulateInundationOptions {
        rain_intensity_cm_h: 5.0,
        rain_duration_h: 2.0,
        ..Default::default()
    };
    let fsm_res = simulate_fill_spill_merge_in_memory(
        &sinks.dem_refined,
        &sinks.sink,
        &mt.depressions,
        &fsm_opts,
        &InundationForcing::default(),
    )
    .unwrap();
    assert_eq!(fsm_res.basins.len(), mt.nodes.len());
    assert!(
        fsm_res.mass_balance.mass_balance_error_m3.abs() < 1e-4,
        "FSM mass balance residual too large: {}",
        fsm_res.mass_balance.mass_balance_error_m3
    );

    let graph = build_connectivity_graph(
        &mt.depressions,
        &[],
        &fsm_res.fill_time,
        &fsm_res.spill_paths,
        fsm_opts.rain_intensity_cm_h,
        None,
        None,
    )
    .unwrap();
    assert_eq!(graph.nodes.len(), mt.nodes.len());

    // Also test directly on test_data/sink.tif.
    let sample_sink = load_raster::<f64>(test_data_dir().join("sink.tif"), None).unwrap();
    let sample_mt = delineate_depressions_merge_tree_in_memory(
        &sample_sink,
        &MergeTreeOptions {
            min_size: 1000,
            min_depth: 0.3,
            topology: Topology::D8,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!sample_mt.nodes.is_empty());
    assert!(sample_mt.depressions.max_level >= 2);
    assert!(!sample_mt.rating_curves.is_empty());
}

// =============================================================================
// 2. Faster Filters & Edge-Preserving DEM Smoothing Tests
// =============================================================================

fn reflect_idx(mut idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let n = len as isize;
    let period = 2 * n;
    idx = idx.rem_euclid(period);
    if idx >= n {
        (period - 1 - idx) as usize
    } else {
        idx as usize
    }
}

fn naive_median_ref(dem: &Array2D<f64>, k: usize) -> Vec<f64> {
    let (w, h) = (dem.width(), dem.height());
    let r_neg = (k / 2) as isize;
    let r_pos = (k as isize) - 1 - r_neg;
    let nd = dem.no_data();
    let src = dem.data();
    let mut out = vec![0.0; w * h];
    let mut win = Vec::with_capacity(k * k);
    for y in 0..h {
        for x in 0..w {
            let center = src[y * w + x];
            if !center.is_finite() || center == nd {
                out[y * w + x] = center;
                continue;
            }
            win.clear();
            for dy in -r_neg..=r_pos {
                let ny = reflect_idx(y as isize + dy, h);
                for dx in -r_neg..=r_pos {
                    let nx = reflect_idx(x as isize + dx, w);
                    let v = src[ny * w + nx];
                    if v.is_finite() && v != nd {
                        win.push(v);
                    }
                }
            }
            win.sort_by(f64::total_cmp);
            let n = win.len();
            out[y * w + x] = if n == 0 {
                center
            } else if n % 2 == 1 {
                win[n / 2]
            } else {
                0.5 * (win[n / 2 - 1] + win[n / 2])
            };
        }
    }
    out
}

fn naive_mean_ref(dem: &Array2D<f64>, k: usize) -> Vec<f64> {
    let (w, h) = (dem.width(), dem.height());
    let r_neg = (k / 2) as isize;
    let r_pos = (k as isize) - 1 - r_neg;
    let nd = dem.no_data();
    let src = dem.data();
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let center = src[y * w + x];
            if !center.is_finite() || center == nd {
                out[y * w + x] = center;
                continue;
            }
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for dy in -r_neg..=r_pos {
                let ny = reflect_idx(y as isize + dy, h);
                for dx in -r_neg..=r_pos {
                    let nx = reflect_idx(x as isize + dx, w);
                    let v = src[ny * w + nx];
                    if v.is_finite() && v != nd {
                        sum += v;
                        cnt += 1;
                    }
                }
            }
            out[y * w + x] = if cnt > 0 { sum / cnt as f64 } else { center };
        }
    }
    out
}

#[test]
fn median_5x5_fast_path_bit_exact_with_nodata_and_signed_zeros() {
    let mut state: u64 = 0x5EED_CAFE_1234_5678;
    let mut next_u64 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    for &(w, h) in &[(1usize, 7usize), (7, 1), (3, 3), (5, 5), (17, 19)] {
        let mut data = Vec::with_capacity(w * h);
        for i in 0..(w * h) {
            let r = next_u64() % 23;
            let v = if r == 0 {
                -9999.0
            } else if r == 1 {
                f64::NAN
            } else if r == 2 {
                -0.0
            } else if r == 3 {
                0.0
            } else {
                ((next_u64() % 10_000) as f64) * 0.01 - 50.0 + (i % 3) as f64
            };
            data.push(v);
        }
        let mut dem = Array2D::from_vec(w, h, data).unwrap();
        dem.set_no_data(-9999.0);

        let got = median_filter(&dem, 5).unwrap();
        let exp = naive_median_ref(&dem, 5);
        for (idx, (&a, &b)) in got.data().iter().zip(exp.iter()).enumerate() {
            if b.is_nan() {
                assert!(a.is_nan(), "expected NaN at idx {idx} ({w}x{h})");
            } else {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "5x5 median bit mismatch at idx {idx} ({w}x{h}): got {a} ({:#x}), expected {b} ({:#x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }

    // Explicitly verify 13 x -0.0 and 12 x +0.0 -> exact -0.0 bit pattern.
    let mut sz_data = vec![0.0f64; 25];
    for slot in sz_data.iter_mut().take(13) {
        *slot = -0.0;
    }
    let mut sz_dem = Array2D::from_vec(5, 5, sz_data).unwrap();
    sz_dem.set_no_data(-9999.0);
    let sz_med = median_filter(&sz_dem, 5).unwrap();
    assert_eq!(sz_med.get(2, 2).to_bits(), (-0.0f64).to_bits());
}

#[test]
fn mean_filter_sliding_window_matches_reference_for_large_kernels_and_extremes() {
    let (w, h) = (65usize, 67usize);
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let v = if idx % 29 == 5 {
                -9999.0
            } else if idx % 53 == 11 {
                f64::NAN
            } else if (x, y) == (15, 25) {
                // Huge transient spike to test sliding-window cancellation resync.
                1e14
            } else {
                250.0 + ((x * 13 + y * 7) % 101) as f64 * 0.125
            };
            data.push(v);
        }
    }
    let mut dem = Array2D::from_vec(w, h, data).unwrap();
    dem.set_no_data(-9999.0);

    for k in [11usize, 21] {
        let got = mean_filter(&dem, k).unwrap();
        let exp = naive_mean_ref(&dem, k);
        for (idx, (&a, &b)) in got.data().iter().zip(exp.iter()).enumerate() {
            if b.is_nan() {
                assert!(a.is_nan());
            } else {
                let tol = 1e-12 * b.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "mean_filter k={k} mismatch at idx {idx}: got {a}, expected {b}, diff={}",
                    (a - b).abs()
                );
            }
        }
    }

    // Extreme +/-1e308 values inside a large 11x11 kernel must stay finite via scaled fallback.
    let mut ext_data = vec![0.0f64; 25 * 25];
    ext_data[12 * 25 + 12] = 1e308;
    ext_data[12 * 25 + 13] = 1e308;
    ext_data[13 * 25 + 12] = -1e308;
    ext_data[13 * 25 + 13] = -1e308;
    let mut ext_dem = Array2D::from_vec(25, 25, ext_data).unwrap();
    ext_dem.set_no_data(-9999.0);
    let ext_mean = mean_filter(&ext_dem, 11).unwrap();
    assert!(
        ext_mean.get(12, 12).is_finite() && ext_mean.get(12, 12).abs() < 1e-6,
        "expected finite 0.0 mean on balanced +/-1e308 grid, got {}",
        ext_mean.get(12, 12)
    );
}

#[test]
fn bilateral_and_feature_preserving_denoise_planar_terrain_and_preserve_step_edge() {
    // 40x24 DEM with two flat terraces separated by a sharp 10 m embankment at x = 20,
    // plus +/-0.2 m deterministic micro-noise on both terraces and a NoData hole.
    let (w, h) = (40usize, 24usize);
    let mut clean = vec![0.0f64; w * h];
    let mut noisy = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let base = if x < 20 { 100.0 } else { 110.0 };
            let noise = if (x + y) % 2 == 0 { 0.2 } else { -0.2 };
            clean[y * w + x] = base;
            noisy[y * w + x] = base + noise;
        }
    }
    noisy[4 * w + 4] = -9999.0;

    let mut dem = Array2D::from_vec(w, h, noisy).unwrap();
    dem.set_no_data(-9999.0);
    dem.geotransform = [0.0, 1.0, 0.0, h as f64, 0.0, -1.0];

    let gauss = gaussian_filter(&dem, 2.0).unwrap();
    let bilat = bilateral_filter(&dem, 2.0, 1.0).unwrap();
    let fps_opts = FeaturePreservingOptions {
        filter_size: 7,
        norm_diff_deg: 15.0,
        num_iter: 4,
        max_z_change: Some(0.5),
        compress: false,
    };
    let fps = feature_preserving_smoothing(&dem, &fps_opts).unwrap();

    // NoData preserved untouched.
    assert_eq!(bilat.get(4, 4), -9999.0);
    assert_eq!(fps.get(4, 4), -9999.0);

    // Measure noise RMSE on flat interior terrace away from edge and NoData.
    let terrace_rmse = |arr: &Array2D<f64>| -> f64 {
        let mut ss = 0.0;
        let mut cnt = 0usize;
        for y in 6..18 {
            for x in 6..15 {
                let diff = arr.get(x, y) - clean[y * w + x];
                ss += diff * diff;
                cnt += 1;
            }
        }
        (ss / cnt as f64).sqrt()
    };
    let raw_rmse = terrace_rmse(&dem);
    let bilat_rmse = terrace_rmse(&bilat);
    let fps_rmse = terrace_rmse(&fps);
    assert!(
        bilat_rmse < raw_rmse * 0.25,
        "bilateral should strongly suppress terrace micro-noise: raw={raw_rmse}, bilat={bilat_rmse}"
    );
    assert!(
        fps_rmse < raw_rmse * 0.5,
        "feature-preserving smoothing should suppress terrace micro-noise: raw={raw_rmse}, fps={fps_rmse}"
    );

    // Measure step sharpness right across the embankment (x=19 vs x=20, true jump = 10.0).
    let edge_jump = |arr: &Array2D<f64>| -> f64 {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for y in 6..18 {
            sum += arr.get(20, y) - arr.get(19, y);
            cnt += 1;
        }
        sum / cnt as f64
    };
    let gauss_jump = edge_jump(&gauss);
    let bilat_jump = edge_jump(&bilat);
    let fps_jump = edge_jump(&fps);
    assert!(
        bilat_jump > gauss_jump + 4.0 && bilat_jump > 9.0,
        "bilateral should preserve step edge much better than Gaussian: gauss={gauss_jump}, bilat={bilat_jump}"
    );
    assert!(
        fps_jump > gauss_jump + 4.0 && fps_jump > 8.5,
        "feature-preserving should preserve step edge much better than Gaussian: gauss={gauss_jump}, fps={fps_jump}"
    );

    // Verify max_z_change clamping on feature_preserving_smoothing.
    let clamped = feature_preserving_smoothing(
        &dem,
        &FeaturePreservingOptions {
            max_z_change: Some(0.05),
            ..fps_opts.clone()
        },
    )
    .unwrap();
    for (&orig, &out) in dem.data().iter().zip(clamped.data().iter()) {
        if orig != -9999.0 {
            assert!((out - orig).abs() <= 0.05 + 1e-12);
        }
    }

    // Verify file wrappers round-trip cleanly by reopening from disk and comparing cells,
    // NoData sentinel, and geotransform against the in-memory filtered rasters.
    let tmp = tempfile::tempdir().unwrap();
    let in_tif = tmp.path().join("noisy.tif");
    let bilat_tif = tmp.path().join("bilat.tif");
    let fps_tif = tmp.path().join("fps.tif");
    save_raster(&dem, &in_tif, None, false).unwrap();
    bilateral_filter_file(&in_tif, 2.0, 1.0, Some(&bilat_tif)).unwrap();
    feature_preserving_smoothing_file(&in_tif, &fps_opts, Some(&fps_tif)).unwrap();
    assert!(bilat_tif.exists());
    assert!(fps_tif.exists());

    let bilat_disk = load_raster::<f64>(&bilat_tif, None).unwrap();
    let fps_disk = load_raster::<f64>(&fps_tif, None).unwrap();
    assert_eq!(bilat_disk.no_data(), bilat.no_data());
    assert_eq!(bilat_disk.geotransform, bilat.geotransform);
    assert_eq!(fps_disk.no_data(), fps.no_data());
    assert_eq!(fps_disk.geotransform, fps.geotransform);
    for (idx, (&d, &m)) in bilat_disk
        .data()
        .iter()
        .zip(bilat.data().iter())
        .enumerate()
    {
        assert!(
            (d - m).abs() < 1e-12,
            "bilateral disk mismatch at {idx}: disk={d}, mem={m}"
        );
    }
    for (idx, (&d, &m)) in fps_disk.data().iter().zip(fps.data().iter()).enumerate() {
        assert!(
            (d - m).abs() < 1e-12,
            "fps disk mismatch at {idx}: disk={d}, mem={m}"
        );
    }
}

// =============================================================================
// 3. Out-of-Core Windowed Tiling & Stitched Spatial Outputs Tests
// =============================================================================

#[test]
fn tiled_out_of_core_stitches_seamless_rasters_and_gpkg_on_synthetic_and_sample_dem() {
    // 1. Multi-tile synthetic 48x48 DEM (2x2 tiles of size 24, overlap 10) with:
    //    - A basin straddling the 4-way tile seam at rows 20..28, cols 20..28
    //    - An interior basin strictly inside tile (0, 0) at rows 4..10, cols 4..10
    let (w, h) = (48usize, 48usize);
    let mut z = vec![25.0f64; w * h];
    for r in 20..28 {
        for c in 20..28 {
            let d = ((r as f64 - 23.5).powi(2) + (c as f64 - 23.5).powi(2)).sqrt();
            z[r * w + c] = (18.0 + d * 1.2).min(24.5);
        }
    }
    for r in 4..10 {
        for c in 4..10 {
            let d = ((r as f64 - 6.5).powi(2) + (c as f64 - 6.5).powi(2)).sqrt();
            z[r * w + c] = (20.0 + d * 1.0).min(24.5);
        }
    }
    let mut dem = Array2D::from_vec(w, h, z).unwrap();
    dem.set_no_data(-9999.0);
    dem.geotransform = [1000.0, 2.0, 0.0, 2000.0, 0.0, -2.0];

    let tmp = tempfile::tempdir().unwrap();
    let dem_path = tmp.path().join("synth_tiled.tif");
    save_raster(&dem, &dem_path, None, false).unwrap();

    let out_dir = tmp.path().join("tiled_spatial");
    let tiled_opts = TiledDelineationOptions {
        tile_size: 24,
        overlap: 10,
        kernel_size: 1,
        min_size: 10,
        min_depth: 0.3,
        interval: 0.5,
        topology: Topology::D8,
        use_f32: false,
    };
    let out_opts = TiledOutputOptions {
        write_rasters: true,
        write_gpkg: true,
        breach: None,
        compress: true,
    };
    let res = delineate_depressions_tiled_with_outputs(&dem_path, &out_dir, &tiled_opts, &out_opts)
        .unwrap();
    assert_eq!(res.num_tiles, 4);
    assert_eq!(res.depressions.len(), 2);

    let id_tif = res.depression_id_tif.as_ref().unwrap();
    let lvl_tif = res.depression_level_tif.as_ref().unwrap();
    let depth_tif = res.depth_tif.as_ref().unwrap();
    let gpkg = res.gpkg_path.as_ref().unwrap();
    assert!(id_tif.exists());
    assert!(lvl_tif.exists());
    assert!(depth_tif.exists());
    assert!(gpkg.exists());

    // Exact full-DEM dimensions & geotransform on all stitched rasters.
    let id_arr = load_raster::<i32>(id_tif, None).unwrap();
    let lvl_arr = load_raster::<i32>(lvl_tif, None).unwrap();
    let depth_arr = load_raster::<f64>(depth_tif, None).unwrap();
    assert_eq!((id_arr.width(), id_arr.height()), (w, h));
    assert_eq!((lvl_arr.width(), lvl_arr.height()), (w, h));
    assert_eq!((depth_arr.width(), depth_arr.height()), (w, h));
    assert_eq!(id_arr.geotransform, dem.geotransform);
    assert_eq!(depth_arr.geotransform, dem.geotransform);

    // Seamless IDs across the 4-way seam at (23, 23), (23, 24), (24, 23), (24, 24):
    // all 4 seam-corner pixels must belong to the EXACT same global depression ID!
    let seam_id = id_arr.get(23, 23);
    assert!(seam_id > 0);
    assert_eq!(id_arr.get(24, 23), seam_id);
    assert_eq!(id_arr.get(23, 24), seam_id);
    assert_eq!(id_arr.get(24, 24), seam_id);
    assert!(depth_arr.get(23, 23) > 4.0);

    // Interior basin has a distinct ID.
    let int_id = id_arr.get(6, 6);
    assert!(int_id > 0 && int_id != seam_id);

    // Stitched GPKG contains features for both depressions.
    let vds = terrio::open_vector(gpkg.to_str().unwrap(), &VectorOpenOptions::default()).unwrap();
    assert_eq!(vds.layers[0].name, "depressions");
    assert_eq!(vds.layers[0].features.len(), 2);

    // Per-tile complete breaching removes all depressions.
    let breached_dir = tmp.path().join("tiled_breached");
    let breached = delineate_depressions_tiled_with_outputs(
        &dem_path,
        &breached_dir,
        &tiled_opts,
        &TiledOutputOptions {
            write_rasters: false,
            write_gpkg: false,
            breach: Some(BreachOptions::for_mode(BreachMode::Complete, None, None)),
            compress: true,
        },
    )
    .unwrap();
    assert!(
        breached.depressions.is_empty(),
        "complete per-tile breaching should drain all synthetic sinks, got {}",
        breached.depressions.len()
    );

    // 2. Real sample DEM (`test_data/dem.tif`) tiled into 4 tiles with full spatial outputs.
    let sample_dem_path = test_data_dir().join("dem.tif");
    let sample_dem = load_raster::<f64>(&sample_dem_path, None).unwrap();
    let sample_out = tmp.path().join("sample_tiled");
    let sample_res = delineate_depressions_tiled_with_outputs(
        &sample_dem_path,
        &sample_out,
        &TiledDelineationOptions {
            tile_size: 200,
            overlap: 50,
            kernel_size: 3,
            min_size: 1000,
            min_depth: 0.3,
            interval: 0.3,
            topology: Topology::D8,
            use_f32: true,
        },
        &out_opts,
    )
    .unwrap();
    assert_eq!(sample_res.num_tiles, 4);
    assert!(!sample_res.depressions.is_empty());
    let s_id = load_raster::<i32>(sample_res.depression_id_tif.as_ref().unwrap(), None).unwrap();
    assert_eq!(
        (s_id.width(), s_id.height()),
        (sample_dem.width(), sample_dem.height())
    );
    assert_eq!(s_id.geotransform, sample_dem.geotransform);
    let s_gpkg = terrio::open_vector(
        sample_res.gpkg_path.as_ref().unwrap().to_str().unwrap(),
        &VectorOpenOptions::default(),
    )
    .unwrap();
    assert_eq!(s_gpkg.layers.len(), 2);
    assert_eq!(s_gpkg.layers[0].name, "depressions");
    assert_eq!(s_gpkg.layers[1].name, "depressions_full");
}

// =============================================================================
// 4. Complete Enclosing Compound Polygons in GeoPackage (`slicing.rs`) Tests
// =============================================================================

#[test]
fn gpkg_enclosing_layer_produces_hole_free_full_footprint_for_compound_depressions() {
    // 2-level nested double bowl: parent rim surrounds two inner leaf bowls.
    let (w, h) = (30usize, 18usize);
    let mut data = vec![0.0f64; w * h];
    for r in 1..h - 1 {
        for c in 1..w - 1 {
            let d1 = ((r as f64 - 8.5).powi(2) + (c as f64 - 8.5).powi(2)).sqrt();
            let d2 = ((r as f64 - 8.5).powi(2) + (c as f64 - 20.5).powi(2)).sqrt();
            data[r * w + c] = (d1.min(d2) * 0.5 + 10.0).min(15.0);
        }
    }
    let mut sink = Array2D::from_vec(w, h, data).unwrap();
    sink.set_no_data(0.0);
    sink.geotransform = [100.0, 2.0, 0.0, 500.0, 0.0, -2.0]; // cell_area = 4.0

    let deps = delineate_depressions_in_memory(&sink, 10, 0.3, 0.3).unwrap();
    assert!(deps.max_level >= 2);
    let parent = deps
        .depressions
        .iter()
        .find(|d| d.level == deps.max_level && !d.children_ids.is_empty())
        .expect("expected compound parent depression");

    let tmp = tempfile::tempdir().unwrap();
    let gpkg_path = tmp.path().join("nested_enclosing.gpkg");
    write_depressions_vector_with_enclosing(
        &deps.depression_id,
        &deps.depressions,
        &gpkg_path,
        true,
    )
    .unwrap();

    let vds =
        terrio::open_vector(gpkg_path.to_str().unwrap(), &VectorOpenOptions::default()).unwrap();
    assert_eq!(vds.layers.len(), 2);
    assert_eq!(vds.layers[0].name, "depressions");
    assert_eq!(vds.layers[1].name, "depressions_full");
    assert_eq!(vds.layers[0].features.len(), deps.depressions.len());
    assert_eq!(vds.layers[1].features.len(), deps.depressions.len());

    fn find_feat(layer: &terrio::Layer, target_id: u32) -> &terrio::Feature {
        layer
            .features
            .iter()
            .find(|f| matches!(f.attributes.first(), Some(FieldValue::Integer(id)) if *id == target_id as i64))
            .expect("missing feature for parent id")
    }

    // (a) In `"depressions"`, the parent polygon is a ring with holes where its children sit,
    // so its polygon area is strictly less than `parent.area`.
    let ring_feat = find_feat(&vds.layers[0], parent.id);
    let (ring_area, ring_holes) = geom_area(ring_feat.geometry.as_ref().unwrap());
    assert!(
        ring_holes >= 1,
        "expected parent in 'depressions' layer to have interior child hole(s), got {ring_holes}"
    );
    assert!(
        ring_area < parent.area - 1.0,
        "expected parent ring area ({ring_area}) < total enclosing area ({})",
        parent.area
    );

    // (b) In `"depressions_full"`, the parent polygon has NO child holes and its geometric
    // area matches `parent.area` (covering both parent rim and all child interiors) exactly!
    let full_feat = find_feat(&vds.layers[1], parent.id);
    let (full_area, full_holes) = geom_area(full_feat.geometry.as_ref().unwrap());
    assert_eq!(
        full_holes, 0,
        "expected parent in 'depressions_full' layer to have 0 holes, got {full_holes}"
    );
    assert!(
        (full_area - parent.area).abs() < 1e-6,
        "expected full enclosing polygon area ({full_area}) == parent.area ({})",
        parent.area
    );

    // Also verify standalone `build_enclosing_depressions_layer` produces the same feature count.
    let standalone =
        build_enclosing_depressions_layer(&deps.depression_id, &deps.depressions).unwrap();
    assert_eq!(standalone.name, "depressions_full");
    assert_eq!(standalone.features.len(), deps.depressions.len());

    // Two-layer export to GeoJSON must be rejected upfront, while single-layer GeoJSON succeeds.
    let bad_geojson = tmp.path().join("two_layer.geojson");
    assert!(
        write_depressions_vector_with_enclosing(
            &deps.depression_id,
            &deps.depressions,
            &bad_geojson,
            true
        )
        .is_err()
    );
    assert!(!bad_geojson.exists());

    let ok_geojson = tmp.path().join("single_layer.geojson");
    write_depressions_vector_with_enclosing(
        &deps.depression_id,
        &deps.depressions,
        &ok_geojson,
        false,
    )
    .unwrap();
    assert!(ok_geojson.exists());
}

#[test]
fn gpkg_enclosing_layer_covers_three_level_merge_tree_hierarchy() {
    // 3-level hierarchy delineated via merge tree -> exported to GPKG with enclosing layer.
    let (w, h) = (25usize, 9usize);
    let mut data = vec![0.0f64; w * h];
    for r in 1..8 {
        for c in 1..24 {
            data[r * w + c] = 12.0;
        }
    }
    for r in 2..7 {
        for c in 2..14 {
            data[r * w + c] = 9.0;
        }
    }
    for r in 3..6 {
        for c in 3..6 {
            data[r * w + c] = 5.0 + ((r as i32 - 4).abs() + (c as i32 - 4).abs()) as f64 * 0.5;
        }
    }
    for r in 3..6 {
        for c in 9..12 {
            data[r * w + c] = 5.5 + ((r as i32 - 4).abs() + (c as i32 - 10).abs()) as f64 * 0.5;
        }
    }
    data[4 * w + 6] = 7.10;
    data[4 * w + 7] = 7.42;
    data[4 * w + 8] = 7.10;
    for r in 3..6 {
        for c in 18..21 {
            data[r * w + c] = 6.0 + ((r as i32 - 4).abs() + (c as i32 - 19).abs()) as f64 * 0.5;
        }
    }
    data[4 * w + 14] = 9.50;
    data[4 * w + 15] = 9.83;
    data[4 * w + 16] = 9.50;
    data[4 * w + 17] = 9.20;

    let mut sink = Array2D::from_vec(w, h, data).unwrap();
    sink.set_no_data(0.0);
    sink.geotransform = [0.0, 2.0, 0.0, 18.0, 0.0, -2.0]; // cell_area = 4.0

    let mt = delineate_depressions_merge_tree_in_memory(
        &sink,
        &MergeTreeOptions {
            min_size: 8,
            min_depth: 0.5,
            topology: Topology::D4,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(mt.depressions.max_level, 3);

    let tmp = tempfile::tempdir().unwrap();
    let gpkg_path = tmp.path().join("three_level.gpkg");
    write_depressions_vector_with_enclosing(
        &mt.depressions.depression_id,
        &mt.depressions.depressions,
        &gpkg_path,
        true,
    )
    .unwrap();

    let vds =
        terrio::open_vector(gpkg_path.to_str().unwrap(), &VectorOpenOptions::default()).unwrap();
    assert_eq!(vds.layers.len(), 2);
    let full_layer = &vds.layers[1];
    assert_eq!(full_layer.name, "depressions_full");
    assert_eq!(full_layer.features.len(), mt.nodes.len());

    // Every node (level 1, 2, AND 3) in `depressions_full` must have polygon area == node.area!
    for node in &mt.nodes {
        let feat = full_layer
            .features
            .iter()
            .find(|f| matches!(f.attributes.first(), Some(FieldValue::Integer(id)) if *id == node.id as i64))
            .expect("missing feature in depressions_full");
        let (area, holes) = geom_area(feat.geometry.as_ref().unwrap());
        assert_eq!(
            holes, 0,
            "node {} (level {}) should have 0 holes",
            node.id, node.level
        );
        assert!(
            (area - node.area).abs() < 1e-6,
            "node {} (level {}) enclosing area mismatch: poly={area}, attr={}",
            node.id,
            node.level,
            node.area
        );
    }
}
