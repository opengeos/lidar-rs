//! Complementary pure-Rust terrain-hydrology derivatives.
//!
//! Provides grid-based hydrologic terrain derivatives operating directly on
//! [`richdem::Array2D<f64>`]:
//!
//! - **D-Infinity (Tarboton, 1997):** [`dinf_flow_direction`] and
//!   [`dinf_flow_accumulation`] with proportional facet partitioning and
//!   Barnes flat-resolution D8 fallback on flats.
//! - **Multiple Flow Direction (Quinn et al., 1991; Freeman, 1991):**
//!   [`mfd_flow_accumulation`] distributing flow across all lower D8 neighbors
//!   proportional to `(slope_i)^p * contour_length_i`.
//! - **Topographic Wetness Index (Beven & Kirkby, 1979):** [`compute_twi`]
//!   (`ln(a / tan(beta))`) using D-Infinity specific catchment area and slope.
//! - **Height Above Nearest Drainage (Renno et al., 2008; Nobre et al., 2011):**
//!   [`compute_hand`] and [`compute_hand_from_threshold`].
//! - **Stream-network burning (AGREE; Hellweger, 1997):** [`burn_streams`] and
//!   [`burn_streams_file`] for conditioning DEMs along vector hydrography.

use std::collections::VecDeque;
use std::f64::consts::{FRAC_PI_2, TAU};
use std::path::Path;

use richdem::{Array2D, D8Flowdir, D8X, D8Y, DemNum, FLOWDIR_NO_DATA, load_raster, save_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use terrio::crs::SpatialRef;
use terrio::{Geometry, VectorOpenOptions};

use crate::catchment::d8_flow_directions_resolved;
use crate::error::{Error, Result};
use crate::inundation::validate_grid_alignment;
use crate::raster_ops::ensure_no_path_collision;
use crate::region::crs_metric_scale;

/// Minimum tangent slope floor used in [`compute_twi`] to avoid `ln(inf)` on flat cells.
const TWI_MIN_TAN_SLOPE: f64 = 1e-4;

/// Default NoData sentinel for float derivative outputs when the source DEM's NoData
/// would collide with valid domain values.
const DEFAULT_OUT_NODATA: f64 = -9999.0;

#[inline]
fn is_valid_dem(z: f64, nd: f64) -> bool {
    z.is_finite() && !z.rd_eq(nd)
}

/// Selects a safe NoData sentinel that never collides with valid values `>= min_reserved`.
#[inline]
fn safe_nodata(dem_nd: f64, min_reserved: f64) -> f64 {
    if dem_nd.is_finite() && dem_nd < min_reserved {
        dem_nd
    } else {
        DEFAULT_OUT_NODATA
    }
}

/// One triangular facet in Tarboton (1997) Table 1.
#[derive(Clone, Copy)]
struct DinfFacet {
    /// Cardinal neighbor offset `(dc, dr)`.
    e1: (isize, isize),
    /// Diagonal neighbor offset `(dc, dr)`.
    e2: (isize, isize),
    /// Whether `e1` is along x (`true`: `d1 = dx, d2 = dy`) or along y (`false`: `d1 = dy, d2 = dx`).
    e1_is_x: bool,
    /// Angle quadrant multiplier `a_c`.
    ac: f64,
    /// Angle orientation sign `a_f` (`+1.0` or `-1.0`).
    af: f64,
}

/// The 8 counterclockwise triangular facets from East (`0`) around to East (`2*pi`),
/// matching Tarboton (1997) Table 1.
const DINF_FACETS: [DinfFacet; 8] = [
    // 0: E -> NE
    DinfFacet {
        e1: (1, 0),
        e2: (1, -1),
        e1_is_x: true,
        ac: 0.0,
        af: 1.0,
    },
    // 1: N -> NE
    DinfFacet {
        e1: (0, -1),
        e2: (1, -1),
        e1_is_x: false,
        ac: 1.0,
        af: -1.0,
    },
    // 2: N -> NW
    DinfFacet {
        e1: (0, -1),
        e2: (-1, -1),
        e1_is_x: false,
        ac: 1.0,
        af: 1.0,
    },
    // 3: W -> NW
    DinfFacet {
        e1: (-1, 0),
        e2: (-1, -1),
        e1_is_x: true,
        ac: 2.0,
        af: -1.0,
    },
    // 4: W -> SW
    DinfFacet {
        e1: (-1, 0),
        e2: (-1, 1),
        e1_is_x: true,
        ac: 2.0,
        af: 1.0,
    },
    // 5: S -> SW
    DinfFacet {
        e1: (0, 1),
        e2: (-1, 1),
        e1_is_x: false,
        ac: 3.0,
        af: -1.0,
    },
    // 6: S -> SE
    DinfFacet {
        e1: (0, 1),
        e2: (1, 1),
        e1_is_x: false,
        ac: 3.0,
        af: 1.0,
    },
    // 7: E -> SE
    DinfFacet {
        e1: (1, 0),
        e2: (1, 1),
        e1_is_x: true,
        ac: 4.0,
        af: -1.0,
    },
];

/// Per-cell D-Infinity facet solution used internally by direction, accumulation, and TWI.
#[derive(Clone, Copy)]
struct DinfCell {
    /// Flow direction angle in `[0, 2*pi)`, `-1.0` for flats/pits, or `out_nd` for NoData.
    angle: f64,
    /// Steepest downslope gradient `tan(beta) >= 0.0`.
    slope: f64,
    /// First receiving neighbor index (`usize::MAX` if none).
    n1: usize,
    /// Flow weight routed to `n1` (`0.0..=1.0`).
    w1: f64,
    /// Second receiving neighbor index (`usize::MAX` if none).
    n2: usize,
    /// Flow weight routed to `n2` (`0.0..=1.0`).
    w2: f64,
}

fn checked_metric_scale(dem: &Array2D<f64>) -> Result<(f64, f64, f64)> {
    let scale = crs_metric_scale(
        &dem.projection,
        &dem.geotransform,
        dem.width(),
        dem.height(),
    );
    if !scale.cell_width_m.is_finite()
        || !scale.cell_height_m.is_finite()
        || !scale.cell_area_m2.is_finite()
        || scale.cell_width_m <= 0.0
        || scale.cell_height_m <= 0.0
    {
        return Err(Error::InvalidArgument(
            "DEM geotransform or CRS produced non-finite or non-positive metric cell spacing"
                .into(),
        ));
    }
    Ok((scale.cell_width_m, scale.cell_height_m, scale.cell_area_m2))
}

fn solve_dinf_grid(dem: &Array2D<f64>, out_nd: f64) -> Result<Vec<DinfCell>> {
    let w = dem.width();
    let h = dem.height();
    let n = w * h;
    let zdata = dem.data();
    let z_nd = dem.no_data();
    let (dx, dy, _) = checked_metric_scale(dem)?;
    let d_diag = dx.hypot(dy);

    let mut out = vec![
        DinfCell {
            angle: out_nd,
            slope: 0.0,
            n1: usize::MAX,
            w1: 0.0,
            n2: usize::MAX,
            w2: 0.0,
        };
        n
    ];

    for r in 0..h {
        let off = r * w;
        for c in 0..w {
            let idx = off + c;
            let e0 = zdata[idx];
            if !is_valid_dem(e0, z_nd) {
                continue;
            }

            let mut best_slope = 0.0f64;
            let mut best_angle = -1.0f64;
            let mut best_n1 = usize::MAX;
            let mut best_w1 = 0.0f64;
            let mut best_n2 = usize::MAX;
            let mut best_w2 = 0.0f64;

            for facet in &DINF_FACETS {
                let (d1, d2) = if facet.e1_is_x { (dx, dy) } else { (dy, dx) };
                let diag_ang = d2.atan2(d1);

                let nr1 = r as isize + facet.e1.1;
                let nc1 = c as isize + facet.e1.0;
                let e1_info = if nr1 >= 0 && nc1 >= 0 && (nr1 as usize) < h && (nc1 as usize) < w {
                    let ni = (nr1 as usize) * w + (nc1 as usize);
                    let zv = zdata[ni];
                    is_valid_dem(zv, z_nd).then_some((ni, zv))
                } else {
                    None
                };

                let nr2 = r as isize + facet.e2.1;
                let nc2 = c as isize + facet.e2.0;
                let e2_info = if nr2 >= 0 && nc2 >= 0 && (nr2 as usize) < h && (nc2 as usize) < w {
                    let ni = (nr2 as usize) * w + (nc2 as usize);
                    let zv = zdata[ni];
                    is_valid_dem(zv, z_nd).then_some((ni, zv))
                } else {
                    None
                };

                let (slope, r_ang, n1, w1, n2, w2) = match (e1_info, e2_info) {
                    (Some((ni1, e1)), Some((ni2, e2))) => {
                        let s1 = (e0 - e1) / d1;
                        let s2 = (e1 - e2) / d2;
                        if s1 > 0.0 {
                            let r_raw = (s2 / s1).atan();
                            if r_raw <= 0.0 {
                                (s1, 0.0, ni1, 1.0, usize::MAX, 0.0)
                            } else if r_raw >= diag_ang {
                                let sd = (e0 - e2) / d_diag;
                                (sd, diag_ang, ni2, 1.0, usize::MAX, 0.0)
                            } else {
                                let frac2 = (r_raw / diag_ang).clamp(0.0, 1.0);
                                let frac1 = 1.0 - frac2;
                                (s1.hypot(s2), r_raw, ni1, frac1, ni2, frac2)
                            }
                        } else {
                            let sd = (e0 - e2) / d_diag;
                            if sd > 0.0 {
                                (sd, diag_ang, ni2, 1.0, usize::MAX, 0.0)
                            } else {
                                (0.0, 0.0, usize::MAX, 0.0, usize::MAX, 0.0)
                            }
                        }
                    }
                    (Some((ni1, e1)), None) => {
                        let s1 = (e0 - e1) / d1;
                        if s1 > 0.0 {
                            (s1, 0.0, ni1, 1.0, usize::MAX, 0.0)
                        } else {
                            (0.0, 0.0, usize::MAX, 0.0, usize::MAX, 0.0)
                        }
                    }
                    (None, Some((ni2, e2))) => {
                        let sd = (e0 - e2) / d_diag;
                        if sd > 0.0 {
                            (sd, diag_ang, ni2, 1.0, usize::MAX, 0.0)
                        } else {
                            (0.0, 0.0, usize::MAX, 0.0, usize::MAX, 0.0)
                        }
                    }
                    (None, None) => (0.0, 0.0, usize::MAX, 0.0, usize::MAX, 0.0),
                };

                if slope > best_slope {
                    best_slope = slope;
                    let mut ang = (facet.ac * FRAC_PI_2 + facet.af * r_ang).rem_euclid(TAU);
                    if ang >= TAU {
                        ang = 0.0;
                    }
                    best_angle = ang;
                    best_n1 = n1;
                    best_w1 = w1;
                    best_n2 = n2;
                    best_w2 = w2;
                }
            }

            out[idx] = DinfCell {
                angle: best_angle,
                slope: best_slope,
                n1: best_n1,
                w1: best_w1,
                n2: best_n2,
                w2: best_w2,
            };
        }
    }

    Ok(out)
}

/// Computes D-Infinity flow directions (Tarboton, 1997) in radians `[0, 2*pi)` measured
/// counterclockwise from East (`0` = East, `pi/2` = North, `pi` = West, `3*pi/2` = South).
///
/// Flat and pit cells (no strictly lower neighbor) are assigned `-1.0`. NoData cells in
/// `dem` are preserved as NoData in the output raster.
pub fn dinf_flow_direction(dem: &Array2D<f64>) -> Result<Array2D<f64>> {
    let w = dem.width();
    let h = dem.height();
    let out_nd = safe_nodata(dem.no_data(), -1.0);
    let cells = solve_dinf_grid(dem, out_nd)?;
    let data: Vec<f64> = cells.into_iter().map(|c| c.angle).collect();
    let mut out = Array2D::from_vec(w, h, data)?;
    out.copy_metadata_from(dem);
    out.set_no_data(out_nd);
    Ok(out)
}

#[inline]
fn step_d8_valid(
    idx: usize,
    w: usize,
    h: usize,
    fdata: &[u8],
    zdata: &[f64],
    z_nd: f64,
) -> Option<usize> {
    let dir = fdata[idx] as usize;
    if dir == 0 || dir == FLOWDIR_NO_DATA as usize || dir > 8 {
        return None;
    }
    let nr = (idx / w) as isize + D8Y[dir] as isize;
    let nc = (idx % w) as isize + D8X[dir] as isize;
    if nr < 0 || nc < 0 || (nr as usize) >= h || (nc as usize) >= w {
        return None;
    }
    let ni = (nr as usize) * w + (nc as usize);
    is_valid_dem(zdata[ni], z_nd).then_some(ni)
}

/// Computes D-Infinity flow accumulation (Tarboton, 1997) in contributing cell-count units.
///
/// Each valid DEM cell contributes `1.0` unit and partitions its accumulated flow across up
/// to two bounding triangular facet neighbors in topological high-to-low order. Flat cells
/// (where D-Infinity slope is zero) route seamlessly using Barnes (2014) flat-resolved D8
/// directions so flow is conserved across flats. NoData cells remain NoData.
pub fn dinf_flow_accumulation(dem: &Array2D<f64>) -> Result<Array2D<f64>> {
    let (acc, _) = dinf_flow_accumulation_and_slope(dem)?;
    Ok(acc)
}

fn dinf_flow_accumulation_and_slope(dem: &Array2D<f64>) -> Result<(Array2D<f64>, Vec<f64>)> {
    let w = dem.width();
    let h = dem.height();
    let n = w * h;
    let zdata = dem.data();
    let z_nd = dem.no_data();
    let out_nd = safe_nodata(z_nd, 0.0);

    let mut cells = solve_dinf_grid(dem, out_nd)?;
    let has_flats = cells
        .iter()
        .enumerate()
        .any(|(i, c)| is_valid_dem(zdata[i], z_nd) && c.angle < 0.0);

    if has_flats {
        let flowdirs = d8_flow_directions_resolved(dem)?;
        let fdata = flowdirs.data();
        for (idx, cell) in cells.iter_mut().enumerate() {
            if is_valid_dem(zdata[idx], z_nd)
                && cell.angle < 0.0
                && let Some(nxt) = step_d8_valid(idx, w, h, fdata, zdata, z_nd)
            {
                cell.n1 = nxt;
                cell.w1 = 1.0;
                cell.n2 = usize::MAX;
                cell.w2 = 0.0;
            }
        }
    }

    let mut indegree = vec![0u32; n];
    for (idx, cell) in cells.iter().enumerate() {
        if !is_valid_dem(zdata[idx], z_nd) {
            continue;
        }
        if cell.n1 != usize::MAX && cell.w1 > 0.0 {
            indegree[cell.n1] += 1;
        }
        if cell.n2 != usize::MAX && cell.w2 > 0.0 {
            indegree[cell.n2] += 1;
        }
    }

    let mut acc = vec![out_nd; n];
    let mut visited = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for idx in 0..n {
        if is_valid_dem(zdata[idx], z_nd) {
            acc[idx] = 1.0;
            if indegree[idx] == 0 {
                queue.push_back(idx);
            }
        }
    }

    while let Some(u) = queue.pop_front() {
        visited[u] = true;
        let flow = acc[u];
        let c = cells[u];
        if c.n1 != usize::MAX && c.w1 > 0.0 {
            acc[c.n1] += flow * c.w1;
            indegree[c.n1] = indegree[c.n1].saturating_sub(1);
            if indegree[c.n1] == 0 {
                queue.push_back(c.n1);
            }
        }
        if c.n2 != usize::MAX && c.w2 > 0.0 {
            acc[c.n2] += flow * c.w2;
            indegree[c.n2] = indegree[c.n2].saturating_sub(1);
            if indegree[c.n2] == 0 {
                queue.push_back(c.n2);
            }
        }
    }

    // Fallback for any degenerate cycle (should not occur with resolved flats).
    let mut remaining: Vec<usize> = (0..n)
        .filter(|&i| is_valid_dem(zdata[i], z_nd) && !visited[i])
        .collect();
    if !remaining.is_empty() {
        remaining.sort_by(|&a, &b| zdata[b].total_cmp(&zdata[a]).then_with(|| a.cmp(&b)));
        for u in remaining {
            visited[u] = true;
            let flow = acc[u];
            let c = cells[u];
            if c.n1 != usize::MAX && c.w1 > 0.0 && !visited[c.n1] {
                acc[c.n1] += flow * c.w1;
            }
            if c.n2 != usize::MAX && c.w2 > 0.0 && !visited[c.n2] {
                acc[c.n2] += flow * c.w2;
            }
        }
    }

    let slopes: Vec<f64> = cells.into_iter().map(|c| c.slope).collect();
    let mut out = Array2D::from_vec(w, h, acc)?;
    out.copy_metadata_from(dem);
    out.set_no_data(out_nd);
    Ok((out, slopes))
}

/// Computes Multiple Flow Direction (MFD) flow accumulation (Quinn et al., 1991;
/// Freeman, 1991) in contributing cell-count units.
///
/// Each valid cell distributes its accumulated flow across all strictly lower D8 neighbors
/// proportional to `(slope_i)^exponent * contour_length_i`, where cardinal neighbors carry
/// contour weight `0.5 * L` and diagonal neighbors carry `0.354 * L` (`sqrt(2)/4 * L`).
/// Typical exponent values are `1.0` (Quinn et al., 1991) or `1.1` (Freeman, 1991).
/// Flat cells without a strictly lower neighbor route using Barnes (2014) resolved D8
/// directions.
pub fn mfd_flow_accumulation(dem: &Array2D<f64>, exponent: f64) -> Result<Array2D<f64>> {
    if !exponent.is_finite() || exponent <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "MFD exponent must be finite and > 0, got {exponent}"
        )));
    }
    let w = dem.width();
    let h = dem.height();
    let n = w * h;
    let zdata = dem.data();
    let z_nd = dem.no_data();
    let out_nd = safe_nodata(z_nd, 0.0);
    let (dx, dy, _) = checked_metric_scale(dem)?;
    let cell_size = 0.5 * (dx + dy);
    let d_diag = dx.hypot(dy);
    let diag_contour = 0.25 * std::f64::consts::SQRT_2 * cell_size;

    // Precompute neighbor step distance and contour length for D8 codes 1..=8.
    let mut d8_dist = [0.0f64; 9];
    let mut d8_contour = [0.0f64; 9];
    for d in 1..=8 {
        let sx = D8X[d];
        let sy = D8Y[d];
        if sx != 0 && sy != 0 {
            d8_dist[d] = d_diag;
            d8_contour[d] = diag_contour;
        } else if sx != 0 {
            d8_dist[d] = dx;
            d8_contour[d] = 0.5 * dy;
        } else {
            d8_dist[d] = dy;
            d8_contour[d] = 0.5 * dx;
        }
    }

    // Store normalized MFD outgoing edges in CSR arrays to avoid per-cell Vec allocations.
    let mut edge_offsets = vec![0usize; n + 1];
    let mut edge_targets: Vec<usize> = Vec::with_capacity(n * 3);
    let mut edge_weights: Vec<f64> = Vec::with_capacity(n * 3);
    let mut flat_cells: Vec<usize> = Vec::new();
    let mut indegree = vec![0u32; n];

    let mut local_slopes = [(0usize, 0.0f64, 0usize); 8];
    let mut local_nbrs = [(0usize, 0.0f64); 8];

    for r in 0..h {
        let off = r * w;
        for c in 0..w {
            let idx = off + c;
            edge_offsets[idx] = edge_targets.len();
            let z0 = zdata[idx];
            if !is_valid_dem(z0, z_nd) {
                continue;
            }
            let mut raw_cnt = 0usize;
            let mut max_slope = 0.0f64;
            for d in 1..=8 {
                let nr = r as isize + D8Y[d] as isize;
                let nc = c as isize + D8X[d] as isize;
                if nr < 0 || nc < 0 || (nr as usize) >= h || (nc as usize) >= w {
                    continue;
                }
                let ni = (nr as usize) * w + (nc as usize);
                let zi = zdata[ni];
                if is_valid_dem(zi, z_nd) && zi < z0 {
                    let slope = (z0 - zi) / d8_dist[d];
                    if slope > 0.0 && slope.is_finite() {
                        local_slopes[raw_cnt] = (ni, slope, d);
                        raw_cnt += 1;
                        if slope > max_slope {
                            max_slope = slope;
                        }
                    }
                }
            }
            let mut cnt = 0usize;
            let mut sum_w = 0.0f64;
            if raw_cnt > 0 && max_slope > 0.0 {
                // Compute normalized slope^exponent in log space relative to max_slope so
                // extreme slope spreads neither overflow sum_w to +inf nor underflow valid
                // branches to 0 when exponent is small.
                let ln_max = max_slope.ln();
                for &(ni, slope, d) in &local_slopes[..raw_cnt] {
                    let weight = (exponent * (slope.ln() - ln_max)).exp() * d8_contour[d];
                    if weight > 0.0 && weight.is_finite() {
                        local_nbrs[cnt] = (ni, weight);
                        cnt += 1;
                        sum_w += weight;
                    }
                }
            }
            if cnt > 0 && sum_w > 0.0 && sum_w.is_finite() {
                for &(ni, weight) in &local_nbrs[..cnt] {
                    edge_targets.push(ni);
                    edge_weights.push(weight / sum_w);
                    indegree[ni] += 1;
                }
            } else {
                flat_cells.push(idx);
            }
        }
    }
    edge_offsets[n] = edge_targets.len();

    // Resolve flat cells (zero downward neighbors) using Barnes D8 flat resolution.
    let mut flat_target = if !flat_cells.is_empty() {
        let flowdirs = d8_flow_directions_resolved(dem)?;
        let fdata = flowdirs.data();
        let mut ft = vec![usize::MAX; n];
        for &idx in &flat_cells {
            if let Some(nxt) = step_d8_valid(idx, w, h, fdata, zdata, z_nd) {
                ft[idx] = nxt;
                indegree[nxt] += 1;
            }
        }
        ft
    } else {
        Vec::new()
    };

    let mut acc = vec![out_nd; n];
    let mut visited = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for idx in 0..n {
        if is_valid_dem(zdata[idx], z_nd) {
            acc[idx] = 1.0;
            if indegree[idx] == 0 {
                queue.push_back(idx);
            }
        }
    }

    while let Some(u) = queue.pop_front() {
        visited[u] = true;
        let flow = acc[u];
        let start = edge_offsets[u];
        let end = edge_offsets[u + 1];
        if start < end {
            for k in start..end {
                let v = edge_targets[k];
                acc[v] += flow * edge_weights[k];
                indegree[v] = indegree[v].saturating_sub(1);
                if indegree[v] == 0 {
                    queue.push_back(v);
                }
            }
        } else if !flat_target.is_empty() {
            let v = flat_target[u];
            if v != usize::MAX {
                acc[v] += flow;
                indegree[v] = indegree[v].saturating_sub(1);
                if indegree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
    }

    // Fallback for any degenerate cycle.
    let mut remaining: Vec<usize> = (0..n)
        .filter(|&i| is_valid_dem(zdata[i], z_nd) && !visited[i])
        .collect();
    if !remaining.is_empty() {
        remaining.sort_by(|&a, &b| zdata[b].total_cmp(&zdata[a]).then_with(|| a.cmp(&b)));
        for u in remaining {
            visited[u] = true;
            let flow = acc[u];
            let start = edge_offsets[u];
            let end = edge_offsets[u + 1];
            if start < end {
                for k in start..end {
                    let v = edge_targets[k];
                    if !visited[v] {
                        acc[v] += flow * edge_weights[k];
                    }
                }
            } else if !flat_target.is_empty() {
                let v = flat_target[u];
                if v != usize::MAX && !visited[v] {
                    acc[v] += flow;
                }
                flat_target[u] = usize::MAX;
            }
        }
    }

    let mut out = Array2D::from_vec(w, h, acc)?;
    out.copy_metadata_from(dem);
    out.set_no_data(out_nd);
    Ok(out)
}

/// Computes the Topographic Wetness Index (`TWI = ln(a / tan(beta))`; Beven & Kirkby, 1979).
///
/// - `a` is specific catchment area in map units (`flow_acc_cells * cell_area / cell_size`),
///   computed from [`dinf_flow_accumulation`] when `flow_acc` is `None`, or from the provided
///   cell-count `flow_acc` raster when `Some`.
/// - `tan(beta)` is the local D-Infinity steepest downslope gradient, clamped to a positive
///   floor of `TWI_MIN_TAN_SLOPE = 1e-4` to prevent `ln(inf)`.
///
/// Flat and pit cells have D-Infinity slope `0.0` and are clamped to `1e-4`, producing a
/// baseline boost of `~ln(1e4) ~= 9.21` on unresolved flats. When flat-resolved downslope
/// gradients are needed, pre-condition the DEM with [`crate::breach_depressions`] first.
pub fn compute_twi(dem: &Array2D<f64>, flow_acc: Option<&Array2D<f64>>) -> Result<Array2D<f64>> {
    if let Some(fa) = flow_acc {
        validate_grid_alignment(dem, fa, "flow_acc")?;
    }
    let w = dem.width();
    let h = dem.height();
    let n = w * h;
    let zdata = dem.data();
    let z_nd = dem.no_data();
    let out_nd = safe_nodata(z_nd, -1000.0);
    let (dx, dy, cell_area) = checked_metric_scale(dem)?;
    let cell_size = 0.5 * (dx + dy);
    let sca_scale = cell_area / cell_size.max(1e-12);

    let computed_acc;
    let (acc_ref, slopes) = match flow_acc {
        Some(fa) => {
            let cells = solve_dinf_grid(dem, out_nd)?;
            let slopes: Vec<f64> = cells.into_iter().map(|c| c.slope).collect();
            (fa, slopes)
        }
        None => {
            let (acc, slopes) = dinf_flow_accumulation_and_slope(dem)?;
            computed_acc = acc;
            (&computed_acc, slopes)
        }
    };
    let acc_data = acc_ref.data();
    let acc_nd = acc_ref.no_data();

    let mut twi = vec![out_nd; n];
    for idx in 0..n {
        if !is_valid_dem(zdata[idx], z_nd) {
            continue;
        }
        let a_cells = acc_data[idx];
        if !a_cells.is_finite() || a_cells.rd_eq(acc_nd) {
            continue;
        }
        let sca = a_cells.max(1.0) * sca_scale;
        let tan_beta = slopes[idx].max(TWI_MIN_TAN_SLOPE);
        twi[idx] = (sca / tan_beta).ln();
    }

    let mut out = Array2D::from_vec(w, h, twi)?;
    out.copy_metadata_from(dem);
    out.set_no_data(out_nd);
    Ok(out)
}

/// Computes Height Above Nearest Drainage (HAND; Renno et al., 2008; Nobre et al., 2011).
///
/// Traces resolved D8 flow paths ([`d8_flow_directions_resolved`]) from every valid DEM cell
/// to its nearest downstream stream cell (`stream_mask > 0`) and computes
/// `(z_cell - z_drainage).max(0.0)`. Stream cells receive `0.0`. Cells whose flow path exits
/// the grid or terminates in an isolated pit without reaching a stream cell receive NoData.
pub fn compute_hand(dem: &Array2D<f64>, stream_mask: &Array2D<u8>) -> Result<Array2D<f64>> {
    validate_grid_alignment(dem, stream_mask, "stream_mask")?;
    let flowdirs = d8_flow_directions_resolved(dem)?;
    compute_hand_with_flowdirs(dem, stream_mask, &flowdirs)
}

fn compute_hand_with_flowdirs(
    dem: &Array2D<f64>,
    stream_mask: &Array2D<u8>,
    flowdirs: &Array2D<D8Flowdir>,
) -> Result<Array2D<f64>> {
    let w = dem.width();
    let h = dem.height();
    let n = w * h;
    let zdata = dem.data();
    let z_nd = dem.no_data();
    let mdata = stream_mask.data();
    let fdata = flowdirs.data();
    let out_nd = safe_nodata(z_nd, 0.0);

    // 0 = unvisited, 1 = visiting, 2 = resolved
    let mut state = vec![0u8; n];
    let mut drain_z = vec![f64::NAN; n];

    for idx in 0..n {
        if !is_valid_dem(zdata[idx], z_nd) {
            state[idx] = 2;
        } else if mdata[idx] > 0 {
            drain_z[idx] = zdata[idx];
            state[idx] = 2;
        }
    }

    let mut path: Vec<usize> = Vec::with_capacity(128);
    for start in 0..n {
        if state[start] == 2 {
            continue;
        }
        path.clear();
        let mut cur = start;
        let resolved_z = loop {
            if state[cur] == 2 {
                break drain_z[cur];
            }
            if state[cur] == 1 {
                break f64::NAN;
            }
            state[cur] = 1;
            path.push(cur);
            match step_d8_valid(cur, w, h, fdata, zdata, z_nd) {
                Some(nxt) => cur = nxt,
                None => break f64::NAN,
            }
        };
        for &p in &path {
            drain_z[p] = resolved_z;
            state[p] = 2;
        }
    }

    let mut hand = vec![out_nd; n];
    for idx in 0..n {
        let z = zdata[idx];
        let dz = drain_z[idx];
        if is_valid_dem(z, z_nd) && dz.is_finite() {
            hand[idx] = (z - dz).max(0.0);
        }
    }

    let mut out = Array2D::from_vec(w, h, hand)?;
    out.copy_metadata_from(dem);
    out.set_no_data(out_nd);
    Ok(out)
}

/// Computes Height Above Nearest Drainage (HAND) by extracting a stream network mask from
/// D8 flow accumulation cells meeting `acc >= acc_threshold_cells`.
pub fn compute_hand_from_threshold(
    dem: &Array2D<f64>,
    acc_threshold_cells: f64,
) -> Result<Array2D<f64>> {
    if !acc_threshold_cells.is_finite() || acc_threshold_cells <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "acc_threshold_cells must be finite and > 0, got {acc_threshold_cells}"
        )));
    }
    let w = dem.width();
    let h = dem.height();
    let n = w * h;
    let zdata = dem.data();
    let z_nd = dem.no_data();
    let flowdirs = d8_flow_directions_resolved(dem)?;
    let fdata = flowdirs.data();

    // O(N) D8 flow accumulation via Kahn's topological pass.
    let mut indegree = vec![0u32; n];
    let mut next_cell = vec![usize::MAX; n];
    for idx in 0..n {
        if is_valid_dem(zdata[idx], z_nd)
            && let Some(nxt) = step_d8_valid(idx, w, h, fdata, zdata, z_nd)
        {
            next_cell[idx] = nxt;
            indegree[nxt] += 1;
        }
    }

    let mut acc = vec![0.0f64; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for idx in 0..n {
        if is_valid_dem(zdata[idx], z_nd) {
            acc[idx] = 1.0;
            if indegree[idx] == 0 {
                queue.push_back(idx);
            }
        }
    }
    while let Some(u) = queue.pop_front() {
        let v = next_cell[u];
        if v != usize::MAX {
            acc[v] += acc[u];
            indegree[v] = indegree[v].saturating_sub(1);
            if indegree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    let mask_data: Vec<u8> = (0..n)
        .map(|i| u8::from(is_valid_dem(zdata[i], z_nd) && acc[i] >= acc_threshold_cells))
        .collect();
    let mut mask = Array2D::from_vec(w, h, mask_data)?;
    mask.copy_metadata_from(dem);
    mask.set_no_data(0);

    compute_hand_with_flowdirs(dem, &mask, &flowdirs)
}

/// Options controlling vector stream-network burning ([`burn_streams`] and
/// [`burn_streams_file`]).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BurnStreamsOptions {
    /// Sharp vertical elevation drop applied directly to stream channel cells
    /// (vertical DEM units, `>= 0.0`, default `1.0`).
    pub burn_depth: f64,
    /// Additional half-width in cells around the rasterized stream centerline receiving the
    /// full `burn_depth` (`0` = 1-cell centerline, `1` = 3-cell channel, default `0`).
    pub stream_width_cells: usize,
    /// Optional AGREE-style (Hellweger, 1997) smooth buffer radius in cells outside the
    /// stream channel (`0` disables the smooth buffer, default `0`).
    pub buffer_cells: usize,
    /// Smooth elevation drop tapering linearly from `smooth_drop` at the stream edge to
    /// `0.0` at distance `buffer_cells` (vertical DEM units, `>= 0.0`, default `0.0`).
    pub smooth_drop: f64,
    /// Compress output GeoTIFF with DEFLATE (used by [`burn_streams_file`]).
    pub compress: bool,
}

impl Default for BurnStreamsOptions {
    fn default() -> Self {
        Self {
            burn_depth: 1.0,
            stream_width_cells: 0,
            buffer_cells: 0,
            smooth_drop: 0.0,
            compress: false,
        }
    }
}

impl BurnStreamsOptions {
    /// Validates numeric burning parameters.
    pub fn validate(&self) -> Result<()> {
        if !self.burn_depth.is_finite() || self.burn_depth < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "burn_depth must be finite and >= 0, got {}",
                self.burn_depth
            )));
        }
        if !self.smooth_drop.is_finite() || self.smooth_drop < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "smooth_drop must be finite and >= 0, got {}",
                self.smooth_drop
            )));
        }
        Ok(())
    }
}

fn world_to_col_row(gt: &[f64; 6], x: f64, y: f64) -> Option<(isize, isize)> {
    let det = gt[1] * gt[5] - gt[2] * gt[4];
    if !det.is_finite() || det.abs() < 1e-18 {
        return None;
    }
    let dx = x - gt[0];
    let dy = y - gt[3];
    let col = (gt[5] * dx - gt[2] * dy) / det;
    let row = (-gt[4] * dx + gt[1] * dy) / det;
    if !col.is_finite() || !row.is_finite() {
        return None;
    }
    Some((col.floor() as isize, row.floor() as isize))
}

/// Liang-Barsky clipped Bresenham line rasterizer returning in-bounds `(col, row)` cells.
fn bresenham_segment(
    c0: isize,
    r0: isize,
    c1: isize,
    r1: isize,
    w: usize,
    h: usize,
) -> Vec<(usize, usize)> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let max_x = (w - 1) as isize;
    let max_y = (h - 1) as isize;
    let dx_f = c1 as f64 - c0 as f64;
    let dy_f = r1 as f64 - r0 as f64;
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    let clip = |p: f64, q: f64, t0: &mut f64, t1: &mut f64| -> bool {
        if p.abs() < 1e-12 {
            return q >= 0.0;
        }
        let r = q / p;
        if p < 0.0 {
            if r > *t1 {
                return false;
            }
            if r > *t0 {
                *t0 = r;
            }
        } else {
            if r < *t0 {
                return false;
            }
            if r < *t1 {
                *t1 = r;
            }
        }
        true
    };
    let ok = clip(-dx_f, c0 as f64 + 0.5, &mut t0, &mut t1)
        && clip(dx_f, max_x as f64 + 0.5 - c0 as f64, &mut t0, &mut t1)
        && clip(-dy_f, r0 as f64 + 0.5, &mut t0, &mut t1)
        && clip(dy_f, max_y as f64 + 0.5 - r0 as f64, &mut t0, &mut t1);
    if !ok {
        return Vec::new();
    }
    let sc0 = (c0 as f64 + t0 * dx_f).round() as isize;
    let sr0 = (r0 as f64 + t0 * dy_f).round() as isize;
    let sc1 = (c0 as f64 + t1 * dx_f).round() as isize;
    let sr1 = (r0 as f64 + t1 * dy_f).round() as isize;

    let mut out = Vec::new();
    let mut x = sc0;
    let mut y = sr0;
    let dx = (sc1 - sc0).abs();
    let dy = -(sr1 - sr0).abs();
    let sx = if sc0 < sc1 { 1 } else { -1 };
    let sy = if sr0 < sr1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            out.push((x as usize, y as usize));
        }
        if x == sc1 && y == sr1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x == sc1 {
                break;
            }
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            if y == sr1 {
                break;
            }
            err += dx;
            y += sy;
        }
    }
    out
}

fn collect_stream_polylines(geom: &Geometry, out: &mut Vec<Vec<(f64, f64)>>) {
    match geom {
        Geometry::LineString(coords) => {
            if coords.len() >= 2 {
                out.push(coords.iter().map(|c| (c.x, c.y)).collect());
            }
        }
        Geometry::MultiLineString(lines) => {
            for line in lines {
                if line.len() >= 2 {
                    out.push(line.iter().map(|c| (c.x, c.y)).collect());
                }
            }
        }
        Geometry::GeometryCollection(geoms) => {
            for g in geoms {
                collect_stream_polylines(g, out);
            }
        }
        _ => {}
    }
}

/// 1D exact squared Euclidean distance transform (Felzenszwalb & Huttenlocher, 2012) in `O(n)`.
fn edt_1d(f: &[f64], out: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        if !f[q].is_finite() {
            continue;
        }
        let mut s = loop {
            let p = v[k];
            if !f[p].is_finite() {
                v[k] = q;
                break f64::NEG_INFINITY;
            }
            let qf = q as f64;
            let pf = p as f64;
            let num = (f[q] + qf * qf) - (f[p] + pf * pf);
            let den = 2.0 * (qf - pf);
            let s_cand = num / den;
            if s_cand <= z[k] && k > 0 {
                k -= 1;
            } else {
                break s_cand;
            }
        };
        if s.is_finite() {
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
        } else {
            // Replaced an initial infinite seed at k == 0.
            s = f64::NEG_INFINITY;
            z[0] = s;
            z[1] = f64::INFINITY;
        }
    }
    let mut j = 0usize;
    for (q, dst) in out.iter_mut().enumerate() {
        let qf = q as f64;
        while z[j + 1] < qf {
            j += 1;
        }
        let p = v[j];
        let diff = qf - p as f64;
        *dst = diff * diff + f[p];
    }
}

/// Exact 2D Euclidean distance transform in `O(W * H)` from `seeds == true`.
fn euclidean_distance_2d(seeds: &[bool], w: usize, h: usize) -> Vec<f64> {
    let n = w * h;
    let mut g = vec![f64::INFINITY; n];
    let max_dim = w.max(h);
    let mut v = vec![0usize; max_dim];
    let mut z = vec![0.0f64; max_dim + 1];
    let mut col_in = vec![0.0f64; h];
    let mut col_out = vec![0.0f64; h];

    // Pass 1: columns.
    for c in 0..w {
        for r in 0..h {
            col_in[r] = if seeds[r * w + c] { 0.0 } else { f64::INFINITY };
        }
        edt_1d(&col_in, &mut col_out, &mut v, &mut z);
        for r in 0..h {
            g[r * w + c] = col_out[r];
        }
    }

    // Pass 2: rows.
    let mut row_out = vec![0.0f64; w];
    for r in 0..h {
        let off = r * w;
        edt_1d(&g[off..off + w], &mut row_out, &mut v, &mut z);
        for c in 0..w {
            g[off + c] = row_out[c].sqrt();
        }
    }
    g
}

/// Burns vector stream polylines into an in-memory DEM with configurable sharp channel
/// carving (`burn_depth`, `stream_width_cells`) and optional AGREE-style smooth buffer
/// lowering (`buffer_cells`, `smooth_drop`).
///
/// Accepts any polyline vector format supported by `terrio` (`.gpkg`, `.shp`, `.geojson`,
/// etc.) in the same CRS as `dem`. When `dem.projection` is non-empty it must parse as a
/// recognized CRS, and CRS equivalence is verified whenever both the DEM and the stream
/// layer carry CRS metadata. Stream layers without CRS metadata (for example, a `.shp`
/// missing `.prj`) or DEMs with an empty `projection` are assumed to already share the same
/// coordinate system. Returns `(conditioned_dem, modified_cell_count)`.
pub fn burn_streams(
    dem: &Array2D<f64>,
    streams_path: impl AsRef<Path>,
    options: &BurnStreamsOptions,
) -> Result<(Array2D<f64>, usize)> {
    options.validate()?;
    let vec_str = streams_path.as_ref().to_str().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "non-UTF-8 stream vector path: {}",
            streams_path.as_ref().display()
        ))
    })?;
    let ds = terrio::open_vector(vec_str, &VectorOpenOptions::default())?;
    let dem_srs = if dem.projection.trim().is_empty() {
        None
    } else {
        let parsed = SpatialRef::parse(&dem.projection).map_err(|e| {
            Error::InvalidArgument(format!(
                "invalid DEM CRS metadata '{}': {e}",
                dem.projection
            ))
        })?;
        if parsed.is_unknown() {
            return Err(Error::InvalidArgument(format!(
                "unrecognized DEM CRS metadata '{}'",
                dem.projection
            )));
        }
        Some(parsed)
    };

    let mut polylines: Vec<Vec<(f64, f64)>> = Vec::new();
    for layer in &ds.layers {
        if let Some(ref d_srs) = dem_srs {
            let l_srs = SpatialRef::from_layer(layer);
            if !l_srs.is_unknown() && !d_srs.same_as(&l_srs) {
                return Err(Error::InvalidArgument(format!(
                    "stream vector layer '{}' CRS ({}) does not match DEM CRS ({})",
                    layer.name,
                    l_srs.label(),
                    d_srs.label()
                )));
            }
        }
        for feat in &layer.features {
            if let Some(ref geom) = feat.geometry {
                collect_stream_polylines(geom, &mut polylines);
            }
        }
    }

    let w = dem.width();
    let h = dem.height();
    let n = w * h;
    let mut out = dem.clone();
    if n == 0 || polylines.is_empty() {
        return Ok((out, 0));
    }

    let gt = dem.geotransform;
    let mut centerline = vec![false; n];
    let mut any_stream = false;
    for line in &polylines {
        for seg in line.windows(2) {
            let Some((c0, r0)) = world_to_col_row(&gt, seg[0].0, seg[0].1) else {
                continue;
            };
            let Some((c1, r1)) = world_to_col_row(&gt, seg[1].0, seg[1].1) else {
                continue;
            };
            for (c, r) in bresenham_segment(c0, r0, c1, r1, w, h) {
                centerline[r * w + c] = true;
                any_stream = true;
            }
        }
    }
    if !any_stream {
        return Ok((out, 0));
    }

    // Expand centerline by `stream_width_cells` if requested.
    let stream_mask = if options.stream_width_cells > 0 {
        let dist_center = euclidean_distance_2d(&centerline, w, h);
        let max_r = options.stream_width_cells as f64 + 1e-9;
        dist_center.into_iter().map(|d| d <= max_r).collect()
    } else {
        centerline
    };

    let use_buffer = options.buffer_cells > 0 && options.smooth_drop > 0.0;
    let dist_from_stream = if use_buffer {
        Some(euclidean_distance_2d(&stream_mask, w, h))
    } else {
        None
    };

    let nd = out.no_data();
    let buf_f = options.buffer_cells as f64;
    let mut modified = 0usize;
    for (idx, val) in out.data_mut().iter_mut().enumerate() {
        if !is_valid_dem(*val, nd) {
            continue;
        }
        let drop = if stream_mask[idx] {
            options.burn_depth + if use_buffer { options.smooth_drop } else { 0.0 }
        } else if let Some(ref dists) = dist_from_stream {
            let d = dists[idx];
            if d < buf_f {
                options.smooth_drop * (1.0 - d / buf_f)
            } else {
                0.0
            }
        } else {
            0.0
        };
        if drop > 0.0 {
            let mut new_val = *val - drop;
            if nd.is_finite() && new_val.rd_eq(nd) {
                new_val = new_val.next_down();
                if !new_val.is_finite() {
                    new_val = nd.next_up();
                }
            }
            *val = new_val;
            modified += 1;
        }
    }

    Ok((out, modified))
}

/// File-based wrapper around [`burn_streams`] with input/output path collision checks.
///
/// Inherits [`burn_streams`] CRS validation: non-empty DEM CRS metadata must parse as a
/// recognized CRS, CRS equivalence is verified whenever both the DEM and the stream layer
/// carry CRS metadata, and layers or DEMs lacking CRS metadata (such as a `.shp` without a
/// `.prj`) are assumed to already share the same coordinate system.
pub fn burn_streams_file(
    in_dem: impl AsRef<Path>,
    streams_path: impl AsRef<Path>,
    out_dem: impl AsRef<Path>,
    options: &BurnStreamsOptions,
) -> Result<(Array2D<f64>, usize)> {
    let in_dem = in_dem.as_ref();
    let streams_path = streams_path.as_ref();
    let out_dem = out_dem.as_ref();
    ensure_no_path_collision(in_dem, &[out_dem])?;
    ensure_no_path_collision(streams_path, &[out_dem])?;
    if let Some(parent) = out_dem.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let dem = load_raster::<f64>(in_dem, None)?;
    let (out, modified) = burn_streams(&dem, streams_path, options)?;
    save_raster(&out, out_dem, Some("burn_streams"), options.compress)?;
    Ok((out, modified))
}
