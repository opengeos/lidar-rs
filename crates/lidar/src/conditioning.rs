//! Hydro-conditioning utilities: least-cost depression breaching (Lindsay, 2016)
//! and vector-guided culvert burning across embankments.
//!
//! Powered by [`richdem::lindsay2016`] and [`terrio`] with zero C/C++ or GDAL
//! dependencies.

use std::path::Path;
use std::str::FromStr;

use richdem::{
    Array2D, DemNum, LindsayMode, Topology, complete_breaching_lindsay2016, lindsay2016,
    load_raster, save_raster,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use terrio::crs::{SpatialRef, Transformer};
use terrio::vector::Coord;
use terrio::{
    Feature, FieldDef, FieldType, FieldValue, Geometry, GeometryType, Layer, VectorOpenOptions,
    VectorWriteOptions,
};

use crate::error::{Error, Result};
use crate::raster_ops::ensure_no_path_collision;

/// Breaching mode for [`breach_depressions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BreachMode {
    /// Breach all depressions completely regardless of depth or path length.
    Complete,
    /// Breach a depression only if its breach path satisfies both `max_depth` and `max_length`.
    #[default]
    Selective,
    /// Drill a breach path up to `max_depth` and `max_length`, leaving deeper/longer sinks intact.
    Constrained,
}

impl FromStr for BreachMode {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "complete" => Ok(Self::Complete),
            "selective" => Ok(Self::Selective),
            "constrained" => Ok(Self::Constrained),
            other => Err(Error::InvalidArgument(format!(
                "unknown breach mode '{other}'; expected complete, selective, or constrained"
            ))),
        }
    }
}

impl From<BreachMode> for LindsayMode {
    fn from(m: BreachMode) -> Self {
        match m {
            BreachMode::Complete => LindsayMode::CompleteBreaching,
            BreachMode::Selective => LindsayMode::SelectiveBreaching,
            BreachMode::Constrained => LindsayMode::ConstrainedBreaching,
        }
    }
}

/// Options controlling [`breach_depressions`] and [`breach_depressions_file`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BreachOptions {
    /// Breaching strategy (`BreachMode::Selective` by default).
    pub mode: BreachMode,
    /// Maximum breach depth in vertical elevation units (`None` = unconstrained / `f64::INFINITY`).
    pub max_depth: Option<f64>,
    /// Maximum breach channel length in grid cells (`None` = unconstrained / `u32::MAX`).
    pub max_length: Option<u32>,
    /// Enforce monotonic epsilon downslope gradients along breached paths (default `true`).
    pub eps_gradients: bool,
    /// Fill any remaining unbreached depressions after selective/constrained breaching (default `false`).
    pub fill_remaining: bool,
    /// Grid connectivity topology used when `mode == BreachMode::Complete` and `eps_gradients == false`.
    pub topology: Topology,
    /// Compress output GeoTIFF with DEFLATE (file wrapper only).
    pub compress: bool,
}

impl BreachOptions {
    /// Builds `BreachOptions` resolving mode-appropriate defaults when `max_depth` or
    /// `max_length` is omitted: `Complete` leaves omitted bounds unconstrained (`None`),
    /// whereas `Selective` and `Constrained` default omitted bounds to `1.5` and `50`.
    pub fn for_mode(mode: BreachMode, max_depth: Option<f64>, max_length: Option<u32>) -> Self {
        let def = Self::default();
        let (eff_depth, eff_len) = match mode {
            BreachMode::Complete => (max_depth, max_length),
            BreachMode::Selective | BreachMode::Constrained => {
                (max_depth.or(def.max_depth), max_length.or(def.max_length))
            }
        };
        Self {
            mode,
            max_depth: eff_depth,
            max_length: eff_len,
            ..def
        }
    }
}

impl Default for BreachOptions {
    fn default() -> Self {
        Self {
            mode: BreachMode::Selective,
            max_depth: Some(1.5),
            max_length: Some(50),
            eps_gradients: true,
            fill_remaining: false,
            topology: Topology::D8,
            compress: false,
        }
    }
}

/// Performs pure-Rust least-cost depression breaching (Lindsay, 2016) on an in-memory DEM.
///
/// Using [`BreachMode::Selective`] with small `max_depth` / `max_length` removes
/// narrow road-embankment impoundments and micro-pits while preserving true large
/// wetlands and karst dolines.
pub fn breach_depressions(dem: &Array2D<f64>, options: &BreachOptions) -> Result<Array2D<f64>> {
    if let Some(md) = options.max_depth
        && (!md.is_finite() || md < 0.0)
    {
        return Err(Error::InvalidArgument(format!(
            "max_depth must be finite and >= 0, got {md}"
        )));
    }
    let mut out = dem.clone();
    if options.mode == BreachMode::Complete && !options.eps_gradients && !options.fill_remaining {
        complete_breaching_lindsay2016(&mut out, options.topology);
        return Ok(out);
    }

    let max_depth = options.max_depth.unwrap_or(f64::INFINITY);
    let max_len = options.max_length.unwrap_or(u32::MAX);
    lindsay2016(
        &mut out,
        options.mode.into(),
        options.eps_gradients,
        options.fill_remaining,
        max_len,
        max_depth,
    )?;
    Ok(out)
}

/// File-based wrapper around [`breach_depressions`].
pub fn breach_depressions_file(
    in_dem: impl AsRef<Path>,
    out_dem: impl AsRef<Path>,
    options: &BreachOptions,
) -> Result<Array2D<f64>> {
    let in_dem = in_dem.as_ref();
    let out_dem = out_dem.as_ref();
    ensure_no_path_collision(in_dem, &[out_dem])?;
    if let Some(parent) = out_dem.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let dem = load_raster::<f64>(in_dem, None)?;
    let out = breach_depressions(&dem, options)?;
    save_raster(&out, out_dem, Some("breach_depressions"), options.compress)?;
    Ok(out)
}

/// Options controlling [`burn_culverts`] and [`burn_culverts_file`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BurnCulvertsOptions {
    /// Half-width of the carved culvert channel in grid cells (`0` = 1-cell wide, `1` = 3-cell wide).
    pub half_width_cells: usize,
    /// Additional elevation drop carved below the minimum endpoint elevation (in vertical DEM units, default `0.05`).
    pub depth_offset: f64,
    /// Search radius (cells) around culvert point features when carving across an embankment (default `5`).
    pub point_radius_cells: usize,
    /// Compress output GeoTIFF with DEFLATE (file wrapper only).
    pub compress: bool,
}

impl Default for BurnCulvertsOptions {
    fn default() -> Self {
        Self {
            half_width_cells: 0,
            depth_offset: 0.05,
            point_radius_cells: 5,
            compress: false,
        }
    }
}

#[inline]
fn world_to_col_row(gt: &[f64; 6], x: f64, y: f64) -> Option<(isize, isize)> {
    let det = gt[1] * gt[5] - gt[2] * gt[4];
    if !det.is_finite() || det.abs() < 1e-18 {
        return None;
    }
    let dx = x - gt[0];
    let dy = y - gt[3];
    let c = (dx * gt[5] - dy * gt[2]) / det;
    let r = (gt[1] * dy - gt[4] * dx) / det;
    if !c.is_finite() || !r.is_finite() {
        return None;
    }
    Some((c.floor() as isize, r.floor() as isize))
}

/// Rasterizes a line segment using Bresenham's algorithm, returning `(col, row)` grid coordinates
/// clipped to `(0..w, 0..h)`.
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
    let max_x = w as isize - 1;
    let max_y = h as isize - 1;
    // Reject segments whose bounding box is completely disjoint from the raster.
    if c0.max(c1) < 0 || c0.min(c1) > max_x || r0.max(r1) < 0 || r0.min(r1) > max_y {
        return Vec::new();
    }
    // Liang-Barsky parametric clip to [-0.5, max_x + 0.5] x [-0.5, max_y + 0.5] so huge
    // off-grid segments never iterate millions of off-grid cells. Subtract in `f64` to avoid
    // signed integer overflow on extreme coordinates.
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

/// Carves a linear culvert polyline into `dem`, interpolating linearly between the
/// upstream and downstream endpoint elevations (minus `depth_offset`) so water flows
/// monotonically across road embankments without creating artificial pits.
fn carve_polyline(
    dem: &mut Array2D<f64>,
    coords: &[(f64, f64)],
    options: &BurnCulvertsOptions,
) -> usize {
    if coords.len() < 2 {
        return 0;
    }
    let w = dem.width();
    let h = dem.height();
    let nd = dem.no_data();
    let gt = dem.geotransform;

    let mut path_cells: Vec<(usize, usize)> = Vec::new();
    for seg in coords.windows(2) {
        let Some((c0, r0)) = world_to_col_row(&gt, seg[0].0, seg[0].1) else {
            continue;
        };
        let Some((c1, r1)) = world_to_col_row(&gt, seg[1].0, seg[1].1) else {
            continue;
        };
        for cell in bresenham_segment(c0, r0, c1, r1, w, h) {
            if path_cells.last() != Some(&cell) {
                path_cells.push(cell);
            }
        }
    }
    if path_cells.is_empty() {
        return 0;
    }

    // Determine valid start and end elevations along the culvert polyline.
    let mut z_start: Option<f64> = None;
    let mut z_end: Option<f64> = None;
    for &(c, r) in &path_cells {
        let v = dem.get(c, r);
        if v.is_finite() && !v.rd_eq(nd) {
            z_start = Some(v);
            break;
        }
    }
    for &(c, r) in path_cells.iter().rev() {
        let v = dem.get(c, r);
        if v.is_finite() && !v.rd_eq(nd) {
            z_end = Some(v);
            break;
        }
    }
    let (Some(za), Some(zb)) = (z_start, z_end) else {
        return 0;
    };

    let n = path_cells.len();
    let hw = options.half_width_cells;
    let mut carved = 0usize;
    for (i, &(c, r)) in path_cells.iter().enumerate() {
        let t = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        // Keep the two endpoints tied to the receiving bed so carving never creates an outlet pit.
        let offset = if i == 0 || i + 1 == n {
            0.0
        } else {
            options.depth_offset
        };
        let target_z = za + t * (zb - za) - offset;
        let r_lo = r.saturating_sub(hw);
        let r_hi = r.saturating_add(hw).min(h - 1);
        let c_lo = c.saturating_sub(hw);
        let c_hi = c.saturating_add(hw).min(w - 1);
        for nru in r_lo..=r_hi {
            for ncu in c_lo..=c_hi {
                let cur = dem.get(ncu, nru);
                if cur.is_finite() && !cur.rd_eq(nd) && cur > target_z {
                    dem.set(ncu, nru, target_z);
                    carved += 1;
                }
            }
        }
    }
    carved
}

/// Carves a culvert point by finding the two lowest perimeter cells on opposite sides
/// of a local `(2 * radius + 1)` window and carving a straight minimum-elevation breach
/// connecting them through the culvert center.
fn carve_point(dem: &mut Array2D<f64>, x: f64, y: f64, options: &BurnCulvertsOptions) -> usize {
    let w = dem.width();
    let h = dem.height();
    let nd = dem.no_data();
    let Some((cc, cr)) = world_to_col_row(&dem.geotransform, x, y) else {
        return 0;
    };
    if cc < 0 || cr < 0 || (cc as usize) >= w || (cr as usize) >= h {
        return 0;
    }
    let rad = options.point_radius_cells.max(1) as isize;
    let r0 = (cr - rad).max(0) as usize;
    let r1 = (cr + rad).min(h as isize - 1) as usize;
    let c0 = (cc - rad).max(0) as usize;
    let c1 = (cc + rad).min(w as isize - 1) as usize;

    // Collect perimeter cells and find the lowest cell p1, then the lowest cell p2 on the
    // opposite hemisphere relative to the culvert center.
    let mut perim: Vec<(usize, usize, f64)> = Vec::new();
    for r in r0..=r1 {
        for c in c0..=c1 {
            if r != r0 && r != r1 && c != c0 && c != c1 {
                continue;
            }
            let z = dem.get(c, r);
            if z.is_finite() && !z.rd_eq(nd) {
                perim.push((c, r, z));
            }
        }
    }
    if perim.len() < 2 {
        return 0;
    }
    perim.sort_by(|a, b| a.2.total_cmp(&b.2));
    let (p1c, p1r, z1) = perim[0];
    let v1x = p1c as f64 - cc as f64;
    let v1y = p1r as f64 - cr as f64;
    // Strictly require the opposite hemisphere (`dot < 0.0`) so a point culvert cuts across the
    // embankment rather than grazing a perpendicular perimeter cell on the same ridge.
    let Some((p2c, p2r, z2)) = perim
        .iter()
        .skip(1)
        .find(|&&(c, r, _)| (c as f64 - cc as f64) * v1x + (r as f64 - cr as f64) * v1y < 0.0)
        .copied()
    else {
        return 0;
    };

    let mut path = bresenham_segment(p1c as isize, p1r as isize, cc, cr, w, h);
    for cell in bresenham_segment(cc, cr, p2c as isize, p2r as isize, w, h) {
        if path.last() != Some(&cell) {
            path.push(cell);
        }
    }
    let hw = options.half_width_cells;
    let mut carved = 0usize;
    let n = path.len();
    for (i, &(c, r)) in path.iter().enumerate() {
        let t = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        let offset = if i == 0 || i + 1 == n {
            0.0
        } else {
            options.depth_offset
        };
        let z_interp = z1 + t * (z2 - z1) - offset;
        let r_lo = r.saturating_sub(hw);
        let r_hi = r.saturating_add(hw).min(h - 1);
        let c_lo = c.saturating_sub(hw);
        let c_hi = c.saturating_add(hw).min(w - 1);
        for nru in r_lo..=r_hi {
            for ncu in c_lo..=c_hi {
                let cur = dem.get(ncu, nru);
                if cur.is_finite() && !cur.rd_eq(nd) && cur > z_interp {
                    dem.set(ncu, nru, z_interp);
                    carved += 1;
                }
            }
        }
    }
    carved
}

/// Burns vector-guided culverts (polylines or points) into an in-memory DEM.
///
/// Accepts any vector format supported by `terrio` (`.gpkg`, `.shp`, `.geojson`, etc.)
/// in the same CRS as `dem`. Returns `(conditioned_dem, carved_cell_count)`.
pub fn burn_culverts(
    dem: &Array2D<f64>,
    culverts_vector: impl AsRef<Path>,
    options: &BurnCulvertsOptions,
) -> Result<(Array2D<f64>, usize)> {
    if !options.depth_offset.is_finite() || options.depth_offset < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "depth_offset must be finite and >= 0, got {}",
            options.depth_offset
        )));
    }
    let vec_str = culverts_vector.as_ref().to_str().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "non-UTF-8 culvert vector path: {}",
            culverts_vector.as_ref().display()
        ))
    })?;
    let ds = terrio::open_vector(vec_str, &VectorOpenOptions::default())?;
    let dem_srs = if dem.projection.is_empty() {
        None
    } else {
        SpatialRef::parse(&dem.projection)
            .ok()
            .filter(|s| !s.is_unknown())
    };
    let mut out = dem.clone();
    let mut total_carved = 0usize;

    for layer in &ds.layers {
        if let Some(ref d_srs) = dem_srs {
            let l_srs = SpatialRef::from_layer(layer);
            if !l_srs.is_unknown() && !d_srs.same_as(&l_srs) {
                return Err(Error::InvalidArgument(format!(
                    "culvert vector layer '{}' CRS ({}) does not match DEM CRS ({})",
                    layer.name,
                    l_srs.label(),
                    d_srs.label()
                )));
            }
        }
        for feat in &layer.features {
            let Some(ref geom) = feat.geometry else {
                continue;
            };
            total_carved += burn_geometry(&mut out, geom, options);
        }
    }
    Ok((out, total_carved))
}

fn burn_geometry(dem: &mut Array2D<f64>, geom: &Geometry, options: &BurnCulvertsOptions) -> usize {
    match geom {
        Geometry::Point(c) => carve_point(dem, c.x, c.y, options),
        Geometry::MultiPoint(pts) => pts
            .iter()
            .map(|c| carve_point(dem, c.x, c.y, options))
            .sum(),
        Geometry::LineString(coords) => {
            let pts: Vec<(f64, f64)> = coords.iter().map(|c| (c.x, c.y)).collect();
            carve_polyline(dem, &pts, options)
        }
        Geometry::MultiLineString(lines) => lines
            .iter()
            .map(|line| {
                let pts: Vec<(f64, f64)> = line.iter().map(|c| (c.x, c.y)).collect();
                carve_polyline(dem, &pts, options)
            })
            .sum(),
        Geometry::GeometryCollection(geoms) => {
            geoms.iter().map(|g| burn_geometry(dem, g, options)).sum()
        }
        _ => 0,
    }
}

/// File-based wrapper around [`burn_culverts`].
pub fn burn_culverts_file(
    in_dem: impl AsRef<Path>,
    culverts_vector: impl AsRef<Path>,
    out_dem: impl AsRef<Path>,
    options: &BurnCulvertsOptions,
) -> Result<(Array2D<f64>, usize)> {
    let in_dem = in_dem.as_ref();
    let out_dem = out_dem.as_ref();
    ensure_no_path_collision(in_dem, &[out_dem])?;
    if let Some(parent) = out_dem.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let dem = load_raster::<f64>(in_dem, None)?;
    let (out, carved) = burn_culverts(&dem, culverts_vector, options)?;
    save_raster(&out, out_dem, Some("burn_culverts"), options.compress)?;
    Ok((out, carved))
}

/// Options controlling [`synthesize_culverts`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SynthesizeCulvertsOptions {
    /// Half-length (in meters) extracted along each stream polyline on both sides of a
    /// road crossing (total target breach segment length is `2 * half_length_m`; default `25.0`).
    pub half_length_m: f64,
    /// Minimum Euclidean distance (in meters) between distinct synthesized culvert crossings;
    /// clustered intersections closer than this threshold are deduplicated (default `15.0`).
    pub snap_distance_m: f64,
    /// Minimum stream polyline length (in meters) eligible for crossing synthesis (default `5.0`).
    pub min_stream_length_m: f64,
    /// Minimum extracted culvert breach segment length (in meters) kept in output (default `6.0`).
    pub min_segment_length_m: f64,
    /// Optional target projected CRS (e.g. `"EPSG:26914"`). When `None`, uses the stream layer's
    /// projected CRS or automatically selects the local WGS 84 UTM zone if inputs are geographic.
    pub target_crs: Option<String>,
}

impl Default for SynthesizeCulvertsOptions {
    fn default() -> Self {
        Self {
            half_length_m: 25.0,
            snap_distance_m: 15.0,
            min_stream_length_m: 5.0,
            min_segment_length_m: 6.0,
            target_crs: None,
        }
    }
}

fn collect_polylines(geom: &Geometry, out: &mut Vec<Vec<(f64, f64)>>) {
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
                collect_polylines(g, out);
            }
        }
        _ => {}
    }
}

fn transform_polylines(
    lines: &mut [Vec<(f64, f64)>],
    src: &SpatialRef,
    dst: &SpatialRef,
) -> Result<()> {
    if src.is_unknown() || dst.is_unknown() || src.same_as(dst) {
        return Ok(());
    }
    let tx = Transformer::new(src, dst)?;
    for line in lines.iter_mut() {
        tx.transform_slice(line)?;
    }
    Ok(())
}

/// Computes the exact 2D intersection point between closed line segments `ab` and `cd`.
fn segment_intersection(
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    d: (f64, f64),
) -> Option<(f64, f64, f64)> {
    let r = (b.0 - a.0, b.1 - a.1);
    let s = (d.0 - c.0, d.1 - c.1);
    let denom = r.0 * s.1 - r.1 * s.0;
    if denom.abs() < 1e-12 {
        return None;
    }
    let qmp = (c.0 - a.0, c.1 - a.1);
    let t = (qmp.0 * s.1 - qmp.1 * s.0) / denom;
    let u = (qmp.0 * r.1 - qmp.1 * r.0) / denom;
    if (-1e-9..=1.0 + 1e-9).contains(&t) && (-1e-9..=1.0 + 1e-9).contains(&u) {
        let tc = t.clamp(0.0, 1.0);
        Some((a.0 + tc * r.0, a.1 + tc * r.1, tc))
    } else {
        None
    }
}

fn polyline_cumulative_lengths(pts: &[(f64, f64)]) -> Vec<f64> {
    let mut cum = Vec::with_capacity(pts.len());
    cum.push(0.0);
    let mut acc = 0.0;
    for w in pts.windows(2) {
        acc += (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
        cum.push(acc);
    }
    cum
}

fn interpolate_along_polyline(pts: &[(f64, f64)], cum: &[f64], dist: f64) -> (f64, f64) {
    if dist <= 0.0 {
        return pts[0];
    }
    let total = *cum.last().unwrap_or(&0.0);
    if dist >= total {
        return *pts.last().unwrap_or(&pts[0]);
    }
    let idx = cum.partition_point(|&d| d < dist).saturating_sub(1);
    let d0 = cum[idx];
    let d1 = cum[idx + 1];
    let seg_len = d1 - d0;
    let t = if seg_len > 1e-12 {
        ((dist - d0) / seg_len).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let p0 = pts[idx];
    let p1 = pts[idx + 1];
    (p0.0 + t * (p1.0 - p0.0), p0.1 + t * (p1.1 - p0.1))
}

fn extract_polyline_substring(
    pts: &[(f64, f64)],
    cum: &[f64],
    s0: f64,
    s1: f64,
) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    out.push(interpolate_along_polyline(pts, cum, s0));
    for (i, &d) in cum.iter().enumerate() {
        if d > s0 + 1e-9 && d < s1 - 1e-9 {
            out.push(pts[i]);
        }
    }
    out.push(interpolate_along_polyline(pts, cum, s1));
    out
}

fn utm_epsg_from_lon_lat(lon: f64, lat: f64) -> u32 {
    let zone = (((lon + 180.0) / 6.0).floor() as i32).clamp(0, 59) as u32 + 1;
    if lat >= 0.0 {
        32600 + zone
    } else {
        32700 + zone
    }
}

/// Synthesizes deterministic road-stream culvert breach polylines by intersecting stream
/// flowlines (e.g., USGS NHD) with road/railway centerlines (e.g., USGS National Transportation).
///
/// Clustered intersections within [`SynthesizeCulvertsOptions::snap_distance_m`] are deduplicated,
/// and for each crossing a breach polyline of length `±half_length_m` is extracted along the
/// local stream direction so it can be passed directly into [`burn_culverts`].
///
/// When input layers are geographic (e.g., `EPSG:4326`) and `target_crs` is `None`, synthesis
/// automatically runs in the local WGS 84 UTM zone so distances are exact meters, and outputs
/// the synthesized culverts in that projected CRS (or `target_crs` if specified).
///
/// Returns the number of synthesized culvert polylines written to `out_culverts`.
pub fn synthesize_culverts(
    streams_vector: impl AsRef<Path>,
    roads_vector: impl AsRef<Path>,
    out_culverts: impl AsRef<Path>,
    options: &SynthesizeCulvertsOptions,
) -> Result<usize> {
    if !options.half_length_m.is_finite() || options.half_length_m <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "half_length_m must be finite and > 0, got {}",
            options.half_length_m
        )));
    }
    if !options.snap_distance_m.is_finite() || options.snap_distance_m < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "snap_distance_m must be finite and >= 0, got {}",
            options.snap_distance_m
        )));
    }

    let streams_path = streams_vector.as_ref();
    let roads_path = roads_vector.as_ref();
    let out_path = out_culverts.as_ref();
    ensure_no_path_collision(streams_path, &[out_path])?;
    ensure_no_path_collision(roads_path, &[out_path])?;
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let streams_str = streams_path.to_str().ok_or_else(|| {
        Error::InvalidArgument(format!("non-UTF-8 path: {}", streams_path.display()))
    })?;
    let roads_str = roads_path.to_str().ok_or_else(|| {
        Error::InvalidArgument(format!("non-UTF-8 path: {}", roads_path.display()))
    })?;
    let out_str = out_path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("non-UTF-8 path: {}", out_path.display())))?;

    let streams_ds = terrio::open_vector(streams_str, &VectorOpenOptions::default())?;
    let roads_ds = terrio::open_vector(roads_str, &VectorOpenOptions::default())?;

    let streams_srs = streams_ds
        .layers
        .first()
        .map(SpatialRef::from_layer)
        .unwrap_or_else(|| SpatialRef::from_epsg(4326).expect("EPSG:4326"));
    let roads_srs = roads_ds
        .layers
        .first()
        .map(SpatialRef::from_layer)
        .unwrap_or_else(|| streams_srs.clone());

    let mut stream_lines: Vec<Vec<(f64, f64)>> = Vec::new();
    for layer in &streams_ds.layers {
        for feat in &layer.features {
            if let Some(ref g) = feat.geometry {
                collect_polylines(g, &mut stream_lines);
            }
        }
    }
    let mut road_lines: Vec<Vec<(f64, f64)>> = Vec::new();
    for layer in &roads_ds.layers {
        for feat in &layer.features {
            if let Some(ref g) = feat.geometry {
                collect_polylines(g, &mut road_lines);
            }
        }
    }

    // Determine working metric CRS
    let work_srs = if let Some(ref crs_str) = options.target_crs {
        SpatialRef::parse(crs_str)?
    } else if !streams_srs.is_unknown() && !streams_srs.is_geographic() {
        streams_srs.clone()
    } else {
        // Compute mean lon/lat across stream vertices to choose local UTM zone
        let (mut sx, mut sy, mut cnt) = (0.0f64, 0.0f64, 0usize);
        for line in &stream_lines {
            for &(x, y) in line {
                if x.is_finite() && y.is_finite() {
                    sx += x;
                    sy += y;
                    cnt += 1;
                }
            }
        }
        if cnt > 0 && streams_srs.is_geographic() {
            let epsg = utm_epsg_from_lon_lat(sx / cnt as f64, sy / cnt as f64);
            SpatialRef::from_epsg(epsg)?
        } else {
            streams_srs.clone()
        }
    };

    transform_polylines(&mut stream_lines, &streams_srs, &work_srs)?;
    transform_polylines(&mut road_lines, &roads_srs, &work_srs)?;

    // Build a uniform spatial grid index over road segments for O(1) crossing lookup
    #[derive(Clone, Copy)]
    struct RoadSeg {
        a: (f64, f64),
        b: (f64, f64),
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    }
    let mut road_segs: Vec<RoadSeg> = Vec::new();
    for line in &road_lines {
        for w in line.windows(2) {
            road_segs.push(RoadSeg {
                a: w[0],
                b: w[1],
                min_x: w[0].0.min(w[1].0),
                min_y: w[0].1.min(w[1].1),
                max_x: w[0].0.max(w[1].0),
                max_y: w[0].1.max(w[1].1),
            });
        }
    }

    let cell_size = (options.half_length_m * 10.0).max(250.0);
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, seg) in road_segs.iter().enumerate() {
        let gx0 = (seg.min_x / cell_size).floor() as i64;
        let gx1 = (seg.max_x / cell_size).floor() as i64;
        let gy0 = (seg.min_y / cell_size).floor() as i64;
        let gy1 = (seg.max_y / cell_size).floor() as i64;
        // Guard against extreme degenerate segments spanning huge grids
        if (gx1 - gx0).saturating_mul(gy1 - gy0) > 10_000 {
            continue;
        }
        for gx in gx0..=gx1 {
            for gy in gy0..=gy1 {
                grid.entry((gx, gy)).or_default().push(idx);
            }
        }
    }

    struct SynthesizedSegment {
        stream_idx: usize,
        length_m: f64,
        pts: Vec<(f64, f64)>,
    }
    let snap_sq = options.snap_distance_m * options.snap_distance_m;
    let mut seen_pts: Vec<(f64, f64)> = Vec::new();
    let mut culvert_geoms: Vec<SynthesizedSegment> = Vec::new();
    let mut visited_seg_stamp: Vec<u32> = vec![0; road_segs.len()];
    let mut stamp: u32 = 0;

    for (stream_idx, sline) in stream_lines.iter().enumerate() {
        let cum = polyline_cumulative_lengths(sline);
        let total_len = *cum.last().unwrap_or(&0.0);
        if total_len < options.min_stream_length_m {
            continue;
        }
        for (si, w) in sline.windows(2).enumerate() {
            let a = w[0];
            let b = w[1];
            let seg_len = cum[si + 1] - cum[si];
            let smin_x = a.0.min(b.0);
            let smax_x = a.0.max(b.0);
            let smin_y = a.1.min(b.1);
            let smax_y = a.1.max(b.1);
            let gx0 = (smin_x / cell_size).floor() as i64;
            let gx1 = (smax_x / cell_size).floor() as i64;
            let gy0 = (smin_y / cell_size).floor() as i64;
            let gy1 = (smax_y / cell_size).floor() as i64;
            if (gx1 - gx0).saturating_mul(gy1 - gy0) > 10_000 {
                continue;
            }
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                visited_seg_stamp.fill(0);
                stamp = 1;
            }
            for gx in gx0..=gx1 {
                for gy in gy0..=gy1 {
                    let Some(bucket) = grid.get(&(gx, gy)) else {
                        continue;
                    };
                    for &ridx in bucket {
                        if visited_seg_stamp[ridx] == stamp {
                            continue;
                        }
                        visited_seg_stamp[ridx] = stamp;
                        let rseg = road_segs[ridx];
                        if rseg.max_x < smin_x
                            || rseg.min_x > smax_x
                            || rseg.max_y < smin_y
                            || rseg.min_y > smax_y
                        {
                            continue;
                        }
                        let Some((ix, iy, t)) = segment_intersection(a, b, rseg.a, rseg.b) else {
                            continue;
                        };
                        if seen_pts
                            .iter()
                            .any(|&(px, py)| (px - ix).powi(2) + (py - iy).powi(2) < snap_sq)
                        {
                            continue;
                        }
                        let dist_along = cum[si] + t * seg_len;
                        let s0 = (dist_along - options.half_length_m).max(0.0);
                        let s1 = (dist_along + options.half_length_m).min(total_len);
                        if s1 - s0 < options.min_segment_length_m {
                            continue;
                        }
                        seen_pts.push((ix, iy));
                        let sub = extract_polyline_substring(sline, &cum, s0, s1);
                        culvert_geoms.push(SynthesizedSegment {
                            stream_idx,
                            length_m: s1 - s0,
                            pts: sub,
                        });
                    }
                }
            }
        }
    }

    let mut out_layer = Layer::new("culverts".to_string()).with_geom_type(GeometryType::LineString);
    if !work_srs.is_unknown() {
        work_srs.apply_to_layer(&mut out_layer);
    }
    out_layer
        .schema
        .add_field(FieldDef::new("culvert_id".to_string(), FieldType::Integer));
    out_layer
        .schema
        .add_field(FieldDef::new("stream_idx".to_string(), FieldType::Integer));
    out_layer
        .schema
        .add_field(FieldDef::new("length_m".to_string(), FieldType::Float));
    for (idx, seg) in culvert_geoms.iter().enumerate() {
        let fid = (idx + 1) as u64;
        out_layer.features.push(Feature {
            fid,
            geometry: Some(Geometry::LineString(
                seg.pts.iter().map(|&(x, y)| Coord::xy(x, y)).collect(),
            )),
            attributes: vec![
                FieldValue::Integer(fid as i64),
                FieldValue::Integer(seg.stream_idx as i64),
                FieldValue::Float(seg.length_m),
            ],
        });
    }
    let count = out_layer.features.len();
    terrio::write_vector(&[out_layer], out_str, &VectorWriteOptions::default())?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use terrio::vector::Coord;
    use terrio::{Feature, GeometryType, Layer, VectorWriteOptions};

    #[test]
    fn selective_breach_removes_shallow_pit_keeps_deep_basin() {
        // 19x19 surface at 20.0 m with a lower outlet margin at col 0 (19.0 m) and two 3x3 basins:
        // - shallow impoundment (19.6 m, depth 0.4 m <= 1.0 m) centered at (4, 9)
        // - deep basin (15.0 m, depth 5.0 m > 1.0 m) centered at (13, 9)
        let w = 19usize;
        let h = 19usize;
        let mut z = vec![20.0f64; w * h];
        for r in 0..h {
            z[r * w] = 19.0;
        }
        for r in 8..=10 {
            for c in 3..=5 {
                z[r * w + c] = 19.6;
            }
            for c in 12..=14 {
                z[r * w + c] = 15.0;
            }
        }

        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        let opts = BreachOptions {
            mode: BreachMode::Selective,
            max_depth: Some(1.0),
            max_length: Some(5),
            eps_gradients: false,
            fill_remaining: false,
            ..Default::default()
        };
        let breached = breach_depressions(&dem, &opts).unwrap();
        // Shallow impoundment should be breached out toward col 0 (at least one barrier cell lowered below 20.0)
        let carved_shallow = (7..=11)
            .flat_map(|r| (1..=2).map(move |c| (c, r)))
            .any(|(c, r)| breached.get(c, r) < 20.0);
        assert!(carved_shallow, "expected shallow basin to breach to col 0");
        // Deep basin floor remains untouched at 15.0 because breaching it requires 5.0 m > 1.0 m
        assert!((breached.get(13, 9) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn burn_culverts_carves_road_embankment_polyline() {
        // 11x11 flat valley at 10.0 m split north-south by a 15.0 m road embankment at col 5
        let w = 11usize;
        let h = 11usize;
        let mut z = vec![10.0f64; w * h];
        for r in 0..h {
            z[r * w + 5] = 15.0;
        }
        let mut dem = Array2D::from_vec(w, h, z).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [0.0, 1.0, 0.0, 11.0, 0.0, -1.0];

        let tmp = tempfile::tempdir().unwrap();
        let culvert_path = tmp.path().join("culverts.geojson");
        let mut layer = Layer::new("culverts".to_string()).with_geom_type(GeometryType::LineString);
        layer.features.push(Feature {
            fid: 1,
            geometry: Some(Geometry::LineString(vec![
                Coord::xy(3.5, 5.5),
                Coord::xy(7.5, 5.5),
            ])),
            attributes: vec![],
        });
        terrio::write_vector(
            &[layer],
            culvert_path.to_str().unwrap(),
            &VectorWriteOptions::default(),
        )
        .unwrap();

        let (conditioned, carved) =
            burn_culverts(&dem, &culvert_path, &BurnCulvertsOptions::default()).unwrap();
        assert!(carved >= 1);
        // Embankment at col 5, row 5 must be breached down to ~9.95 m
        assert!(conditioned.get(5, 5) < 10.0);
        // Embankment away from culvert (row 1) remains at 15.0 m
        assert!((conditioned.get(5, 1) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn burn_culverts_rejects_crs_mismatch() {
        let mut dem = Array2D::from_vec(5, 5, vec![10.0f64; 25]).unwrap();
        dem.set_no_data(-9999.0);
        dem.geotransform = [500_000.0, 1.0, 0.0, 4_000_005.0, 0.0, -1.0];
        dem.projection = "EPSG:26916".to_string();

        let tmp = tempfile::tempdir().unwrap();
        let culvert_path = tmp.path().join("culverts_4326.gpkg");
        let mut layer = Layer::new("culverts".to_string()).with_geom_type(GeometryType::LineString);
        SpatialRef::parse("EPSG:4326")
            .unwrap()
            .apply_to_layer(&mut layer);
        layer.features.push(Feature {
            fid: 1,
            geometry: Some(Geometry::LineString(vec![
                Coord::xy(-86.0, 37.0),
                Coord::xy(-85.99, 37.0),
            ])),
            attributes: vec![],
        });
        terrio::write_vector(
            &[layer],
            culvert_path.to_str().unwrap(),
            &VectorWriteOptions::default(),
        )
        .unwrap();

        let err = burn_culverts(&dem, &culvert_path, &BurnCulvertsOptions::default());
        assert!(
            matches!(err, Err(Error::InvalidArgument(ref msg)) if msg.contains("CRS")),
            "expected InvalidArgument mentioning CRS mismatch, got: {err:?}"
        );
    }

    #[test]
    fn synthesize_culverts_intersects_snaps_and_burns() {
        let tmp = tempfile::tempdir().unwrap();
        let streams_path = tmp.path().join("streams.gpkg");
        let roads_path = tmp.path().join("roads.gpkg");
        let culverts_path = tmp.path().join("culverts.gpkg");

        let utm = SpatialRef::from_epsg(26914).unwrap();
        let mut s_layer =
            Layer::new("streams".to_string()).with_geom_type(GeometryType::LineString);
        utm.apply_to_layer(&mut s_layer);
        // East-west stream from x=500000 to x=500200 at y=5200100
        s_layer.features.push(Feature {
            fid: 1,
            geometry: Some(Geometry::LineString(vec![
                Coord::xy(500_000.0, 5_200_100.0),
                Coord::xy(500_200.0, 5_200_100.0),
            ])),
            attributes: vec![],
        });
        terrio::write_vector(
            &[s_layer],
            streams_path.to_str().unwrap(),
            &VectorWriteOptions::default(),
        )
        .unwrap();

        let mut r_layer = Layer::new("roads".to_string()).with_geom_type(GeometryType::LineString);
        utm.apply_to_layer(&mut r_layer);
        // Two close parallel lanes 5 m apart at x=500080 and x=500085 (should snap to 1 culvert),
        // plus a second crossing 60 m away at x=500145 (distinct culvert).
        for (fid, x) in [(1, 500_080.0), (2, 500_085.0), (3, 500_145.0)] {
            r_layer.features.push(Feature {
                fid,
                geometry: Some(Geometry::LineString(vec![
                    Coord::xy(x, 5_200_050.0),
                    Coord::xy(x, 5_200_150.0),
                ])),
                attributes: vec![],
            });
        }
        terrio::write_vector(
            &[r_layer],
            roads_path.to_str().unwrap(),
            &VectorWriteOptions::default(),
        )
        .unwrap();

        let count = synthesize_culverts(
            &streams_path,
            &roads_path,
            &culverts_path,
            &SynthesizeCulvertsOptions {
                half_length_m: 20.0,
                snap_distance_m: 15.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(count, 2, "expected 2 snapped culverts, got {count}");

        // Verify output CRS matches UTM and segments have ~40 m length
        let ds = terrio::open_vector(
            culverts_path.to_str().unwrap(),
            &VectorOpenOptions::default(),
        )
        .unwrap();
        assert_eq!(ds.layers[0].features.len(), 2);
        let out_srs = SpatialRef::from_layer(&ds.layers[0]);
        assert!(out_srs.same_as(&utm));
    }

    #[test]
    fn synthesize_culverts_handles_wgs84_inputs_automatically() {
        let tmp = tempfile::tempdir().unwrap();
        let streams_path = tmp.path().join("streams_wgs84.geojson");
        let roads_path = tmp.path().join("roads_wgs84.geojson");
        let culverts_path = tmp.path().join("culverts_utm.gpkg");

        let wgs = SpatialRef::from_epsg(4326).unwrap();
        let mut s_layer =
            Layer::new("streams".to_string()).with_geom_type(GeometryType::LineString);
        wgs.apply_to_layer(&mut s_layer);
        s_layer.features.push(Feature {
            fid: 1,
            geometry: Some(Geometry::LineString(vec![
                Coord::xy(-99.205, 47.100),
                Coord::xy(-99.195, 47.100),
            ])),
            attributes: vec![],
        });
        terrio::write_vector(
            &[s_layer],
            streams_path.to_str().unwrap(),
            &VectorWriteOptions::default(),
        )
        .unwrap();

        let mut r_layer = Layer::new("roads".to_string()).with_geom_type(GeometryType::LineString);
        wgs.apply_to_layer(&mut r_layer);
        r_layer.features.push(Feature {
            fid: 1,
            geometry: Some(Geometry::LineString(vec![
                Coord::xy(-99.200, 47.095),
                Coord::xy(-99.200, 47.105),
            ])),
            attributes: vec![],
        });
        terrio::write_vector(
            &[r_layer],
            roads_path.to_str().unwrap(),
            &VectorWriteOptions::default(),
        )
        .unwrap();

        let count = synthesize_culverts(
            &streams_path,
            &roads_path,
            &culverts_path,
            &SynthesizeCulvertsOptions::default(),
        )
        .unwrap();
        assert_eq!(count, 1);

        let ds = terrio::open_vector(
            culverts_path.to_str().unwrap(),
            &VectorOpenOptions::default(),
        )
        .unwrap();
        let out_srs = SpatialRef::from_layer(&ds.layers[0]);
        assert!(!out_srs.is_geographic(), "expected projected metric CRS");
    }
}
