//! Level-set surface inundation dynamics simulation.
//!
//! Pure-Rust port of `6_Simulate Inundation.py` from `opengeos/lidar`'s ArcGIS
//! Hydrology Analyst toolbox.
//!
//! Given a sink elevation raster (`sink.tif`) and a catchment hierarchy raster
//! (`catchment_hir.tif`, where each depression pixel stores its contributing
//! catchment area in map units squared), [`simulate_inundation`] computes the
//! time-to-inundation for every depression cell in a **single parallel level-set pass**
//! (avoiding redundant re-slicing across timesteps) and emits time-series inundation
//! GeoTIFFs (`flood_01.tif`, `flood_02.tif`, …), `fill_time.tif`, and
//! `inundation_summary.csv`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use richdem::{Array2D, DemNum, load_raster, save_raster};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::raster_ops::ensure_no_path_collision;
use crate::region::{region_group, region_props_xy};

/// Maximum vertical slices permitted per region to guard against tiny intervals.
const MAX_SLICES_PER_REGION: usize = 100_000;
/// Maximum number of output simulation frames permitted in one run.
const MAX_SIMULATION_STEPS: usize = 10_000;

/// Options controlling [`simulate_inundation`] and [`simulate_inundation_in_memory`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulateInundationOptions {
    /// Minimum number of pixels for a valid depression slice (default `1000`).
    pub min_size: usize,
    /// Minimum depth for a valid depression slice (default `0.2`).
    pub min_depth: f64,
    /// Vertical level-set slicing interval (positive; sliced top-down, default `0.2`).
    pub interval: f64,
    /// Rainfall intensity in **cm/h** (default `5.0`, converted to m/h internally).
    pub rain_intensity_cm_h: f64,
    /// Total rainfall duration in **hours** (default `50.0`).
    pub rain_duration_h: f64,
    /// Output simulation time step in **hours** (default `1.0`).
    pub time_step_h: f64,
    /// Compress output GeoTIFFs with DEFLATE.
    pub compress: bool,
}

impl Default for SimulateInundationOptions {
    fn default() -> Self {
        Self {
            min_size: 1000,
            min_depth: 0.2,
            interval: 0.2,
            rain_intensity_cm_h: 5.0,
            rain_duration_h: 50.0,
            time_step_h: 1.0,
            compress: false,
        }
    }
}

/// Summary metrics for one time step of [`simulate_inundation`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InundationStep {
    /// 1-based iteration index.
    pub step: usize,
    /// Elapsed rainfall duration in hours.
    pub time_hours: f64,
    /// Number of inundated pixels at this time step.
    pub inundated_cells: usize,
    /// Inundated surface area in map units squared.
    pub inundated_area: f64,
    /// Fraction of total valid sink area inundated (`0.0..=1.0`).
    pub inundated_fraction: f64,
    /// Path to `flood_<step>.tif`.
    pub raster_path: PathBuf,
}

/// Output paths and per-step summary produced by [`simulate_inundation`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulateInundationResult {
    /// Time-ordered paths to `flood_01.tif`, `flood_02.tif`, …
    pub frame_paths: Vec<PathBuf>,
    /// Path to `fill_time.tif` (`f64`, hours until each sink pixel inundates; `-1.0` = dry/NoData).
    pub fill_time_path: PathBuf,
    /// Path to `inundation_summary.csv`.
    pub summary_csv_path: PathBuf,
    /// Number of top-level sink regions simulated.
    pub num_regions: usize,
    /// Per-timestep inundation summary records.
    pub steps: Vec<InundationStep>,
}

fn validate_options(opts: &SimulateInundationOptions) -> Result<usize> {
    if !opts.min_depth.is_finite() || opts.min_depth < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "min_depth must be finite and >= 0, got {}",
            opts.min_depth
        )));
    }
    if !opts.interval.is_finite() || opts.interval == 0.0 {
        return Err(Error::InvalidArgument(format!(
            "interval must be finite and non-zero, got {}",
            opts.interval
        )));
    }
    if !opts.rain_intensity_cm_h.is_finite() || opts.rain_intensity_cm_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "rain_intensity_cm_h must be finite and > 0, got {}",
            opts.rain_intensity_cm_h
        )));
    }
    if !opts.rain_duration_h.is_finite() || opts.rain_duration_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "rain_duration_h must be finite and > 0, got {}",
            opts.rain_duration_h
        )));
    }
    if !opts.time_step_h.is_finite() || opts.time_step_h <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "time_step_h must be finite and > 0, got {}",
            opts.time_step_h
        )));
    }
    let n_steps_f = ((opts.rain_duration_h + 1e-9) / opts.time_step_h).floor();
    if !n_steps_f.is_finite() || n_steps_f < 1.0 || n_steps_f > MAX_SIMULATION_STEPS as f64 {
        return Err(Error::InvalidArgument(format!(
            "simulation produces {n_steps_f} steps; must be between 1 and {MAX_SIMULATION_STEPS}"
        )));
    }
    Ok(n_steps_f as usize)
}

fn affine_metrics(gt: &[f64; 6]) -> (f64, f64, f64) {
    let res_x = gt[1].hypot(gt[4]);
    let res_y = gt[2].hypot(gt[5]);
    let det = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    let rx = if res_x > 0.0 && res_x.is_finite() {
        res_x
    } else {
        1.0
    };
    let ry = if res_y > 0.0 && res_y.is_finite() {
        res_y
    } else {
        1.0
    };
    let area = if det > 0.0 && det.is_finite() {
        det
    } else {
        rx * ry
    };
    (rx, ry, area)
}

struct RegionFillTask {
    bbox: (usize, usize, usize, usize),
    roi_elev: Vec<f64>,
    roi_catchment: Vec<f64>,
}

/// Computes the minimum inundation time (in hours) for every cell in a single sink region
/// in one top-down level-set pass.
fn region_fill_times(
    task: &RegionFillTask,
    options: &SimulateInundationOptions,
    metrics: (f64, f64, f64),
) -> Result<Vec<f64>> {
    let min_size = options.min_size;
    let min_depth = options.min_depth;
    let step = options.interval.abs();
    let rain_m_h = options.rain_intensity_cm_h / 100.0;
    let (res_x, res_y, cell_area) = metrics;
    let (r0, c0, r1, c1) = task.bbox;
    let bh = r1 - r0;
    let bw = c1 - c0;
    let mut img = task.roi_elev.clone();
    let mut fill_time = vec![f64::INFINITY; bw * bh];

    let mut min_elev = f64::INFINITY;
    let mut max_elev = f64::NEG_INFINITY;
    for &v in &img {
        if v.is_finite() {
            if v < min_elev {
                min_elev = v;
            }
            if v > max_elev {
                max_elev = v;
            }
        }
    }
    if !min_elev.is_finite() || !max_elev.is_finite() {
        return Ok(fill_time);
    }

    let elev_span = (max_elev - min_elev).max(0.0);
    let est_slices = (elev_span / step).ceil();
    if !est_slices.is_finite() || est_slices > MAX_SLICES_PER_REGION as f64 {
        return Err(Error::InvalidArgument(format!(
            "interval={step} requires {est_slices} slices (exceeds MAX_SLICES_PER_REGION={MAX_SLICES_PER_REGION})"
        )));
    }

    let mut slice_idx = 0usize;
    loop {
        let elev = max_elev - (slice_idx as f64) * step;
        if slice_idx > 0 && elev <= min_elev {
            break;
        }
        slice_idx += 1;

        for v in &mut img {
            if v.is_finite() && *v > elev {
                *v = f64::NAN;
            }
        }

        let (label_objects, nb_labels) = region_group(&img, bw, bh, min_size, |v| v.is_finite());
        if nb_labels == 0 {
            break;
        }

        let objects = region_props_xy(
            &label_objects,
            nb_labels,
            &img,
            bw,
            bh,
            res_x,
            res_y,
            cell_area,
        );
        for obj in &objects {
            if obj.max_depth < min_depth {
                continue;
            }
            let (fr, fc) = obj.first_pixel;
            let raw_cat = task.roi_catchment[fr * bw + fc];
            // If catchment raster is 0/missing at this cell, fall back to direct rainfall
            // over the depression's own footprint so valid depressions still simulate cleanly.
            let cat_area = if raw_cat.is_finite() && raw_cat > 0.0 {
                raw_cat
            } else {
                obj.area
            };
            if cat_area <= 0.0 {
                continue;
            }
            let t_fill = obj.volume / (cat_area * rain_m_h);
            let (or0, oc0, or1, oc1) = obj.bbox;
            for r in or0..or1 {
                let off = r * bw;
                for c in oc0..oc1 {
                    if label_objects[off + c] == obj.label && t_fill < fill_time[off + c] {
                        fill_time[off + c] = t_fill;
                    }
                }
            }
        }
    }

    Ok(fill_time)
}

/// Computes the per-pixel minimum time to inundation (hours) in memory.
///
/// Pixels that never inundate (or lie outside valid sink regions) are assigned `-1.0`
/// (the output raster's NoData value).
pub fn simulate_inundation_in_memory(
    sink: &Array2D<f64>,
    catchment_area: &Array2D<f64>,
    options: &SimulateInundationOptions,
) -> Result<(Array2D<f64>, usize)> {
    let _ = validate_options(options)?;
    let w = sink.width();
    let h = sink.height();
    if catchment_area.width() != w || catchment_area.height() != h {
        return Err(Error::DimensionMismatch(format!(
            "catchment raster ({}x{}) does not match sink raster ({w}x{h})",
            catchment_area.width(),
            catchment_area.height()
        )));
    }
    for i in 0..6 {
        let diff = (sink.geotransform[i] - catchment_area.geotransform[i]).abs();
        let tol = 1e-6 * sink.geotransform[i].abs().max(1.0);
        if diff > tol {
            return Err(Error::InvalidArgument(format!(
                "catchment geotransform differs from sink raster at index {i}: {} vs {}",
                catchment_area.geotransform[i], sink.geotransform[i]
            )));
        }
    }

    let metrics = affine_metrics(&sink.geotransform);
    let nd = sink.no_data();
    let c_nd = catchment_area.no_data();

    let (labels, nb_regions_u32) = region_group(sink.data(), w, h, options.min_size, |v| {
        v.is_finite() && !v.rd_eq(nd)
    });
    let nb_regions = nb_regions_u32 as usize;

    let mut fill_time_data = vec![-1.0f64; w * h];
    if nb_regions == 0 {
        let mut out = Array2D::from_vec(w, h, fill_time_data)?;
        out.copy_metadata_from(sink);
        out.set_no_data(-1.0);
        return Ok((out, 0));
    }

    // Compute per-region bounding boxes.
    let mut bboxes = vec![(usize::MAX, usize::MAX, 0usize, 0usize); nb_regions];
    for r in 0..h {
        let off = r * w;
        for c in 0..w {
            let lbl = labels[off + c];
            if lbl > 0 {
                let b = &mut bboxes[(lbl - 1) as usize];
                if r < b.0 {
                    b.0 = r;
                }
                if c < b.1 {
                    b.1 = c;
                }
                if r + 1 > b.2 {
                    b.2 = r + 1;
                }
                if c + 1 > b.3 {
                    b.3 = c + 1;
                }
            }
        }
    }

    let sdata = sink.data();
    let cdata = catchment_area.data();
    let tasks: Vec<RegionFillTask> = (0..nb_regions)
        .map(|ri| {
            let lbl = (ri + 1) as u32;
            let bbox = bboxes[ri];
            let (r0, c0, r1, c1) = bbox;
            let bh = r1 - r0;
            let bw = c1 - c0;
            let mut roi_elev = vec![f64::NAN; bw * bh];
            let mut roi_catchment = vec![0.0f64; bw * bh];
            for r in r0..r1 {
                let g_off = r * w;
                let l_off = (r - r0) * bw;
                for c in c0..c1 {
                    if labels[g_off + c] == lbl {
                        roi_elev[l_off + (c - c0)] = sdata[g_off + c];
                        let cv = cdata[g_off + c];
                        roi_catchment[l_off + (c - c0)] =
                            if cv.is_finite() && !cv.rd_eq(c_nd) && cv > 0.0 {
                                cv
                            } else {
                                0.0
                            };
                    }
                }
            }
            RegionFillTask {
                bbox,
                roi_elev,
                roi_catchment,
            }
        })
        .collect();

    #[cfg(feature = "parallel")]
    let region_results: Result<Vec<Vec<f64>>> = tasks
        .par_iter()
        .map(|t| region_fill_times(t, options, metrics))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let region_results: Result<Vec<Vec<f64>>> = tasks
        .iter()
        .map(|t| region_fill_times(t, options, metrics))
        .collect();

    let region_results = region_results?;
    for (task, times) in tasks.iter().zip(region_results) {
        let (r0, c0, r1, c1) = task.bbox;
        let bw = c1 - c0;
        for r in r0..r1 {
            let g_off = r * w;
            let l_off = (r - r0) * bw;
            for c in c0..c1 {
                let t = times[l_off + (c - c0)];
                if t.is_finite() {
                    fill_time_data[g_off + c] = t;
                }
            }
        }
    }

    let mut out = Array2D::from_vec(w, h, fill_time_data)?;
    out.copy_metadata_from(sink);
    out.set_no_data(-1.0);
    Ok((out, nb_regions))
}

/// Simulates surface inundation dynamics over time and writes `flood_*.tif`,
/// `fill_time.tif`, and `inundation_summary.csv` into `out_dir`.
///
/// Pure-Rust port of `6_Simulate Inundation.py`.
pub fn simulate_inundation(
    in_sink: impl AsRef<Path>,
    in_catchment: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    options: &SimulateInundationOptions,
) -> Result<SimulateInundationResult> {
    let n_steps = validate_options(options)?;
    let in_sink = in_sink.as_ref();
    let in_catchment = in_catchment.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let fill_time_path = out_dir.join("fill_time.tif");
    let summary_csv_path = out_dir.join("inundation_summary.csv");

    let digits = (n_steps as f64).log10().floor() as usize + 1;
    let frame_paths: Vec<PathBuf> = (1..=n_steps)
        .map(|k| out_dir.join(format!("flood_{k:0digits$}.tif")))
        .collect();

    let mut protected: Vec<&Path> = Vec::with_capacity(frame_paths.len() + 2);
    protected.push(&fill_time_path);
    protected.push(&summary_csv_path);
    for p in &frame_paths {
        protected.push(p.as_path());
    }
    ensure_no_path_collision(in_sink, &protected)?;
    ensure_no_path_collision(in_catchment, &protected)?;

    let sink = load_raster::<f64>(in_sink, None)?;
    let catchment = load_raster::<f64>(in_catchment, None)?;
    let (fill_time, num_regions) = simulate_inundation_in_memory(&sink, &catchment, options)?;

    // Remove stale flood_*.tif(f) frames from any prior run in out_dir so rerun outputs
    // with a different step count cannot interleave with old frames.
    if let Ok(entries) = std::fs::read_dir(out_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file()
                && let Some(name) = p.file_name().and_then(|n| n.to_str())
            {
                let lower = name.to_ascii_lowercase();
                if lower.starts_with("flood_")
                    && (lower.ends_with(".tif") || lower.ends_with(".tiff"))
                {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
    }

    save_raster(
        &fill_time,
        &fill_time_path,
        Some("simulate_inundation: fill_time_hours"),
        options.compress,
    )?;

    let w = sink.width();
    let h = sink.height();
    let (_, _, cell_area) = affine_metrics(&sink.geotransform);
    let nd = sink.no_data();
    let total_sink_cells = sink
        .data()
        .iter()
        .filter(|&&v| v.is_finite() && !v.rd_eq(nd))
        .count();

    let ft_data = fill_time.data();
    let mut steps = Vec::with_capacity(n_steps);
    let mut flood_buf = vec![0u32; w * h];

    for (i, frame_path) in frame_paths.iter().enumerate() {
        let step_num = i + 1;
        let t_hours = (step_num as f64) * options.time_step_h;
        let mut inundated_cells = 0usize;
        for (dst, &ft) in flood_buf.iter_mut().zip(ft_data.iter()) {
            if ft >= 0.0 && ft.is_finite() && ft <= t_hours {
                *dst = 1;
                inundated_cells += 1;
            } else {
                *dst = 0;
            }
        }
        let mut frame_arr = Array2D::from_vec(w, h, flood_buf.clone())?;
        frame_arr.copy_metadata_from(&sink);
        frame_arr.set_no_data(0);
        save_raster(
            &frame_arr,
            frame_path,
            Some(&format!("simulate_inundation(t={t_hours:.2}h)")),
            options.compress,
        )?;

        let inundated_area = inundated_cells as f64 * cell_area;
        let inundated_fraction = if total_sink_cells > 0 {
            inundated_cells as f64 / total_sink_cells as f64
        } else {
            0.0
        };
        steps.push(InundationStep {
            step: step_num,
            time_hours: t_hours,
            inundated_cells,
            inundated_area,
            inundated_fraction,
            raster_path: frame_path.clone(),
        });
    }

    write_inundation_summary_csv(&steps, &summary_csv_path)?;

    Ok(SimulateInundationResult {
        frame_paths,
        fill_time_path,
        summary_csv_path,
        num_regions,
        steps,
    })
}

/// Writes `inundation_summary.csv`.
pub fn write_inundation_summary_csv(
    steps: &[InundationStep],
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(
        w,
        "step,time_hours,inundated_cells,inundated_area,inundated_fraction"
    )?;
    for s in steps {
        writeln!(
            w,
            "{},{:.4},{},{:.4},{:.6}",
            s.step, s.time_hours, s.inundated_cells, s.inundated_area, s.inundated_fraction
        )?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_bowl_inundates_monotonically_from_bottom_up() {
        // 10x10 sink with a deep inner 4x4 core (elev 10.0) inside a shallower rim (elev 12.0).
        let mut sink_data = vec![12.0; 100];
        for r in 3..7 {
            for c in 3..7 {
                sink_data[r * 10 + c] = 10.0;
            }
        }
        let mut sink = Array2D::from_vec(10, 10, sink_data).unwrap();
        sink.set_no_data(0.0);
        let mut cat = Array2D::from_vec(10, 10, vec![200.0; 100]).unwrap();
        cat.set_no_data(0.0);

        let opts = SimulateInundationOptions {
            min_size: 4,
            min_depth: 0.0,
            interval: 0.5,
            rain_intensity_cm_h: 5.0,
            rain_duration_h: 20.0,
            time_step_h: 2.0,
            compress: false,
        };
        let (ft, n_reg) = simulate_inundation_in_memory(&sink, &cat, &opts).unwrap();
        assert_eq!(n_reg, 1);
        let core_t = ft.get(4, 4);
        let rim_t = ft.get(0, 0);
        assert!(core_t >= 0.0 && rim_t >= 0.0);
        assert!(
            core_t <= rim_t,
            "inner core ({core_t}h) should inundate no later than outer rim ({rim_t}h)"
        );
    }
}
