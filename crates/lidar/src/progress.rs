//! Lightweight, zero-dependency, WASM-safe progress reporting and cooperative cancellation.
//!
//! Provides [`ProgressEvent`], [`ProgressCallback`], [`FnProgress`], [`CancellationToken`],
//! and [`ExecutionControl`] so Rust CLI tools, Python (`tqdm`) bindings, and WebAssembly
//! Web Workers can observe long-running workflows and cancel them cleanly between stages
//! or tiles.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Structured progress update emitted by long-running `lidar-rs` workflows.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProgressEvent {
    /// Short machine-readable stage identifier (e.g. `"load"`, `"extract_sinks"`, `"tile"`).
    pub stage: String,
    /// Normalized overall completion fraction in `[0.0, 1.0]`.
    pub fraction: f64,
    /// Optional human-readable progress detail message.
    pub message: Option<String>,
}

impl ProgressEvent {
    /// Creates a new [`ProgressEvent`], clamping `fraction` into `[0.0, 1.0]`
    /// (replacing non-finite values with `0.0`).
    pub fn new(stage: impl Into<String>, fraction: f64, message: Option<String>) -> Self {
        let clamped = if fraction.is_finite() {
            fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            stage: stage.into(),
            fraction: clamped,
            message,
        }
    }
}

/// Callback sink receiving [`ProgressEvent`] updates from long-running operations.
///
/// Implementations must be `Send + Sync` so parallel tile workers and background
/// executors can report progress safely across threads.
pub trait ProgressCallback: Send + Sync {
    /// Handles a single progress notification.
    fn on_progress(&self, event: &ProgressEvent);
}

/// Adapter turning any `Fn(&ProgressEvent) + Send + Sync` closure into a [`ProgressCallback`].
pub struct FnProgress<F>(pub F);

impl<F> ProgressCallback for FnProgress<F>
where
    F: Fn(&ProgressEvent) + Send + Sync,
{
    fn on_progress(&self, event: &ProgressEvent) {
        (self.0)(event);
    }
}

impl<F> fmt::Debug for FnProgress<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("FnProgress").field(&"<closure>").finish()
    }
}

/// Cooperative cancellation token shared across threads via `Arc<AtomicBool>`.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates a new, uncancelled [`CancellationToken`].
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signals cancellation to any operation polling this token or its clones.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Returns `true` if [`Self::cancel`] has been called on this token or any clone.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Returns `Err(Error::Cancelled)` if cancelled, or `Ok(())` otherwise.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Optional progress reporting and cooperative cancellation controls passed to
/// `*_with_control` workflow entry points.
#[derive(Clone, Default)]
pub struct ExecutionControl {
    /// Optional progress sink notified as stages or tiles complete.
    pub progress: Option<Arc<dyn ProgressCallback>>,
    /// Optional cancellation token polled between stages or tiles.
    pub cancel: Option<CancellationToken>,
}

impl fmt::Debug for ExecutionControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionControl")
            .field("has_progress", &self.progress.is_some())
            .field("cancel", &self.cancel)
            .finish()
    }
}

impl ExecutionControl {
    /// Creates an empty [`ExecutionControl`] with neither progress callback nor cancellation token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches a [`ProgressCallback`] sink.
    pub fn with_progress(mut self, progress: Arc<dyn ProgressCallback>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Attaches a closure as a [`ProgressCallback`] via [`FnProgress`].
    pub fn with_progress_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&ProgressEvent) + Send + Sync + 'static,
    {
        self.progress = Some(Arc::new(FnProgress(f)));
        self
    }

    /// Attaches a [`CancellationToken`].
    pub fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Emits a [`ProgressEvent`] to the registered callback (no-op if none is attached).
    pub fn report(&self, stage: impl Into<String>, fraction: f64, message: Option<String>) {
        if let Some(ref cb) = self.progress {
            let event = ProgressEvent::new(stage, fraction, message);
            cb.on_progress(&event);
        }
    }

    /// Returns `true` if a cancellation token is attached and has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }

    /// Returns `Err(Error::Cancelled)` if cancelled, or `Ok(())` otherwise.
    pub fn check_cancelled(&self) -> Result<()> {
        if let Some(ref token) = self.cancel {
            token.check()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn progress_event_clamps_fractions() {
        assert_eq!(ProgressEvent::new("a", -0.5, None).fraction, 0.0);
        assert_eq!(ProgressEvent::new("b", 1.5, None).fraction, 1.0);
        assert_eq!(ProgressEvent::new("c", f64::NAN, None).fraction, 0.0);
        assert_eq!(ProgressEvent::new("d", f64::INFINITY, None).fraction, 0.0);
        assert_eq!(ProgressEvent::new("e", 0.42, None).fraction, 0.42);
    }

    #[test]
    fn cancellation_token_clones_share_state() {
        let tok = CancellationToken::new();
        let clone = tok.clone();
        assert!(!tok.is_cancelled());
        assert!(tok.check().is_ok());

        clone.cancel();
        assert!(tok.is_cancelled());
        assert!(matches!(tok.check(), Err(Error::Cancelled)));
    }

    #[test]
    fn execution_control_reports_and_checks_cancellation() {
        let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let token = CancellationToken::new();

        let ctrl = ExecutionControl::new()
            .with_progress_fn(move |ev| {
                captured.lock().unwrap().push(ev.clone());
            })
            .with_cancel(token.clone());

        assert!(ctrl.check_cancelled().is_ok());
        ctrl.report("load", 0.1, Some("loading DEM".into()));
        ctrl.report("complete", 1.0, None);

        token.cancel();
        assert!(ctrl.is_cancelled());
        assert!(matches!(ctrl.check_cancelled(), Err(Error::Cancelled)));

        let recorded = events.lock().unwrap();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].stage, "load");
        assert_eq!(recorded[0].fraction, 0.1);
        assert_eq!(recorded[0].message.as_deref(), Some("loading DEM"));
        assert_eq!(recorded[1].stage, "complete");
        assert_eq!(recorded[1].fraction, 1.0);
    }
}
