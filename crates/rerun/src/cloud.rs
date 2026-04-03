use std::{
    env,
    path::{Path, PathBuf},
    time::{Instant, SystemTime},
};

use anyhow::Result;
use structura_geometry::point::WorldPoint;

#[derive(Debug, Clone, Copy)]
pub struct SparseCloudFrame<'a> {
    pub step: i64,
    pub registered_views: usize,
    pub stage: &'a str,
    pub mean_reprojection_error: Option<f64>,
}

pub struct SparseCloudLogger {
    recording: Option<rerun::RecordingStream>,
    started_at: Instant,
}

impl SparseCloudLogger {
    pub fn from_env(application_id: &str) -> Self {
        let recording = match env::var_os("STRUCTURA_RERUN_SAVE") {
            Some(path) => Self::open_save(application_id, Path::new(&path)),
            None => match env::var_os("STRUCTURA_RERUN") {
                Some(_) => Self::open_spawn(application_id),
                None => {
                    eprintln!(
                        "rerun logging disabled: set STRUCTURA_RERUN=1 to spawn the viewer or STRUCTURA_RERUN_SAVE=/tmp/{application_id}.rrd to save the recording"
                    );
                    None
                }
            },
        };

        Self {
            recording,
            started_at: Instant::now(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.recording.is_some()
    }

    pub fn log_sparse_cloud(
        &self,
        frame: SparseCloudFrame<'_>,
        points: &[WorldPoint],
    ) -> Result<()> {
        let Some(recording) = &self.recording else {
            return Ok(());
        };

        let positions = points
            .iter()
            .map(|point| [point.x as f32, point.y as f32, point.z as f32])
            .collect::<Vec<_>>();
        let elapsed = self.started_at.elapsed();

        recording.set_time_sequence("pipeline_step", frame.step);
        recording.set_time_sequence("registered_views", frame.registered_views as i64);
        recording.set_time("elapsed", elapsed);
        recording.set_time("wall_clock", SystemTime::now());

        recording.log(
            "reconstruction/sparse_points",
            &rerun::Points3D::new(positions),
        )?;

        let status = match frame.mean_reprojection_error {
            Some(mean_reprojection_error) => format!(
                "{}: {} views, {} points, mean reproj {:.3}",
                frame.stage,
                frame.registered_views,
                points.len(),
                mean_reprojection_error
            ),
            None => format!(
                "{}: {} views, {} points",
                frame.stage,
                frame.registered_views,
                points.len()
            ),
        };
        recording.log("reconstruction/status", &rerun::TextLog::new(status))?;

        Ok(())
    }

    fn open_spawn(application_id: &str) -> Option<rerun::RecordingStream> {
        match rerun::RecordingStreamBuilder::new(application_id).spawn() {
            Ok(recording) => Some(recording),
            Err(error) => {
                eprintln!("rerun logging disabled: failed to spawn viewer: {error}");
                None
            }
        }
    }

    fn open_save(application_id: &str, path: &Path) -> Option<rerun::RecordingStream> {
        match rerun::RecordingStreamBuilder::new(application_id).save(path) {
            Ok(recording) => {
                eprintln!(
                    "rerun logging enabled: writing sparse cloud to {}",
                    path.display()
                );
                Some(recording)
            }
            Err(error) => {
                eprintln!(
                    "rerun logging disabled: failed to save recording to {}: {error}",
                    path.display()
                );
                None
            }
        }
    }
}

pub fn default_rrd_path(name: &str) -> PathBuf {
    PathBuf::from(format!("/tmp/{name}.rrd"))
}
