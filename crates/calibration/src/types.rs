use std::path::PathBuf;

use structura_core::camera::{Distortion, Intrinsics};
use structura_core::math::{ImageSize, Point2, Point3};
use structura_core::pose::Pose3;

use crate::patterns::CalibrationPattern;

#[derive(Debug, Clone)]
pub struct CalibrationInput {
    pub image_paths: Vec<PathBuf>,
    pub pattern: CalibrationPattern,
}

#[derive(Debug, Clone)]
pub struct PatternObservation {
    pub image_path: PathBuf,
    pub image_size: ImageSize,
    pub image_points: Vec<Point2>,
    pub object_points: Vec<Point3>,
}

#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    pub pattern: CalibrationPattern,
    pub observations: Vec<PatternObservation>,
}

#[derive(Debug, Clone)]
pub struct CalibrationResult {
    pub intrinsics: Intrinsics,
    pub distortion: Distortion,
    pub poses: Vec<Pose3>,
    pub image_size: ImageSize,
}

#[derive(Debug, Clone)]
pub struct ReprojectionReport {
    pub per_view_rmse: Vec<f64>,
    pub mean_rmse: f64,
}

#[derive(Debug, Clone)]
pub struct CalibrationOutput {
    pub dataset: CalibrationDataset,
    pub calibration: CalibrationResult,
    pub reprojection: ReprojectionReport,
}
