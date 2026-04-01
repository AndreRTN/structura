use structura_core::error::StructuraError;

use crate::detectors::PatternDetector;
use crate::reprojection::compute_reprojection_report;
use crate::solvers::CalibrationSolver;
use crate::types::{CalibrationDataset, CalibrationInput, CalibrationOutput};

pub struct CalibrateCameraService<D, S> {
    detector: D,
    solver: S,
}

impl<D, S> CalibrateCameraService<D, S>
where
    D: PatternDetector,
    S: CalibrationSolver,
{
    pub fn new(detector: D, solver: S) -> Self {
        Self { detector, solver }
    }

    pub fn description(&self) -> &'static str {
        "calibration service: detect -> calibrate -> reprojection"
    }

    pub fn execute(&self, input: CalibrationInput) -> Result<CalibrationOutput, StructuraError> {
        if input.image_paths.is_empty() {
            return Err(StructuraError::InvalidInput(
                "at least one image is required".to_string(),
            ));
        }

        let mut observations = Vec::with_capacity(input.image_paths.len());
        for image_path in &input.image_paths {
            if let Some(observation) = self.detector.detect(image_path, &input.pattern)? {
                observations.push(observation);
            }
        }

        if observations.is_empty() {
            return Err(StructuraError::InvalidInput(
                "no calibration patterns were detected in the provided images".to_string(),
            ));
        }

        let dataset = CalibrationDataset {
            pattern: input.pattern,
            observations,
        };
        let calibration = self.solver.calibrate(&dataset)?;
        let reprojection = compute_reprojection_report(&dataset, &calibration)?;

        Ok(CalibrationOutput {
            dataset,
            calibration,
            reprojection,
        })
    }
}
