use structura_calibration::dataset::load_images_from_directory;
use structura_calibration::detectors::opencv::OpenCvPatternDetector;
use structura_calibration::patterns::{CalibrationPattern, ChessboardSpec};
use structura_calibration::service::CalibrateCameraService;
use structura_calibration::solvers::opencv::OpenCvCalibrationSolver;
use structura_core::error::StructuraError;
use structura_sfm::service::RunSfmService;
use structura_vision::features::extractor::{FeatureExtractor, FeatureMatcher};
use structura_vision::features::types::{FeatureSet, MatchSet};

fn main() {
    let calibration =
        CalibrateCameraService::new(OpenCvPatternDetector::default(), OpenCvCalibrationSolver::default());
    let sfm = RunSfmService::new(StubFeatureExtractor, StubFeatureMatcher);

    println!("Structura workspace OK");
    println!("calibration: {}", calibration.description());
    println!("sfm: {}", sfm.description());

    let dataset_dir = std::path::Path::new("data/chessboard");
    if let Ok(image_paths) = load_images_from_directory(dataset_dir) {
        let input = structura_calibration::types::CalibrationInput {
            image_paths,
            pattern: CalibrationPattern::Chessboard(ChessboardSpec {
                columns: 9,
                rows: 6,
                square_size_meters: 0.025,
            }),
        };
        println!("calibration dataset images: {}", input.image_paths.len());
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct StubFeatureExtractor;

impl FeatureExtractor for StubFeatureExtractor {
    fn extract(&self, _image_path: &str) -> Result<FeatureSet, StructuraError> {
        Err(StructuraError::Unsupported(
            "feature extraction backend is not wired yet",
        ))
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct StubFeatureMatcher;

impl FeatureMatcher for StubFeatureMatcher {
    fn match_features(
        &self,
        _lhs: &FeatureSet,
        _rhs: &FeatureSet,
    ) -> Result<MatchSet, StructuraError> {
        Err(StructuraError::Unsupported(
            "feature matcher backend is not wired yet",
        ))
    }
}
