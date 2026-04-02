use structura_feature::matching::{LoweRatioConfig, PointMatch};
use structura_geometry::{
    camera::CameraExtrinsics,
    point::{ImagePoint64, WorldPoint},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialPairSelectionConfig {
    pub lowe_ratio: LoweRatioConfig,
    pub min_matches: usize,
}

impl Default for InitialPairSelectionConfig {
    fn default() -> Self {
        Self {
            lowe_ratio: LoweRatioConfig::default(),
            min_matches: 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialPairScore {
    pub overlap_score: f64,
    pub baseline_score: f64,
    pub total_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialPairCandidate {
    pub source_view_index: usize,
    pub target_view_index: usize,
    pub matches: Vec<PointMatch>,
    pub score: InitialPairScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkObservation {
    pub view_index: usize,
    pub feature_index: usize,
    pub image_point: ImagePoint64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkTrack {
    pub position: WorldPoint,
    pub observations: Vec<LandmarkObservation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureTrack {
    pub observations: Vec<LandmarkObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackTriangulationConfig {
    pub min_observations: usize,
    pub max_mean_reprojection_error: f64,
    pub require_positive_depth: bool,
}

impl Default for TrackTriangulationConfig {
    fn default() -> Self {
        Self {
            min_observations: 2,
            max_mean_reprojection_error: 2.0,
            require_positive_depth: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredView {
    pub view_index: usize,
    pub extrinsics: CameraExtrinsics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialReconstruction {
    pub views: [RegisteredView; 2],
    pub landmarks: Vec<LandmarkTrack>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NextViewRegistration {
    pub view: RegisteredView,
    pub used_correspondence_count: usize,
}
