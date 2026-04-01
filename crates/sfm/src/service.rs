use structura_core::error::StructuraError;
use structura_vision::features::extractor::{FeatureExtractor, FeatureMatcher};

use crate::types::{SfmInput, SfmResult};

pub struct RunSfmService<E, M> {
    extractor: E,
    matcher: M,
}

impl<E, M> RunSfmService<E, M>
where
    E: FeatureExtractor,
    M: FeatureMatcher,
{
    pub fn new(extractor: E, matcher: M) -> Self {
        Self { extractor, matcher }
    }

    pub fn description(&self) -> &'static str {
        "sfm service: extract -> match -> estimate -> triangulate -> optimize"
    }

    pub fn execute(&self, input: SfmInput) -> Result<SfmResult, StructuraError> {
        if input.images.is_empty() {
            return Err(StructuraError::InvalidInput(
                "at least one image is required".to_string(),
            ));
        }

        if input.images.len() < 2 {
            return Err(StructuraError::InvalidInput(
                "at least two images are required for matching".to_string(),
            ));
        }

        let features0 = self.extractor.extract(&input.images[0])?;
        let features1 = self.extractor.extract(&input.images[1])?;
        let _matches = self.matcher.match_features(&features0, &features1)?;

        Err(StructuraError::Unsupported(
            "SfM pipeline is not implemented yet",
        ))
    }
}
