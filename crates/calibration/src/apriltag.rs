use anyhow::{Result, anyhow};
use kornia::image::{ImageSize, allocator::ImageAllocator};
use kornia_apriltag::{
    AprilTagDecoder, DecodeTagsConfig, decoder::Detection, family::TagFamilyKind,
};

use crate::detector::{Detector, GrayImage, ImagePoint};

#[derive(Debug, Clone, PartialEq)]
pub struct AprilTagDetection {
    pub id: u16,
    pub family: TagFamilyKind,
    pub hamming: u8,
    pub decision_margin: f32,
    pub center: ImagePoint,
    pub corners: [ImagePoint; 4],
}

impl From<Detection> for AprilTagDetection {
    fn from(value: Detection) -> Self {
        Self {
            id: value.id,
            family: value.tag_family_kind,
            hamming: value.hamming,
            decision_margin: value.decision_margin,
            center: ImagePoint::new(value.center.x, value.center.y),
            corners: value
                .quad
                .corners
                .map(|corner| ImagePoint::new(corner.x, corner.y)),
        }
    }
}

pub struct AprilTagDetector {
    config: DecodeTagsConfig,
    decoder: Option<AprilTagDecoder>,
    decoder_size: Option<ImageSize>,
}

impl AprilTagDetector {
    pub fn new(config: DecodeTagsConfig) -> Self {
        Self {
            config,
            decoder: None,
            decoder_size: None,
        }
    }

    pub fn all() -> Self {
        Self::new(DecodeTagsConfig::all())
    }

    pub fn with_families(tag_family_kinds: Vec<TagFamilyKind>) -> Self {
        Self::new(DecodeTagsConfig::new(tag_family_kinds))
    }

    pub fn config(&self) -> &DecodeTagsConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut DecodeTagsConfig {
        &mut self.config
    }

    pub fn add_family(&mut self, family: TagFamilyKind) {
        self.config.add(family.into());
        self.decoder = None;
        self.decoder_size = None;
    }

    fn ensure_decoder(&mut self, image_size: ImageSize) -> Result<&mut AprilTagDecoder> {
        let needs_new_decoder = self.decoder.is_none() || self.decoder_size != Some(image_size);

        if needs_new_decoder {
            self.decoder = Some(
                AprilTagDecoder::new(self.config.clone(), image_size)
                    .map_err(|error| anyhow!("failed to create apriltag decoder: {error}"))?,
            );
            self.decoder_size = Some(image_size);
        }

        self.decoder
            .as_mut()
            .ok_or_else(|| anyhow!("apriltag decoder was not initialized"))
    }
}

impl Detector for AprilTagDetector {
    type Detection = AprilTagDetection;

    fn detect<A>(&mut self, image: &GrayImage<A>) -> Result<Vec<Self::Detection>>
    where
        A: ImageAllocator,
    {
        let image_size = image.size();
        let decoder = self.ensure_decoder(image_size)?;
        decoder.clear();

        Ok(decoder
            .decode(image)
            .map_err(|error| anyhow!("failed to decode apriltag image: {error}"))?
            .into_iter()
            .map(Into::into)
            .collect())
    }
}

pub fn detect_apriltag_corners<A>(
    image: &GrayImage<A>,
    detector: &mut AprilTagDetector,
) -> Result<Vec<AprilTagDetection>>
where
    A: ImageAllocator,
{
    detector.detect(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia::image::{Image, ImageSize, allocator::CpuAllocator};

    #[test]
    fn detects_from_kornia_gray_image() {
        let image = Image::new(
            ImageSize {
                width: 32,
                height: 32,
            },
            vec![0_u8; 32 * 32],
            CpuAllocator,
        )
        .unwrap();

        let mut detector = AprilTagDetector::with_families(vec![TagFamilyKind::Tag36H11]);
        let detections = detector.detect(&image).unwrap();

        assert!(detections.is_empty());
    }
}
