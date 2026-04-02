use anyhow::{Result, anyhow};
use chess_corners::{ChessConfig, CornerDescriptor, find_chess_corners_u8};
use kornia::image::allocator::ImageAllocator;

use crate::detector::{Detector, GrayImage, ImagePoint};

#[derive(Debug, Clone, PartialEq)]
pub struct ChessboardCorner {
    pub point: ImagePoint,
    pub response: f32,
    pub orientation: f32,
}

impl From<CornerDescriptor> for ChessboardCorner {
    fn from(value: CornerDescriptor) -> Self {
        Self {
            point: ImagePoint::new(value.x, value.y),
            response: value.response,
            orientation: value.orientation,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChessCornersDetector {
    config: ChessConfig,
}

impl ChessCornersDetector {
    pub fn new(config: ChessConfig) -> Self {
        Self { config }
    }

    pub fn single_scale() -> Self {
        Self::new(ChessConfig::single_scale())
    }

    pub fn multiscale() -> Self {
        Self::new(ChessConfig::multiscale())
    }

    pub fn config(&self) -> &ChessConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut ChessConfig {
        &mut self.config
    }
}

impl Detector for ChessCornersDetector {
    type Detection = ChessboardCorner;

    fn detect<A>(&mut self, image: &GrayImage<A>) -> Result<Vec<Self::Detection>>
    where
        A: ImageAllocator,
    {
        let width = u32::try_from(image.width())
            .map_err(|_| anyhow!("image width does not fit into u32"))?;
        let height = u32::try_from(image.height())
            .map_err(|_| anyhow!("image height does not fit into u32"))?;

        Ok(
            find_chess_corners_u8(image.as_slice(), width, height, &self.config)
                .into_iter()
                .map(Into::into)
                .collect(),
        )
    }
}

pub fn detect_chessboard_corners<A>(
    image: &GrayImage<A>,
    detector: &mut ChessCornersDetector,
) -> Result<Vec<ChessboardCorner>>
where
    A: ImageAllocator,
{
    detector.detect(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia::image::{ImageSize, allocator::CpuAllocator};

    #[test]
    fn detects_from_kornia_gray_image() {
        let image = GrayImage::new(
            ImageSize {
                width: 32,
                height: 32,
            },
            vec![0_u8; 32 * 32],
            CpuAllocator,
        )
        .unwrap();

        let mut detector = ChessCornersDetector::single_scale();
        let corners = detector.detect(&image).unwrap();

        assert!(corners.is_empty());
    }
}
