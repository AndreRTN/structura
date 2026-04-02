use std::path::Path;

use anyhow::{Context, Result, ensure};
use image::RgbImage;
use opencv::{
    core::{self, KeyPoint, Mat, Vector},
    features2d,
    prelude::*,
};

pub use crate::descriptor::lowe_ratio_match_descriptor_features as lowe_ratio_match_features;
use crate::descriptor::{DescriptorFeature, DescriptorFeatureSet};

pub type SiftFeature = DescriptorFeature;
pub type SiftFeatureSet = DescriptorFeatureSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiftConfig {
    pub nfeatures: i32,
    pub n_octave_layers: i32,
    pub contrast_threshold: f64,
    pub edge_threshold: f64,
    pub sigma: f64,
    pub enable_precise_upscale: bool,
}

impl Default for SiftConfig {
    fn default() -> Self {
        Self {
            nfeatures: 0,
            n_octave_layers: 3,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            sigma: 1.6,
            enable_precise_upscale: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SiftExtractor {
    config: SiftConfig,
}

impl SiftExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: SiftConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> SiftConfig {
        self.config
    }

    pub fn extract_from_path(&self, image_path: impl AsRef<Path>) -> Result<SiftFeatureSet> {
        let gray = image::open(image_path.as_ref())
            .with_context(|| format!("failed to open image {}", image_path.as_ref().display()))?
            .to_luma8();
        self.extract_from_luma_image(gray.width() as i32, gray.height() as i32, gray.as_raw())
    }

    pub fn extract_from_rgb_image(&self, image: &RgbImage) -> Result<SiftFeatureSet> {
        let gray = image::DynamicImage::ImageRgb8(image.clone()).into_luma8();
        self.extract_from_luma_image(gray.width() as i32, gray.height() as i32, gray.as_raw())
    }

    fn extract_from_luma_image(
        &self,
        width: i32,
        height: i32,
        pixels: &[u8],
    ) -> Result<SiftFeatureSet> {
        let image = Mat::new_rows_cols_with_data(height, width, pixels)
            .context("failed to wrap grayscale image for OpenCV")?;
        let mut detector = features2d::SIFT::create(
            self.config.nfeatures,
            self.config.n_octave_layers,
            self.config.contrast_threshold,
            self.config.edge_threshold,
            self.config.sigma,
            self.config.enable_precise_upscale,
        )
        .context("failed to create OpenCV SIFT extractor")?;
        let mut keypoints = Vector::<KeyPoint>::new();
        let mut descriptors = Mat::default();

        detector
            .detect_and_compute(
                &image,
                &core::no_array(),
                &mut keypoints,
                &mut descriptors,
                false,
            )
            .context("OpenCV SIFT detect_and_compute failed")?;

        ensure!(
            descriptors.rows() == keypoints.len() as i32,
            "SIFT returned {} descriptors for {} keypoints",
            descriptors.rows(),
            keypoints.len()
        );
        ensure!(
            descriptors.cols() >= 0,
            "SIFT returned invalid descriptor matrix with {} columns",
            descriptors.cols()
        );

        let descriptor_len = descriptors.cols() as usize;
        let descriptor_data = descriptors
            .data_typed::<f32>()
            .context("failed to read SIFT descriptor matrix as f32")?;

        let features = keypoints
            .iter()
            .enumerate()
            .map(|(index, keypoint)| {
                let point = keypoint.pt();
                let start = index * descriptor_len;
                let end = start + descriptor_len;
                DescriptorFeature {
                    point: structura_geometry::point::ImagePoint::new(point.x, point.y),
                    score: keypoint.response(),
                    descriptor: descriptor_data[start..end].to_vec(),
                }
            })
            .collect();

        Ok(SiftFeatureSet { features })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn extracts_sift_features_from_checkerboard() {
        let image = RgbImage::from_fn(160, 160, |x, y| {
            let tile = ((x / 20) + (y / 20)) % 2;
            if tile == 0 {
                Rgb([24, 24, 24])
            } else {
                Rgb([232, 232, 232])
            }
        });

        let features = SiftExtractor::new().extract_from_rgb_image(&image).unwrap();

        assert!(!features.is_empty());
    }
}
