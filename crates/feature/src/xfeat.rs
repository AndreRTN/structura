use std::path::Path;

use anyhow::{Context, Result, anyhow};
use image::{RgbImage, imageops::FilterType};
use ndarray::{Array1, Array4, ArrayView2, ArrayView3, Axis, Ix4};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use structura_geometry::point::ImagePoint;

pub use crate::descriptor::{
    DescriptorFeature as XfeatFeature, DescriptorFeatureSet as XfeatFeatureSet,
    lowe_ratio_match_descriptor_features as lowe_ratio_match_features,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XfeatConfig {
    pub detection_threshold: f32,
    pub top_k: usize,
}

impl Default for XfeatConfig {
    fn default() -> Self {
        Self {
            detection_threshold: 0.05,
            top_k: 4096,
        }
    }
}

#[derive(Debug)]
pub struct XfeatExtractor {
    session: Session,
    config: XfeatConfig,
}

impl XfeatExtractor {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(model_path, XfeatConfig::default())
    }

    pub fn with_config(model_path: impl AsRef<Path>, config: XfeatConfig) -> Result<Self> {
        Ok(Self {
            session: build_session(model_path.as_ref())?,
            config,
        })
    }

    pub fn config(&self) -> XfeatConfig {
        self.config
    }

    pub fn extract_from_path(&mut self, image_path: impl AsRef<Path>) -> Result<XfeatFeatureSet> {
        let rgb = image::open(image_path.as_ref())
            .with_context(|| format!("failed to open image {}", image_path.as_ref().display()))?
            .to_rgb8();
        self.extract_from_rgb_image(&rgb)
    }

    pub fn extract_from_rgb_image(&mut self, image: &RgbImage) -> Result<XfeatFeatureSet> {
        let prepared = prepare_xfeat_input(image)?;
        run_xfeat_backbone(
            &mut self.session,
            prepared.tensor.view(),
            self.config,
            prepared.scale_x,
            prepared.scale_y,
        )
    }
}

#[derive(Debug, Clone)]
struct PreparedImage {
    tensor: Array4<f32>,
    scale_x: f32,
    scale_y: f32,
}

#[derive(Debug, Clone)]
struct Candidate {
    x: usize,
    y: usize,
    score: f32,
}

fn build_session(model_path: &Path) -> Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)
        .with_context(|| format!("failed to load ONNX model at {}", model_path.display()))
}

fn prepare_xfeat_input(image: &RgbImage) -> Result<PreparedImage> {
    let src_width = image.width() as usize;
    let src_height = image.height() as usize;
    let width = (src_width / 32) * 32;
    let height = (src_height / 32) * 32;
    anyhow::ensure!(
        width > 0 && height > 0,
        "image is too small after 32-pixel alignment"
    );

    let resized = if width == src_width && height == src_height {
        image.clone()
    } else {
        image::imageops::resize(image, width as u32, height as u32, FilterType::Triangle)
    };

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));
    resized.enumerate_pixels().for_each(|(x, y, pixel)| {
        let x = x as usize;
        let y = y as usize;
        tensor[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
        tensor[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
        tensor[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
    });

    Ok(PreparedImage {
        tensor,
        scale_x: src_width as f32 / width as f32,
        scale_y: src_height as f32 / height as f32,
    })
}

fn run_xfeat_backbone(
    session: &mut Session,
    image: ndarray::ArrayView4<'_, f32>,
    config: XfeatConfig,
    scale_x: f32,
    scale_y: f32,
) -> Result<XfeatFeatureSet> {
    let inputs = ort::inputs![TensorRef::from_array_view(image)?];
    let outputs = session.run(inputs)?;

    let feats = outputs["feats"]
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix4>()
        .context("feats output did not have rank 4")?
        .to_owned();
    let keypoint_logits = outputs["keypoint_logits"]
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix4>()
        .context("keypoint_logits output did not have rank 4")?
        .to_owned();
    let reliability = outputs["reliability"]
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix4>()
        .context("reliability output did not have rank 4")?
        .to_owned();

    decode_sparse_features(
        &feats,
        &keypoint_logits,
        &reliability,
        config.detection_threshold,
        config.top_k,
        scale_x,
        scale_y,
    )
}

fn decode_sparse_features(
    feats: &Array4<f32>,
    keypoint_logits: &Array4<f32>,
    reliability: &Array4<f32>,
    detection_threshold: f32,
    top_k: usize,
    scale_x: f32,
    scale_y: f32,
) -> Result<XfeatFeatureSet> {
    let batch = feats.shape()[0];
    anyhow::ensure!(batch == 1, "xfeat decoder currently assumes batch size 1");

    let heatmap = keypoint_logits_to_heatmap(keypoint_logits)?;
    let candidates = nms_candidates(
        &heatmap.index_axis(Axis(0), 0).index_axis(Axis(0), 0),
        detection_threshold,
    );
    let scored = score_candidates(
        &candidates,
        &heatmap.index_axis(Axis(0), 0).index_axis(Axis(0), 0),
        &reliability.index_axis(Axis(0), 0).index_axis(Axis(0), 0),
    );
    let best = top_k_candidates(scored, top_k);
    let descriptor_map = feats.index_axis(Axis(0), 0);

    Ok(XfeatFeatureSet {
        features: best
            .iter()
            .map(|candidate| {
                let descriptor = bilinear_sample_descriptor(
                    &descriptor_map,
                    candidate.x as f32,
                    candidate.y as f32,
                )
                .to_vec();

                XfeatFeature {
                    point: ImagePoint::new(
                        candidate.x as f32 * scale_x,
                        candidate.y as f32 * scale_y,
                    ),
                    score: candidate.score,
                    descriptor,
                }
            })
            .collect(),
    })
}

fn keypoint_logits_to_heatmap(keypoint_logits: &Array4<f32>) -> Result<Array4<f32>> {
    let [batch, channels, h8, w8]: [usize; 4] = keypoint_logits
        .shape()
        .try_into()
        .map_err(|_| anyhow!("keypoint_logits must have rank 4"))?;
    anyhow::ensure!(
        channels >= 64,
        "keypoint_logits must expose at least 64 channels"
    );

    let mut out = Array4::<f32>::zeros((batch, 1, h8 * 8, w8 * 8));
    (0..batch).for_each(|b| {
        (0..h8).for_each(|gy| {
            (0..w8).for_each(|gx| {
                let logits = (0..64)
                    .map(|channel| keypoint_logits[[b, channel, gy, gx]])
                    .collect::<Vec<_>>();
                let probs = softmax_1d(&logits);
                (0..8).for_each(|ky| {
                    (0..8).for_each(|kx| {
                        let channel = ky * 8 + kx;
                        out[[b, 0, gy * 8 + ky, gx * 8 + kx]] = probs[channel];
                    });
                });
            });
        });
    });

    Ok(out)
}

fn softmax_1d(values: &[f32]) -> Vec<f32> {
    let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = values
        .iter()
        .map(|value| (value - max_value).exp())
        .collect::<Vec<_>>();
    let denominator = exponentials.iter().copied().sum::<f32>().max(1e-12);

    exponentials
        .into_iter()
        .map(|value| value / denominator)
        .collect()
}

fn nms_candidates(heatmap: &ArrayView2<'_, f32>, threshold: f32) -> Vec<(usize, usize)> {
    let (height, width) = heatmap.dim();
    let radius = 2usize;

    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let center = heatmap[[y, x]];
            center > threshold
                && (y.saturating_sub(radius)..(y + radius + 1).min(height)).all(|yy| {
                    (x.saturating_sub(radius)..(x + radius + 1).min(width))
                        .all(|xx| heatmap[[yy, xx]] <= center || (xx == x && yy == y))
                })
        })
        .collect()
}

fn score_candidates(
    candidates: &[(usize, usize)],
    heatmap: &ArrayView2<'_, f32>,
    reliability: &ArrayView2<'_, f32>,
) -> Vec<Candidate> {
    candidates
        .iter()
        .map(|&(x, y)| Candidate {
            x,
            y,
            score: heatmap[[y, x]]
                * bilinear_sample_scalar(reliability, x as f32 / 8.0, y as f32 / 8.0),
        })
        .collect()
}

fn top_k_candidates(mut candidates: Vec<Candidate>, top_k: usize) -> Vec<Candidate> {
    candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
    candidates.truncate(top_k.min(candidates.len()));
    candidates
}

fn bilinear_sample_scalar(map: &ArrayView2<'_, f32>, x: f32, y: f32) -> f32 {
    let (height, width) = map.dim();
    let x = x.clamp(0.0, width.saturating_sub(1) as f32);
    let y = y.clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let v00 = map[[y0, x0]];
    let v10 = map[[y0, x1]];
    let v01 = map[[y1, x0]];
    let v11 = map[[y1, x1]];
    let top = v00 * (1.0 - dx) + v10 * dx;
    let bottom = v01 * (1.0 - dx) + v11 * dx;

    top * (1.0 - dy) + bottom * dy
}

fn bilinear_sample_descriptor(map: &ArrayView3<'_, f32>, x: f32, y: f32) -> Array1<f32> {
    let channels = map.shape()[0];
    let mut descriptor = Array1::<f32>::zeros(channels);
    (0..channels).for_each(|channel| {
        descriptor[channel] =
            bilinear_sample_scalar(&map.index_axis(Axis(0), channel), x / 8.0, y / 8.0);
    });

    let norm = descriptor
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    descriptor.mapv_inplace(|value| value / norm);
    descriptor
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use std::path::PathBuf;

    use crate::matching::LoweRatioConfig;

    #[test]
    fn decodes_sparse_features_from_synthetic_logits() {
        let mut feats = Array4::<f32>::zeros((1, 4, 1, 1));
        feats[[0, 0, 0, 0]] = 3.0;
        feats[[0, 1, 0, 0]] = 4.0;

        let mut keypoint_logits = Array4::<f32>::from_elem((1, 65, 1, 1), -10.0);
        keypoint_logits[[0, 9, 0, 0]] = 10.0;

        let reliability = Array4::<f32>::from_elem((1, 1, 1, 1), 1.0);

        let features =
            decode_sparse_features(&feats, &keypoint_logits, &reliability, 0.01, 16, 1.0, 1.0)
                .unwrap();

        assert_eq!(features.len(), 1);
        assert_eq!(features.features[0].point, ImagePoint::new(1.0, 1.0));
        assert!(features.features[0].score > 0.9);
        assert!((features.features[0].descriptor[0] - 0.6).abs() < 1e-4);
        assert!((features.features[0].descriptor[1] - 0.8).abs() < 1e-4);
    }

    #[test]
    fn lowe_ratio_matches_synthetic_feature_sets() {
        let source = XfeatFeatureSet {
            features: vec![
                XfeatFeature {
                    point: ImagePoint::new(10.0, 12.0),
                    score: 0.9,
                    descriptor: vec![1.0, 0.0, 0.0, 0.0],
                },
                XfeatFeature {
                    point: ImagePoint::new(30.0, 40.0),
                    score: 0.85,
                    descriptor: vec![0.0, 1.0, 0.0, 0.0],
                },
            ],
        };
        let target = XfeatFeatureSet {
            features: vec![
                XfeatFeature {
                    point: ImagePoint::new(13.0, 14.0),
                    score: 0.91,
                    descriptor: vec![0.99, 0.01, 0.0, 0.0],
                },
                XfeatFeature {
                    point: ImagePoint::new(33.0, 42.0),
                    score: 0.82,
                    descriptor: vec![0.02, 0.98, 0.0, 0.0],
                },
                XfeatFeature {
                    point: ImagePoint::new(100.0, 120.0),
                    score: 0.3,
                    descriptor: vec![0.5, 0.5, 0.5, 0.5],
                },
            ],
        };

        let matches =
            lowe_ratio_match_features(&source, &target, LoweRatioConfig::new(0.8)).unwrap();

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].source_index, 0);
        assert_eq!(matches[0].target_index, 0);
        assert_eq!(matches[1].source_index, 1);
        assert_eq!(matches[1].target_index, 1);
    }

    #[test]
    fn extracts_and_matches_translated_checkerboard_with_onnx_backbone() {
        let model_path = workspace_root().join("models").join("xfeat_backbone.onnx");
        let mut extractor = XfeatExtractor::with_config(
            &model_path,
            XfeatConfig {
                detection_threshold: 0.03,
                top_k: 1024,
            },
        )
        .unwrap();
        let image0 = synthetic_checkerboard_image(256, 256, 32, (0, 0));
        let image1 = synthetic_checkerboard_image(256, 256, 32, (8, 6));

        let features0 = extractor.extract_from_rgb_image(&image0).unwrap();
        let features1 = extractor.extract_from_rgb_image(&image1).unwrap();
        let matches =
            lowe_ratio_match_features(&features0, &features1, LoweRatioConfig::new(0.9)).unwrap();

        assert!(
            features0.len() >= 32,
            "expected rich features in image0, got {}",
            features0.len()
        );
        assert!(
            features1.len() >= 32,
            "expected rich features in image1, got {}",
            features1.len()
        );
        assert!(
            matches.len() >= 12,
            "expected enough matches, got {}",
            matches.len()
        );

        let consistent_translations = matches
            .iter()
            .filter(|matched| {
                let dx = matched.target_point.x - matched.source_point.x;
                let dy = matched.target_point.y - matched.source_point.y;
                (dx - 8.0).abs() <= 3.0 && (dy - 6.0).abs() <= 3.0
            })
            .count();

        assert!(
            consistent_translations >= 8,
            "expected translated correspondences, got {consistent_translations} / {}",
            matches.len()
        );
    }

    fn synthetic_checkerboard_image(
        width: u32,
        height: u32,
        cell_size: u32,
        offset: (u32, u32),
    ) -> RgbImage {
        let mut image = RgbImage::from_pixel(width, height, Rgb([245, 245, 245]));

        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .for_each(|(x, y)| {
                let shifted_x = x.saturating_sub(offset.0);
                let shifted_y = y.saturating_sub(offset.1);
                let checker = ((shifted_x / cell_size) + (shifted_y / cell_size)).is_multiple_of(2);
                let pixel = if checker {
                    Rgb([25, 25, 25])
                } else {
                    Rgb([230, 230, 230])
                };
                image.put_pixel(x, y, pixel);
            });

        image
    }

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }
}
