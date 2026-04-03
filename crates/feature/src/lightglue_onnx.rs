use std::{env, path::Path};

use anyhow::{Context, Result, anyhow, ensure};
use image::{RgbImage, imageops::FilterType};
use ndarray::{Array1, Array2, Array3, Array4, Ix1, Ix2, Ix3};
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    tensor::TensorElementType,
    value::TensorRef,
};
use structura_geometry::point::ImagePoint;

use crate::matching::PointMatch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightGlueOnnxConfig {
    pub resize_width: u32,
    pub resize_height: u32,
}

impl Default for LightGlueOnnxConfig {
    fn default() -> Self {
        Self {
            resize_width: 1024,
            resize_height: 1024,
        }
    }
}

#[derive(Debug)]
pub struct LightGlueOnnxMatcher {
    session: Session,
    config: LightGlueOnnxConfig,
}

impl LightGlueOnnxMatcher {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(model_path, LightGlueOnnxConfig::default())
    }

    pub fn with_config(
        model_path: impl AsRef<Path>,
        config: LightGlueOnnxConfig,
    ) -> Result<Self> {
        let session = build_session(model_path.as_ref())?;
        Ok(Self {
            session,
            config,
        })
    }

    pub fn match_from_rgb_images(
        &mut self,
        left: &RgbImage,
        right: &RgbImage,
    ) -> Result<Vec<PointMatch>> {
        let input = prepare_input(left, right, self.config);
        let output_metadata = self
            .session
            .outputs()
            .iter()
            .map(|outlet| format!("{}:{:?}", outlet.name(), outlet.dtype()))
            .collect::<Vec<_>>()
            .join(", ");
        let outputs = self
            .session
            .run(ort::inputs![TensorRef::from_array_view(input.view())?])?;

        let mut keypoints = None;
        let mut matches = None;
        let mut scores = None;
        for (name, value) in &outputs {
            let dtype = value.dtype();
            let shape = dtype.tensor_shape().map_or(0, |shape| shape.len());
            match (dtype.tensor_type(), shape) {
                (Some(TensorElementType::Int64), 3) => {
                    keypoints = Some(
                        value
                            .try_extract_array::<i64>()?
                            .into_dimensionality::<Ix3>()
                            .with_context(|| {
                                format!("LightGlue output `{name}` expected rank-3 keypoints")
                            })?
                            .to_owned(),
                    );
                }
                (Some(TensorElementType::Int64), 2) => {
                    matches = Some(
                        value
                            .try_extract_array::<i64>()?
                            .into_dimensionality::<Ix2>()
                            .with_context(|| {
                                format!("LightGlue output `{name}` expected rank-2 matches")
                            })?
                            .to_owned(),
                    );
                }
                (Some(TensorElementType::Float32), 1) => {
                    scores = Some(
                        value
                            .try_extract_array::<f32>()?
                            .into_dimensionality::<Ix1>()
                            .with_context(|| {
                                format!("LightGlue output `{name}` expected rank-1 scores")
                            })?
                            .to_owned(),
                    );
                }
                _ => {}
            }
        }
        let keypoints = keypoints.ok_or_else(|| {
            anyhow!("failed to locate LightGlue keypoints output; outputs: {output_metadata}")
        })?;
        let matches = matches.ok_or_else(|| {
            anyhow!("failed to locate LightGlue matches output; outputs: {output_metadata}")
        })?;
        let scores = scores.ok_or_else(|| {
            anyhow!("failed to locate LightGlue scores output; outputs: {output_metadata}")
        })?;

        decode_matches(&keypoints, &matches, &scores)
    }
}

fn build_session(model_path: &Path) -> Result<Session> {
    let builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    let builder = configure_execution_providers(builder)?;
    builder.commit_from_file(model_path).with_context(|| {
        format!(
            "failed to load LightGlue ONNX model at {}",
            model_path.display()
        )
    })
}

fn configure_execution_providers(
    builder: ort::session::builder::SessionBuilder,
) -> Result<ort::session::builder::SessionBuilder> {
    if env::var_os("STRUCTURA_ORT_ROCM").is_some() {
        let tuning = env::var_os("STRUCTURA_ORT_ROCM_TUNING").is_some();
        let hip_graph = env::var_os("STRUCTURA_ORT_ROCM_HIP_GRAPH").is_some();
        let exhaustive = env::var_os("STRUCTURA_ORT_ROCM_EXHAUSTIVE").is_some();
        eprintln!("lightglue_onnx: requesting ROCm execution provider");
        Ok(builder.with_execution_providers([
            ep::ROCm::default()
                .with_device_id(0)
                .with_exhaustive_conv_search(exhaustive)
                .with_conv_use_max_workspace(true)
                .with_hip_graph(hip_graph)
                .with_tunable_op(tuning)
                .with_tuning(tuning)
                .with_max_tuning_duration(100)
                .build(),
        ])?)
    } else if env::var_os("STRUCTURA_ORT_MIGRAPHX").is_some() {
        eprintln!("lightglue_onnx: requesting MIGraphX execution provider");
        Ok(builder.with_execution_providers([ep::MIGraphX::default().with_device_id(0).build()])?)
    } else {
        Ok(builder)
    }
}

fn prepare_input(
    left: &RgbImage,
    right: &RgbImage,
    config: LightGlueOnnxConfig,
) -> Array4<f32> {
    let left = image::imageops::resize(
        left,
        config.resize_width,
        config.resize_height,
        FilterType::Triangle,
    );
    let right = image::imageops::resize(
        right,
        config.resize_width,
        config.resize_height,
        FilterType::Triangle,
    );

    let height = config.resize_height as usize;
    let width = config.resize_width as usize;
    let mut input = Array4::<f32>::zeros((2, 1, height, width));

    for (batch, image) in [left, right].into_iter().enumerate() {
        image.enumerate_pixels().for_each(|(x, y, pixel)| {
            let x = x as usize;
            let y = y as usize;
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            input[[batch, 0, y, x]] = 0.299 * r + 0.587 * g + 0.114 * b;
        });
    }

    input
}

fn decode_matches(
    keypoints: &Array3<i64>,
    matches: &Array2<i64>,
    scores: &Array1<f32>,
) -> Result<Vec<PointMatch>> {
    ensure!(
        keypoints.shape()[0] == 2,
        "LightGlue wrapper expects a single image pair batched as 2 images"
    );
    ensure!(
        matches.shape().get(1).copied() == Some(3),
        "LightGlue matches output must have shape [M, 3]"
    );
    ensure!(
        scores.len() == matches.shape()[0],
        "LightGlue produced {} scores for {} matches",
        scores.len(),
        matches.shape()[0]
    );

    matches
        .outer_iter()
        .zip(scores.iter().copied())
        .filter(|(matched, _)| matched[0] == 0)
        .map(|(matched, score)| {
            let source_index = usize::try_from(matched[1])
                .map_err(|_| anyhow!("invalid left match index {}", matched[1]))?;
            let target_index = usize::try_from(matched[2])
                .map_err(|_| anyhow!("invalid right match index {}", matched[2]))?;
            ensure!(
                source_index < keypoints.shape()[1] && target_index < keypoints.shape()[1],
                "LightGlue match indices out of range: left={} right={} keypoints={}",
                source_index,
                target_index,
                keypoints.shape()[1]
            );

            Ok(PointMatch {
                source_index,
                target_index,
                source_point: ImagePoint::new(
                    keypoints[[0, source_index, 0]] as f32,
                    keypoints[[0, source_index, 1]] as f32,
                ),
                target_point: ImagePoint::new(
                    keypoints[[1, target_index, 0]] as f32,
                    keypoints[[1, target_index, 1]] as f32,
                ),
                distance: 1.0 - score,
                ratio: 0.0,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn matches_dino_pair_with_superpoint_lightglue_onnx() {
        let model_path = workspace_root()
            .join("models")
            .join("superpoint_lightglue_pipeline.ort.onnx");
        let mut matcher = LightGlueOnnxMatcher::with_config(
            &model_path,
            LightGlueOnnxConfig {
                resize_width: 1024,
                resize_height: 1024,
            },
        )
        .unwrap();
        let image0 = image::open(workspace_root().join("data/images/viff.000.ppm"))
            .unwrap()
            .to_rgb8();
        let image1 = image::open(workspace_root().join("data/images/viff.001.ppm"))
            .unwrap()
            .to_rgb8();

        let matches = matcher.match_from_rgb_images(&image0, &image1).unwrap();
        eprintln!("superpoint+lightglue dino matches={}", matches.len());

        assert!(
            matches.len() >= 128,
            "expected enough LightGlue matches, got {}",
            matches.len()
        );
    }

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }
}
