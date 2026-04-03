use std::{cmp::Ordering, env, path::Path};

use anyhow::{Context, Result, anyhow, ensure};
use image::{RgbImage, imageops::FilterType};
use ndarray::{Array1, Array2, Array4, ArrayD, Axis, Ix1, Ix2};
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    tensor::TensorElementType,
    value::{TensorRef, ValueRef},
};
use structura_geometry::point::ImagePoint;

use crate::descriptor::{DescriptorFeature, DescriptorFeatureSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlikedOnnxConfig {
    pub min_score: f32,
    pub max_num_features: usize,
    pub input_width: u32,
    pub input_height: u32,
}

impl Default for AlikedOnnxConfig {
    fn default() -> Self {
        Self {
            min_score: 0.2,
            max_num_features: 2048,
            input_width: 640,
            input_height: 640,
        }
    }
}

#[derive(Debug)]
pub struct AlikedOnnxExtractor {
    session: Session,
    config: AlikedOnnxConfig,
}

impl AlikedOnnxExtractor {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(model_path, AlikedOnnxConfig::default())
    }

    pub fn with_config(model_path: impl AsRef<Path>, config: AlikedOnnxConfig) -> Result<Self> {
        let session = build_session(model_path.as_ref())?;
        Ok(Self { session, config })
    }

    pub fn extract_from_rgb_image(&mut self, image: &RgbImage) -> Result<DescriptorFeatureSet> {
        let input = prepare_input(image, self.config);
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
        let mut descriptors = None;
        let mut scores = None;

        for (name, value) in &outputs {
            let lower_name = name.to_ascii_lowercase();
            let dtype = value.dtype();

            if lower_name.contains("desc") && descriptors.is_none() {
                descriptors = Some(extract_rank2_f32(&value).with_context(|| {
                    format!("ALIKED output `{name}` could not be decoded as descriptors")
                })?);
                continue;
            }

            if (lower_name.contains("score") || lower_name.contains("conf")) && scores.is_none() {
                scores = Some(extract_rank1_f32(&value).with_context(|| {
                    format!("ALIKED output `{name}` could not be decoded as scores")
                })?);
                continue;
            }

            if (lower_name.contains("kpt")
                || lower_name.contains("keypoint")
                || lower_name.contains("point"))
                && keypoints.is_none()
            {
                keypoints = Some(extract_rank2_points_f32(&value).with_context(|| {
                    format!("ALIKED output `{name}` could not be decoded as keypoints")
                })?);
                continue;
            }

            match (
                dtype.tensor_type(),
                dtype.tensor_shape().map_or(0, |shape| shape.len()),
            ) {
                (Some(TensorElementType::Float32), 3) if keypoints.is_none() => {
                    let candidate = extract_rank2_points_f32(&value)?;
                    if candidate.shape()[1] >= 2 {
                        keypoints = Some(candidate);
                    }
                }
                (Some(TensorElementType::Float32), 3 | 2) if descriptors.is_none() => {
                    let candidate = extract_rank2_f32(&value)?;
                    if candidate.shape()[1] > 2 {
                        descriptors = Some(candidate);
                    } else if keypoints.is_none() && candidate.shape()[1] == 2 {
                        keypoints = Some(candidate);
                    }
                }
                (Some(TensorElementType::Float32), 2 | 1) if scores.is_none() => {
                    let candidate = extract_rank1_f32(&value)?;
                    scores = Some(candidate);
                }
                _ => {}
            }
        }

        let keypoints = keypoints.ok_or_else(|| {
            anyhow!("failed to locate ALIKED keypoints output; outputs: {output_metadata}")
        })?;
        let descriptors = descriptors.ok_or_else(|| {
            anyhow!("failed to locate ALIKED descriptors output; outputs: {output_metadata}")
        })?;
        let scores = scores.ok_or_else(|| {
            anyhow!("failed to locate ALIKED scores output; outputs: {output_metadata}")
        })?;

        decode_feature_set(
            &keypoints,
            &descriptors,
            &scores,
            image.width() as f32 / self.config.input_width as f32,
            image.height() as f32 / self.config.input_height as f32,
            self.config,
        )
    }
}

fn build_session(model_path: &Path) -> Result<Session> {
    let builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    let builder = configure_execution_providers(builder)?;
    builder.commit_from_file(model_path).with_context(|| {
        format!(
            "failed to load ALIKED ONNX model at {}",
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
        eprintln!("aliked_onnx: requesting ROCm execution provider");
        Ok(builder.with_execution_providers([ep::ROCm::default()
            .with_device_id(0)
            .with_exhaustive_conv_search(exhaustive)
            .with_conv_use_max_workspace(true)
            .with_hip_graph(hip_graph)
            .with_tunable_op(tuning)
            .with_tuning(tuning)
            .with_max_tuning_duration(100)
            .build()])?)
    } else if env::var_os("STRUCTURA_ORT_MIGRAPHX").is_some() {
        eprintln!("aliked_onnx: requesting MIGraphX execution provider");
        Ok(
            builder
                .with_execution_providers([ep::MIGraphX::default().with_device_id(0).build()])?,
        )
    } else {
        Ok(builder)
    }
}

fn prepare_input(image: &RgbImage, config: AlikedOnnxConfig) -> Array4<f32> {
    let resized = image::imageops::resize(
        image,
        config.input_width,
        config.input_height,
        FilterType::Triangle,
    );
    let height = config.input_height as usize;
    let width = config.input_width as usize;
    let mut input = Array4::<f32>::zeros((1, 3, height, width));

    resized.enumerate_pixels().for_each(|(x, y, pixel)| {
        let x = x as usize;
        let y = y as usize;
        input[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
        input[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
        input[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
    });

    input
}

fn extract_rank2_points_f32(value: &ValueRef<'_>) -> Result<Array2<f32>> {
    let array = value.try_extract_array::<f32>()?.to_owned().into_dyn();
    let array = squeeze_leading_axis(array)?;

    match array.ndim() {
        2 => {
            let array = array.into_dimensionality::<Ix2>()?;
            ensure!(
                array.shape()[1] >= 2,
                "expected keypoints last dimension >= 2, got {:?}",
                array.shape()
            );
            Ok(array)
        }
        _ => Err(anyhow!(
            "expected rank-2 keypoints, got shape {:?}",
            array.shape()
        )),
    }
}

fn extract_rank2_f32(value: &ValueRef<'_>) -> Result<Array2<f32>> {
    let array = value.try_extract_array::<f32>()?.to_owned().into_dyn();
    let array = squeeze_leading_axis(array)?;

    match array.ndim() {
        2 => Ok(array.into_dimensionality::<Ix2>()?),
        _ => Err(anyhow!(
            "expected rank-2 tensor, got shape {:?}",
            array.shape()
        )),
    }
}

fn extract_rank1_f32(value: &ValueRef<'_>) -> Result<Array1<f32>> {
    let array = value.try_extract_array::<f32>()?.to_owned().into_dyn();
    let array = squeeze_leading_axis(array)?;

    match array.ndim() {
        1 => Ok(array.into_dimensionality::<Ix1>()?),
        2 => {
            let array = array.into_dimensionality::<Ix2>()?;
            if array.shape()[1] == 1 {
                Ok(array.index_axis_move(Axis(1), 0))
            } else if array.shape()[0] == 1 {
                Ok(array.index_axis_move(Axis(0), 0))
            } else {
                Err(anyhow!(
                    "expected scores vector, got shape {:?}",
                    array.shape()
                ))
            }
        }
        _ => Err(anyhow!(
            "expected rank-1 tensor, got shape {:?}",
            array.shape()
        )),
    }
}

fn squeeze_leading_axis(mut array: ArrayD<f32>) -> Result<ArrayD<f32>> {
    while array.ndim() > 0 && array.shape()[0] == 1 {
        array = array.index_axis_move(Axis(0), 0).into_dyn();
    }
    Ok(array)
}

fn decode_feature_set(
    keypoints: &Array2<f32>,
    descriptors: &Array2<f32>,
    scores: &Array1<f32>,
    scale_x: f32,
    scale_y: f32,
    config: AlikedOnnxConfig,
) -> Result<DescriptorFeatureSet> {
    ensure!(
        keypoints.shape()[0] == descriptors.shape()[0] && keypoints.shape()[0] == scores.len(),
        "ALIKED outputs have incompatible shapes: keypoints={:?} descriptors={:?} scores={}",
        keypoints.shape(),
        descriptors.shape(),
        scores.len()
    );

    let mut features = keypoints
        .outer_iter()
        .zip(descriptors.outer_iter())
        .zip(scores.iter().copied())
        .filter_map(|((point, descriptor), score)| {
            (score >= config.min_score).then(|| DescriptorFeature {
                point: ImagePoint::new(point[0] * scale_x, point[1] * scale_y),
                score,
                descriptor: descriptor.to_vec(),
            })
        })
        .collect::<Vec<_>>();

    features.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| Ordering::Equal)
    });
    features.truncate(config.max_num_features);

    Ok(DescriptorFeatureSet { features })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, array};

    #[test]
    fn decodes_and_filters_feature_set() {
        let feature_set = decode_feature_set(
            &array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            &array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            &array![0.1, 0.9, 0.7],
            2.0,
            3.0,
            AlikedOnnxConfig {
                min_score: 0.2,
                max_num_features: 2,
                input_width: 640,
                input_height: 640,
            },
        )
        .unwrap();

        assert_eq!(feature_set.features.len(), 2);
        assert_eq!(feature_set.features[0].point, ImagePoint::new(6.0, 12.0));
        assert_eq!(feature_set.features[1].point, ImagePoint::new(10.0, 18.0));
    }

    #[test]
    fn squeezes_leading_batch_axes() {
        let squeezed = squeeze_leading_axis(ArrayD::from_elem(vec![1, 1, 3], 0.5_f32)).unwrap();

        assert_eq!(squeezed.shape(), &[3]);
    }
}
