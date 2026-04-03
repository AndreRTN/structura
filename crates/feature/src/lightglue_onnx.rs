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

#[derive(Debug, Clone)]
pub struct LightGlueOnnxMatchResult {
    pub left_keypoints: Vec<ImagePoint>,
    pub right_keypoints: Vec<ImagePoint>,
    pub matches: Vec<PointMatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightGlueOnnxConfig {
    pub resize_width: u32,
    pub resize_height: u32,
    pub crop_right_border: u32,
}

impl Default for LightGlueOnnxConfig {
    fn default() -> Self {
        Self {
            resize_width: 1024,
            resize_height: 1024,
            crop_right_border: 35,
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
        Ok(self.match_with_keypoints_from_rgb_images(left, right)?.matches)
    }

    pub fn match_with_keypoints_from_rgb_images(
        &mut self,
        left: &RgbImage,
        right: &RgbImage,
    ) -> Result<LightGlueOnnxMatchResult> {
        let left = crop_image_for_model(left, self.config)?;
        let right = crop_image_for_model(right, self.config)?;
        let input = prepare_input(&left, &right, self.config);
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

        decode_match_result(
            &keypoints,
            &matches,
            &scores,
            self.config,
            left.width(),
            left.height(),
            right.width(),
            right.height(),
        )
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

fn crop_image_for_model(image: &RgbImage, config: LightGlueOnnxConfig) -> Result<RgbImage> {
    ensure!(
        image.width() > config.crop_right_border,
        "image width {} must be larger than right crop {}",
        image.width(),
        config.crop_right_border
    );
    let cropped_width = image.width() - config.crop_right_border;
    Ok(image::imageops::crop_imm(image, 0, 0, cropped_width, image.height()).to_image())
}

fn decode_match_result(
    keypoints: &Array3<i64>,
    matches: &Array2<i64>,
    scores: &Array1<f32>,
    config: LightGlueOnnxConfig,
    left_width: u32,
    left_height: u32,
    right_width: u32,
    right_height: u32,
) -> Result<LightGlueOnnxMatchResult> {
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

    let left_keypoints = keypoints
        .outer_iter()
        .next()
        .expect("validated rank-3 keypoints")
        .outer_iter()
        .map(|point| {
            remap_keypoint(
                point[0] as f32,
                point[1] as f32,
                config.resize_width,
                config.resize_height,
                left_width,
                left_height,
            )
        })
        .collect::<Vec<_>>();
    let right_keypoints = keypoints
        .outer_iter()
        .nth(1)
        .expect("validated rank-3 keypoints")
        .outer_iter()
        .map(|point| {
            remap_keypoint(
                point[0] as f32,
                point[1] as f32,
                config.resize_width,
                config.resize_height,
                right_width,
                right_height,
            )
        })
        .collect::<Vec<_>>();

    let matches = matches
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
                source_point: remap_keypoint(
                    keypoints[[0, source_index, 0]] as f32,
                    keypoints[[0, source_index, 1]] as f32,
                    config.resize_width,
                    config.resize_height,
                    left_width,
                    left_height,
                ),
                target_point: remap_keypoint(
                    keypoints[[1, target_index, 0]] as f32,
                    keypoints[[1, target_index, 1]] as f32,
                    config.resize_width,
                    config.resize_height,
                    right_width,
                    right_height,
                ),
                distance: 1.0 - score,
                ratio: 0.0,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LightGlueOnnxMatchResult {
        left_keypoints,
        right_keypoints,
        matches,
    })
}

fn remap_keypoint(
    x: f32,
    y: f32,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
) -> ImagePoint {
    let scale_x = output_width as f32 / input_width as f32;
    let scale_y = output_height as f32 / input_height as f32;
    ImagePoint::new(x * scale_x, y * scale_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use std::{fs, path::PathBuf};

    #[test]
    fn matches_dino_pair_with_superpoint_lightglue_onnx() {
        let (image0, image1, result) = run_match_test("viff.000.ppm", "viff.001.ppm", "viff_000_001");
        eprintln!(
            "superpoint+lightglue dino keypoints=({},{}) matches={}",
            result.left_keypoints.len(),
            result.right_keypoints.len(),
            result.matches.len()
        );

        save_visualizations(
            &image0,
            &image1,
            &result,
            "viff_000",
            "viff_001",
            "viff_000_001_matches",
        )
        .unwrap();

        assert!(
            result.matches.len() >= 128,
            "expected enough LightGlue matches, got {}",
            result.matches.len()
        );
    }

    #[test]
    fn matches_dino_wide_baseline_pair_with_superpoint_lightglue_onnx() {
        let (image0, image1, result) = run_match_test("viff.000.ppm", "viff.030.ppm", "viff_000_030");
        eprintln!(
            "superpoint+lightglue wide-baseline keypoints=({},{}) matches={}",
            result.left_keypoints.len(),
            result.right_keypoints.len(),
            result.matches.len()
        );

        save_visualizations(
            &image0,
            &image1,
            &result,
            "viff_000",
            "viff_030",
            "viff_000_030_matches",
        )
        .unwrap();

        assert!(
            !result.matches.is_empty(),
            "expected at least one LightGlue match for wide-baseline pair"
        );
    }

    #[test]
    fn matches_temple_ring_neighbor_pair_with_superpoint_lightglue_onnx() {
        let (image0, image1, result) = run_match_test_in_dir(
            "data/templeRing/templeR0001.png",
            "data/templeRing/templeR0002.png",
            "templeR0001_0002",
        );
        eprintln!(
            "superpoint+lightglue temple neighbor keypoints=({},{}) matches={}",
            result.left_keypoints.len(),
            result.right_keypoints.len(),
            result.matches.len()
        );

        save_visualizations(
            &image0,
            &image1,
            &result,
            "templeR0001",
            "templeR0002",
            "templeR0001_0002_matches",
        )
        .unwrap();

        assert!(
            result.matches.len() >= 128,
            "expected enough LightGlue matches for temple neighbor pair, got {}",
            result.matches.len()
        );
    }

    #[test]
    fn matches_temple_ring_wide_baseline_pair_with_superpoint_lightglue_onnx() {
        let (image0, image1, result) = run_match_test_in_dir(
            "data/templeRing/templeR0001.png",
            "data/templeRing/templeR0035.png",
            "templeR0001_0035",
        );
        eprintln!(
            "superpoint+lightglue temple wide-baseline keypoints=({},{}) matches={}",
            result.left_keypoints.len(),
            result.right_keypoints.len(),
            result.matches.len()
        );

        save_visualizations(
            &image0,
            &image1,
            &result,
            "templeR0001",
            "templeR0035",
            "templeR0001_0035_matches",
        )
        .unwrap();

        assert!(
            !result.matches.is_empty(),
            "expected at least one LightGlue match for temple wide-baseline pair"
        );
    }

    fn run_match_test(
        left_name: &str,
        right_name: &str,
        artifact_prefix: &str,
    ) -> (RgbImage, RgbImage, LightGlueOnnxMatchResult) {
        run_match_test_in_dir(
            &format!("data/images/{left_name}"),
            &format!("data/images/{right_name}"),
            artifact_prefix,
        )
    }

    fn run_match_test_in_dir(
        left_relative_path: &str,
        right_relative_path: &str,
        artifact_prefix: &str,
    ) -> (RgbImage, RgbImage, LightGlueOnnxMatchResult) {
        let model_path = workspace_root()
            .join("models")
            .join("superpoint_lightglue_pipeline.ort.onnx");
        let mut matcher = LightGlueOnnxMatcher::with_config(
            &model_path,
            LightGlueOnnxConfig {
                resize_width: 1024,
                resize_height: 1024,
                crop_right_border: 5,
            },
        )
        .unwrap();
        let image0 = image::open(workspace_root().join(left_relative_path))
            .unwrap()
            .to_rgb8();
        let image1 = image::open(workspace_root().join(right_relative_path))
            .unwrap()
            .to_rgb8();

        let result = matcher
            .match_with_keypoints_from_rgb_images(&image0, &image1)
            .unwrap();
        eprintln!(
            "saved LightGlue artifacts for {artifact_prefix} under {}",
            artifacts_dir().display()
        );
        (image0, image1, result)
    }

    fn save_visualizations(
        image0: &RgbImage,
        image1: &RgbImage,
        result: &LightGlueOnnxMatchResult,
        left_prefix: &str,
        right_prefix: &str,
        matches_prefix: &str,
    ) -> Result<()> {
        fs::create_dir_all(artifacts_dir())?;
        remove_legacy_artifacts()?;

        let config = LightGlueOnnxConfig::default();
        let image0 = crop_image_for_model(image0, config)?;
        let image1 = crop_image_for_model(image1, config)?;
        let image0_points = draw_points(&image0, result.left_keypoints.iter().copied());
        let image1_points = draw_points(&image1, result.right_keypoints.iter().copied());
        let matches_image = draw_matches(&image0, &image1, &result.matches);

        image0_points.save(artifacts_dir().join(format!("{left_prefix}_keypoints.jpg")))?;
        image1_points.save(artifacts_dir().join(format!("{right_prefix}_keypoints.jpg")))?;
        matches_image.save(artifacts_dir().join(format!("{matches_prefix}.jpg")))?;
        Ok(())
    }

    fn remove_legacy_artifacts() -> Result<()> {
        for name in [
            "viff_000_001_left_points.png",
            "viff_000_001_right_points.png",
            "viff_000_001_left_all_keypoints.png",
            "viff_000_001_right_all_keypoints.png",
            "viff_000_001_left_matched_keypoints.png",
            "viff_000_001_right_matched_keypoints.png",
            "viff_000_001_matches.png",
            "viff_000_030_left_points.png",
            "viff_000_030_right_points.png",
            "viff_000_030_left_all_keypoints.png",
            "viff_000_030_right_all_keypoints.png",
            "viff_000_030_left_matched_keypoints.png",
            "viff_000_030_right_matched_keypoints.png",
            "viff_000_030_matches.png",
        ] {
            let path = artifacts_dir().join(name);
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    fn draw_points(
        image: &RgbImage,
        points: impl IntoIterator<Item = ImagePoint>,
    ) -> RgbImage {
        let mut output = image.clone();
        for point in points {
            draw_cross(&mut output, point, Rgb([255, 64, 64]));
        }
        output
    }

    fn draw_matches(image0: &RgbImage, image1: &RgbImage, matches: &[PointMatch]) -> RgbImage {
        let width = image0.width() + image1.width();
        let height = image0.height().max(image1.height());
        let mut canvas = RgbImage::from_pixel(width, height, Rgb([16, 16, 16]));

        blit(&mut canvas, image0, 0, 0);
        blit(&mut canvas, image1, image0.width(), 0);

        for (index, matched) in matches.iter().enumerate() {
            let color = palette_color(index);
            draw_cross(&mut canvas, matched.source_point, color);
            let shifted_target =
                ImagePoint::new(matched.target_point.x + image0.width() as f32, matched.target_point.y);
            draw_cross(&mut canvas, shifted_target, color);
            draw_line(&mut canvas, matched.source_point, shifted_target, color);
        }

        canvas
    }

    fn blit(dst: &mut RgbImage, src: &RgbImage, offset_x: u32, offset_y: u32) {
        for (x, y, pixel) in src.enumerate_pixels() {
            dst.put_pixel(offset_x + x, offset_y + y, *pixel);
        }
    }

    fn draw_cross(image: &mut RgbImage, point: ImagePoint, color: Rgb<u8>) {
        let cx = point.x.round() as i32;
        let cy = point.y.round() as i32;
        for delta in -3..=3 {
            put_pixel_checked(image, cx + delta, cy, color);
            put_pixel_checked(image, cx, cy + delta, color);
        }
    }

    fn draw_line(image: &mut RgbImage, start: ImagePoint, end: ImagePoint, color: Rgb<u8>) {
        let mut x0 = start.x.round() as i32;
        let mut y0 = start.y.round() as i32;
        let x1 = end.x.round() as i32;
        let y1 = end.y.round() as i32;

        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;

        loop {
            put_pixel_checked(image, x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let twice_error = 2 * error;
            if twice_error >= dy {
                error += dy;
                x0 += sx;
            }
            if twice_error <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    fn put_pixel_checked(image: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
        if x >= 0 && y >= 0 && (x as u32) < image.width() && (y as u32) < image.height() {
            image.put_pixel(x as u32, y as u32, color);
        }
    }

    fn palette_color(index: usize) -> Rgb<u8> {
        const PALETTE: [Rgb<u8>; 8] = [
            Rgb([255, 99, 71]),
            Rgb([255, 215, 0]),
            Rgb([50, 205, 50]),
            Rgb([0, 191, 255]),
            Rgb([255, 105, 180]),
            Rgb([255, 140, 0]),
            Rgb([138, 43, 226]),
            Rgb([64, 224, 208]),
        ];
        PALETTE[index % PALETTE.len()]
    }

    fn artifacts_dir() -> PathBuf {
        workspace_root().join("target").join("lightglue-onnx-tests")
    }

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }
}
