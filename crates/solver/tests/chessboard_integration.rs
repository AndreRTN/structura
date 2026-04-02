use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use approx::assert_relative_eq;
use image::ImageReader;
use kornia::image::{ImageSize, allocator::CpuAllocator};
use nalgebra::{Point2, Point3, Rotation3, Vector3};
use structura_calibration::{
    chessboard::{ChessCornersDetector, detect_chessboard_corners},
    detector::GrayImage,
    zhang::{ZhangDistortion, calibrate_from_homographies_with_radial_distortion, distort_point},
};
use structura_geometry::{
    camera::{CameraExtrinsics, CameraIntrinsics},
    homography::{HomographyMatrix, estimate_homography_4pt},
    point::{ImagePoint, PointCorrespondence2D3D},
};
use structura_solver::{compute_reprojection_error, from_zhang_initialization, optimize};

const BOARD_WIDTH: usize = 9;
const BOARD_HEIGHT: usize = 6;
const BOARD_POINT_COUNT: usize = BOARD_WIDTH * BOARD_HEIGHT;
const MATCH_THRESHOLD_PX: f64 = 18.0;

#[test]
fn calibrates_chessboard_dataset_close_to_ground_truth() -> Result<()> {
    let dataset = ChessboardDataset::load()?;
    let views = dataset.detect_views()?;

    assert_eq!(views.len(), dataset.gt_extrinsics.len());

    let homographies = views
        .iter()
        .map(|view| build_homography(&view.matches))
        .collect::<Result<Vec<_>>>()?;
    let observations = views
        .iter()
        .map(|view| view.matches.clone())
        .collect::<Vec<_>>();

    let (zhang_calibration, zhang_distortion) =
        calibrate_from_homographies_with_radial_distortion(&homographies, &observations)?;
    let zhang_initial = from_zhang_initialization(&zhang_calibration, Some(zhang_distortion));
    let refined = optimize(&observations, &zhang_initial)?;

    let zhang_error = compute_reprojection_error(
        &observations,
        &zhang_initial.intrinsics,
        &zhang_initial.distortion,
        &zhang_initial.extrinsics,
    )?;

    log_estimated_vs_ground_truth(
        &refined.intrinsics,
        &refined.distortion,
        &dataset.gt_intrinsics,
        dataset.gt_distortion,
        zhang_error,
        refined.reprojection_error,
    );

    assert!(
        zhang_error < 3.5,
        "zhang reprojection error too high: {zhang_error}"
    );
    assert!(
        refined.reprojection_error < zhang_error,
        "solver refinement should reduce reprojection error: zhang={zhang_error}, refined={}",
        refined.reprojection_error
    );
    assert!(
        refined.reprojection_error < 1.0,
        "refined reprojection error too high: {}",
        refined.reprojection_error
    );

    assert_intrinsics_close(&refined.intrinsics, &dataset.gt_intrinsics);
    assert_distortion_close(&refined.distortion, &dataset.gt_distortion);

    Ok(())
}

#[derive(Debug, Clone)]
struct ChessboardDataset {
    image_paths: Vec<PathBuf>,
    gt_intrinsics: CameraIntrinsics,
    gt_distortion: ZhangDistortion,
    gt_extrinsics: Vec<CameraExtrinsics>,
    world_points: Vec<Point3<f64>>,
}

#[derive(Debug, Clone)]
struct DetectedView {
    matches: Vec<PointCorrespondence2D3D>,
}

impl ChessboardDataset {
    fn load() -> Result<Self> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
            .join("chessboard");
        let yaml = fs::read_to_string(root.join("left_intrinsics.yml"))
            .context("failed to read chessboard ground-truth yaml")?;
        let gt_intrinsics = parse_intrinsics(&yaml)?;
        let gt_distortion = parse_distortion(&yaml)?;
        let gt_extrinsics = parse_extrinsics(&yaml)?;
        let square_size = parse_scalar(&yaml, "square_size")?;
        let board_width = parse_usize(&yaml, "board_width")?;
        let board_height = parse_usize(&yaml, "board_height")?;
        let mut image_paths = fs::read_dir(&root)
            .with_context(|| format!("failed to read {}", root.display()))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()
            .with_context(|| format!("failed to enumerate {}", root.display()))?
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("left") && name.ends_with(".jpg"))
            })
            .collect::<Vec<_>>();
        image_paths.sort();
        let world_points = board_points(board_width, board_height, square_size);

        Ok(Self {
            image_paths,
            gt_intrinsics,
            gt_distortion,
            gt_extrinsics,
            world_points,
        })
    }

    fn detect_views(&self) -> Result<Vec<DetectedView>> {
        let mut detector = ChessCornersDetector::multiscale();
        let detected = self
            .image_paths
            .iter()
            .map(|path| self.detect_image(path, &mut detector))
            .collect::<Result<Vec<_>>>()?;
        let selected = detected
            .into_iter()
            .zip(self.gt_extrinsics.iter().enumerate())
            .map(|(detections, (gt_index, gt_extrinsics))| {
                let matches = match_corners_to_world(
                    &detections.1,
                    &self.gt_intrinsics,
                    self.gt_distortion,
                    gt_extrinsics,
                    &self.world_points,
                )
                .with_context(|| {
                    format!(
                        "failed to match {} against gt extrinsic row {}",
                        detections.0.display(),
                        gt_index
                    )
                })?;
                Ok(DetectedView { matches })
            })
            .collect::<Result<Vec<_>>>()?;

        assert!(
            selected
                .iter()
                .all(|view| view.matches.len() == BOARD_POINT_COUNT),
            "all selected views must contain the full board"
        );

        Ok(selected)
    }

    fn detect_image(
        &self,
        image_path: &Path,
        detector: &mut ChessCornersDetector,
    ) -> Result<(PathBuf, Vec<Point2<f64>>)> {
        let gray = ImageReader::open(image_path)
            .with_context(|| format!("failed to open {}", image_path.display()))?
            .decode()
            .with_context(|| format!("failed to decode {}", image_path.display()))?
            .to_luma8();
        let image = GrayImage::new(
            ImageSize {
                width: gray.width() as usize,
                height: gray.height() as usize,
            },
            gray.into_raw(),
            CpuAllocator,
        )
        .map_err(|error| anyhow!("failed to build kornia image: {error}"))?;
        let corners = detect_chessboard_corners(&image, detector)?;

        Ok((
            image_path.to_path_buf(),
            corners
                .into_iter()
                .map(|corner| Point2::new(corner.point.x as f64, corner.point.y as f64))
                .collect(),
        ))
    }
}

fn match_corners_to_world(
    detected: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: ZhangDistortion,
    extrinsics: &CameraExtrinsics,
    world_points: &[Point3<f64>],
) -> Result<Vec<PointCorrespondence2D3D>> {
    if detected.len() < BOARD_POINT_COUNT {
        return Err(anyhow!(
            "detector returned {} corners, expected at least {BOARD_POINT_COUNT}",
            detected.len()
        ));
    }

    let expected = world_points
        .iter()
        .copied()
        .map(|world| {
            project_world_to_pixel(intrinsics, distortion, extrinsics, world)
                .map(|image| (world, image))
        })
        .collect::<Result<Vec<_>>>()?;

    expected
        .into_iter()
        .try_fold(
            (vec![false; detected.len()], Vec::with_capacity(world_points.len())),
            |(mut used, mut matches), (world, projected)| {
                let (best_index, best_distance) = detected
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !used[*index])
                    .map(|(index, detected)| (index, point_distance(projected, *detected)))
                    .min_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                    .ok_or_else(|| anyhow!("failed to find a corner assignment"))?;

                if best_distance > MATCH_THRESHOLD_PX {
                    return Err(anyhow!(
                        "nearest detected corner is too far from GT projection: {best_distance:.3}px"
                    ));
                }

                used[best_index] = true;
                matches.push(PointCorrespondence2D3D::new(detected[best_index], world));
                Ok((used, matches))
            },
        )
        .map(|(_, matches)| matches)
}

fn build_homography(matches: &[PointCorrespondence2D3D]) -> Result<HomographyMatrix> {
    let corners = [
        0,
        BOARD_WIDTH - 1,
        BOARD_POINT_COUNT - BOARD_WIDTH,
        BOARD_POINT_COUNT - 1,
    ]
    .map(|index| &matches[index]);
    let source = corners.map(|entry| ImagePoint::new(entry.world.x as f32, entry.world.y as f32));
    let target = corners.map(|entry| ImagePoint::new(entry.image.x as f32, entry.image.y as f32));

    estimate_homography_4pt(&source, &target)
}

fn board_points(board_width: usize, board_height: usize, square_size: f64) -> Vec<Point3<f64>> {
    (0..board_height)
        .flat_map(|row| {
            (0..board_width).map(move |col| {
                Point3::new(col as f64 * square_size, row as f64 * square_size, 0.0)
            })
        })
        .collect()
}

fn project_world_to_pixel(
    intrinsics: &CameraIntrinsics,
    distortion: ZhangDistortion,
    extrinsics: &CameraExtrinsics,
    world: Point3<f64>,
) -> Result<Point2<f64>> {
    let camera = extrinsics.rotation * world.coords + extrinsics.translation;
    if camera.z.abs() <= 1e-12 {
        return Err(anyhow!("cannot project point with near-zero camera depth"));
    }

    let ideal = Point2::new(
        intrinsics.alpha * (camera.x / camera.z)
            + intrinsics.gamma * (camera.y / camera.z)
            + intrinsics.u0,
        intrinsics.beta * (camera.y / camera.z) + intrinsics.v0,
    );

    Ok(distort_point(intrinsics, distortion, ideal))
}

fn point_distance(lhs: Point2<f64>, rhs: Point2<f64>) -> f64 {
    ((lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y)).sqrt()
}

fn log_estimated_vs_ground_truth(
    estimated_intrinsics: &CameraIntrinsics,
    estimated_distortion: &structura_solver::BrownConradyDistortion,
    ground_truth_intrinsics: &CameraIntrinsics,
    ground_truth_distortion: ZhangDistortion,
    zhang_error: f64,
    refined_error: f64,
) {
    let rows = [
        (
            "fx",
            estimated_intrinsics.alpha,
            ground_truth_intrinsics.alpha,
        ),
        (
            "fy",
            estimated_intrinsics.beta,
            ground_truth_intrinsics.beta,
        ),
        (
            "skew",
            estimated_intrinsics.gamma,
            ground_truth_intrinsics.gamma,
        ),
        ("cx", estimated_intrinsics.u0, ground_truth_intrinsics.u0),
        ("cy", estimated_intrinsics.v0, ground_truth_intrinsics.v0),
        ("k1", estimated_distortion.k1, ground_truth_distortion.k1),
        ("k2", estimated_distortion.k2, ground_truth_distortion.k2),
        ("k3", estimated_distortion.k3, ground_truth_distortion.k3),
        ("p1", estimated_distortion.p1, ground_truth_distortion.p1),
        ("p2", estimated_distortion.p2, ground_truth_distortion.p2),
    ];

    eprintln!();
    eprintln!(
        "{:<8} | {:>14} | {:>14} | {:>14}",
        "param", "estimated", "ground_truth", "delta"
    );
    eprintln!("{}", "-".repeat(61));
    rows.iter().for_each(|(name, estimated, ground_truth)| {
        eprintln!(
            "{:<8} | {:>14.8} | {:>14.8} | {:>14.8}",
            name,
            estimated,
            ground_truth,
            estimated - ground_truth
        );
    });
    eprintln!("{}", "-".repeat(61));
    eprintln!(
        "{:<8} | {:>14.8} | {:>14} | {:>14}",
        "zhang_rmse", zhang_error, "-", "-"
    );
    eprintln!(
        "{:<8} | {:>14.8} | {:>14} | {:>14}",
        "solver_rmse", refined_error, "-", "-"
    );
    eprintln!();
}

fn assert_intrinsics_close(estimated: &CameraIntrinsics, ground_truth: &CameraIntrinsics) {
    assert_relative_eq!(estimated.alpha, ground_truth.alpha, epsilon = 20.0);
    assert_relative_eq!(estimated.beta, ground_truth.beta, epsilon = 20.0);
    assert_relative_eq!(estimated.gamma, ground_truth.gamma, epsilon = 15.0);
    assert_relative_eq!(estimated.u0, ground_truth.u0, epsilon = 20.0);
    assert_relative_eq!(estimated.v0, ground_truth.v0, epsilon = 20.0);
}

fn assert_distortion_close(
    estimated: &structura_solver::BrownConradyDistortion,
    ground_truth: &ZhangDistortion,
) {
    assert_relative_eq!(estimated.k1, ground_truth.k1, epsilon = 0.15);
    assert_relative_eq!(estimated.k2, ground_truth.k2, epsilon = 0.15);
    assert_relative_eq!(estimated.k3, ground_truth.k3, epsilon = 0.35);
    assert_relative_eq!(estimated.p1, ground_truth.p1, epsilon = 0.01);
    assert_relative_eq!(estimated.p2, ground_truth.p2, epsilon = 0.01);
}

fn parse_intrinsics(yaml: &str) -> Result<CameraIntrinsics> {
    let values = parse_matrix_data(yaml, "camera_matrix")?;
    if values.len() != 9 {
        return Err(anyhow!("camera_matrix must contain 9 values"));
    }

    Ok(CameraIntrinsics {
        alpha: values[0],
        beta: values[4],
        gamma: values[1],
        u0: values[2],
        v0: values[5],
    })
}

fn parse_distortion(yaml: &str) -> Result<ZhangDistortion> {
    let values = parse_matrix_data(yaml, "distortion_coefficients")?;
    if values.len() != 5 {
        return Err(anyhow!("distortion_coefficients must contain 5 values"));
    }

    Ok(ZhangDistortion {
        k1: values[0],
        k2: values[1],
        p1: values[2],
        p2: values[3],
        k3: values[4],
    })
}

fn parse_extrinsics(yaml: &str) -> Result<Vec<CameraExtrinsics>> {
    let values = parse_matrix_data(yaml, "extrinsic_parameters")?;
    values
        .chunks_exact(6)
        .map(|chunk| {
            Ok(CameraExtrinsics {
                rotation: Rotation3::from_scaled_axis(Vector3::new(chunk[0], chunk[1], chunk[2]))
                    .into_inner(),
                translation: Vector3::new(chunk[3], chunk[4], chunk[5]),
            })
        })
        .collect()
}

fn parse_matrix_data(yaml: &str, key: &str) -> Result<Vec<f64>> {
    yaml.lines()
        .enumerate()
        .find(|(_, line)| line.trim_start().starts_with(&format!("{key}:")))
        .map(|(index, _)| yaml.lines().skip(index).collect::<Vec<_>>())
        .ok_or_else(|| anyhow!("missing yaml key {key}"))
        .and_then(|lines| {
            lines
                .iter()
                .find_map(|line| {
                    line.split_once("data:").map(|(_, rest)| {
                        collect_bracketed_values(
                            rest,
                            lines
                                .iter()
                                .skip_while(|candidate| *candidate != line)
                                .skip(1),
                        )
                    })
                })
                .ok_or_else(|| anyhow!("missing data array for yaml key {key}"))?
        })
}

fn collect_bracketed_values<'a, I>(first_line: &'a str, remaining: I) -> Result<Vec<f64>>
where
    I: Iterator<Item = &'a &'a str>,
{
    let combined = std::iter::once(first_line.trim())
        .chain(remaining.copied().map(str::trim))
        .scan(String::new(), |state, line| {
            if !state.is_empty() {
                state.push(' ');
            }
            state.push_str(line);
            Some(state.clone())
        })
        .find(|text| text.contains(']'))
        .ok_or_else(|| anyhow!("unterminated matrix data array"))?;
    let start = combined
        .find('[')
        .ok_or_else(|| anyhow!("matrix data array is missing '['"))?;
    let end = combined
        .find(']')
        .ok_or_else(|| anyhow!("matrix data array is missing ']'"))?;

    combined[start + 1..end]
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f64>()
                .map_err(|error| anyhow!("failed to parse matrix value '{value}': {error}"))
        })
        .collect()
}

fn parse_scalar(yaml: &str, key: &str) -> Result<f64> {
    yaml.lines()
        .find_map(|line| {
            line.trim_start()
                .split_once(':')
                .filter(|(name, _)| *name == key)
                .map(|(_, value)| value.trim())
        })
        .ok_or_else(|| anyhow!("missing scalar key {key}"))?
        .parse::<f64>()
        .map_err(|error| anyhow!("failed to parse scalar key {key}: {error}"))
}

fn parse_usize(yaml: &str, key: &str) -> Result<usize> {
    parse_scalar(yaml, key).map(|value| value as usize)
}
