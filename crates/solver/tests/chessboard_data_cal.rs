use std::{fs, path::{Path, PathBuf}};

use anyhow::{Context, Result, anyhow, ensure};
use image::ImageReader;
use kornia::image::{ImageSize, allocator::CpuAllocator};
use nalgebra::{Point2, Point3};
use structura_calibration::{
    chessboard::{ChessCornersDetector, detect_chessboard_corners, order_chessboard_corners},
    detector::GrayImage,
    zhang::calibrate_from_homographies_with_radial_distortion,
};
use structura_geometry::{
    homography::{PointMatchLike, estimate_homography_from_matches},
    point::{ImagePoint, PointCorrespondence2D3D},
};
use structura_solver::{compute_reprojection_error, from_zhang_initialization, optimize};

const BOARD_WIDTH: usize = 9;
const BOARD_HEIGHT: usize = 6;
const BOARD_POINT_COUNT: usize = BOARD_WIDTH * BOARD_HEIGHT;
const SQUARE_SIZE_MM: f64 = 1.0;

#[test]
fn calibrates_data_cal_with_kornia_corners() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
        .join("cal");
    let image_paths = collect_image_paths(&root)?;
    ensure!(
        !image_paths.is_empty(),
        "no calibration images found in {}",
        root.display()
    );

    let world_points = board_points();
    let mut detector = ChessCornersDetector::multiscale();
    let observations = image_paths
        .iter()
        .map(|path| detect_ordered_matches(path, &mut detector, &world_points))
        .collect::<Result<Vec<_>>>()?;
    let homographies = observations
        .iter()
        .map(|matches| estimate_homography_from_board(matches))
        .collect::<Result<Vec<_>>>()?;

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

    eprintln!(
        "zhang rmse={:.6} solver rmse={:.6} fx={:.6} fy={:.6} cx={:.6} cy={:.6}",
        zhang_error,
        refined.reprojection_error,
        refined.intrinsics.alpha,
        refined.intrinsics.beta,
        refined.intrinsics.u0,
        refined.intrinsics.v0
    );

    assert!(zhang_error < 2.0, "zhang reprojection error too high: {zhang_error}");
    assert!(
        refined.reprojection_error < 1.0,
        "refined reprojection error too high: {}",
        refined.reprojection_error
    );

    Ok(())
}

#[derive(Clone, Copy)]
struct Match2D {
    source: ImagePoint,
    target: ImagePoint,
}

impl PointMatchLike for Match2D {
    fn source_point(&self) -> ImagePoint {
        self.source
    }

    fn target_point(&self) -> ImagePoint {
        self.target
    }
}

fn collect_image_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn detect_ordered_matches(
    image_path: &Path,
    detector: &mut ChessCornersDetector,
    world_points: &[Point3<f64>],
) -> Result<Vec<PointCorrespondence2D3D>> {
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
    ensure!(
        corners.len() == BOARD_POINT_COUNT,
        "{} returned {} corners, expected {BOARD_POINT_COUNT}",
        image_path.display(),
        corners.len()
    );
    let detected = corners
        .into_iter()
        .map(|corner| Point2::new(corner.point.x as f64, corner.point.y as f64))
        .collect::<Vec<_>>();
    let ordered = order_chessboard_corners(&detected, BOARD_WIDTH, BOARD_HEIGHT)?;

    Ok(ordered
        .into_iter()
        .zip(world_points.iter().copied())
        .map(|(image, world)| PointCorrespondence2D3D::new(image, world))
        .collect())
}

fn estimate_homography_from_board(
    matches: &[PointCorrespondence2D3D],
) -> Result<structura_geometry::homography::HomographyMatrix> {
    let planar_matches = matches
        .iter()
        .map(|entry| Match2D {
            source: ImagePoint::new(entry.world.x as f32, entry.world.y as f32),
            target: ImagePoint::new(entry.image.x as f32, entry.image.y as f32),
        })
        .collect::<Vec<_>>();
    estimate_homography_from_matches(&planar_matches)
}

fn board_points() -> Vec<Point3<f64>> {
    (0..BOARD_HEIGHT)
        .flat_map(|row| {
            (0..BOARD_WIDTH).map(move |col| {
                Point3::new(col as f64 * SQUARE_SIZE_MM, row as f64 * SQUARE_SIZE_MM, 0.0)
            })
        })
        .collect()
}
