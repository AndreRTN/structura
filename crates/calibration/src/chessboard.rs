use anyhow::{Result, anyhow};
use chess_corners::{ChessConfig, CornerDescriptor, find_chess_corners_u8};
use kornia::image::allocator::ImageAllocator;
use nalgebra::{Point2, Vector3};
use structura_geometry::homography::{HomographyMatrix, estimate_homography_4pt};

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

pub fn order_chessboard_corners(
    points: &[Point2<f64>],
    board_width: usize,
    board_height: usize,
) -> Result<Vec<Point2<f64>>> {
    let point_count = board_width * board_height;
    if points.len() != point_count {
        return Err(anyhow!(
            "expected {point_count} points for a {board_width}x{board_height} board, got {}",
            points.len()
        ));
    }

    let image_corners = extreme_corners(points);
    let mut best: Option<(f64, Vec<Point2<f64>>)> = None;

    for target_corners in canonical_corner_sets(board_width, board_height) {
        let source = image_corners.map(|point| ImagePoint::new(point.x as f32, point.y as f32));
        let target = target_corners.map(|point| ImagePoint::new(point.x as f32, point.y as f32));
        let homography = estimate_homography_4pt(&source, &target)?;
        let candidate = assign_points_to_grid(points, &homography, board_width, board_height)?;
        let score = grid_assignment_score(&candidate, &homography, board_width)?;

        if best.as_ref().is_none_or(|(best_score, _)| score < *best_score) {
            best = Some((score, candidate));
        }
    }

    best.map(|(_, ordered)| ordered)
        .ok_or_else(|| anyhow!("failed to order chessboard corners"))
}

fn extreme_corners(points: &[Point2<f64>]) -> [Point2<f64>; 4] {
    let tl = *points
        .iter()
        .min_by(|a, b| (a.x + a.y).total_cmp(&(b.x + b.y)))
        .expect("point set must be non-empty");
    let tr = *points
        .iter()
        .max_by(|a, b| (a.x - a.y).total_cmp(&(b.x - b.y)))
        .expect("point set must be non-empty");
    let bl = *points
        .iter()
        .min_by(|a, b| (a.x - a.y).total_cmp(&(b.x - b.y)))
        .expect("point set must be non-empty");
    let br = *points
        .iter()
        .max_by(|a, b| (a.x + a.y).total_cmp(&(b.x + b.y)))
        .expect("point set must be non-empty");
    [tl, tr, bl, br]
}

fn canonical_corner_sets(board_width: usize, board_height: usize) -> Vec<[Point2<f64>; 4]> {
    let width = (board_width - 1) as f64;
    let height = (board_height - 1) as f64;
    vec![
        corners_for_transform(|x, y| Point2::new(x, y), width, height),
        corners_for_transform(|x, y| Point2::new(width - x, y), width, height),
        corners_for_transform(|x, y| Point2::new(x, height - y), width, height),
        corners_for_transform(|x, y| Point2::new(width - x, height - y), width, height),
    ]
}

fn corners_for_transform<F>(transform: F, width: f64, height: f64) -> [Point2<f64>; 4]
where
    F: Fn(f64, f64) -> Point2<f64>,
{
    [
        transform(0.0, 0.0),
        transform(width, 0.0),
        transform(0.0, height),
        transform(width, height),
    ]
}

fn assign_points_to_grid(
    points: &[Point2<f64>],
    homography: &HomographyMatrix,
    board_width: usize,
    board_height: usize,
) -> Result<Vec<Point2<f64>>> {
    let mut assigned = vec![None; board_width * board_height];

    for &point in points {
        let warped = apply_homography(homography, point);
        let col = warped.x.round() as isize;
        let row = warped.y.round() as isize;
        if !(0..board_width as isize).contains(&col) || !(0..board_height as isize).contains(&row)
        {
            return Err(anyhow!(
                "projected point out of board bounds: ({:.3}, {:.3})",
                warped.x,
                warped.y
            ));
        }

        let index = row as usize * board_width + col as usize;
        if assigned[index].is_some() {
            return Err(anyhow!("duplicate assignment for board cell ({col}, {row})"));
        }
        assigned[index] = Some(point);
    }

    assigned
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow!("incomplete board assignment"))
}

fn grid_assignment_score(
    ordered: &[Point2<f64>],
    homography: &HomographyMatrix,
    board_width: usize,
) -> Result<f64> {
    if ordered.is_empty() {
        return Err(anyhow!("ordered point set must not be empty"));
    }

    Ok(ordered
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let col = (index % board_width) as f64;
            let row = (index / board_width) as f64;
            let warped = apply_homography(homography, *point);
            (warped.x - col).powi(2) + (warped.y - row).powi(2)
        })
        .sum())
}

fn apply_homography(homography: &HomographyMatrix, point: Point2<f64>) -> Point2<f64> {
    let projected = homography * Vector3::new(point.x, point.y, 1.0);
    Point2::new(projected.x / projected.z, projected.y / projected.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia::image::{ImageSize, allocator::CpuAllocator};
    use nalgebra::Point2;

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

    #[test]
    fn orders_permuted_grid_points() {
        let mut ordered = Vec::new();
        for row in 0..3 {
            for col in 0..4 {
                ordered.push(Point2::new(col as f64 * 10.0, row as f64 * 20.0));
            }
        }

        let permuted = vec![
            ordered[11], ordered[9], ordered[10], ordered[8], ordered[7], ordered[5], ordered[6],
            ordered[4], ordered[3], ordered[1], ordered[2], ordered[0],
        ];

        let recovered = order_chessboard_corners(&permuted, 4, 3).unwrap();

        assert_eq!(recovered, ordered);
    }
}
