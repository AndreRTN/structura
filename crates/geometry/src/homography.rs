use anyhow::{Result, anyhow};
use kornia::k3d::pose::homography_4pt2d;
use nalgebra::{DMatrix, Matrix3, Point2, Vector3, linalg::SVD};

use crate::point::{ImagePoint, PointCorrespondence2D3D};

pub type HomographyMatrix = Matrix3<f64>;

pub trait PointMatchLike {
    fn source_point(&self) -> ImagePoint;
    fn target_point(&self) -> ImagePoint;
}

impl PointMatchLike for PointCorrespondence2D3D {
    fn source_point(&self) -> ImagePoint {
        ImagePoint::new(self.world.x as f32, self.world.y as f32)
    }

    fn target_point(&self) -> ImagePoint {
        ImagePoint::new(self.image.x as f32, self.image.y as f32)
    }
}

pub fn estimate_homography_4pt(
    source_points: &[ImagePoint; 4],
    target_points: &[ImagePoint; 4],
) -> Result<HomographyMatrix> {
    let mut homo = [[0.0_f64; 3]; 3];
    let source = source_points.map(to_point2d);
    let target = target_points.map(to_point2d);

    homography_4pt2d(&source, &target, &mut homo)
        .map_err(|error| anyhow!("failed to estimate 2d homography: {error}"))?;

    Ok(Matrix3::new(
        homo[0][0], homo[0][1], homo[0][2], homo[1][0], homo[1][1], homo[1][2], homo[2][0],
        homo[2][1], homo[2][2],
    ))
}

pub fn estimate_homography_from_matches<T>(matches: &[T]) -> Result<HomographyMatrix>
where
    T: PointMatchLike,
{
    if matches.len() < 4 {
        return Err(anyhow!(
            "at least four matches are required to estimate a 2d homography"
        ));
    }

    if matches.len() == 4 {
        let source = [
            matches[0].source_point(),
            matches[1].source_point(),
            matches[2].source_point(),
            matches[3].source_point(),
        ];
        let target = [
            matches[0].target_point(),
            matches[1].target_point(),
            matches[2].target_point(),
            matches[3].target_point(),
        ];
        return estimate_homography_4pt(&source, &target);
    }

    let source = matches
        .iter()
        .map(|entry| to_point64(entry.source_point()))
        .collect::<Vec<_>>();
    let target = matches
        .iter()
        .map(|entry| to_point64(entry.target_point()))
        .collect::<Vec<_>>();
    let (source_norm, source_t) = normalize_points(&source)?;
    let (target_norm, target_t) = normalize_points(&target)?;
    let mut a = DMatrix::<f64>::zeros(matches.len() * 2, 9);

    source_norm
        .iter()
        .zip(target_norm.iter())
        .enumerate()
        .for_each(|(index, (source, target))| {
            let row0 = index * 2;
            let row1 = row0 + 1;

            a[(row0, 0)] = -source.x;
            a[(row0, 1)] = -source.y;
            a[(row0, 2)] = -1.0;
            a[(row0, 6)] = target.x * source.x;
            a[(row0, 7)] = target.x * source.y;
            a[(row0, 8)] = target.x;

            a[(row1, 3)] = -source.x;
            a[(row1, 4)] = -source.y;
            a[(row1, 5)] = -1.0;
            a[(row1, 6)] = target.y * source.x;
            a[(row1, 7)] = target.y * source.y;
            a[(row1, 8)] = target.y;
        });

    let svd = SVD::new(a, false, true);
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("homography DLT SVD did not produce V^T"))?;
    let h = v_t.row(v_t.nrows() - 1);
    let normalized = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    let target_t_inv = target_t
        .try_inverse()
        .ok_or_else(|| anyhow!("target normalization transform is not invertible"))?;
    let homography = target_t_inv * normalized * source_t;

    let scale = if homography[(2, 2)].abs() > 1e-12 {
        homography[(2, 2)]
    } else {
        1.0
    };

    Ok(homography / scale)
}

fn to_point2d(point: ImagePoint) -> [f64; 2] {
    [point.x as f64, point.y as f64]
}

fn to_point64(point: ImagePoint) -> Point2<f64> {
    Point2::new(point.x as f64, point.y as f64)
}

fn normalize_points(points: &[Point2<f64>]) -> Result<(Vec<Point2<f64>>, Matrix3<f64>)> {
    if points.is_empty() {
        return Err(anyhow!("at least one point is required for normalization"));
    }

    let centroid = points
        .iter()
        .fold(Point2::new(0.0, 0.0), |acc, point| {
            Point2::new(acc.x + point.x, acc.y + point.y)
        });
    let centroid = Point2::new(
        centroid.x / points.len() as f64,
        centroid.y / points.len() as f64,
    );
    let mean_distance = points
        .iter()
        .map(|point| ((point.x - centroid.x).powi(2) + (point.y - centroid.y).powi(2)).sqrt())
        .sum::<f64>()
        / points.len() as f64;
    if mean_distance <= 1e-12 {
        return Err(anyhow!("point set is degenerate for homography normalization"));
    }

    let scale = 2.0_f64.sqrt() / mean_distance;
    let transform = Matrix3::new(
        scale,
        0.0,
        -scale * centroid.x,
        0.0,
        scale,
        -scale * centroid.y,
        0.0,
        0.0,
        1.0,
    );

    let normalized = points
        .iter()
        .map(|point| {
            let projected = transform * Vector3::new(point.x, point.y, 1.0);
            Point2::new(projected.x / projected.z, projected.y / projected.z)
        })
        .collect();

    Ok((normalized, transform))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[derive(Debug, Clone, Copy)]
    struct TestMatch {
        source: ImagePoint,
        target: ImagePoint,
    }

    impl PointMatchLike for TestMatch {
        fn source_point(&self) -> ImagePoint {
            self.source
        }

        fn target_point(&self) -> ImagePoint {
            self.target
        }
    }

    #[test]
    fn estimates_translation_homography() {
        let source = [
            ImagePoint::new(0.0, 0.0),
            ImagePoint::new(1.0, 0.0),
            ImagePoint::new(0.0, 1.0),
            ImagePoint::new(1.0, 1.0),
        ];
        let target = [
            ImagePoint::new(2.0, 3.0),
            ImagePoint::new(3.0, 3.0),
            ImagePoint::new(2.0, 4.0),
            ImagePoint::new(3.0, 4.0),
        ];

        let homo = estimate_homography_4pt(&source, &target).unwrap();

        assert_relative_eq!(homo[(0, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(1, 1)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(0, 2)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(1, 2)], 3.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(2, 2)], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn estimates_identity_homography_from_matches() {
        let matches = [
            test_match((0.0, 0.0), (0.0, 0.0)),
            test_match((1.0, 0.0), (1.0, 0.0)),
            test_match((0.0, 1.0), (0.0, 1.0)),
            test_match((1.0, 1.0), (1.0, 1.0)),
        ];

        let homo = estimate_homography_from_matches(&matches).unwrap();

        assert_relative_eq!(homo[(0, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(0, 1)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(0, 2)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(1, 0)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(1, 1)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(1, 2)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(2, 0)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(2, 1)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(2, 2)], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn estimates_scale_homography() {
        let source = [
            ImagePoint::new(0.0, 0.0),
            ImagePoint::new(2.0, 0.0),
            ImagePoint::new(0.0, 2.0),
            ImagePoint::new(2.0, 2.0),
        ];
        let target = [
            ImagePoint::new(0.0, 0.0),
            ImagePoint::new(4.0, 0.0),
            ImagePoint::new(0.0, 6.0),
            ImagePoint::new(4.0, 6.0),
        ];

        let homo = estimate_homography_4pt(&source, &target).unwrap();

        assert_relative_eq!(homo[(0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(1, 1)], 3.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(0, 2)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[(1, 2)], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn estimates_projective_homography() {
        let expected = Matrix3::new(1.0, 0.1, 2.0, 0.2, 1.0, 3.0, 0.01, 0.02, 1.0);
        let source = [
            ImagePoint::new(0.0, 0.0),
            ImagePoint::new(1.0, 0.0),
            ImagePoint::new(0.0, 1.0),
            ImagePoint::new(1.0, 1.0),
        ];
        let target = source.map(|point| apply_homography(&expected, point));

        let homo = estimate_homography_4pt(&source, &target).unwrap();

        for row in 0..3 {
            for col in 0..3 {
                assert_relative_eq!(homo[(row, col)], expected[(row, col)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn fails_with_fewer_than_four_matches() {
        let matches = [
            test_match((0.0, 0.0), (1.0, 1.0)),
            test_match((1.0, 0.0), (2.0, 1.0)),
            test_match((0.0, 1.0), (1.0, 2.0)),
        ];

        let error = estimate_homography_from_matches(&matches).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("at least four matches are required")
        );
    }

    fn test_match(source: (f32, f32), target: (f32, f32)) -> TestMatch {
        TestMatch {
            source: ImagePoint::new(source.0, source.1),
            target: ImagePoint::new(target.0, target.1),
        }
    }

    fn apply_homography(h: &HomographyMatrix, point: ImagePoint) -> ImagePoint {
        let x = point.x as f64;
        let y = point.y as f64;
        let w = h[(2, 0)] * x + h[(2, 1)] * y + h[(2, 2)];
        let x_prime = (h[(0, 0)] * x + h[(0, 1)] * y + h[(0, 2)]) / w;
        let y_prime = (h[(1, 0)] * x + h[(1, 1)] * y + h[(1, 2)]) / w;

        ImagePoint::new(x_prime as f32, y_prime as f32)
    }
}
