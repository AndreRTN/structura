use anyhow::{Result, anyhow};
use kornia::k3d::pose::homography_4pt2d;

use crate::point::ImagePoint;

pub type HomographyMatrix = [[f64; 3]; 3];

pub trait PointMatchLike {
    fn source_point(&self) -> ImagePoint;
    fn target_point(&self) -> ImagePoint;
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

    Ok(homo)
}

pub fn estimate_homography_from_matches<T>(matches: &[T]) -> Result<HomographyMatrix>
where
    T: PointMatchLike + Copy,
{
    if matches.len() < 4 {
        return Err(anyhow!(
            "at least four matches are required to estimate a 2d homography"
        ));
    }

    let selected: [T; 4] = matches[..4]
        .try_into()
        .map_err(|_| anyhow!("failed to select four matches for homography estimation"))?;
    let source = selected.map(|entry| entry.source_point());
    let target = selected.map(|entry| entry.target_point());

    estimate_homography_4pt(&source, &target)
}

fn to_point2d(point: ImagePoint) -> [f64; 2] {
    [point.x as f64, point.y as f64]
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

        assert_relative_eq!(homo[0][0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(homo[1][1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(homo[0][2], 2.0, epsilon = 1e-6);
        assert_relative_eq!(homo[1][2], 3.0, epsilon = 1e-6);
        assert_relative_eq!(homo[2][2], 1.0, epsilon = 1e-6);
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

        assert_relative_eq!(homo[0][0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(homo[0][1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[0][2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[1][0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[1][1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(homo[1][2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[2][0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[2][1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[2][2], 1.0, epsilon = 1e-6);
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

        assert_relative_eq!(homo[0][0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(homo[1][1], 3.0, epsilon = 1e-6);
        assert_relative_eq!(homo[0][2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(homo[1][2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn estimates_projective_homography() {
        let expected = [[1.0, 0.1, 2.0], [0.2, 1.0, 3.0], [0.01, 0.02, 1.0]];
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
                assert_relative_eq!(homo[row][col], expected[row][col], epsilon = 1e-6);
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
        let w = h[2][0] * x + h[2][1] * y + h[2][2];
        let x_prime = (h[0][0] * x + h[0][1] * y + h[0][2]) / w;
        let y_prime = (h[1][0] * x + h[1][1] * y + h[1][2]) / w;

        ImagePoint::new(x_prime as f32, y_prime as f32)
    }
}
