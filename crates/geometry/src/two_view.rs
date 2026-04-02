use anyhow::{Result, anyhow, ensure};
use nalgebra::{DMatrix, Matrix3};

use crate::{camera::CameraIntrinsics, point::ImagePoint64};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FundamentalRansacConfig {
    pub iterations: usize,
    pub inlier_threshold: f64,
    pub min_inlier_count: usize,
}

impl Default for FundamentalRansacConfig {
    fn default() -> Self {
        Self {
            iterations: 1024,
            inlier_threshold: 1.0,
            min_inlier_count: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FundamentalRansacResult {
    pub fundamental_matrix: Matrix3<f64>,
    pub inlier_indices: Vec<usize>,
}

pub fn estimate_fundamental(
    points_a: &[ImagePoint64],
    points_b: &[ImagePoint64],
) -> Result<Matrix3<f64>> {
    ensure!(
        points_a.len() == points_b.len(),
        "fundamental estimation requires the same number of points in both views"
    );
    ensure!(
        points_a.len() >= 8,
        "fundamental estimation requires at least eight correspondences, got {}",
        points_a.len()
    );

    let norm_a = normalize_points(points_a);
    let norm_b = normalize_points(points_b);
    let mut a = DMatrix::zeros(points_a.len(), 9);

    norm_a
        .points
        .iter()
        .zip(norm_b.points.iter())
        .enumerate()
        .for_each(|(index, (point_a, point_b))| {
            a[(index, 0)] = point_b.x * point_a.x;
            a[(index, 1)] = point_b.x * point_a.y;
            a[(index, 2)] = point_b.x;
            a[(index, 3)] = point_b.y * point_a.x;
            a[(index, 4)] = point_b.y * point_a.y;
            a[(index, 5)] = point_b.y;
            a[(index, 6)] = point_a.x;
            a[(index, 7)] = point_a.y;
            a[(index, 8)] = 1.0;
        });

    let v_t = a
        .svd(false, true)
        .v_t
        .ok_or_else(|| anyhow!("fundamental DLT SVD did not produce V^T"))?;
    let row = v_t.row(v_t.nrows() - 1);
    let raw = Matrix3::new(
        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
    );
    let rank2 = enforce_rank2(&raw)?;
    let denormalized = norm_b.transform.transpose() * rank2 * norm_a.transform;
    let scale = denormalized.norm();

    ensure!(
        scale.is_finite() && scale > f64::EPSILON,
        "estimated fundamental matrix is degenerate"
    );

    Ok(denormalized / scale)
}

pub fn estimate_fundamental_ransac(
    points_a: &[ImagePoint64],
    points_b: &[ImagePoint64],
    config: FundamentalRansacConfig,
) -> Result<FundamentalRansacResult> {
    ensure!(
        points_a.len() == points_b.len(),
        "fundamental RANSAC requires the same number of points in both views"
    );
    ensure!(
        points_a.len() >= 8,
        "fundamental RANSAC requires at least eight correspondences, got {}",
        points_a.len()
    );

    let mut best_model = None::<(Matrix3<f64>, Vec<usize>)>;

    for iteration in 0..config.iterations.max(1) {
        let sample = sample_indices(points_a.len(), iteration as u64, 8);
        let sample_a = sample
            .iter()
            .map(|&index| points_a[index])
            .collect::<Vec<_>>();
        let sample_b = sample
            .iter()
            .map(|&index| points_b[index])
            .collect::<Vec<_>>();
        let Ok(candidate) = estimate_fundamental(&sample_a, &sample_b) else {
            continue;
        };

        let inliers = points_a
            .iter()
            .zip(points_b.iter())
            .enumerate()
            .filter_map(|(index, (point_a, point_b))| {
                (sampson_error(&candidate, point_a, point_b) <= config.inlier_threshold)
                    .then_some(index)
            })
            .collect::<Vec<_>>();

        match &best_model {
            Some((_, best_inliers)) if best_inliers.len() >= inliers.len() => {}
            _ => best_model = Some((candidate, inliers)),
        }
    }

    let (_, best_inliers) =
        best_model.ok_or_else(|| anyhow!("RANSAC failed to estimate a fundamental matrix"))?;
    ensure!(
        best_inliers.len() >= config.min_inlier_count.max(8),
        "RANSAC only found {} inliers, expected at least {}",
        best_inliers.len(),
        config.min_inlier_count.max(8)
    );

    let inlier_points_a = best_inliers
        .iter()
        .map(|&index| points_a[index])
        .collect::<Vec<_>>();
    let inlier_points_b = best_inliers
        .iter()
        .map(|&index| points_b[index])
        .collect::<Vec<_>>();
    let refined = estimate_fundamental(&inlier_points_a, &inlier_points_b)?;

    Ok(FundamentalRansacResult {
        fundamental_matrix: refined,
        inlier_indices: best_inliers,
    })
}

pub fn essential_from_fundamental(
    fundamental_matrix: &Matrix3<f64>,
    intrinsics: &CameraIntrinsics,
) -> Matrix3<f64> {
    let k = intrinsics.matrix();
    k.transpose() * fundamental_matrix * k
}

pub fn sampson_error(
    fundamental_matrix: &Matrix3<f64>,
    point_a: &ImagePoint64,
    point_b: &ImagePoint64,
) -> f64 {
    let x_a = point_a.to_homogeneous();
    let x_b = point_b.to_homogeneous();
    let fx_a = fundamental_matrix * x_a;
    let f_t_x_b = fundamental_matrix.transpose() * x_b;
    let numerator = x_b.dot(&(fundamental_matrix * x_a));
    let denominator =
        fx_a.x * fx_a.x + fx_a.y * fx_a.y + f_t_x_b.x * f_t_x_b.x + f_t_x_b.y * f_t_x_b.y;

    if denominator <= f64::EPSILON {
        f64::INFINITY
    } else {
        (numerator * numerator) / denominator
    }
}

fn enforce_rank2(fundamental_matrix: &Matrix3<f64>) -> Result<Matrix3<f64>> {
    let svd = fundamental_matrix.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| anyhow!("rank-2 enforcement SVD did not produce U"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("rank-2 enforcement SVD did not produce V^T"))?;
    let mut sigma = svd.singular_values;
    sigma[2] = 0.0;
    Ok(u * Matrix3::from_diagonal(&sigma) * v_t)
}

#[derive(Debug, Clone)]
struct PointNormalization {
    points: Vec<ImagePoint64>,
    transform: Matrix3<f64>,
}

fn normalize_points(points: &[ImagePoint64]) -> PointNormalization {
    let count = points.len() as f64;
    let centroid = points
        .iter()
        .fold(ImagePoint64::origin(), |acc, point| acc + point.coords)
        / count;
    let mean_distance = points
        .iter()
        .map(|point| {
            let dx = point.x - centroid.x;
            let dy = point.y - centroid.y;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>()
        / count;
    let scale = if mean_distance > f64::EPSILON {
        std::f64::consts::SQRT_2 / mean_distance
    } else {
        1.0
    };

    PointNormalization {
        points: points
            .iter()
            .map(|point| {
                ImagePoint64::new(
                    (point.x - centroid.x) * scale,
                    (point.y - centroid.y) * scale,
                )
            })
            .collect(),
        transform: Matrix3::new(
            scale,
            0.0,
            -scale * centroid.x,
            0.0,
            scale,
            -scale * centroid.y,
            0.0,
            0.0,
            1.0,
        ),
    }
}

fn sample_indices(population: usize, seed: u64, sample_size: usize) -> Vec<usize> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut chosen = Vec::with_capacity(sample_size);

    while chosen.len() < sample_size {
        state = state
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let candidate = (state % population as u64) as usize;
        if !chosen.contains(&candidate) {
            chosen.push(candidate);
        }
    }

    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector3};

    fn sample_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics {
            alpha: 800.0,
            beta: 800.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        }
    }

    fn project(
        intrinsics: &CameraIntrinsics,
        rotation: &Matrix3<f64>,
        translation: &Vector3<f64>,
        world: &Vector3<f64>,
    ) -> ImagePoint64 {
        let camera = rotation * world + translation;
        ImagePoint64::new(
            intrinsics.alpha * (camera[0] / camera[2]) + intrinsics.u0,
            intrinsics.beta * (camera[1] / camera[2]) + intrinsics.v0,
        )
    }

    fn sample_scene() -> (Vec<Vector3<f64>>, Matrix3<f64>, Vector3<f64>) {
        let points = vec![
            Vector3::new(-1.0, -1.0, 5.0),
            Vector3::new(1.0, -1.0, 5.0),
            Vector3::new(1.0, 1.0, 5.0),
            Vector3::new(-1.0, 1.0, 5.0),
            Vector3::new(0.0, 0.0, 4.0),
            Vector3::new(-0.5, 0.5, 6.0),
            Vector3::new(0.5, -0.5, 5.5),
            Vector3::new(-1.0, 0.0, 4.5),
            Vector3::new(0.0, -1.0, 5.5),
            Vector3::new(0.5, 0.5, 4.5),
            Vector3::new(-0.5, -0.5, 5.0),
            Vector3::new(1.0, 0.0, 5.5),
        ];
        let rotation = Rotation3::from_euler_angles(0.0, 0.1, 0.0).into_inner();
        let translation = Vector3::new(0.5, 0.0, 0.0);
        (points, rotation, translation)
    }

    #[test]
    fn estimates_rank_two_fundamental_matrix() {
        let intrinsics = sample_intrinsics();
        let (world_points, rotation, translation) = sample_scene();
        let first = Matrix3::identity();
        let second = rotation;
        let first_translation = Vector3::zeros();
        let points_a = world_points
            .iter()
            .map(|world| project(&intrinsics, &first, &first_translation, world))
            .collect::<Vec<_>>();
        let points_b = world_points
            .iter()
            .map(|world| project(&intrinsics, &second, &translation, world))
            .collect::<Vec<_>>();

        let fundamental = estimate_fundamental(&points_a, &points_b).unwrap();
        let singular_values = fundamental.svd(false, false).singular_values;

        assert!(singular_values[2].abs() < 1e-8);
    }

    #[test]
    fn ransac_rejects_large_outlier_set() {
        let intrinsics = sample_intrinsics();
        let (world_points, rotation, translation) = sample_scene();
        let first = Matrix3::identity();
        let first_translation = Vector3::zeros();
        let mut points_a = world_points
            .iter()
            .map(|world| project(&intrinsics, &first, &first_translation, world))
            .collect::<Vec<_>>();
        let mut points_b = world_points
            .iter()
            .map(|world| project(&intrinsics, &rotation, &translation, world))
            .collect::<Vec<_>>();

        points_a.extend([
            ImagePoint64::new(20.0, 30.0),
            ImagePoint64::new(200.0, 500.0),
            ImagePoint64::new(620.0, 120.0),
            ImagePoint64::new(400.0, 320.0),
        ]);
        points_b.extend([
            ImagePoint64::new(600.0, 50.0),
            ImagePoint64::new(30.0, 90.0),
            ImagePoint64::new(80.0, 540.0),
            ImagePoint64::new(100.0, 100.0),
        ]);

        let result = estimate_fundamental_ransac(
            &points_a,
            &points_b,
            FundamentalRansacConfig {
                iterations: 256,
                inlier_threshold: 4.0,
                min_inlier_count: 8,
            },
        )
        .unwrap();

        assert!(result.inlier_indices.len() >= 8);
    }
}
