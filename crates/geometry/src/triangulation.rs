use anyhow::{Result, anyhow, ensure};
use nalgebra::{Matrix3, Vector3};

use crate::{
    camera::{CameraExtrinsics, CameraIntrinsics},
    point::{ImagePoint64, WorldPoint},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangulatedPoint {
    pub position: WorldPoint,
    pub mean_reprojection_error: f64,
    pub max_reprojection_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TriangulationObservation {
    pub image: ImagePoint64,
    pub extrinsics: CameraExtrinsics,
}

impl TriangulationObservation {
    pub const fn new(image: ImagePoint64, extrinsics: CameraExtrinsics) -> Self {
        Self { image, extrinsics }
    }
}

pub fn triangulate_two_views(
    intrinsics: &CameraIntrinsics,
    first_extrinsics: &CameraExtrinsics,
    first_image: ImagePoint64,
    second_extrinsics: &CameraExtrinsics,
    second_image: ImagePoint64,
) -> Result<WorldPoint> {
    triangulate_observations(
        intrinsics,
        &[
            TriangulationObservation::new(first_image, first_extrinsics.clone()),
            TriangulationObservation::new(second_image, second_extrinsics.clone()),
        ],
    )
}

pub fn triangulate_observations(
    intrinsics: &CameraIntrinsics,
    observations: &[TriangulationObservation],
) -> Result<WorldPoint> {
    Ok(triangulate_observations_with_stats(intrinsics, observations)?.position)
}

pub fn triangulate_observations_with_stats(
    intrinsics: &CameraIntrinsics,
    observations: &[TriangulationObservation],
) -> Result<TriangulatedPoint> {
    ensure!(
        observations.len() >= 2,
        "triangulation requires at least two observations"
    );

    let mut normal_matrix = Matrix3::<f64>::zeros();
    let mut rhs = Vector3::<f64>::zeros();

    for observation in observations {
        let center = camera_center(&observation.extrinsics);
        let direction =
            viewing_ray_direction(intrinsics, &observation.extrinsics, observation.image)?;
        let projector = Matrix3::<f64>::identity() - direction * direction.transpose();

        normal_matrix += projector;
        rhs += projector * center;
    }

    let point = normal_matrix
        .lu()
        .solve(&rhs)
        .ok_or_else(|| anyhow!("triangulation failed because the viewing rays are degenerate"))?;

    let position = WorldPoint::from(point);
    let errors = observations
        .iter()
        .map(|observation| {
            reprojection_error(
                intrinsics,
                &observation.extrinsics,
                &position,
                observation.image,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mean_reprojection_error = errors.iter().sum::<f64>() / errors.len().max(1) as f64;
    let max_reprojection_error = errors.into_iter().fold(0.0_f64, f64::max);

    Ok(TriangulatedPoint {
        position,
        mean_reprojection_error,
        max_reprojection_error,
    })
}

pub fn normalized_image_point(
    intrinsics: &CameraIntrinsics,
    image: ImagePoint64,
) -> Result<Vector3<f64>> {
    let k_inv = intrinsics
        .matrix()
        .try_inverse()
        .ok_or_else(|| anyhow!("camera intrinsics matrix is not invertible"))?;
    Ok(k_inv * Vector3::new(image.x, image.y, 1.0))
}

pub fn camera_center(extrinsics: &CameraExtrinsics) -> Vector3<f64> {
    -extrinsics.rotation.transpose() * extrinsics.translation
}

pub fn point_depth(extrinsics: &CameraExtrinsics, world: &WorldPoint) -> f64 {
    let camera = extrinsics.rotation * world.coords + extrinsics.translation;
    camera.z
}

pub fn has_positive_depth(extrinsics: &CameraExtrinsics, world: &WorldPoint) -> bool {
    point_depth(extrinsics, world) > 0.0
}

pub fn project_world_to_image(
    intrinsics: &CameraIntrinsics,
    extrinsics: &CameraExtrinsics,
    world: &WorldPoint,
) -> Result<ImagePoint64> {
    let camera = extrinsics.rotation * world.coords + extrinsics.translation;
    ensure!(camera.z > 1e-12, "point projects behind the camera");

    Ok(ImagePoint64::new(
        intrinsics.alpha * (camera.x / camera.z)
            + intrinsics.gamma * (camera.y / camera.z)
            + intrinsics.u0,
        intrinsics.beta * (camera.y / camera.z) + intrinsics.v0,
    ))
}

pub fn reprojection_error(
    intrinsics: &CameraIntrinsics,
    extrinsics: &CameraExtrinsics,
    world: &WorldPoint,
    observed: ImagePoint64,
) -> Result<f64> {
    let projected = project_world_to_image(intrinsics, extrinsics, world)?;
    let dx = projected.x - observed.x;
    let dy = projected.y - observed.y;
    Ok((dx * dx + dy * dy).sqrt())
}

fn viewing_ray_direction(
    intrinsics: &CameraIntrinsics,
    extrinsics: &CameraExtrinsics,
    image: ImagePoint64,
) -> Result<Vector3<f64>> {
    let camera_direction = normalized_image_point(intrinsics, image)?;
    let world_direction = extrinsics.rotation.transpose() * camera_direction;
    let norm = world_direction.norm();
    ensure!(
        norm > f64::EPSILON,
        "triangulation ray direction has zero norm"
    );
    Ok(world_direction / norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Point2, Rotation3};

    #[test]
    fn triangulates_exact_point_from_two_views() {
        let intrinsics = sample_intrinsics();
        let first = CameraExtrinsics {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        };
        let second = CameraExtrinsics {
            rotation: Rotation3::from_euler_angles(0.01, -0.03, 0.02).into_inner(),
            translation: Vector3::new(0.4, 0.02, 0.01),
        };
        let world = WorldPoint::new(0.2, -0.1, 4.5);

        let triangulated = triangulate_two_views(
            &intrinsics,
            &first,
            project(&intrinsics, &first, world),
            &second,
            project(&intrinsics, &second, world),
        )
        .unwrap();

        assert_relative_eq!(triangulated.x, world.x, epsilon = 1e-6);
        assert_relative_eq!(triangulated.y, world.y, epsilon = 1e-6);
        assert_relative_eq!(triangulated.z, world.z, epsilon = 1e-6);
    }

    #[test]
    fn triangulates_point_from_multiple_noisy_views() {
        let intrinsics = sample_intrinsics();
        let views = [
            CameraExtrinsics {
                rotation: Matrix3::identity(),
                translation: Vector3::zeros(),
            },
            CameraExtrinsics {
                rotation: Rotation3::from_euler_angles(0.01, -0.03, 0.02).into_inner(),
                translation: Vector3::new(0.4, 0.02, 0.01),
            },
            CameraExtrinsics {
                rotation: Rotation3::from_euler_angles(-0.02, 0.01, -0.04).into_inner(),
                translation: Vector3::new(-0.25, 0.05, 0.03),
            },
        ];
        let world = WorldPoint::new(-0.15, 0.22, 5.2);
        let observations = views
            .iter()
            .enumerate()
            .map(|(index, extrinsics)| {
                let projected = project(&intrinsics, extrinsics, world);
                let noise = 0.2 * index as f64;
                TriangulationObservation::new(
                    Point2::new(projected.x + noise, projected.y - noise * 0.5),
                    extrinsics.clone(),
                )
            })
            .collect::<Vec<_>>();

        let triangulated = triangulate_observations(&intrinsics, &observations).unwrap();

        assert!((triangulated - world).norm() < 0.03);
    }

    #[test]
    fn reports_reprojection_statistics() {
        let intrinsics = sample_intrinsics();
        let first = CameraExtrinsics {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        };
        let second = CameraExtrinsics {
            rotation: Rotation3::from_euler_angles(0.01, -0.03, 0.02).into_inner(),
            translation: Vector3::new(0.4, 0.02, 0.01),
        };
        let world = WorldPoint::new(0.2, -0.1, 4.5);
        let observations = vec![
            TriangulationObservation::new(project(&intrinsics, &first, world), first),
            TriangulationObservation::new(
                Point2::new(
                    project(&intrinsics, &second, world).x + 0.2,
                    project(&intrinsics, &second, world).y - 0.1,
                ),
                second,
            ),
        ];

        let triangulated = triangulate_observations_with_stats(&intrinsics, &observations).unwrap();

        assert!(triangulated.mean_reprojection_error > 0.0);
        assert!(triangulated.max_reprojection_error >= triangulated.mean_reprojection_error);
    }

    #[test]
    fn rejects_single_observation() {
        let intrinsics = sample_intrinsics();
        let observation = TriangulationObservation::new(
            Point2::new(320.0, 240.0),
            CameraExtrinsics {
                rotation: Matrix3::identity(),
                translation: Vector3::zeros(),
            },
        );

        let error = triangulate_observations(&intrinsics, &[observation]).unwrap_err();

        assert!(error.to_string().contains("at least two observations"));
    }

    #[test]
    fn rejects_degenerate_parallel_rays() {
        let intrinsics = sample_intrinsics();
        let extrinsics = CameraExtrinsics {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        };
        let observations = vec![
            TriangulationObservation::new(Point2::new(320.0, 240.0), extrinsics.clone()),
            TriangulationObservation::new(Point2::new(320.0, 240.0), extrinsics),
        ];

        let error = triangulate_observations(&intrinsics, &observations).unwrap_err();

        assert!(error.to_string().contains("degenerate"));
    }

    #[test]
    fn reports_positive_depth_for_front_facing_point() {
        let extrinsics = CameraExtrinsics {
            rotation: Matrix3::identity(),
            translation: Vector3::new(0.0, 0.0, 1.0),
        };
        let world = WorldPoint::new(0.1, 0.2, 4.0);

        assert!(has_positive_depth(&extrinsics, &world));
        assert!(point_depth(&extrinsics, &world) > 0.0);
    }

    #[test]
    fn projects_world_point_back_to_image() {
        let intrinsics = sample_intrinsics();
        let extrinsics = CameraExtrinsics {
            rotation: Matrix3::identity(),
            translation: Vector3::new(0.1, -0.1, 1.0),
        };
        let world = WorldPoint::new(0.3, 0.2, 4.0);

        let projected = project_world_to_image(&intrinsics, &extrinsics, &world).unwrap();

        assert!(projected.x.is_finite());
        assert!(projected.y.is_finite());
    }

    fn sample_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics {
            alpha: 800.0,
            beta: 810.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        }
    }

    fn project(
        intrinsics: &CameraIntrinsics,
        extrinsics: &CameraExtrinsics,
        world: WorldPoint,
    ) -> Point2<f64> {
        let camera = extrinsics.rotation * world.coords + extrinsics.translation;
        Point2::new(
            intrinsics.alpha * (camera.x / camera.z)
                + intrinsics.gamma * (camera.y / camera.z)
                + intrinsics.u0,
            intrinsics.beta * (camera.y / camera.z) + intrinsics.v0,
        )
    }
}
