use anyhow::{Result, anyhow};
use factrs::{
    linalg::{Const, ForwardProp, Numeric, VectorX},
    residuals::Residual3,
    variables::{MatrixLieGroup, SE3, Variable, VectorVar2, VectorVar3, VectorVar5},
};
use nalgebra::{Point2, Point3};
use structura_geometry::camera::{CameraExtrinsics, CameraIntrinsics};

use crate::calibration::BrownConradyDistortion;

#[derive(Clone, Debug)]
pub struct ReprojectionResidual {
    observed: VectorVar2,
    world: VectorVar3,
}

impl ReprojectionResidual {
    pub fn new(observed: Point2<f64>, world: Point3<f64>) -> Self {
        Self {
            observed: VectorVar2::new(observed.x, observed.y),
            world: VectorVar3::new(world.x, world.y, world.z),
        }
    }
}

#[factrs::mark]
impl Residual3 for ReprojectionResidual {
    type Differ = ForwardProp<Const<16>>;
    type V1 = SE3;
    type V2 = VectorVar5;
    type V3 = VectorVar5;
    type DimIn = Const<16>;
    type DimOut = Const<2>;

    fn residual3<T: Numeric>(
        &self,
        pose: SE3<T>,
        intrinsics: VectorVar5<T>,
        distortion: VectorVar5<T>,
    ) -> VectorX<T> {
        let world: VectorVar3<T> = self.world.cast();
        let observed: VectorVar2<T> = self.observed.cast();
        let projected = project_with_variable_model(&pose, &intrinsics, &distortion, &world);

        VectorX::from_iterator(
            2,
            [projected[0] - observed[0], projected[1] - observed[1]],
        )
    }
}

pub fn project_world_to_pixel(
    intrinsics: &CameraIntrinsics,
    distortion: &BrownConradyDistortion,
    extrinsics: &CameraExtrinsics,
    world: Point3<f64>,
) -> Result<Point2<f64>> {
    let camera = extrinsics.rotation * world.coords + extrinsics.translation;

    if camera.z.abs() <= 1e-12 {
        return Err(anyhow!("cannot project point with near-zero camera depth"));
    }

    Ok(project_point_from_normalized(
        intrinsics,
        distortion,
        Point2::new(camera.x / camera.z, camera.y / camera.z),
    ))
}

pub(crate) fn project_point_from_normalized(
    intrinsics: &CameraIntrinsics,
    distortion: &BrownConradyDistortion,
    normalized: Point2<f64>,
) -> Point2<f64> {
    let (x_distorted, y_distorted) =
        apply_brown_conrady_distortion(normalized.x, normalized.y, distortion);

    Point2::new(
        intrinsics.alpha * x_distorted + intrinsics.gamma * y_distorted + intrinsics.u0,
        intrinsics.beta * y_distorted + intrinsics.v0,
    )
}

fn project_with_variable_model<T: Numeric>(
    pose: &SE3<T>,
    intrinsics: &VectorVar5<T>,
    distortion: &VectorVar5<T>,
    world: &VectorVar3<T>,
) -> VectorVar2<T> {
    let camera = pose.apply(world.0.as_view());
    let x = camera[0] / camera[2];
    let y = camera[1] / camera[2];
    let (x_distorted, y_distorted) = apply_variable_brown_conrady_distortion(x, y, distortion);

    VectorVar2::new(
        intrinsics[0] * x_distorted + intrinsics[2] * y_distorted + intrinsics[3],
        intrinsics[1] * y_distorted + intrinsics[4],
    )
}

fn apply_brown_conrady_distortion(
    x: f64,
    y: f64,
    distortion: &BrownConradyDistortion,
) -> (f64, f64) {
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let radial = 1.0 + distortion.k1 * r2 + distortion.k2 * r4 + distortion.k3 * r6;

    (
        x * radial + 2.0 * distortion.p1 * x * y + distortion.p2 * (r2 + 2.0 * x * x),
        y * radial + distortion.p1 * (r2 + 2.0 * y * y) + 2.0 * distortion.p2 * x * y,
    )
}

fn apply_variable_brown_conrady_distortion<T: Numeric>(
    x: T,
    y: T,
    distortion: &VectorVar5<T>,
) -> (T, T) {
    let one = T::from(1.0);
    let two = T::from(2.0);
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let radial = one + distortion[0] * r2 + distortion[1] * r4 + distortion[2] * r6;

    (
        x * radial + two * distortion[3] * x * y + distortion[4] * (r2 + two * x * x),
        y * radial + distortion[3] * (r2 + two * y * y) + two * distortion[4] * x * y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use factrs::{
        linalg::Vector3,
        variables::{SO3, Variable},
    };
    use nalgebra::Vector3 as NaVector3;

    fn make_pose(translation: NaVector3<f64>) -> SE3 {
        SE3::from_rot_trans(SO3::identity(), Vector3::new(translation.x, translation.y, translation.z))
    }

    fn make_extrinsics(translation: NaVector3<f64>) -> CameraExtrinsics {
        CameraExtrinsics {
            rotation: nalgebra::Matrix3::identity(),
            translation,
        }
    }

    #[test]
    fn residual_is_zero_for_perfect_projection() {
        let intrinsics = VectorVar5::new(800.0, 790.0, 2.0, 320.0, 240.0);
        let distortion = VectorVar5::new(-0.04, 0.01, 0.002, 0.001, -0.0005);
        let world = Point3::new(0.1, -0.05, 0.0);
        let pose = make_pose(NaVector3::new(0.0, 0.0, 1.2));
        let observed = project_with_variable_model(
            &pose,
            &intrinsics,
            &distortion,
            &VectorVar3::new(world.x, world.y, world.z),
        );
        let residual = ReprojectionResidual::new(Point2::new(observed[0], observed[1]), world);
        let values = residual.residual3(pose, intrinsics, distortion);

        assert_relative_eq!(values[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(values[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn residual_matches_pixel_offset() {
        let pose = make_pose(NaVector3::new(0.0, 0.0, 1.0));
        let intrinsics = VectorVar5::new(800.0, 800.0, 0.0, 320.0, 240.0);
        let distortion = VectorVar5::new(0.0, 0.0, 0.0, 0.0, 0.0);
        let world = Point3::new(0.1, 0.2, 0.0);
        let residual = ReprojectionResidual::new(Point2::new(405.0, 385.0), world);
        let values = residual.residual3(pose, intrinsics, distortion);

        assert_relative_eq!(values[0], -5.0, epsilon = 1e-12);
        assert_relative_eq!(values[1], 15.0, epsilon = 1e-12);
    }

    #[test]
    fn projection_uses_full_brown_conrady_terms() {
        let intrinsics = CameraIntrinsics {
            alpha: 700.0,
            beta: 710.0,
            gamma: 4.0,
            u0: 320.0,
            v0: 240.0,
        };
        let distortion = BrownConradyDistortion {
            k1: -0.04,
            k2: 0.01,
            k3: 0.002,
            p1: 0.001,
            p2: -0.0005,
        };
        let normalized = Point2::new(0.12, -0.08);
        let projected = project_point_from_normalized(&intrinsics, &distortion, normalized);

        let r2 = normalized.x * normalized.x + normalized.y * normalized.y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + distortion.k1 * r2 + distortion.k2 * r4 + distortion.k3 * r6;
        let x_distorted = normalized.x * radial
            + 2.0 * distortion.p1 * normalized.x * normalized.y
            + distortion.p2 * (r2 + 2.0 * normalized.x * normalized.x);
        let y_distorted = normalized.y * radial
            + distortion.p1 * (r2 + 2.0 * normalized.y * normalized.y)
            + 2.0 * distortion.p2 * normalized.x * normalized.y;
        let expected_x =
            intrinsics.alpha * x_distorted + intrinsics.gamma * y_distorted + intrinsics.u0;
        let expected_y = intrinsics.beta * y_distorted + intrinsics.v0;

        assert_relative_eq!(projected.x, expected_x, epsilon = 1e-12);
        assert_relative_eq!(projected.y, expected_y, epsilon = 1e-12);
    }

    #[test]
    fn world_projection_matches_residual_projection() {
        let intrinsics = CameraIntrinsics {
            alpha: 800.0,
            beta: 805.0,
            gamma: 1.5,
            u0: 320.0,
            v0: 240.0,
        };
        let distortion = BrownConradyDistortion {
            k1: -0.05,
            k2: 0.02,
            k3: -0.003,
            p1: 0.001,
            p2: 0.0008,
        };
        let extrinsics = make_extrinsics(NaVector3::new(0.02, -0.03, 1.3));
        let world = Point3::new(0.15, -0.04, 0.0);
        let projected = project_world_to_pixel(&intrinsics, &distortion, &extrinsics, world).unwrap();

        let pose = make_pose(extrinsics.translation);
        let projected_var = project_with_variable_model(
            &pose,
            &VectorVar5::new(
                intrinsics.alpha,
                intrinsics.beta,
                intrinsics.gamma,
                intrinsics.u0,
                intrinsics.v0,
            ),
            &VectorVar5::new(
                distortion.k1,
                distortion.k2,
                distortion.k3,
                distortion.p1,
                distortion.p2,
            ),
            &VectorVar3::new(world.x, world.y, world.z),
        );

        assert_relative_eq!(projected.x, projected_var[0], epsilon = 1e-12);
        assert_relative_eq!(projected.y, projected_var[1], epsilon = 1e-12);
    }
}
