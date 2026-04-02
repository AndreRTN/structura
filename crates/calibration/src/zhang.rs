use anyhow::{Result, anyhow};
use nalgebra::{DMatrix, DVector, Matrix3, Point2, Point3, SVector, Vector3, linalg::SVD};
use structura_geometry::{
    camera::{CameraExtrinsics, CameraIntrinsics},
    homography::HomographyMatrix,
    point::PointCorrespondence2D3D,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ZhangCalibration {
    pub intrinsics: CameraIntrinsics,
    pub extrinsics: Vec<CameraExtrinsics>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZhangDistortion {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub p1: f64,
    pub p2: f64,
}

pub fn distort_point(
    intrinsics: &CameraIntrinsics,
    distortion: ZhangDistortion,
    point: Point2<f64>,
) -> Point2<f64> {
    project_normalized_to_pixel(
        intrinsics,
        distort_normalized_point(project_pixel_to_normalized(intrinsics, point), distortion),
    )
}

pub fn undistort_point_iterative(
    intrinsics: &CameraIntrinsics,
    distortion: ZhangDistortion,
    point: Point2<f64>,
) -> Point2<f64> {
    let distorted_normalized = project_pixel_to_normalized(intrinsics, point);
    let undistorted = (0..8).fold(distorted_normalized, |estimate, _| {
        let distorted_estimate = distort_normalized_point(estimate, distortion);
        Point2::new(
            distorted_normalized.x + estimate.x - distorted_estimate.x,
            distorted_normalized.y + estimate.y - distorted_estimate.y,
        )
    });

    project_normalized_to_pixel(intrinsics, undistorted)
}

pub fn undistort_points(
    intrinsics: &CameraIntrinsics,
    distortion: ZhangDistortion,
    points: &[Point2<f64>],
) -> Vec<Point2<f64>> {
    points
        .iter()
        .copied()
        .map(|point| undistort_point_iterative(intrinsics, distortion, point))
        .collect()
}

pub fn estimate_intrinsics_from_homographies(
    homographies: &[HomographyMatrix],
) -> Result<CameraIntrinsics> {
    if homographies.len() < 3 {
        return Err(anyhow!(
            "at least three homographies are required for Zhang calibration"
        ));
    }

    let rows = homographies
        .iter()
        .flat_map(|homography| {
            let v12 = vij(homography, 0, 1);
            let v11 = vij(homography, 0, 0);
            let v22 = vij(homography, 1, 1);
            [v12, v11 - v22]
        })
        .collect::<Vec<_>>();
    let mut v = DMatrix::<f64>::zeros(rows.len(), 6);

    rows.iter()
        .enumerate()
        .for_each(|(row, values)| write_row(&mut v, row, values));

    let b = solve_homogeneous_system(&v)?;
    intrinsics_from_b(&b)
}

pub fn estimate_extrinsics_from_homographies(
    intrinsics: &CameraIntrinsics,
    homographies: &[HomographyMatrix],
) -> Result<Vec<CameraExtrinsics>> {
    let a_inv = intrinsics
        .matrix()
        .try_inverse()
        .ok_or_else(|| anyhow!("intrinsics matrix is not invertible"))?;

    homographies
        .iter()
        .map(|homography| extrinsics_from_homography(&a_inv, homography))
        .collect()
}

pub fn calibrate_from_homographies(homographies: &[HomographyMatrix]) -> Result<ZhangCalibration> {
    let intrinsics = estimate_intrinsics_from_homographies(homographies)?;
    let extrinsics = estimate_extrinsics_from_homographies(&intrinsics, homographies)?;

    Ok(ZhangCalibration {
        intrinsics,
        extrinsics,
    })
}

pub fn estimate_radial_distortion(
    intrinsics: &CameraIntrinsics,
    extrinsics: &[CameraExtrinsics],
    observations: &[Vec<PointCorrespondence2D3D>],
) -> Result<ZhangDistortion> {
    if extrinsics.len() != observations.len() {
        return Err(anyhow!(
            "the number of extrinsic views must match the number of observation sets"
        ));
    }

    let num_equations = observations.iter().map(|view| view.len() * 2).sum::<usize>();
    if num_equations < 5 {
        return Err(anyhow!(
            "at least three 2d/3d correspondences are required to estimate Brown-Conrady distortion"
        ));
    }

    let equations = extrinsics
        .iter()
        .zip(observations)
        .map(|(extrinsics, view_observations)| {
            view_observations
                .iter()
                .map(|observation| radial_distortion_equations(intrinsics, extrinsics, observation))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();

    let mut d = DMatrix::<f64>::zeros(num_equations, 5);
    let mut residuals = DVector::<f64>::zeros(num_equations);
    equations
        .iter()
        .enumerate()
        .for_each(|(row, (coefficients, residual))| {
            coefficients
                .iter()
                .enumerate()
                .for_each(|(col, value)| d[(row, col)] = *value);
            residuals[row] = *residual;
        });

    let normal_matrix = d.transpose() * &d;
    let normal_rhs = d.transpose() * residuals;
    let solution = normal_matrix.try_inverse().ok_or_else(|| {
        anyhow!("failed to estimate radial distortion: normal matrix is singular")
    })? * normal_rhs;

    Ok(ZhangDistortion {
        k1: solution[0],
        k2: solution[1],
        k3: solution[2],
        p1: solution[3],
        p2: solution[4],
    })
}

pub fn calibrate_from_homographies_with_radial_distortion(
    homographies: &[HomographyMatrix],
    observations: &[Vec<PointCorrespondence2D3D>],
) -> Result<(ZhangCalibration, ZhangDistortion)> {
    let calibration = calibrate_from_homographies(homographies)?;
    let radial_distortion = estimate_radial_distortion(
        &calibration.intrinsics,
        &calibration.extrinsics,
        observations,
    )?;

    Ok((calibration, radial_distortion))
}

fn intrinsics_from_b(b: &SVector<f64, 6>) -> Result<CameraIntrinsics> {
    intrinsics_from_b_with_scale(b).or_else(|_| intrinsics_from_b_with_scale(&(-b)))
}

fn intrinsics_from_b_with_scale(b: &SVector<f64, 6>) -> Result<CameraIntrinsics> {
    let [b11, b12, b22, b13, b23, b33] = [b[0], b[1], b[2], b[3], b[4], b[5]];
    let denominator = b11 * b22 - b12 * b12;
    if denominator.abs() < 1e-12 {
        return Err(anyhow!(
            "invalid homography constraints: singular denominator"
        ));
    }

    if b11.abs() < 1e-12 {
        return Err(anyhow!("invalid homography constraints: b11 is too small"));
    }

    let v0 = (b12 * b13 - b11 * b23) / denominator;
    let lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;

    if lambda <= 0.0 {
        return Err(anyhow!(
            "invalid homography constraints: lambda must be positive"
        ));
    }

    let alpha = (lambda / b11).sqrt();
    let beta = (lambda * b11 / denominator).sqrt();
    let gamma = -b12 * alpha * alpha * beta / lambda;
    let u0 = gamma * v0 / beta - b13 * alpha * alpha / lambda;

    Ok(CameraIntrinsics {
        alpha,
        beta,
        gamma,
        u0,
        v0,
    })
}

fn extrinsics_from_homography(
    a_inv: &Matrix3<f64>,
    homography: &HomographyMatrix,
) -> Result<CameraExtrinsics> {
    let h1 = homography.column(0).into_owned();
    let h2 = homography.column(1).into_owned();
    let h3 = homography.column(2).into_owned();

    let r1_tilde = a_inv * h1;
    let r2_tilde = a_inv * h2;
    let norm1 = r1_tilde.norm();
    let norm2 = r2_tilde.norm();
    if norm1 <= 1e-12 || norm2 <= 1e-12 {
        return Err(anyhow!("degenerate homography for extrinsics estimation"));
    }

    let lambda = 2.0 / (norm1 + norm2);
    let r1 = lambda * r1_tilde;
    let r2 = lambda * r2_tilde;
    let r3 = r1.cross(&r2);
    let t = lambda * (a_inv * h3);

    let r_tilde = Matrix3::from_columns(&[r1, r2, r3]);
    let svd = SVD::new(r_tilde, true, true);
    let u = svd
        .u
        .ok_or_else(|| anyhow!("failed to compute left singular vectors for rotation"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("failed to compute right singular vectors for rotation"))?;

    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        let mut u_fixed = u;
        u_fixed
            .column_mut(2)
            .iter_mut()
            .for_each(|value| *value *= -1.0);
        rotation = u_fixed * v_t;
    }

    Ok(CameraExtrinsics {
        rotation,
        translation: t,
    })
}

fn solve_homogeneous_system(v: &DMatrix<f64>) -> Result<SVector<f64, 6>> {
    let svd = SVD::new(v.clone(), false, true);
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("failed to compute right singular vectors"))?;
    let smallest = v_t.row(v_t.nrows() - 1);

    Ok(SVector::<f64, 6>::from_iterator(smallest.iter().copied()))
}

fn vij(h: &HomographyMatrix, i: usize, j: usize) -> SVector<f64, 6> {
    let hi = column(h, i);
    let hj = column(h, j);

    SVector::<f64, 6>::new(
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    )
}

fn column(h: &HomographyMatrix, index: usize) -> Vector3<f64> {
    h.column(index).into_owned()
}

fn write_row(matrix: &mut DMatrix<f64>, row: usize, values: &SVector<f64, 6>) {
    values
        .iter()
        .enumerate()
        .for_each(|(col, value)| matrix[(row, col)] = *value);
}

fn project_world_to_normalized(
    extrinsics: &CameraExtrinsics,
    world: Point3<f64>,
) -> Result<Point2<f64>> {
    let camera = extrinsics.rotation * world.coords + extrinsics.translation;

    if camera.z.abs() <= 1e-12 {
        return Err(anyhow!("cannot project point with near-zero camera depth"));
    }

    Ok(Point2::new(camera.x / camera.z, camera.y / camera.z))
}

fn project_normalized_to_pixel(
    intrinsics: &CameraIntrinsics,
    normalized: Point2<f64>,
) -> Point2<f64> {
    Point2::new(
        intrinsics.alpha * normalized.x + intrinsics.gamma * normalized.y + intrinsics.u0,
        intrinsics.beta * normalized.y + intrinsics.v0,
    )
}

fn project_pixel_to_normalized(intrinsics: &CameraIntrinsics, point: Point2<f64>) -> Point2<f64> {
    let y = (point.y - intrinsics.v0) / intrinsics.beta;
    let x = (point.x - intrinsics.u0 - intrinsics.gamma * y) / intrinsics.alpha;
    Point2::new(x, y)
}

fn radial_distance_squared(normalized: Point2<f64>) -> f64 {
    normalized.x * normalized.x + normalized.y * normalized.y
}

fn distort_normalized_point(
    normalized: Point2<f64>,
    distortion: ZhangDistortion,
) -> Point2<f64> {
    let radial = radial_distance_squared(normalized);
    let radial_squared = radial * radial;
    let radial_cubed = radial_squared * radial;
    let x = normalized.x;
    let y = normalized.y;
    let radial_factor =
        1.0 + distortion.k1 * radial + distortion.k2 * radial_squared + distortion.k3 * radial_cubed;

    Point2::new(
        x * radial_factor + 2.0 * distortion.p1 * x * y + distortion.p2 * (radial + 2.0 * x * x),
        y * radial_factor + distortion.p1 * (radial + 2.0 * y * y) + 2.0 * distortion.p2 * x * y,
    )
}

fn radial_distortion_equations(
    intrinsics: &CameraIntrinsics,
    extrinsics: &CameraExtrinsics,
    observation: &PointCorrespondence2D3D,
) -> Result<[(SVector<f64, 5>, f64); 2]> {
    let normalized = project_world_to_normalized(extrinsics, observation.world)?;
    let ideal = project_normalized_to_pixel(intrinsics, normalized);
    let radial = radial_distance_squared(normalized);
    let radial_squared = radial * radial;
    let radial_cubed = radial_squared * radial;
    let x = normalized.x;
    let y = normalized.y;
    let delta_u = ideal.x - intrinsics.u0;
    let delta_v = ideal.y - intrinsics.v0;
    let tangential_x = [
        intrinsics.alpha * 2.0 * x * y + intrinsics.gamma * (radial + 2.0 * y * y),
        intrinsics.alpha * (radial + 2.0 * x * x) + intrinsics.gamma * 2.0 * x * y,
    ];
    let tangential_y = [
        intrinsics.beta * (radial + 2.0 * y * y),
        intrinsics.beta * 2.0 * x * y,
    ];

    Ok([
        (
            SVector::<f64, 5>::new(
                delta_u * radial,
                delta_u * radial_squared,
                delta_u * radial_cubed,
                tangential_x[0],
                tangential_x[1],
            ),
            observation.image.x - ideal.x,
        ),
        (
            SVector::<f64, 5>::new(
                delta_v * radial,
                delta_v * radial_squared,
                delta_v * radial_cubed,
                tangential_y[0],
                tangential_y[1],
            ),
            observation.image.y - ideal.y,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Rotation3, UnitQuaternion};

    #[test]
    fn estimates_intrinsics_from_multiple_homographies() {
        let intrinsics = CameraIntrinsics {
            alpha: 800.0,
            beta: 820.0,
            gamma: 12.0,
            u0: 320.0,
            v0: 240.0,
        };

        let homographies = vec![
            homography_from_pose(
                &intrinsics,
                Vector3::new(0.1, -0.2, 0.05),
                Vector3::new(0.2, -0.1, 5.0),
            ),
            homography_from_pose(
                &intrinsics,
                Vector3::new(-0.15, 0.05, 0.12),
                Vector3::new(-0.3, 0.15, 6.0),
            ),
            homography_from_pose(
                &intrinsics,
                Vector3::new(0.08, 0.12, -0.09),
                Vector3::new(0.1, 0.25, 5.5),
            ),
        ];

        let estimated = estimate_intrinsics_from_homographies(&homographies).unwrap();

        assert_relative_eq!(estimated.alpha, intrinsics.alpha, epsilon = 1e-6);
        assert_relative_eq!(estimated.beta, intrinsics.beta, epsilon = 1e-6);
        assert_relative_eq!(estimated.gamma, intrinsics.gamma, epsilon = 1e-6);
        assert_relative_eq!(estimated.u0, intrinsics.u0, epsilon = 1e-6);
        assert_relative_eq!(estimated.v0, intrinsics.v0, epsilon = 1e-6);
    }

    #[test]
    fn estimates_extrinsics_from_homography() {
        let intrinsics = CameraIntrinsics {
            alpha: 700.0,
            beta: 710.0,
            gamma: 5.0,
            u0: 300.0,
            v0: 220.0,
        };
        let rotation_vec = Vector3::new(0.1, -0.05, 0.08);
        let translation = Vector3::new(0.3, -0.2, 4.5);
        let homography = homography_from_pose(&intrinsics, rotation_vec, translation);

        let extrinsics = estimate_extrinsics_from_homographies(&intrinsics, &[homography]).unwrap();
        let extrinsic = &extrinsics[0];
        let expected_rotation = UnitQuaternion::from_scaled_axis(rotation_vec)
            .to_rotation_matrix()
            .into_inner();

        for row in 0..3 {
            for col in 0..3 {
                assert_relative_eq!(
                    extrinsic.rotation[(row, col)],
                    expected_rotation[(row, col)],
                    epsilon = 1e-6
                );
            }
        }

        assert_relative_eq!(extrinsic.translation[0], translation[0], epsilon = 1e-6);
        assert_relative_eq!(extrinsic.translation[1], translation[1], epsilon = 1e-6);
        assert_relative_eq!(extrinsic.translation[2], translation[2], epsilon = 1e-6);
    }

    #[test]
    fn calibrates_intrinsics_and_extrinsics_together() {
        let intrinsics = CameraIntrinsics {
            alpha: 900.0,
            beta: 910.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        };

        let homographies = vec![
            homography_from_pose(
                &intrinsics,
                Vector3::new(0.05, 0.1, -0.02),
                Vector3::new(0.0, 0.0, 5.0),
            ),
            homography_from_pose(
                &intrinsics,
                Vector3::new(-0.07, 0.02, 0.09),
                Vector3::new(0.4, -0.1, 6.0),
            ),
            homography_from_pose(
                &intrinsics,
                Vector3::new(0.03, -0.08, 0.06),
                Vector3::new(-0.2, 0.2, 5.5),
            ),
        ];

        let calibration = calibrate_from_homographies(&homographies).unwrap();

        assert_relative_eq!(
            calibration.intrinsics.alpha,
            intrinsics.alpha,
            epsilon = 1e-6
        );
        assert_relative_eq!(calibration.intrinsics.beta, intrinsics.beta, epsilon = 1e-6);
        assert_eq!(calibration.extrinsics.len(), homographies.len());
    }

    #[test]
    fn estimates_radial_distortion_from_observations() {
        let intrinsics = CameraIntrinsics {
            alpha: 850.0,
            beta: 860.0,
            gamma: 4.0,
            u0: 320.0,
            v0: 240.0,
        };
        let distortion = ZhangDistortion {
            k1: 0.015,
            k2: -0.004,
            k3: 0.002,
            p1: 0.001,
            p2: -0.0005,
        };

        let extrinsics = vec![
            extrinsics_from_pose(
                Vector3::new(0.04, -0.02, 0.03),
                Vector3::new(0.1, -0.05, 4.5),
            ),
            extrinsics_from_pose(
                Vector3::new(-0.06, 0.03, 0.08),
                Vector3::new(-0.2, 0.1, 5.2),
            ),
        ];
        let observations = extrinsics
            .iter()
            .map(|extrinsics| synthetic_observations(&intrinsics, extrinsics, distortion))
            .collect::<Vec<_>>();

        let estimated =
            estimate_radial_distortion(&intrinsics, &extrinsics, &observations).unwrap();

        assert_relative_eq!(estimated.k1, distortion.k1, epsilon = 1e-10);
        assert_relative_eq!(estimated.k2, distortion.k2, epsilon = 1e-10);
        assert_relative_eq!(estimated.k3, distortion.k3, epsilon = 1e-10);
        assert_relative_eq!(estimated.p1, distortion.p1, epsilon = 1e-10);
        assert_relative_eq!(estimated.p2, distortion.p2, epsilon = 1e-10);
    }

    #[test]
    fn calibrates_with_radial_distortion() {
        let intrinsics = CameraIntrinsics {
            alpha: 900.0,
            beta: 910.0,
            gamma: 3.0,
            u0: 320.0,
            v0: 240.0,
        };
        let distortion = ZhangDistortion {
            k1: 0.01,
            k2: -0.002,
            k3: 0.001,
            p1: 0.0008,
            p2: -0.0006,
        };
        let extrinsics = vec![
            extrinsics_from_pose(Vector3::new(0.05, 0.1, -0.02), Vector3::new(0.0, 0.0, 5.0)),
            extrinsics_from_pose(
                Vector3::new(-0.07, 0.02, 0.09),
                Vector3::new(0.4, -0.1, 6.0),
            ),
            extrinsics_from_pose(
                Vector3::new(0.03, -0.08, 0.06),
                Vector3::new(-0.2, 0.2, 5.5),
            ),
        ];
        let homographies = extrinsics
            .iter()
            .map(|extrinsics| homography_from_extrinsics(&intrinsics, extrinsics))
            .collect::<Vec<_>>();
        let observations = extrinsics
            .iter()
            .map(|extrinsics| synthetic_observations(&intrinsics, extrinsics, distortion))
            .collect::<Vec<_>>();

        let (calibration, estimated_distortion) =
            calibrate_from_homographies_with_radial_distortion(&homographies, &observations)
                .unwrap();

        assert_relative_eq!(
            calibration.intrinsics.alpha,
            intrinsics.alpha,
            epsilon = 1e-6
        );
        assert_relative_eq!(calibration.intrinsics.beta, intrinsics.beta, epsilon = 1e-6);
        assert_eq!(calibration.extrinsics.len(), 3);
        assert_relative_eq!(estimated_distortion.k1, distortion.k1, epsilon = 1e-10);
        assert_relative_eq!(estimated_distortion.k2, distortion.k2, epsilon = 1e-10);
        assert_relative_eq!(estimated_distortion.k3, distortion.k3, epsilon = 1e-10);
        assert_relative_eq!(estimated_distortion.p1, distortion.p1, epsilon = 1e-10);
        assert_relative_eq!(estimated_distortion.p2, distortion.p2, epsilon = 1e-10);
    }

    #[test]
    fn distorts_and_undistorts_single_point() {
        let intrinsics = CameraIntrinsics {
            alpha: 900.0,
            beta: 910.0,
            gamma: 3.0,
            u0: 320.0,
            v0: 240.0,
        };
        let distortion = ZhangDistortion {
            k1: 0.01,
            k2: -0.002,
            k3: 0.001,
            p1: 0.0008,
            p2: -0.0006,
        };
        let ideal = Point2::new(412.0, 318.0);

        let distorted = distort_point(&intrinsics, distortion, ideal);
        let undistorted = undistort_point_iterative(&intrinsics, distortion, distorted);

        assert_relative_eq!(undistorted.x, ideal.x, epsilon = 1e-10);
        assert_relative_eq!(undistorted.y, ideal.y, epsilon = 1e-10);
    }

    #[test]
    fn undistorts_multiple_points() {
        let intrinsics = CameraIntrinsics {
            alpha: 850.0,
            beta: 860.0,
            gamma: 4.0,
            u0: 320.0,
            v0: 240.0,
        };
        let distortion = ZhangDistortion {
            k1: 0.015,
            k2: -0.004,
            k3: 0.002,
            p1: 0.001,
            p2: -0.0005,
        };
        let ideal_points = vec![
            Point2::new(320.0, 240.0),
            Point2::new(412.0, 318.0),
            Point2::new(280.0, 190.0),
        ];
        let distorted_points = ideal_points
            .iter()
            .copied()
            .map(|point| distort_point(&intrinsics, distortion, point))
            .collect::<Vec<_>>();

        let undistorted_points = undistort_points(&intrinsics, distortion, &distorted_points);

        undistorted_points
            .iter()
            .zip(ideal_points.iter())
            .for_each(|(undistorted, ideal)| {
                assert_relative_eq!(undistorted.x, ideal.x, epsilon = 1e-10);
                assert_relative_eq!(undistorted.y, ideal.y, epsilon = 1e-10);
            });
    }

    fn homography_from_pose(
        intrinsics: &CameraIntrinsics,
        rotation_vec: Vector3<f64>,
        translation: Vector3<f64>,
    ) -> HomographyMatrix {
        let extrinsics = extrinsics_from_pose(rotation_vec, translation);
        homography_from_extrinsics(intrinsics, &extrinsics)
    }

    fn homography_from_extrinsics(
        intrinsics: &CameraIntrinsics,
        extrinsics: &CameraExtrinsics,
    ) -> HomographyMatrix {
        let r1 = extrinsics.rotation.column(0).into_owned();
        let r2 = extrinsics.rotation.column(1).into_owned();
        intrinsics.matrix() * Matrix3::from_columns(&[r1, r2, extrinsics.translation])
    }

    fn extrinsics_from_pose(
        rotation_vec: Vector3<f64>,
        translation: Vector3<f64>,
    ) -> CameraExtrinsics {
        let rotation = Rotation3::from_scaled_axis(rotation_vec);
        CameraExtrinsics {
            rotation: rotation.into_inner(),
            translation,
        }
    }

    fn synthetic_observations(
        intrinsics: &CameraIntrinsics,
        extrinsics: &CameraExtrinsics,
        distortion: ZhangDistortion,
    ) -> Vec<PointCorrespondence2D3D> {
        let board_points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(0.5, 1.5, 0.0),
            Point3::new(1.5, 1.5, 0.0),
        ];

        board_points
            .into_iter()
            .map(|world| {
                let normalized = project_world_to_normalized(extrinsics, world).unwrap();
                let ideal = project_normalized_to_pixel(intrinsics, normalized);
                let observed = distort_point(intrinsics, distortion, ideal);
                PointCorrespondence2D3D::new(observed, world)
            })
            .collect()
    }
}
