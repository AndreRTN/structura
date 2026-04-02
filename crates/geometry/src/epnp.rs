use anyhow::{Result, anyhow, ensure};
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, SMatrix, SVector, SymmetricEigen, Vector3};

use crate::{
    camera::{CameraExtrinsics, CameraIntrinsics},
    point::{PointCorrespondence2D3D, WorldPoint},
};

type BetaVector = SVector<f64, 4>;
type RhoVector = SVector<f64, 6>;
type LMatrix = SMatrix<f64, 6, 10>;
type AlphaRow = [f64; 4];
type ControlPoints = [Vector3<f64>; 4];
type Basis = [[Vector3<f64>; 4]; 4];

#[derive(Debug, Clone)]
struct PoseCandidate {
    extrinsics: CameraExtrinsics,
    reprojection_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpnpConfig {
    pub beta_iterations: usize,
    pub pose_refinement_iterations: usize,
}

impl Default for EpnpConfig {
    fn default() -> Self {
        Self {
            beta_iterations: 8,
            pose_refinement_iterations: 10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpnpSolver {
    config: EpnpConfig,
}

impl Default for EpnpSolver {
    fn default() -> Self {
        Self {
            config: EpnpConfig::default(),
        }
    }
}

impl EpnpSolver {
    pub fn new(config: EpnpConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> EpnpConfig {
        self.config
    }

    pub fn solve_pose(
        &self,
        intrinsics: &CameraIntrinsics,
        correspondences: &[PointCorrespondence2D3D],
    ) -> Result<CameraExtrinsics> {
        ensure!(
            correspondences.len() >= 4,
            "epnp requires at least four 3d-2d correspondences"
        );

        let control_points = choose_control_points(correspondences)?;
        let alphas = correspondences
            .iter()
            .map(|correspondence| barycentric_coordinates(&control_points, &correspondence.world))
            .collect::<Result<Vec<_>>>()?;
        let normalized_points = correspondences
            .iter()
            .map(|correspondence| {
                pixel_to_normalized(intrinsics, correspondence.image.x, correspondence.image.y)
            })
            .collect::<Result<Vec<_>>>()?;

        let m = build_measurement_matrix(&alphas, &normalized_points);
        let basis = nullspace_basis(&m)?;
        let l = compute_l_6x10(&basis);
        let rho = compute_rho(&control_points);
        let candidates = [
            find_betas_approx_1(&l, &rho),
            find_betas_approx_2(&l, &rho),
            find_betas_approx_3(&l, &rho),
        ]
        .into_iter()
        .filter_map(|betas| {
            let mut betas = betas?;
            gauss_newton_refine_betas(&l, &rho, &mut betas, self.config.beta_iterations)?;
            compute_pose_candidate(
                intrinsics,
                correspondences,
                &alphas,
                &basis,
                betas,
                self.config.pose_refinement_iterations,
            )
        })
        .collect::<Vec<_>>();

        candidates
            .into_iter()
            .min_by(|left, right| left.reprojection_error.total_cmp(&right.reprojection_error))
            .map(|candidate| candidate.extrinsics)
            .ok_or_else(|| anyhow!("failed to recover EPnP pose from any beta hypothesis"))
    }
}

fn choose_control_points(correspondences: &[PointCorrespondence2D3D]) -> Result<ControlPoints> {
    let centroid = correspondences
        .iter()
        .fold(Vector3::zeros(), |acc, correspondence| {
            acc + correspondence.world.coords
        })
        / correspondences.len() as f64;

    let covariance = correspondences
        .iter()
        .fold(Matrix3::zeros(), |acc, correspondence| {
            let centered = correspondence.world.coords - centroid;
            acc + centered * centered.transpose()
        })
        / correspondences.len() as f64;

    let eigen = SymmetricEigen::new(covariance);
    let eigenpairs = (0..3)
        .map(|index| {
            (
                eigen.eigenvalues[index],
                eigen.eigenvectors.column(index).into_owned(),
            )
        })
        .collect::<Vec<_>>();
    let mut eigenpairs = eigenpairs;
    eigenpairs.sort_by(|left, right| right.0.total_cmp(&left.0));

    ensure!(
        eigenpairs[1].0 > f64::EPSILON,
        "epnp control points are degenerate for colinear or collapsed geometry"
    );

    let mut control_points = [centroid; 4];
    for index in 0..3 {
        let scale = (eigenpairs[index].0 / correspondences.len() as f64).sqrt();
        control_points[index + 1] = centroid + eigenpairs[index].1 * scale;
    }

    Ok(control_points)
}

fn barycentric_coordinates(control_points: &ControlPoints, world: &WorldPoint) -> Result<AlphaRow> {
    let basis = Matrix3::from_columns(&[
        control_points[1] - control_points[0],
        control_points[2] - control_points[0],
        control_points[3] - control_points[0],
    ]);
    let svd = basis.svd(true, true);
    let basis_inverse = svd
        .pseudo_inverse(1e-12)
        .map_err(|_| anyhow!("epnp control points do not form a usable basis"))?;
    let weights_123 = basis_inverse * (world.coords - control_points[0]);
    let alpha1 = weights_123[0];
    let alpha2 = weights_123[1];
    let alpha3 = weights_123[2];
    let alpha0 = 1.0 - alpha1 - alpha2 - alpha3;

    Ok([alpha0, alpha1, alpha2, alpha3])
}

fn pixel_to_normalized(intrinsics: &CameraIntrinsics, x: f64, y: f64) -> Result<Vector3<f64>> {
    let k_inv = intrinsics
        .matrix()
        .try_inverse()
        .ok_or_else(|| anyhow!("camera intrinsics matrix is not invertible"))?;
    Ok(k_inv * Vector3::new(x, y, 1.0))
}

fn build_measurement_matrix(
    alphas: &[AlphaRow],
    normalized_points: &[Vector3<f64>],
) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::zeros(alphas.len() * 2, 12);

    for (index, (alpha, normalized)) in alphas.iter().zip(normalized_points.iter()).enumerate() {
        let row_x = index * 2;
        let row_y = row_x + 1;
        for control_index in 0..4 {
            let base = control_index * 3;
            m[(row_x, base)] = alpha[control_index];
            m[(row_x, base + 2)] = -alpha[control_index] * normalized.x;
            m[(row_y, base + 1)] = alpha[control_index];
            m[(row_y, base + 2)] = -alpha[control_index] * normalized.y;
        }
    }

    m
}

fn nullspace_basis(m: &DMatrix<f64>) -> Result<Basis> {
    let svd = m.clone().svd(true, true);
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("measurement matrix SVD did not produce V^T"))?;
    ensure!(
        v_t.nrows() >= 4 && v_t.ncols() == 12,
        "unexpected EPnP nullspace dimensions"
    );

    let mut basis = [[Vector3::zeros(); 4]; 4];
    for basis_index in 0..4 {
        let row = v_t.row(v_t.nrows() - 1 - basis_index);
        for control_index in 0..4 {
            let base = control_index * 3;
            basis[basis_index][control_index] =
                Vector3::new(row[base], row[base + 1], row[base + 2]);
        }
    }
    Ok(basis)
}

fn compute_l_6x10(basis: &Basis) -> LMatrix {
    let mut l = LMatrix::zeros();
    for (row_index, (left, right)) in control_pairs().into_iter().enumerate() {
        let dv = (0..4)
            .map(|basis_index| basis[basis_index][left] - basis[basis_index][right])
            .collect::<Vec<_>>();

        l[(row_index, 0)] = dv[0].dot(&dv[0]);
        l[(row_index, 1)] = 2.0 * dv[0].dot(&dv[1]);
        l[(row_index, 2)] = dv[1].dot(&dv[1]);
        l[(row_index, 3)] = 2.0 * dv[0].dot(&dv[2]);
        l[(row_index, 4)] = 2.0 * dv[1].dot(&dv[2]);
        l[(row_index, 5)] = dv[2].dot(&dv[2]);
        l[(row_index, 6)] = 2.0 * dv[0].dot(&dv[3]);
        l[(row_index, 7)] = 2.0 * dv[1].dot(&dv[3]);
        l[(row_index, 8)] = 2.0 * dv[2].dot(&dv[3]);
        l[(row_index, 9)] = dv[3].dot(&dv[3]);
    }
    l
}

fn compute_rho(control_points: &ControlPoints) -> RhoVector {
    let mut rho = RhoVector::zeros();
    for (row_index, (left, right)) in control_pairs().into_iter().enumerate() {
        rho[row_index] = (control_points[left] - control_points[right]).norm_squared();
    }
    rho
}

fn camera_control_points(basis: &Basis, betas: &BetaVector) -> ControlPoints {
    let mut control_points = [Vector3::zeros(); 4];
    for control_index in 0..4 {
        control_points[control_index] = (0..4).fold(Vector3::zeros(), |acc, basis_index| {
            acc + basis[basis_index][control_index] * betas[basis_index]
        });
    }

    if control_points.iter().map(|point| point.z).sum::<f64>() < 0.0 {
        control_points.iter_mut().for_each(|point| *point = -*point);
    }

    control_points
}

fn compute_pose_candidate(
    intrinsics: &CameraIntrinsics,
    correspondences: &[PointCorrespondence2D3D],
    alphas: &[AlphaRow],
    basis: &Basis,
    betas: BetaVector,
    pose_refinement_iterations: usize,
) -> Option<PoseCandidate> {
    let control_points_camera = camera_control_points(basis, &betas);
    let mut camera_points = camera_points_from_alphas(alphas, &control_points_camera);
    solve_for_sign(&control_points_camera, &mut camera_points);
    let pose = estimate_pose_from_camera_points(correspondences, &camera_points).ok()?;
    let extrinsics = refine_pose(
        intrinsics,
        correspondences,
        pose,
        pose_refinement_iterations,
    )
    .ok()?;
    let reprojection_error = reprojection_error(intrinsics, correspondences, &extrinsics).ok()?;

    Some(PoseCandidate {
        extrinsics,
        reprojection_error,
    })
}

fn solve_for_sign(control_points_camera: &ControlPoints, camera_points: &mut [Vector3<f64>]) {
    if control_points_camera[0].z < 0.0 {
        camera_points.iter_mut().for_each(|point| *point = -*point);
    }
}

fn find_betas_approx_1(l: &LMatrix, rho: &RhoVector) -> Option<BetaVector> {
    let mut reduced = SMatrix::<f64, 6, 4>::zeros();
    for row in 0..6 {
        reduced[(row, 0)] = l[(row, 0)];
        reduced[(row, 1)] = l[(row, 1)];
        reduced[(row, 2)] = l[(row, 3)];
        reduced[(row, 3)] = l[(row, 6)];
    }

    let solution = solve_least_squares(&reduced, rho)?;
    let beta0 = solution[0].abs().sqrt().max(1e-12);
    if solution[0] < 0.0 {
        Some(BetaVector::new(
            beta0,
            -solution[1] / beta0,
            -solution[2] / beta0,
            -solution[3] / beta0,
        ))
    } else {
        Some(BetaVector::new(
            beta0,
            solution[1] / beta0,
            solution[2] / beta0,
            solution[3] / beta0,
        ))
    }
}

fn find_betas_approx_2(l: &LMatrix, rho: &RhoVector) -> Option<BetaVector> {
    let mut reduced = SMatrix::<f64, 6, 3>::zeros();
    for row in 0..6 {
        reduced[(row, 0)] = l[(row, 0)];
        reduced[(row, 1)] = l[(row, 1)];
        reduced[(row, 2)] = l[(row, 2)];
    }

    let solution = solve_least_squares(&reduced, rho)?;
    let mut beta0 = solution[0].abs().sqrt();
    let beta1 = if solution[0] < 0.0 {
        if solution[2] < 0.0 {
            (-solution[2]).sqrt()
        } else {
            0.0
        }
    } else if solution[2] > 0.0 {
        solution[2].sqrt()
    } else {
        0.0
    };
    if solution[1] < 0.0 {
        beta0 = -beta0;
    }

    Some(BetaVector::new(beta0, beta1, 0.0, 0.0))
}

fn find_betas_approx_3(l: &LMatrix, rho: &RhoVector) -> Option<BetaVector> {
    let mut reduced = SMatrix::<f64, 6, 5>::zeros();
    for row in 0..6 {
        reduced[(row, 0)] = l[(row, 0)];
        reduced[(row, 1)] = l[(row, 1)];
        reduced[(row, 2)] = l[(row, 2)];
        reduced[(row, 3)] = l[(row, 3)];
        reduced[(row, 4)] = l[(row, 4)];
    }

    let solution = solve_least_squares(&reduced, rho)?;
    let mut beta0 = solution[0].abs().sqrt();
    let beta1 = if solution[0] < 0.0 {
        if solution[2] < 0.0 {
            (-solution[2]).sqrt()
        } else {
            0.0
        }
    } else if solution[2] > 0.0 {
        solution[2].sqrt()
    } else {
        0.0
    };
    if solution[1] < 0.0 {
        beta0 = -beta0;
    }

    Some(BetaVector::new(
        beta0,
        beta1,
        solution[3] / beta0.max(1e-12),
        0.0,
    ))
}

fn gauss_newton_refine_betas(
    l: &LMatrix,
    rho: &RhoVector,
    betas: &mut BetaVector,
    iterations: usize,
) -> Option<()> {
    for _ in 0..iterations.max(1) {
        let (a, b) = compute_gauss_newton_system(l, rho, betas);
        let delta = solve_least_squares(&a, &b)?;
        *betas += delta;
    }

    Some(())
}

fn compute_gauss_newton_system(
    l: &LMatrix,
    rho: &RhoVector,
    betas: &BetaVector,
) -> (SMatrix<f64, 6, 4>, RhoVector) {
    let mut a = SMatrix::<f64, 6, 4>::zeros();
    let mut b = RhoVector::zeros();

    for row in 0..6 {
        a[(row, 0)] = 2.0 * l[(row, 0)] * betas[0]
            + l[(row, 1)] * betas[1]
            + l[(row, 3)] * betas[2]
            + l[(row, 6)] * betas[3];
        a[(row, 1)] = l[(row, 1)] * betas[0]
            + 2.0 * l[(row, 2)] * betas[1]
            + l[(row, 4)] * betas[2]
            + l[(row, 7)] * betas[3];
        a[(row, 2)] = l[(row, 3)] * betas[0]
            + l[(row, 4)] * betas[1]
            + 2.0 * l[(row, 5)] * betas[2]
            + l[(row, 8)] * betas[3];
        a[(row, 3)] = l[(row, 6)] * betas[0]
            + l[(row, 7)] * betas[1]
            + l[(row, 8)] * betas[2]
            + 2.0 * l[(row, 9)] * betas[3];

        b[row] = rho[row]
            - (l[(row, 0)] * betas[0] * betas[0]
                + l[(row, 1)] * betas[0] * betas[1]
                + l[(row, 2)] * betas[1] * betas[1]
                + l[(row, 3)] * betas[0] * betas[2]
                + l[(row, 4)] * betas[1] * betas[2]
                + l[(row, 5)] * betas[2] * betas[2]
                + l[(row, 6)] * betas[0] * betas[3]
                + l[(row, 7)] * betas[1] * betas[3]
                + l[(row, 8)] * betas[2] * betas[3]
                + l[(row, 9)] * betas[3] * betas[3]);
    }

    (a, b)
}

fn solve_least_squares<const ROWS: usize, const COLS: usize>(
    a: &SMatrix<f64, ROWS, COLS>,
    b: &SVector<f64, ROWS>,
) -> Option<SVector<f64, COLS>> {
    let a_dynamic = DMatrix::<f64>::from_column_slice(ROWS, COLS, a.as_slice());
    let b_dynamic = DVector::<f64>::from_column_slice(b.as_slice());
    let solution = a_dynamic.svd(true, true).solve(&b_dynamic, 1e-12).ok()?;
    Some(SVector::<f64, COLS>::from_column_slice(solution.as_slice()))
}

fn camera_points_from_alphas(
    alphas: &[AlphaRow],
    control_points_camera: &ControlPoints,
) -> Vec<Vector3<f64>> {
    alphas
        .iter()
        .map(|alpha| {
            (0..4).fold(Vector3::zeros(), |acc, control_index| {
                acc + control_points_camera[control_index] * alpha[control_index]
            })
        })
        .collect()
}

fn estimate_pose_from_camera_points(
    correspondences: &[PointCorrespondence2D3D],
    camera_points: &[Vector3<f64>],
) -> Result<CameraExtrinsics> {
    ensure!(
        correspondences.len() == camera_points.len(),
        "epnp correspondences and camera points must have the same length"
    );

    let world_centroid = correspondences
        .iter()
        .fold(Vector3::zeros(), |acc, correspondence| {
            acc + correspondence.world.coords
        })
        / correspondences.len() as f64;
    let camera_centroid = camera_points
        .iter()
        .fold(Vector3::zeros(), |acc, point| acc + point)
        / camera_points.len() as f64;

    let covariance = correspondences.iter().zip(camera_points.iter()).fold(
        Matrix3::zeros(),
        |acc, (correspondence, camera_point)| {
            let world = correspondence.world.coords - world_centroid;
            let camera = camera_point - camera_centroid;
            acc + camera * world.transpose()
        },
    );

    let svd = covariance.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| anyhow!("epnp pose SVD did not produce U"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("epnp pose SVD did not produce V^T"))?;
    let mut correction = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }

    let rotation = u * correction * v_t;
    let translation = camera_centroid - rotation * world_centroid;

    Ok(CameraExtrinsics {
        rotation,
        translation,
    })
}

fn refine_pose(
    intrinsics: &CameraIntrinsics,
    correspondences: &[PointCorrespondence2D3D],
    initial_pose: CameraExtrinsics,
    iterations: usize,
) -> Result<CameraExtrinsics> {
    let mut parameters = pose_to_parameters(&initial_pose)?;
    let mut best_pose = initial_pose.clone();
    let mut best_error = reprojection_error(intrinsics, correspondences, &best_pose)?;

    for _ in 0..iterations {
        let residual = pose_residuals(intrinsics, correspondences, &parameters)?;
        let jacobian = numerical_pose_jacobian(intrinsics, correspondences, &parameters)?;
        let normal = jacobian.transpose() * &jacobian
            + DMatrix::<f64>::identity(parameters.len(), parameters.len()) * 1e-9;
        let rhs = -jacobian.transpose() * residual;
        let Some(delta) = normal.lu().solve(&rhs) else {
            break;
        };

        parameters += &delta;
        let candidate = parameters_to_pose(&parameters);
        let candidate_error = reprojection_error(intrinsics, correspondences, &candidate)?;
        if candidate_error < best_error {
            best_error = candidate_error;
            best_pose = candidate;
        }

        if delta.norm() < 1e-10 {
            break;
        }
    }

    Ok(best_pose)
}

fn pose_to_parameters(pose: &CameraExtrinsics) -> Result<DVector<f64>> {
    let rotation = Rotation3::from_matrix(&pose.rotation);
    let scaled_axis = rotation.scaled_axis();
    Ok(DVector::from_row_slice(&[
        scaled_axis.x,
        scaled_axis.y,
        scaled_axis.z,
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
    ]))
}

fn parameters_to_pose(parameters: &DVector<f64>) -> CameraExtrinsics {
    let rotation =
        Rotation3::from_scaled_axis(Vector3::new(parameters[0], parameters[1], parameters[2]))
            .into_inner();

    CameraExtrinsics {
        rotation,
        translation: Vector3::new(parameters[3], parameters[4], parameters[5]),
    }
}

fn pose_residuals(
    intrinsics: &CameraIntrinsics,
    correspondences: &[PointCorrespondence2D3D],
    parameters: &DVector<f64>,
) -> Result<DVector<f64>> {
    let pose = parameters_to_pose(parameters);
    let residuals = correspondences
        .iter()
        .flat_map(|correspondence| {
            let projected = project_world_to_pixel(intrinsics, &pose, &correspondence.world);
            projected
                .map(|projected| {
                    [
                        projected.x - correspondence.image.x,
                        projected.y - correspondence.image.y,
                    ]
                })
                .unwrap_or([1e6, 1e6])
        })
        .collect::<Vec<_>>();

    Ok(DVector::from_vec(residuals))
}

fn numerical_pose_jacobian(
    intrinsics: &CameraIntrinsics,
    correspondences: &[PointCorrespondence2D3D],
    parameters: &DVector<f64>,
) -> Result<DMatrix<f64>> {
    let base = pose_residuals(intrinsics, correspondences, parameters)?;
    let mut jacobian = DMatrix::<f64>::zeros(base.len(), parameters.len());
    let step = 1e-6;

    for column in 0..parameters.len() {
        let mut perturbed = parameters.clone();
        perturbed[column] += step;
        let residual = pose_residuals(intrinsics, correspondences, &perturbed)?;
        let difference = (residual - &base) / step;
        jacobian.set_column(column, &difference);
    }

    Ok(jacobian)
}

fn reprojection_error(
    intrinsics: &CameraIntrinsics,
    correspondences: &[PointCorrespondence2D3D],
    pose: &CameraExtrinsics,
) -> Result<f64> {
    let total = correspondences
        .iter()
        .map(|correspondence| {
            project_world_to_pixel(intrinsics, pose, &correspondence.world).map(|projected| {
                let dx = projected.x - correspondence.image.x;
                let dy = projected.y - correspondence.image.y;
                (dx * dx + dy * dy).sqrt()
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<f64>();

    Ok(total / correspondences.len().max(1) as f64)
}

fn project_world_to_pixel(
    intrinsics: &CameraIntrinsics,
    pose: &CameraExtrinsics,
    world: &WorldPoint,
) -> Result<Vector3<f64>> {
    let camera = pose.rotation * world.coords + pose.translation;
    ensure!(camera.z > 1e-12, "point projects behind the camera");

    let normalized = camera.xy() / camera.z;
    Ok(Vector3::new(
        intrinsics.alpha * normalized.x + intrinsics.gamma * normalized.y + intrinsics.u0,
        intrinsics.beta * normalized.y + intrinsics.v0,
        1.0,
    ))
}

fn control_pairs() -> [(usize, usize); 6] {
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Rotation3};

    use super::*;

    #[test]
    fn recovers_pose_from_synthetic_correspondences() {
        assert_pose_close(
            &solve_pose(sample_intrinsics(), nominal_pose(), sample_world_points()),
            &nominal_pose(),
            1e-3,
            1e-3,
        );
    }

    #[test]
    fn recovers_pose_with_mild_non_zero_skew_intrinsics() {
        let intrinsics = CameraIntrinsics {
            alpha: 905.0,
            beta: 890.0,
            gamma: 4.0,
            u0: 319.0,
            v0: 241.0,
        };
        let expected = CameraExtrinsics {
            rotation: Rotation3::from_euler_angles(-0.02, 0.04, -0.015).into_inner(),
            translation: Vector3::new(-0.08, 0.05, 1.6),
        };

        let correspondences = make_correspondences(&intrinsics, &expected, &dense_world_points());
        let pose = EpnpSolver::default()
            .solve_pose(&intrinsics, &correspondences)
            .unwrap();

        assert_pose_close(&pose, &expected, 5e-3, 5e-3);
    }

    #[test]
    fn recovers_pose_from_depth_rich_configuration() {
        let intrinsics = sample_intrinsics();
        let expected = CameraExtrinsics {
            rotation: Rotation3::from_euler_angles(0.015, -0.025, 0.03).into_inner(),
            translation: Vector3::new(0.06, -0.02, 2.3),
        };
        let world_points = dense_world_points();

        let pose = solve_pose(intrinsics, expected.clone(), world_points);

        assert_pose_close(&pose, &expected, 1e-3, 1e-3);
    }

    #[test]
    fn remains_accurate_with_small_image_noise() {
        let intrinsics = sample_intrinsics();
        let expected = nominal_pose();
        let correspondences = sample_world_points()
            .into_iter()
            .enumerate()
            .map(|(index, world)| {
                let projected = project(&intrinsics, &expected, world);
                let noisy = Point2::new(
                    projected.x + (index as f64 * 0.17 - 0.5) * 0.35,
                    projected.y - (index as f64 * 0.11 - 0.4) * 0.35,
                );
                PointCorrespondence2D3D::new(noisy, world)
            })
            .collect::<Vec<_>>();

        let pose = EpnpSolver::default()
            .solve_pose(&intrinsics, &correspondences)
            .unwrap();

        assert_pose_close(&pose, &expected, 6e-3, 2e-2);
    }

    #[test]
    fn rejects_insufficient_correspondences() {
        let intrinsics = sample_intrinsics();
        let expected = nominal_pose();
        let correspondences = sample_world_points()[..3]
            .iter()
            .copied()
            .map(|world| {
                PointCorrespondence2D3D::new(project(&intrinsics, &expected, world), world)
            })
            .collect::<Vec<_>>();

        let error = EpnpSolver::default()
            .solve_pose(&intrinsics, &correspondences)
            .unwrap_err();

        assert!(error.to_string().contains("at least four"));
    }

    #[test]
    fn recovers_pose_from_planar_correspondences() {
        let intrinsics = sample_intrinsics();
        let expected = nominal_pose();
        let world_points = vec![
            WorldPoint::new(-0.3, -0.2, 0.0),
            WorldPoint::new(0.2, -0.1, 0.0),
            WorldPoint::new(0.4, 0.3, 0.0),
            WorldPoint::new(-0.2, 0.5, 0.0),
            WorldPoint::new(0.1, 0.1, 0.0),
            WorldPoint::new(-0.4, 0.0, 0.0),
        ];
        let correspondences = make_correspondences(&intrinsics, &expected, &world_points);

        let pose = EpnpSolver::default()
            .solve_pose(&intrinsics, &correspondences)
            .unwrap();

        assert_pose_close(&pose, &expected, 1e-3, 1e-3);
    }

    #[test]
    fn rejects_colinear_correspondences_as_degenerate() {
        let intrinsics = sample_intrinsics();
        let expected = nominal_pose();
        let world_points = vec![
            WorldPoint::new(-0.4, 0.0, 0.2),
            WorldPoint::new(-0.2, 0.0, 0.5),
            WorldPoint::new(0.0, 0.0, 0.8),
            WorldPoint::new(0.2, 0.0, 1.1),
            WorldPoint::new(0.4, 0.0, 1.4),
        ];
        let correspondences = make_correspondences(&intrinsics, &expected, &world_points);

        let error = EpnpSolver::default()
            .solve_pose(&intrinsics, &correspondences)
            .unwrap_err();

        assert!(error.to_string().contains("degenerate"));
    }

    #[test]
    fn rejects_collapsed_correspondences_as_degenerate() {
        let intrinsics = sample_intrinsics();
        let expected = nominal_pose();
        let repeated = WorldPoint::new(0.1, -0.2, 0.5);
        let correspondences = vec![
            PointCorrespondence2D3D::new(project(&intrinsics, &expected, repeated), repeated),
            PointCorrespondence2D3D::new(project(&intrinsics, &expected, repeated), repeated),
            PointCorrespondence2D3D::new(project(&intrinsics, &expected, repeated), repeated),
            PointCorrespondence2D3D::new(project(&intrinsics, &expected, repeated), repeated),
        ];

        let error = EpnpSolver::default()
            .solve_pose(&intrinsics, &correspondences)
            .unwrap_err();

        assert!(error.to_string().contains("degenerate"));
    }

    #[test]
    fn returns_error_when_any_point_projects_behind_camera() {
        let intrinsics = sample_intrinsics();
        let pose = CameraExtrinsics {
            rotation: Matrix3::identity(),
            translation: Vector3::new(0.0, 0.0, 0.2),
        };
        let world_points = vec![
            WorldPoint::new(-0.2, -0.1, 0.5),
            WorldPoint::new(0.3, 0.1, 0.8),
            WorldPoint::new(0.1, -0.2, -0.4),
            WorldPoint::new(-0.1, 0.2, 0.7),
        ];
        let correspondences = world_points
            .iter()
            .map(|world| {
                let image = project_world_to_pixel(&intrinsics, &pose, world)?;
                Ok(PointCorrespondence2D3D::new(
                    Point2::new(image.x, image.y),
                    *world,
                ))
            })
            .collect::<Result<Vec<_>>>()
            .unwrap_err();

        assert!(correspondences.to_string().contains("behind the camera"));
    }

    fn sample_world_points() -> Vec<WorldPoint> {
        vec![
            WorldPoint::new(-0.2, -0.3, 0.4),
            WorldPoint::new(0.1, -0.1, 0.7),
            WorldPoint::new(0.3, 0.2, 0.5),
            WorldPoint::new(-0.4, 0.1, 0.8),
            WorldPoint::new(0.2, 0.4, 0.9),
            WorldPoint::new(-0.1, 0.3, 0.6),
            WorldPoint::new(0.5, -0.2, 1.0),
            WorldPoint::new(-0.3, -0.4, 1.1),
        ]
    }

    fn dense_world_points() -> Vec<WorldPoint> {
        vec![
            WorldPoint::new(-0.35, -0.25, 0.45),
            WorldPoint::new(-0.15, 0.10, 0.70),
            WorldPoint::new(0.05, -0.30, 0.95),
            WorldPoint::new(0.25, 0.20, 1.20),
            WorldPoint::new(0.40, -0.05, 1.45),
            WorldPoint::new(-0.30, 0.35, 1.70),
            WorldPoint::new(0.15, 0.45, 1.95),
            WorldPoint::new(-0.05, -0.40, 2.20),
            WorldPoint::new(0.32, 0.28, 2.45),
            WorldPoint::new(-0.22, 0.05, 2.70),
        ]
    }

    fn sample_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics {
            alpha: 900.0,
            beta: 880.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        }
    }

    fn nominal_pose() -> CameraExtrinsics {
        CameraExtrinsics {
            rotation: Rotation3::from_euler_angles(0.04, -0.03, 0.02).into_inner(),
            translation: Vector3::new(0.2, -0.1, 1.4),
        }
    }

    fn solve_pose(
        intrinsics: CameraIntrinsics,
        expected: CameraExtrinsics,
        world_points: Vec<WorldPoint>,
    ) -> CameraExtrinsics {
        let correspondences = make_correspondences(&intrinsics, &expected, &world_points);
        EpnpSolver::default()
            .solve_pose(&intrinsics, &correspondences)
            .unwrap()
    }

    fn make_correspondences(
        intrinsics: &CameraIntrinsics,
        pose: &CameraExtrinsics,
        world_points: &[WorldPoint],
    ) -> Vec<PointCorrespondence2D3D> {
        world_points
            .iter()
            .copied()
            .map(|world| PointCorrespondence2D3D::new(project(intrinsics, pose, world), world))
            .collect()
    }

    fn assert_pose_close(
        actual: &CameraExtrinsics,
        expected: &CameraExtrinsics,
        max_rotation_angle: f64,
        max_translation_error: f64,
    ) {
        let rotation_delta =
            Rotation3::from_matrix(&(actual.rotation * expected.rotation.transpose()));
        assert!(rotation_delta.angle() < max_rotation_angle);
        assert!((actual.translation - expected.translation).norm() < max_translation_error);
    }

    fn project(
        intrinsics: &CameraIntrinsics,
        pose: &CameraExtrinsics,
        world: WorldPoint,
    ) -> Point2<f64> {
        let camera = pose.rotation * world.coords + pose.translation;
        Point2::new(
            intrinsics.alpha * (camera.x / camera.z) + intrinsics.u0,
            intrinsics.beta * (camera.y / camera.z) + intrinsics.v0,
        )
    }
}
