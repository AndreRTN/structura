use anyhow::{Result, anyhow};
use factrs::{
    assign_symbols,
    core::{Graph, Values},
    fac,
    optimizers::{LevenMarquardt, LevenParams},
    traits::Optimizer,
    variables::{MatrixLieGroup, SE3, SO3, VectorVar5},
};
use nalgebra::{Point2, Vector3};
use structura_calibration::zhang::{ZhangCalibration, ZhangDistortion};
use structura_geometry::{
    camera::{CameraExtrinsics, CameraIntrinsics},
    point::PointCorrespondence2D3D,
};

use crate::residuals::reprojection::{ReprojectionResidual, project_world_to_pixel};

assign_symbols!(P: SE3; K: VectorVar5; D: VectorVar5);

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BrownConradyDistortion {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub p1: f64,
    pub p2: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverCalibration {
    pub intrinsics: CameraIntrinsics,
    pub distortion: BrownConradyDistortion,
    pub extrinsics: Vec<CameraExtrinsics>,
    pub reprojection_error: f64,
}

pub fn from_zhang_initialization(
    calibration: &ZhangCalibration,
    distortion: Option<ZhangDistortion>,
) -> SolverCalibration {
    let distortion = distortion
        .map(|distortion| BrownConradyDistortion {
            k1: distortion.k1,
            k2: distortion.k2,
            k3: distortion.k3,
            p1: distortion.p1,
            p2: distortion.p2,
        })
        .unwrap_or_default();

    SolverCalibration {
        intrinsics: calibration.intrinsics.clone(),
        distortion,
        extrinsics: calibration.extrinsics.clone(),
        reprojection_error: 0.0,
    }
}

pub fn optimize(
    observations: &[Vec<PointCorrespondence2D3D>],
    initial: &SolverCalibration,
) -> Result<SolverCalibration> {
    if observations.is_empty() {
        return Err(anyhow!(
            "at least one observation set is required for optimization"
        ));
    }

    if observations.len() != initial.extrinsics.len() {
        return Err(anyhow!(
            "the number of observation sets must match the number of extrinsic views"
        ));
    }

    let mut graph = Graph::new();
    observations.iter().enumerate().for_each(|(index, view)| {
        view.iter().for_each(|correspondence| {
            graph.add_factor(fac![
                ReprojectionResidual::new(correspondence.image, correspondence.world),
                (P(index as u32), K(0), D(0))
            ]);
        });
    });

    let mut values = build_pose_values(initial);
    values.insert(
        K(0),
        VectorVar5::new(
            initial.intrinsics.alpha,
            initial.intrinsics.beta,
            initial.intrinsics.gamma,
            initial.intrinsics.u0,
            initial.intrinsics.v0,
        ),
    );
    values.insert(
        D(0),
        VectorVar5::new(
            initial.distortion.k1,
            initial.distortion.k2,
            initial.distortion.k3,
            initial.distortion.p1,
            initial.distortion.p2,
        ),
    );

    let params = LevenParams {
        lambda_min: 1e-20,
        lambda_max: 1e10,
        lambda_factor: 8.0,
        min_model_fidelity: 1e-4,
        diagonal_damping: true,
        base: factrs::optimizers::BaseOptParams {
            max_iterations: 200,
            error_tol_relative: 1e-10,
            error_tol_absolute: 1e-10,
            error_tol: 0.0,
        },
    };

    let mut optimizer = LevenMarquardt::new(params, graph);
    let result = optimizer
        .optimize(values)
        .map_err(|error| anyhow!("solver optimization failed: {error:?}"))?;

    let intrinsics = result
        .get(K(0))
        .map(|intrinsics| CameraIntrinsics {
            alpha: intrinsics[0],
            beta: intrinsics[1],
            gamma: intrinsics[2],
            u0: intrinsics[3],
            v0: intrinsics[4],
        })
        .ok_or_else(|| anyhow!("optimized intrinsics were not found in the result set"))?;
    let distortion = result
        .get(D(0))
        .map(|distortion| BrownConradyDistortion {
            k1: distortion[0],
            k2: distortion[1],
            k3: distortion[2],
            p1: distortion[3],
            p2: distortion[4],
        })
        .ok_or_else(|| anyhow!("optimized distortion was not found in the result set"))?;
    let extrinsics = extract_extrinsics(&result, observations.len())
        .ok_or_else(|| anyhow!("optimized extrinsics were not found in the result set"))?;
    let reprojection_error =
        compute_reprojection_error(observations, &intrinsics, &distortion, &extrinsics)?;

    Ok(SolverCalibration {
        intrinsics,
        distortion,
        extrinsics,
        reprojection_error,
    })
}

pub fn compute_reprojection_error(
    observations: &[Vec<PointCorrespondence2D3D>],
    intrinsics: &CameraIntrinsics,
    distortion: &BrownConradyDistortion,
    extrinsics: &[CameraExtrinsics],
) -> Result<f64> {
    if observations.len() != extrinsics.len() {
        return Err(anyhow!(
            "the number of observation sets must match the number of extrinsic views"
        ));
    }

    let residuals = observations
        .iter()
        .zip(extrinsics.iter())
        .map(|(view, extrinsics)| {
            view.iter()
                .map(|correspondence| {
                    project_world_to_pixel(intrinsics, distortion, extrinsics, correspondence.world)
                        .map(|predicted| pixel_distance(predicted, correspondence.image))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    if residuals.is_empty() {
        return Err(anyhow!(
            "at least one 2d/3d correspondence is required to compute reprojection error"
        ));
    }

    Ok(residuals.iter().sum::<f64>() / residuals.len() as f64)
}

fn build_pose_values(initial: &SolverCalibration) -> Values {
    let mut values = Values::new();

    initial
        .extrinsics
        .iter()
        .enumerate()
        .for_each(|(index, extrinsics)| {
            values.insert(P(index as u32), extrinsics_to_se3(extrinsics));
        });

    values
}

fn extract_extrinsics(result: &Values, count: usize) -> Option<Vec<CameraExtrinsics>> {
    (0..count)
        .map(|index| result.get(P(index as u32)).map(se3_to_extrinsics))
        .collect()
}

fn extrinsics_to_se3(extrinsics: &CameraExtrinsics) -> SE3 {
    SE3::from_rot_trans(
        SO3::from_matrix(extrinsics.rotation.as_view()),
        extrinsics.translation,
    )
}

fn se3_to_extrinsics(se3: &SE3) -> CameraExtrinsics {
    CameraExtrinsics {
        rotation: se3.rot().to_matrix(),
        translation: Vector3::new(se3.xyz()[0], se3.xyz()[1], se3.xyz()[2]),
    }
}

fn pixel_distance(a: Point2<f64>, b: Point2<f64>) -> f64 {
    ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Rotation3};

    fn make_grid_points(rows: usize, cols: usize, spacing: f64) -> Vec<Point3<f64>> {
        (0..rows)
            .flat_map(|row| {
                (0..cols)
                    .map(move |col| Point3::new(col as f64 * spacing, row as f64 * spacing, 0.0))
            })
            .collect()
    }

    fn extrinsics_from_pose(
        rotation_vec: Vector3<f64>,
        translation: Vector3<f64>,
    ) -> CameraExtrinsics {
        CameraExtrinsics {
            rotation: Rotation3::from_scaled_axis(rotation_vec).into_inner(),
            translation,
        }
    }

    fn synthetic_observations(
        intrinsics: &CameraIntrinsics,
        distortion: &BrownConradyDistortion,
        extrinsics: &CameraExtrinsics,
        world_points: &[Point3<f64>],
    ) -> Vec<PointCorrespondence2D3D> {
        world_points
            .iter()
            .copied()
            .map(|world| {
                PointCorrespondence2D3D::new(
                    project_world_to_pixel(intrinsics, distortion, extrinsics, world).unwrap(),
                    world,
                )
            })
            .collect()
    }

    fn make_problem(
        intrinsics: &CameraIntrinsics,
        distortion: &BrownConradyDistortion,
        image_count: usize,
    ) -> (Vec<Vec<PointCorrespondence2D3D>>, Vec<CameraExtrinsics>) {
        let world_points = make_grid_points(6, 9, 0.025);
        let rotations = [
            Vector3::new(0.10, 0.20, 0.05),
            Vector3::new(-0.15, 0.10, -0.10),
            Vector3::new(0.20, -0.10, 0.15),
            Vector3::new(0.05, 0.30, -0.05),
            Vector3::new(-0.10, -0.20, 0.20),
        ];
        let translations = [
            Vector3::new(0.00, 0.00, 0.50),
            Vector3::new(0.05, -0.03, 0.60),
            Vector3::new(-0.04, 0.02, 0.55),
            Vector3::new(0.03, 0.04, 0.45),
            Vector3::new(-0.02, -0.01, 0.65),
        ];

        (0..image_count)
            .map(|index| {
                let extrinsics = extrinsics_from_pose(
                    rotations[index % rotations.len()],
                    translations[index % translations.len()],
                );
                (
                    synthetic_observations(intrinsics, distortion, &extrinsics, &world_points),
                    extrinsics,
                )
            })
            .unzip()
    }

    #[test]
    fn zhang_initialization_seeds_all_zhang_terms() {
        let zhang = ZhangCalibration {
            intrinsics: CameraIntrinsics {
                alpha: 800.0,
                beta: 805.0,
                gamma: 2.0,
                u0: 320.0,
                v0: 240.0,
            },
            extrinsics: vec![CameraExtrinsics {
                rotation: nalgebra::Matrix3::identity(),
                translation: Vector3::new(0.0, 0.0, 1.0),
            }],
        };
        let distortion = ZhangDistortion {
            k1: -0.05,
            k2: 0.01,
            k3: -0.003,
            p1: 0.001,
            p2: -0.001,
        };

        let initial = from_zhang_initialization(&zhang, Some(distortion));

        assert_eq!(initial.intrinsics, zhang.intrinsics);
        assert_eq!(initial.extrinsics, zhang.extrinsics);
        assert_relative_eq!(initial.distortion.k1, distortion.k1, epsilon = 1e-12);
        assert_relative_eq!(initial.distortion.k2, distortion.k2, epsilon = 1e-12);
        assert_relative_eq!(initial.distortion.k3, distortion.k3, epsilon = 1e-12);
        assert_relative_eq!(initial.distortion.p1, distortion.p1, epsilon = 1e-12);
        assert_relative_eq!(initial.distortion.p2, distortion.p2, epsilon = 1e-12);
    }

    #[test]
    fn optimize_improves_perturbed_intrinsics() {
        let true_intrinsics = CameraIntrinsics {
            alpha: 800.0,
            beta: 805.0,
            gamma: 2.0,
            u0: 320.0,
            v0: 240.0,
        };
        let true_distortion = BrownConradyDistortion::default();
        let (observations, extrinsics) = make_problem(&true_intrinsics, &true_distortion, 5);
        let initial = SolverCalibration {
            intrinsics: CameraIntrinsics {
                alpha: 810.0,
                beta: 795.0,
                gamma: 0.5,
                u0: 325.0,
                v0: 235.0,
            },
            distortion: BrownConradyDistortion::default(),
            extrinsics,
            reprojection_error: 0.0,
        };

        let optimized = optimize(&observations, &initial).unwrap();

        assert!((optimized.intrinsics.alpha - true_intrinsics.alpha).abs() < 1.0);
        assert!((optimized.intrinsics.beta - true_intrinsics.beta).abs() < 1.0);
        assert!((optimized.intrinsics.u0 - true_intrinsics.u0).abs() < 1.0);
        assert!((optimized.intrinsics.v0 - true_intrinsics.v0).abs() < 1.0);
    }

    #[test]
    fn optimize_recovers_full_brown_conrady_distortion() {
        let true_intrinsics = CameraIntrinsics {
            alpha: 800.0,
            beta: 800.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        };
        let true_distortion = BrownConradyDistortion {
            k1: -0.05,
            k2: 0.01,
            k3: -0.003,
            p1: 0.001,
            p2: -0.001,
        };
        let (observations, extrinsics) = make_problem(&true_intrinsics, &true_distortion, 5);
        let initial = SolverCalibration {
            intrinsics: true_intrinsics.clone(),
            distortion: BrownConradyDistortion::default(),
            extrinsics,
            reprojection_error: 0.0,
        };

        let optimized = optimize(&observations, &initial).unwrap();

        assert!((optimized.distortion.k1 - true_distortion.k1).abs() < 0.01);
        assert!((optimized.distortion.k2 - true_distortion.k2).abs() < 0.01);
        assert!((optimized.distortion.k3 - true_distortion.k3).abs() < 0.01);
        assert!((optimized.distortion.p1 - true_distortion.p1).abs() < 0.005);
        assert!((optimized.distortion.p2 - true_distortion.p2).abs() < 0.005);
    }

    #[test]
    fn optimize_reduces_reprojection_error() {
        let true_intrinsics = CameraIntrinsics {
            alpha: 800.0,
            beta: 800.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        };
        let true_distortion = BrownConradyDistortion {
            k1: -0.03,
            k2: 0.005,
            k3: 0.001,
            p1: 0.0008,
            p2: -0.0006,
        };
        let (observations, extrinsics) = make_problem(&true_intrinsics, &true_distortion, 5);
        let initial = SolverCalibration {
            intrinsics: CameraIntrinsics {
                alpha: 810.0,
                beta: 790.0,
                gamma: 0.0,
                u0: 325.0,
                v0: 235.0,
            },
            distortion: BrownConradyDistortion::default(),
            extrinsics,
            reprojection_error: 100.0,
        };

        let optimized = optimize(&observations, &initial).unwrap();

        assert!(optimized.reprojection_error < 1.0);
    }

    #[test]
    fn optimize_preserves_good_estimate() {
        let true_intrinsics = CameraIntrinsics {
            alpha: 800.0,
            beta: 800.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        };
        let true_distortion = BrownConradyDistortion::default();
        let (observations, extrinsics) = make_problem(&true_intrinsics, &true_distortion, 5);
        let initial = SolverCalibration {
            intrinsics: true_intrinsics.clone(),
            distortion: true_distortion,
            extrinsics,
            reprojection_error: 0.0,
        };

        let optimized = optimize(&observations, &initial).unwrap();

        assert!((optimized.intrinsics.alpha - true_intrinsics.alpha).abs() < 0.1);
        assert!((optimized.intrinsics.beta - true_intrinsics.beta).abs() < 0.1);
        assert!(optimized.reprojection_error < 0.1);
    }
}
