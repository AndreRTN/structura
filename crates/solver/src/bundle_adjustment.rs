//! Generic bundle-adjustment problem definition and optimizer.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow, ensure};
use factrs::{
    assign_symbols,
    core::{Graph, Values},
    fac,
    optimizers::{LevenMarquardt, LevenParams, OptError},
    robust::Huber,
    traits::Optimizer,
    variables::{MatrixLieGroup, SE3, SO3, VectorVar3},
};
use structura_geometry::{
    camera::{CameraExtrinsics, CameraIntrinsics},
    point::{ImagePoint64, WorldPoint},
};

use crate::{
    calibration::BrownConradyDistortion,
    residuals::reprojection::{
        BundleAdjustmentReprojectionResidual, FixedCameraReprojectionResidual,
        project_world_to_pixel,
    },
};

assign_symbols!(P: SE3; L: VectorVar3);

#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustmentProblem {
    pub intrinsics: CameraIntrinsics,
    pub distortion: BrownConradyDistortion,
    pub cameras: Vec<BundleAdjustmentCamera>,
    pub landmarks: Vec<BundleAdjustmentLandmark>,
    pub observations: Vec<BundleAdjustmentObservation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustmentCamera {
    pub view_index: usize,
    pub extrinsics: CameraExtrinsics,
    pub fixed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustmentLandmark {
    pub id: usize,
    pub position: WorldPoint,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustmentObservation {
    pub camera_id: usize,
    pub landmark_id: usize,
    pub image: ImagePoint64,
}

#[derive(Debug, Clone)]
pub struct BundleAdjustmentConfig {
    pub huber_delta: f64,
    pub levenberg_params: LevenParams,
}

impl Default for BundleAdjustmentConfig {
    fn default() -> Self {
        Self {
            huber_delta: 1.5,
            levenberg_params: LevenParams {
                lambda_min: 1e-20,
                lambda_max: 1e10,
                lambda_factor: 4.0,
                min_model_fidelity: 1e-6,
                diagonal_damping: true,
                base: factrs::optimizers::BaseOptParams {
                    max_iterations: 400,
                    error_tol_relative: 1e-6,
                    error_tol_absolute: 1e-6,
                    error_tol: 0.0,
                },
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustmentResult {
    pub cameras: Vec<BundleAdjustmentCamera>,
    pub landmarks: Vec<BundleAdjustmentLandmark>,
    pub mean_reprojection_error: f64,
}

pub fn optimize_bundle_adjustment(
    problem: &BundleAdjustmentProblem,
    config: &BundleAdjustmentConfig,
) -> Result<BundleAdjustmentResult> {
    let validated = validate_problem(problem)?;
    let mut graph = Graph::new();
    let mut values = Values::new();

    validated
        .variable_cameras
        .iter()
        .for_each(|(&camera_id, &symbol_index)| {
            let camera = validated.cameras_by_id[&camera_id];
            values.insert(P(symbol_index), extrinsics_to_se3(&camera.extrinsics));
        });

    validated
        .landmarks_by_id
        .iter()
        .for_each(|(&landmark_id, landmark)| {
            let symbol_index = validated.landmark_symbols[&landmark_id];
            values.insert(
                L(symbol_index),
                VectorVar3::new(
                    landmark.position.x,
                    landmark.position.y,
                    landmark.position.z,
                ),
            );
        });

    for observation in &problem.observations {
        let camera = validated.cameras_by_id[&observation.camera_id];
        let landmark_symbol = L(validated.landmark_symbols[&observation.landmark_id]);
        let robust = Huber::new(config.huber_delta);

        if camera.fixed {
            graph.add_factor(fac![
                FixedCameraReprojectionResidual::new(
                    observation.image,
                    &camera.extrinsics,
                    &problem.intrinsics,
                    &problem.distortion
                ),
                landmark_symbol,
                _,
                robust
            ]);
        } else {
            graph.add_factor(fac![
                BundleAdjustmentReprojectionResidual::new(
                    observation.image,
                    &problem.intrinsics,
                    &problem.distortion
                ),
                (
                    P(validated.variable_cameras[&observation.camera_id]),
                    landmark_symbol
                ),
                _,
                robust
            ]);
        }
    }

    let mut optimizer = LevenMarquardt::new(config.levenberg_params.clone(), graph);
    let result = optimizer
        .optimize(values)
        .map_err(|error| summarize_optimizer_error(error, problem, config))?;

    let cameras = problem
        .cameras
        .iter()
        .map(|camera| {
            if camera.fixed {
                Ok(camera.clone())
            } else {
                let symbol_index = validated.variable_cameras[&camera.view_index];
                let pose = result.get(P(symbol_index)).ok_or_else(|| {
                    anyhow!("optimized camera pose missing for {}", camera.view_index)
                })?;
                Ok(BundleAdjustmentCamera {
                    view_index: camera.view_index,
                    extrinsics: se3_to_extrinsics(pose),
                    fixed: false,
                })
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let landmarks = problem
        .landmarks
        .iter()
        .map(|landmark| {
            let point = result
                .get(L(validated.landmark_symbols[&landmark.id]))
                .ok_or_else(|| anyhow!("optimized landmark missing for {}", landmark.id))?;
            Ok(BundleAdjustmentLandmark {
                id: landmark.id,
                position: WorldPoint::new(point[0], point[1], point[2]),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mean_reprojection_error = compute_mean_reprojection_error(
        &problem.intrinsics,
        &problem.distortion,
        &cameras,
        &landmarks,
        &problem.observations,
    )?;

    Ok(BundleAdjustmentResult {
        cameras,
        landmarks,
        mean_reprojection_error,
    })
}

fn summarize_optimizer_error(
    error: OptError,
    problem: &BundleAdjustmentProblem,
    config: &BundleAdjustmentConfig,
) -> anyhow::Error {
    match error {
        OptError::MaxIterations(values) => anyhow!(
            "bundle adjustment optimization failed: reached max iterations ({}) with {} state variables for {} cameras, {} landmarks, and {} observations",
            config.levenberg_params.base.max_iterations,
            values.len(),
            problem.cameras.len(),
            problem.landmarks.len(),
            problem.observations.len(),
        ),
        OptError::InvalidSystem => anyhow!(
            "bundle adjustment optimization failed: optimizer produced an invalid linear system"
        ),
        OptError::FailedToStep => anyhow!(
            "bundle adjustment optimization failed: optimizer could not find a valid LM step within lambda range [{:.1e}, {:.1e}]",
            config.levenberg_params.lambda_min,
            config.levenberg_params.lambda_max,
        ),
    }
}

fn compute_mean_reprojection_error(
    intrinsics: &CameraIntrinsics,
    distortion: &BrownConradyDistortion,
    cameras: &[BundleAdjustmentCamera],
    landmarks: &[BundleAdjustmentLandmark],
    observations: &[BundleAdjustmentObservation],
) -> Result<f64> {
    ensure!(
        !observations.is_empty(),
        "at least one observation is required to compute reprojection error"
    );

    let cameras_by_id = cameras
        .iter()
        .map(|camera| (camera.view_index, camera))
        .collect::<HashMap<_, _>>();
    let landmarks_by_id = landmarks
        .iter()
        .map(|landmark| (landmark.id, landmark))
        .collect::<HashMap<_, _>>();

    let total = observations
        .iter()
        .map(|observation| {
            let camera = cameras_by_id.get(&observation.camera_id).ok_or_else(|| {
                anyhow!(
                    "missing camera {} while computing reprojection error",
                    observation.camera_id
                )
            })?;
            let landmark = landmarks_by_id
                .get(&observation.landmark_id)
                .ok_or_else(|| {
                    anyhow!(
                        "missing landmark {} while computing reprojection error",
                        observation.landmark_id
                    )
                })?;
            let projected = project_world_to_pixel(
                intrinsics,
                distortion,
                &camera.extrinsics,
                landmark.position,
            )?;
            let dx = projected.x - observation.image.x;
            let dy = projected.y - observation.image.y;
            Ok((dx * dx + dy * dy).sqrt())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<f64>();

    Ok(total / observations.len() as f64)
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
        translation: nalgebra::Vector3::new(se3.xyz()[0], se3.xyz()[1], se3.xyz()[2]),
    }
}

struct ValidatedProblem<'a> {
    cameras_by_id: HashMap<usize, &'a BundleAdjustmentCamera>,
    landmarks_by_id: HashMap<usize, &'a BundleAdjustmentLandmark>,
    variable_cameras: HashMap<usize, u32>,
    landmark_symbols: HashMap<usize, u32>,
}

fn validate_problem(problem: &BundleAdjustmentProblem) -> Result<ValidatedProblem<'_>> {
    ensure!(
        problem.cameras.len() >= 2,
        "bundle adjustment requires at least two cameras"
    );
    ensure!(
        !problem.landmarks.is_empty(),
        "bundle adjustment requires at least one landmark"
    );
    ensure!(
        !problem.observations.is_empty(),
        "bundle adjustment requires at least one observation"
    );

    let mut cameras_by_id = HashMap::new();
    let mut variable_cameras = HashMap::new();
    for camera in &problem.cameras {
        ensure!(
            cameras_by_id.insert(camera.view_index, camera).is_none(),
            "duplicate camera id {} in bundle adjustment problem",
            camera.view_index
        );
        if !camera.fixed {
            let symbol_index = variable_cameras.len() as u32;
            variable_cameras.insert(camera.view_index, symbol_index);
        }
    }

    let mut landmarks_by_id = HashMap::new();
    let mut landmark_symbols = HashMap::new();
    for landmark in &problem.landmarks {
        ensure!(
            landmarks_by_id.insert(landmark.id, landmark).is_none(),
            "duplicate landmark id {} in bundle adjustment problem",
            landmark.id
        );
        landmark_symbols.insert(landmark.id, landmark_symbols.len() as u32);
    }

    let mut observation_views = HashMap::<usize, HashSet<usize>>::new();
    for observation in &problem.observations {
        ensure!(
            cameras_by_id.contains_key(&observation.camera_id),
            "observation references missing camera {}",
            observation.camera_id
        );
        ensure!(
            landmarks_by_id.contains_key(&observation.landmark_id),
            "observation references missing landmark {}",
            observation.landmark_id
        );
        observation_views
            .entry(observation.landmark_id)
            .or_default()
            .insert(observation.camera_id);
    }

    for landmark in &problem.landmarks {
        let count = observation_views
            .get(&landmark.id)
            .map(HashSet::len)
            .unwrap_or_default();
        ensure!(
            count >= 2,
            "landmark {} must have observations in at least two cameras",
            landmark.id
        );
    }

    Ok(ValidatedProblem {
        cameras_by_id,
        landmarks_by_id,
        variable_cameras,
        landmark_symbols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix3, Rotation3, Vector3};

    fn sample_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics {
            alpha: 900.0,
            beta: 880.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        }
    }

    fn project(
        intrinsics: &CameraIntrinsics,
        extrinsics: &CameraExtrinsics,
        world: WorldPoint,
    ) -> ImagePoint64 {
        project_world_to_pixel(
            intrinsics,
            &BrownConradyDistortion::default(),
            extrinsics,
            world,
        )
        .unwrap()
    }

    fn perturb_pose(extrinsics: &CameraExtrinsics, delta: Vector3<f64>) -> CameraExtrinsics {
        CameraExtrinsics {
            rotation: (Rotation3::from_matrix(&extrinsics.rotation)
                * Rotation3::from_scaled_axis(Vector3::new(0.01, -0.015, 0.008)))
            .into_inner(),
            translation: extrinsics.translation + delta,
        }
    }

    #[test]
    fn rejects_invalid_problem_references() {
        let problem = BundleAdjustmentProblem {
            intrinsics: sample_intrinsics(),
            distortion: BrownConradyDistortion::default(),
            cameras: vec![
                BundleAdjustmentCamera {
                    view_index: 0,
                    extrinsics: CameraExtrinsics {
                        rotation: Matrix3::identity(),
                        translation: Vector3::zeros(),
                    },
                    fixed: true,
                },
                BundleAdjustmentCamera {
                    view_index: 1,
                    extrinsics: CameraExtrinsics {
                        rotation: Matrix3::identity(),
                        translation: Vector3::new(0.2, 0.0, 0.0),
                    },
                    fixed: true,
                },
            ],
            landmarks: vec![BundleAdjustmentLandmark {
                id: 0,
                position: WorldPoint::new(0.0, 0.0, 3.0),
            }],
            observations: vec![BundleAdjustmentObservation {
                camera_id: 42,
                landmark_id: 0,
                image: ImagePoint64::new(320.0, 240.0),
            }],
        };

        let error =
            optimize_bundle_adjustment(&problem, &BundleAdjustmentConfig::default()).unwrap_err();

        assert!(error.to_string().contains("missing camera 42"));
    }

    #[test]
    fn rejects_landmarks_with_fewer_than_two_views() {
        let problem = BundleAdjustmentProblem {
            intrinsics: sample_intrinsics(),
            distortion: BrownConradyDistortion::default(),
            cameras: vec![
                BundleAdjustmentCamera {
                    view_index: 0,
                    extrinsics: CameraExtrinsics {
                        rotation: Matrix3::identity(),
                        translation: Vector3::zeros(),
                    },
                    fixed: true,
                },
                BundleAdjustmentCamera {
                    view_index: 1,
                    extrinsics: CameraExtrinsics {
                        rotation: Matrix3::identity(),
                        translation: Vector3::new(0.2, 0.0, 0.0),
                    },
                    fixed: true,
                },
            ],
            landmarks: vec![BundleAdjustmentLandmark {
                id: 0,
                position: WorldPoint::new(0.0, 0.0, 3.0),
            }],
            observations: vec![BundleAdjustmentObservation {
                camera_id: 0,
                landmark_id: 0,
                image: ImagePoint64::new(320.0, 240.0),
            }],
        };

        let error =
            optimize_bundle_adjustment(&problem, &BundleAdjustmentConfig::default()).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("observations in at least two cameras")
        );
    }

    #[test]
    fn reduces_reprojection_error_on_synthetic_problem() {
        let intrinsics = sample_intrinsics();
        let gt_cameras = vec![
            BundleAdjustmentCamera {
                view_index: 0,
                extrinsics: CameraExtrinsics {
                    rotation: Matrix3::identity(),
                    translation: Vector3::zeros(),
                },
                fixed: true,
            },
            BundleAdjustmentCamera {
                view_index: 1,
                extrinsics: CameraExtrinsics {
                    rotation: Rotation3::from_euler_angles(0.01, -0.02, 0.015).into_inner(),
                    translation: Vector3::new(0.35, 0.02, 0.01),
                },
                fixed: true,
            },
            BundleAdjustmentCamera {
                view_index: 2,
                extrinsics: CameraExtrinsics {
                    rotation: Rotation3::from_euler_angles(-0.02, 0.03, -0.01).into_inner(),
                    translation: Vector3::new(0.18, -0.03, 0.08),
                },
                fixed: false,
            },
        ];
        let gt_points = [
            WorldPoint::new(-0.2, -0.15, 3.4),
            WorldPoint::new(0.1, -0.05, 3.8),
            WorldPoint::new(0.25, 0.2, 4.1),
            WorldPoint::new(-0.1, 0.18, 3.6),
            WorldPoint::new(0.05, 0.06, 4.4),
        ];

        let intrinsics_ref = &intrinsics;
        let observations = gt_points
            .iter()
            .enumerate()
            .flat_map(|(landmark_id, point)| {
                gt_cameras
                    .iter()
                    .map(move |camera| BundleAdjustmentObservation {
                        camera_id: camera.view_index,
                        landmark_id,
                        image: project(intrinsics_ref, &camera.extrinsics, *point),
                    })
            })
            .collect::<Vec<_>>();

        let perturbed_cameras = gt_cameras
            .iter()
            .map(|camera| BundleAdjustmentCamera {
                view_index: camera.view_index,
                extrinsics: if camera.fixed {
                    camera.extrinsics.clone()
                } else {
                    perturb_pose(&camera.extrinsics, Vector3::new(0.08, -0.05, 0.03))
                },
                fixed: camera.fixed,
            })
            .collect::<Vec<_>>();
        let perturbed_landmarks = gt_points
            .iter()
            .enumerate()
            .map(|(id, point)| BundleAdjustmentLandmark {
                id,
                position: WorldPoint::new(point.x + 0.12, point.y - 0.08, point.z + 0.18),
            })
            .collect::<Vec<_>>();
        let problem = BundleAdjustmentProblem {
            intrinsics: intrinsics.clone(),
            distortion: BrownConradyDistortion::default(),
            cameras: perturbed_cameras.clone(),
            landmarks: perturbed_landmarks.clone(),
            observations,
        };

        let initial_error = compute_mean_reprojection_error(
            &intrinsics,
            &BrownConradyDistortion::default(),
            &perturbed_cameras,
            &perturbed_landmarks,
            &problem.observations,
        )
        .unwrap();
        let optimized =
            optimize_bundle_adjustment(&problem, &BundleAdjustmentConfig::default()).unwrap();

        assert!(optimized.mean_reprojection_error < initial_error);
        let optimized_camera = optimized
            .cameras
            .iter()
            .find(|camera| camera.view_index == 2)
            .unwrap();
        assert!(
            (optimized_camera.extrinsics.translation - gt_cameras[2].extrinsics.translation).norm()
                < (perturbed_cameras[2].extrinsics.translation
                    - gt_cameras[2].extrinsics.translation)
                    .norm()
        );
        let landmark_error = optimized
            .landmarks
            .iter()
            .zip(gt_points.iter())
            .map(|(optimized, gt)| (optimized.position - *gt).norm())
            .sum::<f64>()
            / gt_points.len() as f64;
        assert!(landmark_error < 1e-2);
    }

    #[test]
    fn reprojection_error_matches_ground_truth_solution() {
        let intrinsics = sample_intrinsics();
        let cameras = vec![
            BundleAdjustmentCamera {
                view_index: 0,
                extrinsics: CameraExtrinsics {
                    rotation: Matrix3::identity(),
                    translation: Vector3::zeros(),
                },
                fixed: true,
            },
            BundleAdjustmentCamera {
                view_index: 1,
                extrinsics: CameraExtrinsics {
                    rotation: Rotation3::from_euler_angles(0.0, 0.02, 0.0).into_inner(),
                    translation: Vector3::new(0.3, 0.0, 0.0),
                },
                fixed: true,
            },
        ];
        let landmarks = vec![BundleAdjustmentLandmark {
            id: 0,
            position: WorldPoint::new(0.1, -0.05, 3.2),
        }];
        let observations = cameras
            .iter()
            .map(|camera| BundleAdjustmentObservation {
                camera_id: camera.view_index,
                landmark_id: 0,
                image: project(&intrinsics, &camera.extrinsics, landmarks[0].position),
            })
            .collect::<Vec<_>>();

        let error = compute_mean_reprojection_error(
            &intrinsics,
            &BrownConradyDistortion::default(),
            &cameras,
            &landmarks,
            &observations,
        )
        .unwrap();

        assert_relative_eq!(error, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn optimizer_error_summary_omits_factrs_values_dump() {
        let problem = BundleAdjustmentProblem {
            intrinsics: sample_intrinsics(),
            distortion: BrownConradyDistortion::default(),
            cameras: vec![
                BundleAdjustmentCamera {
                    view_index: 0,
                    extrinsics: CameraExtrinsics {
                        rotation: Matrix3::identity(),
                        translation: Vector3::zeros(),
                    },
                    fixed: true,
                },
                BundleAdjustmentCamera {
                    view_index: 1,
                    extrinsics: CameraExtrinsics {
                        rotation: Matrix3::identity(),
                        translation: Vector3::new(0.2, 0.0, 0.0),
                    },
                    fixed: false,
                },
            ],
            landmarks: vec![BundleAdjustmentLandmark {
                id: 0,
                position: WorldPoint::new(0.0, 0.0, 3.0),
            }],
            observations: vec![
                BundleAdjustmentObservation {
                    camera_id: 0,
                    landmark_id: 0,
                    image: ImagePoint64::new(320.0, 240.0),
                },
                BundleAdjustmentObservation {
                    camera_id: 1,
                    landmark_id: 0,
                    image: ImagePoint64::new(380.0, 240.0),
                },
            ],
        };

        let message = summarize_optimizer_error(
            OptError::MaxIterations(Values::new()),
            &problem,
            &BundleAdjustmentConfig::default(),
        )
        .to_string();

        assert!(message.contains("reached max iterations"));
        assert!(!message.contains("Values"));
    }
}
