//! Adapts the generic solver bundle adjustment to SfM reconstruction types.

use std::collections::HashSet;

use anyhow::{Result, anyhow, ensure};
use structura_geometry::camera::CameraIntrinsics;
use structura_solver::{
    BrownConradyDistortion, BundleAdjustmentCamera, BundleAdjustmentConfig,
    BundleAdjustmentLandmark, BundleAdjustmentObservation, BundleAdjustmentProblem,
    optimize_bundle_adjustment,
};

use crate::types::{LandmarkTrack, RegisteredView};

#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustedReconstruction {
    pub views: Vec<RegisteredView>,
    pub landmarks: Vec<LandmarkTrack>,
    pub mean_reprojection_error: f64,
}

pub fn bundle_adjust_registered_reconstruction(
    intrinsics: &CameraIntrinsics,
    views: &[RegisteredView],
    landmarks: &[LandmarkTrack],
) -> Result<BundleAdjustedReconstruction> {
    bundle_adjust_registered_reconstruction_with_config(
        intrinsics,
        &BrownConradyDistortion::default(),
        views,
        landmarks,
        &BundleAdjustmentConfig::default(),
    )
}

pub fn bundle_adjust_registered_reconstruction_with_config(
    intrinsics: &CameraIntrinsics,
    distortion: &BrownConradyDistortion,
    views: &[RegisteredView],
    landmarks: &[LandmarkTrack],
    config: &BundleAdjustmentConfig,
) -> Result<BundleAdjustedReconstruction> {
    ensure!(
        views.len() >= 2,
        "bundle adjustment requires at least two registered views"
    );

    let problem = build_problem(intrinsics, distortion, views, landmarks)?;
    let optimized = optimize_bundle_adjustment(&problem, config)?;

    let optimized_views = views
        .iter()
        .map(|view| {
            let optimized = optimized
                .cameras
                .iter()
                .find(|camera| camera.view_index == view.view_index)
                .ok_or_else(|| anyhow!("missing optimized view {}", view.view_index))?;
            Ok(RegisteredView {
                view_index: view.view_index,
                extrinsics: optimized.extrinsics.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let optimized_landmarks = landmarks
        .iter()
        .enumerate()
        .map(|(landmark_id, landmark)| {
            let optimized = optimized
                .landmarks
                .iter()
                .find(|candidate| candidate.id == landmark_id)
                .ok_or_else(|| anyhow!("missing optimized landmark {}", landmark_id))?;
            Ok(LandmarkTrack {
                position: optimized.position,
                observations: landmark.observations.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(BundleAdjustedReconstruction {
        views: optimized_views,
        landmarks: optimized_landmarks,
        mean_reprojection_error: optimized.mean_reprojection_error,
    })
}

fn build_problem(
    intrinsics: &CameraIntrinsics,
    distortion: &BrownConradyDistortion,
    views: &[RegisteredView],
    landmarks: &[LandmarkTrack],
) -> Result<BundleAdjustmentProblem> {
    let cameras = views
        .iter()
        .enumerate()
        .map(|(index, view)| BundleAdjustmentCamera {
            view_index: view.view_index,
            extrinsics: view.extrinsics.clone(),
            fixed: index < 2,
        })
        .collect::<Vec<_>>();
    let registered_views = views
        .iter()
        .map(|view| view.view_index)
        .collect::<HashSet<_>>();

    let bundle_landmarks = landmarks
        .iter()
        .enumerate()
        .map(|(id, landmark)| BundleAdjustmentLandmark {
            id,
            position: landmark.position,
        })
        .collect::<Vec<_>>();
    let registered_views_ref = &registered_views;
    let observations = landmarks
        .iter()
        .enumerate()
        .flat_map(|(landmark_id, landmark)| {
            landmark.observations.iter().map(move |observation| {
                registered_views_ref
                    .contains(&observation.view_index)
                    .then(|| BundleAdjustmentObservation {
                        camera_id: observation.view_index,
                        landmark_id,
                        image: observation.image_point,
                    })
                    .ok_or_else(|| {
                        anyhow!(
                            "landmark {} references unregistered view {}",
                            landmark_id,
                            observation.view_index
                        )
                    })
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(BundleAdjustmentProblem {
        intrinsics: intrinsics.clone(),
        distortion: *distortion,
        cameras,
        landmarks: bundle_landmarks,
        observations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Rotation3, Vector3};
    use structura_geometry::{
        camera::CameraExtrinsics,
        point::{ImagePoint64, WorldPoint},
    };

    use crate::types::LandmarkObservation;

    #[test]
    fn adapter_refines_registered_reconstruction_and_preserves_tracks() {
        let intrinsics = CameraIntrinsics {
            alpha: 900.0,
            beta: 880.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        };
        let gt_views = vec![
            RegisteredView {
                view_index: 0,
                extrinsics: CameraExtrinsics {
                    rotation: Matrix3::identity(),
                    translation: Vector3::zeros(),
                },
            },
            RegisteredView {
                view_index: 1,
                extrinsics: CameraExtrinsics {
                    rotation: Rotation3::from_euler_angles(0.01, -0.02, 0.015).into_inner(),
                    translation: Vector3::new(0.35, 0.02, 0.01),
                },
            },
            RegisteredView {
                view_index: 2,
                extrinsics: CameraExtrinsics {
                    rotation: Rotation3::from_euler_angles(-0.02, 0.03, -0.01).into_inner(),
                    translation: Vector3::new(0.18, -0.03, 0.08),
                },
            },
        ];
        let world_points = [
            WorldPoint::new(-0.2, -0.15, 3.4),
            WorldPoint::new(0.1, -0.05, 3.8),
            WorldPoint::new(0.25, 0.2, 4.1),
        ];
        let landmarks = world_points
            .iter()
            .enumerate()
            .map(|(feature_index, world)| LandmarkTrack {
                position: WorldPoint::new(world.x + 0.1, world.y - 0.08, world.z + 0.15),
                observations: gt_views
                    .iter()
                    .map(|view| LandmarkObservation {
                        view_index: view.view_index,
                        feature_index,
                        image_point: project(&intrinsics, &view.extrinsics, *world),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let perturbed_views = vec![
            gt_views[0].clone(),
            gt_views[1].clone(),
            RegisteredView {
                view_index: gt_views[2].view_index,
                extrinsics: CameraExtrinsics {
                    rotation: (Rotation3::from_matrix(&gt_views[2].extrinsics.rotation)
                        * Rotation3::from_scaled_axis(Vector3::new(0.01, -0.015, 0.008)))
                    .into_inner(),
                    translation: gt_views[2].extrinsics.translation
                        + Vector3::new(0.08, -0.05, 0.03),
                },
            },
        ];

        let optimized =
            bundle_adjust_registered_reconstruction(&intrinsics, &perturbed_views, &landmarks)
                .unwrap();

        assert_eq!(optimized.views.len(), perturbed_views.len());
        assert_eq!(optimized.landmarks.len(), landmarks.len());
        assert_eq!(
            optimized.landmarks[0].observations,
            landmarks[0].observations
        );
        assert!(
            (optimized.views[2].extrinsics.translation - gt_views[2].extrinsics.translation).norm()
                < (perturbed_views[2].extrinsics.translation - gt_views[2].extrinsics.translation)
                    .norm()
        );
    }

    #[test]
    fn adapter_rejects_landmarks_with_unregistered_observations() {
        let intrinsics = CameraIntrinsics {
            alpha: 900.0,
            beta: 880.0,
            gamma: 0.0,
            u0: 320.0,
            v0: 240.0,
        };
        let views = vec![
            RegisteredView {
                view_index: 0,
                extrinsics: CameraExtrinsics {
                    rotation: Matrix3::identity(),
                    translation: Vector3::zeros(),
                },
            },
            RegisteredView {
                view_index: 1,
                extrinsics: CameraExtrinsics {
                    rotation: Matrix3::identity(),
                    translation: Vector3::new(0.3, 0.0, 0.0),
                },
            },
        ];
        let landmarks = vec![LandmarkTrack {
            position: WorldPoint::new(0.0, 0.0, 3.0),
            observations: vec![
                LandmarkObservation {
                    view_index: 0,
                    feature_index: 0,
                    image_point: ImagePoint64::new(320.0, 240.0),
                },
                LandmarkObservation {
                    view_index: 99,
                    feature_index: 0,
                    image_point: ImagePoint64::new(330.0, 240.0),
                },
            ],
        }];

        let error =
            bundle_adjust_registered_reconstruction(&intrinsics, &views, &landmarks).unwrap_err();

        assert!(error.to_string().contains("unregistered view 99"));
    }

    fn project(
        intrinsics: &CameraIntrinsics,
        extrinsics: &CameraExtrinsics,
        world: WorldPoint,
    ) -> ImagePoint64 {
        let camera = extrinsics.rotation * world.coords + extrinsics.translation;
        ImagePoint64::new(
            intrinsics.alpha * (camera.x / camera.z)
                + intrinsics.gamma * (camera.y / camera.z)
                + intrinsics.u0,
            intrinsics.beta * (camera.y / camera.z) + intrinsics.v0,
        )
    }
}
