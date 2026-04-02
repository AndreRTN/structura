use anyhow::{Result, anyhow, ensure};
use nalgebra::{Matrix3, Vector3};
use std::collections::HashMap;
use structura_feature::{
    descriptor::{DescriptorFeatureSet, lowe_ratio_match_descriptor_features},
    matching::PointMatch,
};
use structura_geometry::{
    camera::{CameraExtrinsics, CameraIntrinsics},
    epnp::EpnpSolver,
    point::{ImagePoint64, PointCorrespondence2D3D, WorldPoint},
    triangulation::{
        TriangulationObservation, has_positive_depth, triangulate_observations,
        triangulate_observations_with_stats,
    },
};

use crate::types::{
    FeatureTrack, InitialPairCandidate, InitialPairScore, InitialPairSelectionConfig,
    InitialReconstruction, LandmarkObservation, LandmarkTrack, NextViewRegistration,
    RegisteredView, TrackTriangulationConfig,
};

pub trait PnpSolver {
    fn solve_pose(
        &self,
        intrinsics: &CameraIntrinsics,
        correspondences: &[PointCorrespondence2D3D],
    ) -> Result<CameraExtrinsics>;
}

pub fn select_initial_pair(
    feature_sets: &[DescriptorFeatureSet],
    config: InitialPairSelectionConfig,
) -> Result<InitialPairCandidate> {
    ensure!(
        feature_sets.len() >= 2,
        "incremental sfm requires at least two feature sets"
    );

    let mut best_candidate: Option<InitialPairCandidate> = None;

    for source_view_index in 0..feature_sets.len() - 1 {
        for target_view_index in source_view_index + 1..feature_sets.len() {
            let matches = lowe_ratio_match_descriptor_features(
                &feature_sets[source_view_index],
                &feature_sets[target_view_index],
                config.lowe_ratio,
            )?;

            if matches.len() < config.min_matches {
                continue;
            }

            let score = score_initial_pair(
                &feature_sets[source_view_index],
                &feature_sets[target_view_index],
                &matches,
            );

            let candidate = InitialPairCandidate {
                source_view_index,
                target_view_index,
                matches,
                score,
            };

            match &best_candidate {
                Some(best) if best.score.total_score >= candidate.score.total_score => {}
                _ => best_candidate = Some(candidate),
            }
        }
    }

    best_candidate.ok_or_else(|| {
        anyhow!(
            "failed to find an initial pair with at least {} matches",
            config.min_matches
        )
    })
}

pub fn reconstruct_initial_pair(
    intrinsics: &CameraIntrinsics,
    initial_pair: &InitialPairCandidate,
    essential_matrix: &Matrix3<f64>,
) -> Result<InitialReconstruction> {
    ensure!(
        initial_pair.matches.len() >= 5,
        "at least five matches are required to decompose the essential matrix"
    );

    let first_pose = CameraExtrinsics {
        rotation: Matrix3::identity(),
        translation: Vector3::zeros(),
    };
    let second_pose = recover_relative_pose(intrinsics, essential_matrix, &initial_pair.matches)?;

    let views = [
        RegisteredView {
            view_index: initial_pair.source_view_index,
            extrinsics: first_pose,
        },
        RegisteredView {
            view_index: initial_pair.target_view_index,
            extrinsics: second_pose,
        },
    ];
    let tracks = initial_pair
        .matches
        .iter()
        .map(|matched| FeatureTrack {
            observations: vec![
                LandmarkObservation {
                    view_index: initial_pair.source_view_index,
                    feature_index: matched.source_index,
                    image_point: ImagePoint64::new(
                        matched.source_point.x as f64,
                        matched.source_point.y as f64,
                    ),
                },
                LandmarkObservation {
                    view_index: initial_pair.target_view_index,
                    feature_index: matched.target_index,
                    image_point: ImagePoint64::new(
                        matched.target_point.x as f64,
                        matched.target_point.y as f64,
                    ),
                },
            ],
        })
        .collect::<Vec<_>>();

    let landmarks = triangulate_feature_tracks(
        intrinsics,
        &views,
        &tracks,
        TrackTriangulationConfig::default(),
    )?;

    ensure!(
        !landmarks.is_empty(),
        "essential matrix decomposition did not yield any front-facing triangulated landmarks"
    );

    Ok(InitialReconstruction { views, landmarks })
}

pub fn register_next_view(
    intrinsics: &CameraIntrinsics,
    view_index: usize,
    correspondences: &[PointCorrespondence2D3D],
) -> Result<NextViewRegistration> {
    register_next_view_with_solver(
        intrinsics,
        view_index,
        correspondences,
        &EpnpSolver::default(),
    )
}

pub fn register_next_view_with_solver(
    intrinsics: &CameraIntrinsics,
    view_index: usize,
    correspondences: &[PointCorrespondence2D3D],
    pnp_solver: &dyn PnpSolver,
) -> Result<NextViewRegistration> {
    ensure!(
        correspondences.len() >= 4,
        "at least four 3d-2d correspondences are required for pnp registration"
    );

    let extrinsics = pnp_solver.solve_pose(intrinsics, correspondences)?;

    Ok(NextViewRegistration {
        view: RegisteredView {
            view_index,
            extrinsics,
        },
        used_correspondence_count: correspondences.len(),
    })
}

pub fn triangulate_feature_tracks(
    intrinsics: &CameraIntrinsics,
    views: &[RegisteredView],
    tracks: &[FeatureTrack],
    config: TrackTriangulationConfig,
) -> Result<Vec<LandmarkTrack>> {
    let views_by_index = views
        .iter()
        .map(|view| (view.view_index, &view.extrinsics))
        .collect::<HashMap<_, _>>();

    tracks
        .iter()
        .filter(|track| track.observations.len() >= config.min_observations)
        .map(|track| {
            let observations = track
                .observations
                .iter()
                .map(|observation| {
                    let extrinsics =
                        views_by_index.get(&observation.view_index).ok_or_else(|| {
                            anyhow!(
                                "missing camera pose for view {} while triangulating track",
                                observation.view_index
                            )
                        })?;
                    Ok(TriangulationObservation::new(
                        observation.image_point,
                        (*extrinsics).clone(),
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let triangulated = triangulate_observations_with_stats(intrinsics, &observations)?;

            ensure!(
                triangulated.mean_reprojection_error <= config.max_mean_reprojection_error,
                "track reprojection error {:.3} exceeds limit {:.3}",
                triangulated.mean_reprojection_error,
                config.max_mean_reprojection_error
            );

            if config.require_positive_depth {
                ensure!(
                    observations.iter().all(|observation| has_positive_depth(
                        &observation.extrinsics,
                        &triangulated.position
                    )),
                    "triangulated track lies behind at least one camera"
                );
            }

            Ok(LandmarkTrack {
                position: triangulated.position,
                observations: track.observations.clone(),
            })
        })
        .filter_map(|result| match result {
            Ok(track) => Some(Ok(track)),
            Err(_) => None,
        })
        .collect()
}

impl PnpSolver for EpnpSolver {
    fn solve_pose(
        &self,
        intrinsics: &CameraIntrinsics,
        correspondences: &[PointCorrespondence2D3D],
    ) -> Result<CameraExtrinsics> {
        EpnpSolver::solve_pose(self, intrinsics, correspondences)
    }
}

fn score_initial_pair(
    source: &DescriptorFeatureSet,
    target: &DescriptorFeatureSet,
    matches: &[PointMatch],
) -> InitialPairScore {
    let source_count = source.len().max(1) as f64;
    let target_count = target.len().max(1) as f64;
    let overlap_score = matches.len() as f64 / source_count.min(target_count);

    let baseline_score = matches
        .iter()
        .map(|matched| {
            let dx = (matched.source_point.x - matched.target_point.x) as f64;
            let dy = (matched.source_point.y - matched.target_point.y) as f64;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>()
        / matches.len().max(1) as f64;

    InitialPairScore {
        overlap_score,
        baseline_score,
        total_score: overlap_score * baseline_score,
    }
}

fn recover_relative_pose(
    intrinsics: &CameraIntrinsics,
    essential_matrix: &Matrix3<f64>,
    matches: &[PointMatch],
) -> Result<CameraExtrinsics> {
    let essential_matrix = enforce_essential_constraints(essential_matrix)?;
    let svd = essential_matrix.svd(true, true);
    let mut u = svd
        .u
        .ok_or_else(|| anyhow!("missing U from essential matrix SVD"))?;
    let mut v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("missing V^T from essential matrix SVD"))?;

    if u.determinant() < 0.0 {
        u.column_mut(2).neg_mut();
    }

    if v_t.determinant() < 0.0 {
        v_t.row_mut(2).neg_mut();
    }

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let rotations = [u * w * v_t, u * w.transpose() * v_t].map(project_to_rotation);
    let translation = u.column(2).into_owned();

    let mut best = None;
    for rotation in rotations {
        for candidate_translation in [translation, -translation] {
            let candidate = CameraExtrinsics {
                rotation,
                translation: candidate_translation,
            };
            let cheirality_score = cheirality_score(intrinsics, &candidate, matches)?;
            match best {
                Some((best_score, _)) if best_score >= cheirality_score => {}
                _ => best = Some((cheirality_score, candidate)),
            }
        }
    }

    let (score, pose) = best.ok_or_else(|| anyhow!("failed to recover a relative pose"))?;
    ensure!(
        score > 0,
        "none of the essential-matrix pose hypotheses produced front-facing triangulations"
    );
    Ok(pose)
}

fn enforce_essential_constraints(essential_matrix: &Matrix3<f64>) -> Result<Matrix3<f64>> {
    let svd = essential_matrix.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| anyhow!("missing U from essential matrix SVD"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("missing V^T from essential matrix SVD"))?;
    let sigma = (svd.singular_values[0] + svd.singular_values[1]) * 0.5;

    Ok(u * Matrix3::from_diagonal(&Vector3::new(sigma, sigma, 0.0)) * v_t)
}

fn project_to_rotation(rotation: Matrix3<f64>) -> Matrix3<f64> {
    let svd = rotation.svd(true, true);
    let u = svd.u.unwrap_or_else(Matrix3::identity);
    let mut v_t = svd.v_t.unwrap_or_else(Matrix3::identity);
    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        v_t.row_mut(2).neg_mut();
        rotation = u * v_t;
    }
    rotation
}

fn cheirality_score(
    intrinsics: &CameraIntrinsics,
    second_pose: &CameraExtrinsics,
    matches: &[PointMatch],
) -> Result<usize> {
    let first_pose = CameraExtrinsics {
        rotation: Matrix3::identity(),
        translation: Vector3::zeros(),
    };

    Ok(matches
        .iter()
        .filter_map(|matched| triangulate_match(intrinsics, &first_pose, second_pose, matched).ok())
        .filter(|world| {
            has_positive_depth(&first_pose, world) && has_positive_depth(second_pose, world)
        })
        .count())
}

fn triangulate_match(
    intrinsics: &CameraIntrinsics,
    first_pose: &CameraExtrinsics,
    second_pose: &CameraExtrinsics,
    matched: &PointMatch,
) -> Result<WorldPoint> {
    triangulate_observations(
        intrinsics,
        &[
            TriangulationObservation::new(
                ImagePoint64::new(matched.source_point.x as f64, matched.source_point.y as f64),
                first_pose.clone(),
            ),
            TriangulationObservation::new(
                ImagePoint64::new(matched.target_point.x as f64, matched.target_point.y as f64),
                second_pose.clone(),
            ),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Rotation3, Vector2};
    use structura_feature::descriptor::DescriptorFeature;
    use structura_geometry::point::ImagePoint;

    struct StubPnpSolver {
        expected: CameraExtrinsics,
    }

    impl PnpSolver for StubPnpSolver {
        fn solve_pose(
            &self,
            _intrinsics: &CameraIntrinsics,
            _correspondences: &[PointCorrespondence2D3D],
        ) -> Result<CameraExtrinsics> {
            Ok(self.expected.clone())
        }
    }

    #[test]
    fn selects_pair_with_best_overlap_and_baseline_proxy() {
        let first = feature_set(&[(10.0, 10.0), (30.0, 12.0), (60.0, 20.0), (90.0, 22.0)]);
        let second = feature_set(&[(20.0, 10.0), (40.0, 12.0), (70.0, 20.0), (100.0, 22.0)]);
        let third = feature_set(&[(11.0, 10.0), (31.0, 12.0), (61.0, 20.0), (300.0, 300.0)]);

        let selected = select_initial_pair(
            &[first, second, third],
            InitialPairSelectionConfig {
                min_matches: 2,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(selected.source_view_index, 0);
        assert_eq!(selected.target_view_index, 1);
        assert_eq!(selected.matches.len(), 3);
    }

    #[test]
    fn reconstructs_initial_pair_from_essential_matrix() {
        let intrinsics = sample_intrinsics();
        let first_pose = CameraExtrinsics {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        };
        let second_pose = CameraExtrinsics {
            rotation: Rotation3::from_euler_angles(0.02, -0.03, 0.05).into_inner(),
            translation: Vector3::new(0.4, 0.02, 0.03),
        };
        let world_points = sample_world_points();
        let matches = world_points
            .iter()
            .enumerate()
            .map(|(index, world)| {
                let source = project(&intrinsics, &first_pose, world);
                let target = project(&intrinsics, &second_pose, world);
                PointMatch {
                    source_index: index,
                    target_index: index,
                    source_point: source,
                    target_point: target,
                    distance: 0.0,
                    ratio: 0.1,
                }
            })
            .collect::<Vec<_>>();
        let essential_matrix = skew(second_pose.translation.normalize()) * second_pose.rotation;
        let initial_pair = InitialPairCandidate {
            source_view_index: 0,
            target_view_index: 1,
            matches,
            score: InitialPairScore {
                overlap_score: 1.0,
                baseline_score: 1.0,
                total_score: 1.0,
            },
        };

        let reconstruction =
            reconstruct_initial_pair(&intrinsics, &initial_pair, &essential_matrix).unwrap();

        assert_eq!(
            reconstruction.views[0].extrinsics.rotation,
            Matrix3::identity()
        );
        assert_eq!(
            reconstruction.views[0].extrinsics.translation,
            Vector3::zeros()
        );
        assert!(reconstruction.landmarks.len() >= 5);
        assert!(reconstruction.views[1].extrinsics.translation.norm() > 0.0);
        assert!(
            reconstruction
                .landmarks
                .iter()
                .all(|landmark| landmark.observations.len() == 2)
        );
    }

    #[test]
    fn registers_next_view_through_custom_pnp_solver() {
        let intrinsics = sample_intrinsics();
        let expected = CameraExtrinsics {
            rotation: Rotation3::from_euler_angles(-0.01, 0.04, 0.02).into_inner(),
            translation: Vector3::new(0.2, -0.1, 0.4),
        };
        let correspondences = sample_world_points()
            .into_iter()
            .enumerate()
            .map(|(index, world)| {
                PointCorrespondence2D3D::new(ImagePoint64::new(index as f64, index as f64), world)
            })
            .collect::<Vec<_>>();

        let registration = register_next_view_with_solver(
            &intrinsics,
            2,
            &correspondences,
            &StubPnpSolver {
                expected: expected.clone(),
            },
        )
        .unwrap();

        assert_eq!(registration.view.view_index, 2);
        assert_eq!(
            registration.used_correspondence_count,
            correspondences.len()
        );
        assert_eq!(registration.view.extrinsics, expected);
    }

    #[test]
    fn registers_next_view_with_default_epnp_solver() {
        let intrinsics = sample_intrinsics();
        let expected = CameraExtrinsics {
            rotation: Rotation3::from_euler_angles(0.03, -0.02, 0.01).into_inner(),
            translation: Vector3::new(0.1, -0.05, 1.2),
        };
        let correspondences = sample_world_points()
            .into_iter()
            .map(|world| {
                PointCorrespondence2D3D::new(project64(&intrinsics, &expected, world), world)
            })
            .collect::<Vec<_>>();

        let registration = register_next_view(&intrinsics, 2, &correspondences).unwrap();

        assert_eq!(registration.view.view_index, 2);
        assert_eq!(
            registration.used_correspondence_count,
            correspondences.len()
        );
        assert!((registration.view.extrinsics.translation - expected.translation).norm() < 1e-3);
        let rotation_delta = Rotation3::from_matrix(
            &(registration.view.extrinsics.rotation * expected.rotation.transpose()),
        );
        assert!(rotation_delta.angle() < 1e-3);
    }

    #[test]
    fn triangulates_sparse_tracks_from_registered_views() {
        let intrinsics = sample_intrinsics();
        let first = RegisteredView {
            view_index: 0,
            extrinsics: CameraExtrinsics {
                rotation: Matrix3::identity(),
                translation: Vector3::zeros(),
            },
        };
        let second = RegisteredView {
            view_index: 1,
            extrinsics: CameraExtrinsics {
                rotation: Rotation3::from_euler_angles(0.02, -0.03, 0.05).into_inner(),
                translation: Vector3::new(0.4, 0.02, 0.03),
            },
        };
        let world_points = sample_world_points();
        let tracks = world_points
            .iter()
            .enumerate()
            .map(|(index, world)| FeatureTrack {
                observations: vec![
                    LandmarkObservation {
                        view_index: 0,
                        feature_index: index,
                        image_point: project64(&intrinsics, &first.extrinsics, *world),
                    },
                    LandmarkObservation {
                        view_index: 1,
                        feature_index: index,
                        image_point: project64(&intrinsics, &second.extrinsics, *world),
                    },
                ],
            })
            .collect::<Vec<_>>();

        let landmarks = triangulate_feature_tracks(
            &intrinsics,
            &[first, second],
            &tracks,
            TrackTriangulationConfig::default(),
        )
        .unwrap();

        assert_eq!(landmarks.len(), world_points.len());
        assert!(
            landmarks
                .iter()
                .zip(world_points.iter())
                .all(|(landmark, world)| (landmark.position - *world).norm() < 1e-3)
        );
    }

    fn feature_set(points: &[(f32, f32)]) -> DescriptorFeatureSet {
        DescriptorFeatureSet {
            features: points
                .iter()
                .map(|&(x, y)| DescriptorFeature {
                    point: ImagePoint::new(x, y),
                    score: 1.0,
                    descriptor: vec![x, y, x + y],
                })
                .collect(),
        }
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

    fn sample_world_points() -> Vec<WorldPoint> {
        vec![
            WorldPoint::new(-0.3, -0.2, 4.0),
            WorldPoint::new(0.1, -0.1, 4.5),
            WorldPoint::new(0.4, 0.2, 5.0),
            WorldPoint::new(-0.2, 0.3, 5.5),
            WorldPoint::new(0.2, 0.4, 6.0),
            WorldPoint::new(-0.4, 0.1, 6.5),
        ]
    }

    fn project(
        intrinsics: &CameraIntrinsics,
        extrinsics: &CameraExtrinsics,
        world: &WorldPoint,
    ) -> nalgebra::Point2<f32> {
        let camera = extrinsics.rotation * world.coords + extrinsics.translation;
        let normalized = Vector2::new(camera.x / camera.z, camera.y / camera.z);
        nalgebra::Point2::new(
            (intrinsics.alpha * normalized.x + intrinsics.gamma * normalized.y + intrinsics.u0)
                as f32,
            (intrinsics.beta * normalized.y + intrinsics.v0) as f32,
        )
    }

    fn project64(
        intrinsics: &CameraIntrinsics,
        extrinsics: &CameraExtrinsics,
        world: WorldPoint,
    ) -> Point2<f64> {
        let camera = extrinsics.rotation * world.coords + extrinsics.translation;
        Point2::new(
            intrinsics.alpha * (camera.x / camera.z) + intrinsics.u0,
            intrinsics.beta * (camera.y / camera.z) + intrinsics.v0,
        )
    }

    fn skew(translation: Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(
            0.0,
            -translation.z,
            translation.y,
            translation.z,
            0.0,
            -translation.x,
            -translation.y,
            translation.x,
            0.0,
        )
    }
}
