use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use nalgebra::{Matrix3, Matrix3x4, Vector3};
use serde_json::Value;
use structura_feature::{
    descriptor::{DescriptorFeatureSet, lowe_ratio_match_descriptor_features},
    lightglue_onnx::{LightGlueOnnxConfig, LightGlueOnnxMatcher},
    matching::{LoweRatioConfig, PointMatch},
    sift::{SiftConfig, SiftExtractor},
};
use structura_geometry::{
    camera::CameraIntrinsics,
    point::{ImagePoint64, PointCorrespondence2D3D},
    triangulation::{
        TriangulationObservation, has_positive_depth, triangulate_observations_with_stats,
    },
    two_view::{FundamentalRansacConfig, estimate_fundamental, estimate_fundamental_ransac},
};
use structura_rerun::cloud::{SparseCloudFrame, SparseCloudLogger};
use structura_sfm::{
    FeatureTrack, InitialPairCandidate, InitialPairScore, LandmarkObservation, LandmarkTrack,
    RegisteredView, TrackTriangulationConfig,
    bundle_adjustment::bundle_adjust_registered_reconstruction_with_config,
    incremental::{reconstruct_initial_pair, register_next_view},
};
use structura_solver::BundleAdjustmentConfig;

#[derive(Debug, Clone)]
struct SequentialPair {
    source_view_index: usize,
    target_view_index: usize,
    matches: Vec<PointMatch>,
    essential_matrix: Matrix3<f64>,
    score: InitialPairScore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeatureBackend {
    Sift,
    SuperPointLightGlue,
}

impl FeatureBackend {
    fn name(self) -> &'static str {
        match self {
            Self::Sift => "sift",
            Self::SuperPointLightGlue => "superpoint_lightglue",
        }
    }
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            let root = self.find(self.parent[value]);
            self.parent[value] = root;
        }
        self.parent[value]
    }

    fn union_with_roots(&mut self, left: usize, right: usize) -> Option<(usize, usize)> {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return None;
        }

        match self.rank[left_root].cmp(&self.rank[right_root]) {
            std::cmp::Ordering::Less => {
                self.parent[left_root] = right_root;
                Some((right_root, left_root))
            }
            std::cmp::Ordering::Greater => {
                self.parent[right_root] = left_root;
                Some((left_root, right_root))
            }
            std::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] += 1;
                Some((left_root, right_root))
            }
        }
    }
}

#[test]
fn reconstructs_dino_dataset_with_sift_and_gt_intrinsics() {
    let result = reconstruct_dino_dataset(FeatureBackend::Sift).unwrap();

    assert!(
        result.registered_view_count >= 24,
        "registered only {} views",
        result.registered_view_count
    );
    assert!(
        result.landmark_count >= 400,
        "triangulated only {} landmarks",
        result.landmark_count
    );
}

#[test]
fn reconstructs_dino_dataset_with_superpoint_lightglue_and_gt_intrinsics() {
    let result = reconstruct_dino_dataset(FeatureBackend::SuperPointLightGlue).unwrap();

    assert!(
        result.registered_view_count >= 24,
        "registered only {} views",
        result.registered_view_count
    );
    assert!(
        result.landmark_count >= 400,
        "triangulated only {} landmarks",
        result.landmark_count
    );
}

#[derive(Debug, Clone, Copy)]
struct ReconstructionSummary {
    registered_view_count: usize,
    landmark_count: usize,
    bundle_adjustment_runs: usize,
}

#[derive(Debug, Clone)]
struct TrackStatistics {
    track_count: usize,
    mean_observations: f64,
    median_observations: usize,
    max_observations: usize,
    mean_unique_views: f64,
    median_unique_views: usize,
    max_unique_views: usize,
    duplicate_view_tracks: usize,
    duplicate_view_observations: usize,
    large_tracks: usize,
    tracks_ge_3_views: usize,
    tracks_ge_4_views: usize,
    tracks_ge_5_views: usize,
}

fn reconstruct_dino_dataset(backend: FeatureBackend) -> anyhow::Result<ReconstructionSummary> {
    let image_paths =
        discover_viff_images(Path::new("/home/matrix/structura/data/images")).unwrap();
    let gt_cameras =
        parse_ground_truth_cameras(Path::new("/home/matrix/structura/data/dino_gt.json")).unwrap();
    let image_paths = image_paths
        .into_iter()
        .take(gt_cameras.len())
        .collect::<Vec<_>>();
    assert!(image_paths.len() >= 8);

    let intrinsics = load_viff_intrinsics_from_gt(
        Path::new("/home/matrix/structura/data/dino_gt.json"),
        (720, 576),
    )
    .unwrap();
    reconstruct_dataset("dino", image_paths, intrinsics, backend, dino_bundle_adjustment_config())
}

fn reconstruct_temple_ring_dataset(
    backend: FeatureBackend,
) -> anyhow::Result<ReconstructionSummary> {
    let image_paths = discover_temple_ring_images(Path::new("/home/matrix/structura/data/templeRing"))?;
    assert!(image_paths.len() >= 8);

    let intrinsics = CameraIntrinsics {
        alpha: 800.0,
        beta: 800.0,
        gamma: 0.0,
        u0: 400.0,
        v0: 225.0,
    };
    reconstruct_dataset(
        "temple_ring",
        image_paths,
        intrinsics,
        backend,
        dino_bundle_adjustment_config(),
    )
}

fn reconstruct_dataset(
    dataset_name: &str,
    image_paths: Vec<PathBuf>,
    intrinsics: CameraIntrinsics,
    backend: FeatureBackend,
    ba_config: BundleAdjustmentConfig,
) -> anyhow::Result<ReconstructionSummary> {
    let rerun_logger =
        SparseCloudLogger::from_env(&format!("structura_sfm_{dataset_name}_sparse_{}", backend.name()));
    let mut log_step = 0_i64;
    let mut bundle_adjustment_runs = 0usize;
    eprintln!(
        "[{dataset_name}/{}] intrinsics: fx={:.3} fy={:.3} cx={:.3} cy={:.3}",
        backend.name(),
        intrinsics.alpha,
        intrinsics.beta,
        intrinsics.u0,
        intrinsics.v0
    );
    eprintln!(
        "[{dataset_name}/{}] BA config: max_iter={} lambda_factor={:.2} min_fidelity={:.1e} tol_abs={:.1e} tol_rel={:.1e}",
        backend.name(),
        ba_config.levenberg_params.base.max_iterations,
        ba_config.levenberg_params.lambda_factor,
        ba_config.levenberg_params.min_model_fidelity,
        ba_config.levenberg_params.base.error_tol_absolute,
        ba_config.levenberg_params.base.error_tol_relative,
    );

    let (feature_sets, sequential_pairs) = prepare_backend_data(&image_paths, &intrinsics, backend)?;
    eprintln!(
        "[{dataset_name}/{}] built {} sequential two-view models",
        backend.name(),
        sequential_pairs.len()
    );

    let global_tracks = build_global_tracks(&feature_sets, &sequential_pairs);
    eprintln!(
        "[{dataset_name}/{}] global tracks: {}",
        backend.name(),
        global_tracks.len()
    );
    log_track_statistics(backend, &global_tracks);

    let (initial_pair, initial) = choose_initial_pair(&intrinsics, &sequential_pairs).unwrap();
    eprintln!(
        "[{dataset_name}/{}] initial pair: {} -> {} with {} inliers",
        backend.name(),
        initial_pair.source_view_index,
        initial_pair.target_view_index,
        initial_pair.matches.len()
    );

    let mut registered_views = initial.views.into_iter().collect::<Vec<_>>();
    let triangulation_config = TrackTriangulationConfig {
        min_observations: 2,
        max_mean_reprojection_error: 4.0,
        require_positive_depth: true,
    };
    let mut landmarks = triangulate_registered_landmarks(
        &intrinsics,
        &registered_views,
        &global_tracks,
        triangulation_config,
    );
    eprintln!(
        "[{dataset_name}/{}] initial landmarks: {}",
        backend.name(),
        landmarks.len()
    );
    log_landmarks_to_rerun(
        &rerun_logger,
        SparseCloudFrame {
            step: log_step,
            registered_views: registered_views.len(),
            stage: "initial_triangulation",
            mean_reprojection_error: None,
        },
        &landmarks,
    );

    let registration_order = build_registration_order(image_paths.len(), &initial_pair);
    for view_index in registration_order {
        let correspondences =
            build_correspondences_for_view(view_index, &global_tracks, &landmarks);
        eprintln!(
            "[{dataset_name}/{}] view {}: {} 3D-2D correspondences",
            backend.name(),
            view_index,
            correspondences.len()
        );

        if correspondences.len() < 24 {
            continue;
        }

        let registration = match register_next_view(&intrinsics, view_index, &correspondences) {
            Ok(registration) => registration,
            Err(error) => {
                eprintln!(
                    "[{dataset_name}/{}] view {} registration failed: {error}",
                    backend.name(),
                    view_index
                );
                continue;
            }
        };
        registered_views.push(registration.view);

        landmarks = triangulate_registered_landmarks(
            &intrinsics,
            &registered_views,
            &global_tracks,
            triangulation_config,
        );
        log_step += 1;
        log_landmarks_to_rerun(
            &rerun_logger,
            SparseCloudFrame {
                step: log_step,
                registered_views: registered_views.len(),
                stage: "triangulated",
                mean_reprojection_error: None,
            },
            &landmarks,
        );

        if registered_views.len() % 5 == 0 || view_index + 1 == image_paths.len() {
            let ba_landmarks = select_landmarks_for_bundle_adjustment(
                backend,
                &intrinsics,
                &registered_views,
                &landmarks,
            );
            eprintln!(
                "[{dataset_name}/{}] BA selection after {} views: {} / {} landmarks",
                backend.name(),
                registered_views.len(),
                ba_landmarks.len(),
                landmarks.len()
            );
            if ba_landmarks.len() < 32 {
                continue;
            }

            let track_ids = ba_landmarks
                .iter()
                .map(|(track_id, _)| *track_id)
                .collect::<Vec<_>>();
            let current = ba_landmarks
                .iter()
                .map(|(_, landmark)| landmark.clone())
                .collect::<Vec<_>>();
            match bundle_adjust_registered_reconstruction_with_config(
                &intrinsics,
                &structura_solver::BrownConradyDistortion::default(),
                &registered_views,
                &current,
                &ba_config,
            ) {
                Ok(optimized) => {
                    bundle_adjustment_runs += 1;
                    registered_views = optimized.views;
                    let optimized_by_track_id = track_ids
                        .into_iter()
                        .zip(optimized.landmarks.into_iter())
                        .collect::<HashMap<_, _>>();
                    landmarks.iter_mut().for_each(|(track_id, landmark)| {
                        if let Some(optimized) = optimized_by_track_id.get(track_id) {
                            *landmark = optimized.clone();
                        }
                    });
                    eprintln!(
                        "[{dataset_name}/{}] BA after {} views: {} landmarks, mean reproj {:.3}",
                        backend.name(),
                        registered_views.len(),
                        landmarks.len(),
                        optimized.mean_reprojection_error
                    );
                    log_step += 1;
                    log_landmarks_to_rerun(
                        &rerun_logger,
                        SparseCloudFrame {
                            step: log_step,
                            registered_views: registered_views.len(),
                            stage: "bundle_adjustment",
                            mean_reprojection_error: Some(optimized.mean_reprojection_error),
                        },
                        &landmarks,
                    );
                }
                Err(error) => {
                    eprintln!(
                        "[{dataset_name}/{}] bundle adjustment skipped after registration failure: {error}",
                        backend.name()
                    );
                }
            }
        }
    }

    eprintln!(
        "[{dataset_name}/{}] final reconstruction: {} views, {} landmarks",
        backend.name(),
        registered_views.len(),
        landmarks.len()
    );
    log_step += 1;
    log_landmarks_to_rerun(
        &rerun_logger,
        SparseCloudFrame {
            step: log_step,
            registered_views: registered_views.len(),
            stage: "final",
            mean_reprojection_error: None,
        },
        &landmarks,
    );

    Ok(ReconstructionSummary {
        registered_view_count: registered_views.len(),
        landmark_count: landmarks.len(),
        bundle_adjustment_runs,
    })
}

#[test]
fn reconstructs_temple_ring_dataset_with_superpoint_lightglue_and_fixed_intrinsics() {
    let result = reconstruct_temple_ring_dataset(FeatureBackend::SuperPointLightGlue).unwrap();

    assert!(
        result.registered_view_count >= 8,
        "registered only {} views",
        result.registered_view_count
    );
    assert!(
        result.landmark_count >= 1_000,
        "triangulated only {} landmarks",
        result.landmark_count
    );
    assert!(
        result.bundle_adjustment_runs >= 1,
        "bundle adjustment never ran"
    );
}

#[test]
fn reconstructs_temple_ring_dataset_with_sift_and_fixed_intrinsics() {
    let result = reconstruct_temple_ring_dataset(FeatureBackend::Sift).unwrap();

    assert!(
        result.registered_view_count >= 8,
        "registered only {} views",
        result.registered_view_count
    );
    assert!(
        result.landmark_count >= 400,
        "triangulated only {} landmarks",
        result.landmark_count
    );
    assert!(
        result.bundle_adjustment_runs >= 1,
        "bundle adjustment never ran"
    );
}

fn log_track_statistics(backend: FeatureBackend, tracks: &[FeatureTrack]) {
    let stats = summarize_tracks(tracks);
    eprintln!(
        "[{}] track stats: count={} obs_mean={:.2} obs_median={} obs_max={} unique_mean={:.2} unique_median={} unique_max={} dup_tracks={} dup_obs={} large_tracks={} ge3={} ge4={} ge5={}",
        backend.name(),
        stats.track_count,
        stats.mean_observations,
        stats.median_observations,
        stats.max_observations,
        stats.mean_unique_views,
        stats.median_unique_views,
        stats.max_unique_views,
        stats.duplicate_view_tracks,
        stats.duplicate_view_observations,
        stats.large_tracks,
        stats.tracks_ge_3_views,
        stats.tracks_ge_4_views,
        stats.tracks_ge_5_views,
    );
}

fn summarize_tracks(tracks: &[FeatureTrack]) -> TrackStatistics {
    if tracks.is_empty() {
        return TrackStatistics {
            track_count: 0,
            mean_observations: 0.0,
            median_observations: 0,
            max_observations: 0,
            mean_unique_views: 0.0,
            median_unique_views: 0,
            max_unique_views: 0,
            duplicate_view_tracks: 0,
            duplicate_view_observations: 0,
            large_tracks: 0,
            tracks_ge_3_views: 0,
            tracks_ge_4_views: 0,
            tracks_ge_5_views: 0,
        };
    }

    let mut observation_counts = tracks
        .iter()
        .map(|track| track.observations.len())
        .collect::<Vec<_>>();
    let mut unique_view_counts = Vec::with_capacity(tracks.len());
    let mut duplicate_view_tracks = 0usize;
    let mut duplicate_view_observations = 0usize;
    let mut large_tracks = 0usize;
    let mut tracks_ge_3_views = 0usize;
    let mut tracks_ge_4_views = 0usize;
    let mut tracks_ge_5_views = 0usize;

    for track in tracks {
        let unique_views = track
            .observations
            .iter()
            .map(|observation| observation.view_index)
            .collect::<HashSet<_>>();
        let unique_count = unique_views.len();
        unique_view_counts.push(unique_count);
        let duplicate_count = track.observations.len().saturating_sub(unique_count);
        if duplicate_count > 0 {
            duplicate_view_tracks += 1;
            duplicate_view_observations += duplicate_count;
        }
        if unique_count >= 3 {
            tracks_ge_3_views += 1;
        }
        if unique_count >= 4 {
            tracks_ge_4_views += 1;
        }
        if unique_count >= 5 {
            tracks_ge_5_views += 1;
        }
        if track.observations.len() >= 8 {
            large_tracks += 1;
        }
    }

    observation_counts.sort_unstable();
    unique_view_counts.sort_unstable();

    TrackStatistics {
        track_count: tracks.len(),
        mean_observations: observation_counts.iter().sum::<usize>() as f64 / tracks.len() as f64,
        median_observations: observation_counts[observation_counts.len() / 2],
        max_observations: *observation_counts.last().unwrap_or(&0),
        mean_unique_views: unique_view_counts.iter().sum::<usize>() as f64 / tracks.len() as f64,
        median_unique_views: unique_view_counts[unique_view_counts.len() / 2],
        max_unique_views: *unique_view_counts.last().unwrap_or(&0),
        duplicate_view_tracks,
        duplicate_view_observations,
        large_tracks,
        tracks_ge_3_views,
        tracks_ge_4_views,
        tracks_ge_5_views,
    }
}

fn dino_bundle_adjustment_config() -> BundleAdjustmentConfig {
    let mut config = BundleAdjustmentConfig::default();
    config.levenberg_params.base.max_iterations = 600;
    config.levenberg_params.lambda_factor = 3.0;
    config.levenberg_params.min_model_fidelity = 1e-7;
    config
}

fn log_landmarks_to_rerun(
    logger: &SparseCloudLogger,
    frame: SparseCloudFrame<'_>,
    landmarks: &[(usize, LandmarkTrack)],
) {
    if !logger.is_enabled() {
        return;
    }

    let points = landmarks
        .iter()
        .map(|(_, landmark)| landmark.position)
        .collect::<Vec<_>>();
    if let Err(error) = logger.log_sparse_cloud(frame, &points) {
        eprintln!("rerun sparse cloud logging failed: {error}");
    }
}

fn discover_viff_images(images_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut indexed = fs::read_dir(images_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            let stem = name.strip_prefix("viff.")?.strip_suffix(".ppm")?;
            Some((stem.parse::<usize>().ok()?, path))
        })
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(index, _)| *index);
    Ok(indexed.into_iter().map(|(_, path)| path).collect())
}

fn discover_temple_ring_images(images_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut indexed = fs::read_dir(images_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            let stem = name.strip_prefix("templeR")?.strip_suffix(".png")?;
            let index = stem.parse::<usize>().ok()?;
            Some((index, path))
        })
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(index, _)| *index);
    Ok(indexed.into_iter().map(|(_, path)| path).collect())
}

fn prepare_backend_data(
    image_paths: &[PathBuf],
    intrinsics: &CameraIntrinsics,
    backend: FeatureBackend,
) -> anyhow::Result<(Vec<DescriptorFeatureSet>, Vec<SequentialPair>)> {
    match backend {
        FeatureBackend::Sift => {
            let feature_sets = extract_sift_features(image_paths)?;
            let sequential_pairs = build_neighbor_pairs(&feature_sets, intrinsics, backend)?;
            Ok((feature_sets, sequential_pairs))
        }
        FeatureBackend::SuperPointLightGlue => {
            extract_superpoint_lightglue_data(image_paths, intrinsics)
        }
    }
}

fn extract_sift_features(image_paths: &[PathBuf]) -> anyhow::Result<Vec<DescriptorFeatureSet>> {
    let extractor = SiftExtractor::with_config(SiftConfig {
        nfeatures: 4096,
        ..Default::default()
    });

    image_paths
        .iter()
        .map(|path| {
            let features = extractor.extract_from_path(path)?;
            eprintln!("{}: {} sift features", path.display(), features.len());
            Ok(features)
        })
        .collect()
}

fn build_neighbor_pairs(
    feature_sets: &[DescriptorFeatureSet],
    intrinsics: &CameraIntrinsics,
    backend: FeatureBackend,
) -> anyhow::Result<Vec<SequentialPair>> {
    let mut pairs = Vec::new();
    let max_offset = match backend {
        FeatureBackend::Sift => 1,
        FeatureBackend::SuperPointLightGlue => unreachable!("use paired ONNX pipeline instead"),
    };
    let lowe_ratio = match backend {
        FeatureBackend::Sift => LoweRatioConfig::new(0.8),
        FeatureBackend::SuperPointLightGlue => {
            unreachable!("use paired ONNX pipeline instead")
        }
    };

    for source_view_index in 0..feature_sets.len() - 1 {
        for offset in 1..=max_offset {
            let target_view_index = source_view_index + offset;
            if target_view_index >= feature_sets.len() {
                break;
            }

            let mutual_matches = mutual_matches(
                &feature_sets[source_view_index],
                &feature_sets[target_view_index],
                lowe_ratio,
            )?;
            if mutual_matches.len() < 32 {
                continue;
            }

            let points_a = mutual_matches
                .iter()
                .map(|matched| {
                    ImagePoint64::new(matched.source_point.x as f64, matched.source_point.y as f64)
                })
                .collect::<Vec<_>>();
            let points_b = mutual_matches
                .iter()
                .map(|matched| {
                    ImagePoint64::new(matched.target_point.x as f64, matched.target_point.y as f64)
                })
                .collect::<Vec<_>>();
            let ransac = match estimate_fundamental_ransac(
                &points_a,
                &points_b,
                FundamentalRansacConfig {
                    iterations: 1024,
                    inlier_threshold: 1.5,
                    min_inlier_count: 32,
                },
            ) {
                Ok(ransac) => ransac,
                Err(error) => {
                    eprintln!(
                        "pair {} -> {} skipped after geometric verification failed: {error}",
                        source_view_index, target_view_index
                    );
                    continue;
                }
            };
            let matches = ransac
                .inlier_indices
                .iter()
                .map(|&index| mutual_matches[index])
                .collect::<Vec<_>>();
            let score = score_pair(
                &feature_sets[source_view_index],
                &feature_sets[target_view_index],
                &matches,
            );

            eprintln!(
                "pair {} -> {}: {} mutual, {} inliers",
                source_view_index,
                target_view_index,
                mutual_matches.len(),
                matches.len()
            );

            pairs.push(SequentialPair {
                source_view_index,
                target_view_index,
                matches,
                essential_matrix: estimate_essential_from_inliers(
                    &points_a,
                    &points_b,
                    &ransac.inlier_indices,
                    intrinsics,
                )?,
                score,
            });
        }
    }

    Ok(pairs)
}

fn mutual_matches(
    source: &DescriptorFeatureSet,
    target: &DescriptorFeatureSet,
    config: LoweRatioConfig,
) -> anyhow::Result<Vec<PointMatch>> {
    let forward = lowe_ratio_match_descriptor_features(source, target, config)?;
    let reverse = lowe_ratio_match_descriptor_features(target, source, config)?;
    let reverse_pairs = reverse
        .iter()
        .map(|matched| (matched.target_index, matched.source_index))
        .collect::<HashSet<_>>();

    Ok(forward
        .into_iter()
        .filter(|matched| reverse_pairs.contains(&(matched.source_index, matched.target_index)))
        .collect())
}

fn extract_superpoint_lightglue_data(
    image_paths: &[PathBuf],
    intrinsics: &CameraIntrinsics,
) -> anyhow::Result<(Vec<DescriptorFeatureSet>, Vec<SequentialPair>)> {
    let model_path = Path::new("/home/matrix/structura/models/superpoint_lightglue_pipeline.ort.onnx");
    let mut matcher = LightGlueOnnxMatcher::with_config(
        model_path,
        LightGlueOnnxConfig {
            resize_width: 1024,
            resize_height: 1024,
        },
    )?;
    let images = image_paths
        .iter()
        .map(|path| {
            image::open(path)
                .map(|image| image.to_rgb8())
                .with_context(|| format!("failed to open image {}", path.display()))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut features_by_view = vec![Vec::new(); image_paths.len()];
    let mut keypoint_indices = vec![HashMap::<(i32, i32), usize>::new(); image_paths.len()];
    let mut sequential_pairs = Vec::new();

    for source_view_index in 0..image_paths.len() - 1 {
        for offset in 1..=2 {
            let target_view_index = source_view_index + offset;
            if target_view_index >= image_paths.len() {
                break;
            }

            let raw_matches = matcher.match_from_rgb_images(
                &images[source_view_index],
                &images[target_view_index],
            )?;
            if raw_matches.len() < 32 {
                continue;
            }

            let points_a = raw_matches
                .iter()
                .map(|matched| {
                    ImagePoint64::new(matched.source_point.x as f64, matched.source_point.y as f64)
                })
                .collect::<Vec<_>>();
            let points_b = raw_matches
                .iter()
                .map(|matched| {
                    ImagePoint64::new(matched.target_point.x as f64, matched.target_point.y as f64)
                })
                .collect::<Vec<_>>();
            let ransac = match estimate_fundamental_ransac(
                &points_a,
                &points_b,
                FundamentalRansacConfig {
                    iterations: 1024,
                    inlier_threshold: 1.5,
                    min_inlier_count: 32,
                },
            ) {
                Ok(ransac) => ransac,
                Err(error) => {
                    eprintln!(
                        "pair {} -> {} skipped after geometric verification failed: {error}",
                        source_view_index, target_view_index
                    );
                    continue;
                }
            };

            let matches = ransac
                .inlier_indices
                .iter()
                .map(|&index| {
                    let matched = raw_matches[index];
                    let source_index = intern_keypoint(
                        &mut features_by_view[source_view_index],
                        &mut keypoint_indices[source_view_index],
                        matched.source_point,
                    );
                    let target_index = intern_keypoint(
                        &mut features_by_view[target_view_index],
                        &mut keypoint_indices[target_view_index],
                        matched.target_point,
                    );
                    PointMatch {
                        source_index,
                        target_index,
                        ..matched
                    }
                })
                .collect::<Vec<_>>();
            let score = score_pair(
                &DescriptorFeatureSet {
                    features: features_by_view[source_view_index].clone(),
                },
                &DescriptorFeatureSet {
                    features: features_by_view[target_view_index].clone(),
                },
                &matches,
            );

            eprintln!(
                "pair {} -> {}: {} lightglue, {} inliers",
                source_view_index,
                target_view_index,
                raw_matches.len(),
                matches.len()
            );

            sequential_pairs.push(SequentialPair {
                source_view_index,
                target_view_index,
                essential_matrix: estimate_essential_from_inliers(
                    &points_a,
                    &points_b,
                    &ransac.inlier_indices,
                    intrinsics,
                )?,
                matches,
                score,
            });
        }
    }

    let feature_sets: Vec<DescriptorFeatureSet> = features_by_view
        .into_iter()
        .map(|features| DescriptorFeatureSet { features })
        .collect();
    for (path, features) in image_paths.iter().zip(feature_sets.iter()) {
        eprintln!("{}: {} superpoint features", path.display(), features.len());
    }

    Ok((feature_sets, sequential_pairs))
}

fn intern_keypoint(
    features: &mut Vec<structura_feature::descriptor::DescriptorFeature>,
    index_by_point: &mut HashMap<(i32, i32), usize>,
    point: structura_geometry::point::ImagePoint,
) -> usize {
    let key = (point.x.round() as i32, point.y.round() as i32);
    if let Some(&index) = index_by_point.get(&key) {
        return index;
    }

    let index = features.len();
    features.push(structura_feature::descriptor::DescriptorFeature {
        point,
        score: 1.0,
        descriptor: Vec::new(),
    });
    index_by_point.insert(key, index);
    index
}

fn score_pair(
    source: &DescriptorFeatureSet,
    target: &DescriptorFeatureSet,
    matches: &[PointMatch],
) -> InitialPairScore {
    let overlap_score = matches.len() as f64 / source.len().min(target.len()).max(1) as f64;
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

fn choose_initial_pair(
    intrinsics: &CameraIntrinsics,
    pairs: &[SequentialPair],
) -> anyhow::Result<(SequentialPair, structura_sfm::InitialReconstruction)> {
    let mut sorted = pairs
        .iter()
        .filter(|pair| pair.matches.len() >= 64)
        .cloned()
        .collect::<Vec<_>>();
    sorted.sort_by(|left, right| right.score.total_score.total_cmp(&left.score.total_score));

    sorted
        .into_iter()
        .filter_map(|pair| {
            let reconstruction = reconstruct_initial_pair(
                intrinsics,
                &InitialPairCandidate {
                    source_view_index: pair.source_view_index,
                    target_view_index: pair.target_view_index,
                    matches: pair.matches.clone(),
                    score: pair.score,
                },
                &pair.essential_matrix,
            )
            .ok()?;
            Some((pair, reconstruction))
        })
        .max_by(
            |(left_pair, left_reconstruction), (right_pair, right_reconstruction)| {
                left_reconstruction
                    .landmarks
                    .len()
                    .cmp(&right_reconstruction.landmarks.len())
                    .then_with(|| {
                        left_pair
                            .score
                            .total_score
                            .total_cmp(&right_pair.score.total_score)
                    })
            },
        )
        .ok_or_else(|| {
            anyhow::anyhow!("no sequential pair produced a valid initial reconstruction")
        })
}

fn build_registration_order(view_count: usize, initial_pair: &SequentialPair) -> Vec<usize> {
    let mut order = ((initial_pair.target_view_index + 1)..view_count).collect::<Vec<_>>();
    order.extend((0..initial_pair.source_view_index).rev());
    order
}

fn estimate_essential_from_inliers(
    points_a: &[ImagePoint64],
    points_b: &[ImagePoint64],
    inlier_indices: &[usize],
    intrinsics: &CameraIntrinsics,
) -> anyhow::Result<Matrix3<f64>> {
    let k_inv = intrinsics
        .matrix()
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("intrinsics matrix is not invertible"))?;
    let normalized_a = inlier_indices
        .iter()
        .map(|&index| {
            let point = k_inv * points_a[index].to_homogeneous();
            ImagePoint64::new(point[0] / point[2], point[1] / point[2])
        })
        .collect::<Vec<_>>();
    let normalized_b = inlier_indices
        .iter()
        .map(|&index| {
            let point = k_inv * points_b[index].to_homogeneous();
            ImagePoint64::new(point[0] / point[2], point[1] / point[2])
        })
        .collect::<Vec<_>>();
    let essential = estimate_fundamental(&normalized_a, &normalized_b)?;
    let svd = essential.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("essential SVD missing U"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("essential SVD missing V^T"))?;
    let sigma = (svd.singular_values[0] + svd.singular_values[1]) * 0.5;
    Ok(u * Matrix3::from_diagonal(&Vector3::new(sigma, sigma, 0.0)) * v_t)
}

fn build_global_tracks(
    feature_sets: &[DescriptorFeatureSet],
    sequential_pairs: &[SequentialPair],
) -> Vec<FeatureTrack> {
    let mut next_observation_id = 0usize;
    let observation_ids = feature_sets
        .iter()
        .map(|feature_set| {
            (0..feature_set.features.len())
                .map(|_| {
                    let id = next_observation_id;
                    next_observation_id += 1;
                    id
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut union_find = UnionFind::new(next_observation_id);
    let mut component_views = vec![HashSet::new(); next_observation_id];
    for (view_index, ids) in observation_ids.iter().enumerate() {
        for &observation_id in ids {
            component_views[observation_id].insert(view_index);
        }
    }
    let mut rejected_conflicts = 0usize;
    for pair in sequential_pairs {
        for matched in &pair.matches {
            let source_id = observation_ids[pair.source_view_index][matched.source_index];
            let target_id = observation_ids[pair.target_view_index][matched.target_index];
            let source_root = union_find.find(source_id);
            let target_root = union_find.find(target_id);
            if source_root == target_root {
                continue;
            }
            if !component_views[source_root].is_disjoint(&component_views[target_root]) {
                rejected_conflicts += 1;
                continue;
            }
            if let Some((kept_root, merged_root)) = union_find.union_with_roots(source_root, target_root)
            {
                let merged_views = std::mem::take(&mut component_views[merged_root]);
                component_views[kept_root].extend(merged_views);
            }
        }
    }
    eprintln!(
        "track builder rejected {} conflicting unions",
        rejected_conflicts
    );

    let mut groups = HashMap::<usize, Vec<LandmarkObservation>>::new();
    for (view_index, feature_set) in feature_sets.iter().enumerate() {
        for (feature_index, feature) in feature_set.features.iter().enumerate() {
            let root = union_find.find(observation_ids[view_index][feature_index]);
            groups.entry(root).or_default().push(LandmarkObservation {
                view_index,
                feature_index,
                image_point: ImagePoint64::new(feature.point.x as f64, feature.point.y as f64),
            });
        }
    }

    groups
        .into_values()
        .filter_map(|mut observations| {
            observations.sort_by_key(|observation| observation.view_index);
            let unique_views = observations
                .iter()
                .map(|observation| observation.view_index)
                .collect::<HashSet<_>>();
            (unique_views.len() >= 2).then_some(FeatureTrack { observations })
        })
        .collect()
}

fn triangulate_registered_landmarks(
    intrinsics: &CameraIntrinsics,
    registered_views: &[RegisteredView],
    global_tracks: &[FeatureTrack],
    config: TrackTriangulationConfig,
) -> Vec<(usize, LandmarkTrack)> {
    let views_by_index = registered_views
        .iter()
        .map(|view| (view.view_index, &view.extrinsics))
        .collect::<HashMap<_, _>>();

    global_tracks
        .iter()
        .enumerate()
        .filter_map(|(track_id, track)| {
            let registered_observations = track
                .observations
                .iter()
                .filter(|observation| views_by_index.contains_key(&observation.view_index))
                .cloned()
                .collect::<Vec<_>>();
            if registered_observations.len() < config.min_observations {
                return None;
            }

            let observations = registered_observations
                .iter()
                .map(|observation| {
                    TriangulationObservation::new(
                        observation.image_point,
                        views_by_index[&observation.view_index].clone(),
                    )
                })
                .collect::<Vec<_>>();
            let triangulated =
                triangulate_observations_with_stats(intrinsics, &observations).ok()?;
            if triangulated.mean_reprojection_error > config.max_mean_reprojection_error {
                return None;
            }
            if config.require_positive_depth
                && !observations.iter().all(|observation| {
                    has_positive_depth(&observation.extrinsics, &triangulated.position)
                })
            {
                return None;
            }

            Some((
                track_id,
                LandmarkTrack {
                    position: triangulated.position,
                    observations: registered_observations,
                },
            ))
        })
        .collect()
}

fn select_landmarks_for_bundle_adjustment(
    backend: FeatureBackend,
    intrinsics: &CameraIntrinsics,
    registered_views: &[RegisteredView],
    landmarks: &[(usize, LandmarkTrack)],
) -> Vec<(usize, LandmarkTrack)> {
    match backend {
        FeatureBackend::Sift => landmarks.to_vec(),
        FeatureBackend::SuperPointLightGlue => {
            let views_by_index = registered_views
                .iter()
                .map(|view| (view.view_index, &view.extrinsics))
                .collect::<HashMap<_, _>>();

            landmarks
                .iter()
                .filter_map(|(track_id, landmark)| {
                    if landmark.observations.len() < 3 {
                        return None;
                    }
                    let observations = landmark
                        .observations
                        .iter()
                        .map(|observation| {
                            TriangulationObservation::new(
                                observation.image_point,
                                views_by_index[&observation.view_index].clone(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let stats = triangulate_observations_with_stats(intrinsics, &observations).ok()?;
                    (stats.mean_reprojection_error <= 1.5)
                        .then_some((*track_id, landmark.clone()))
                })
                .collect()
        }
    }
}

fn build_correspondences_for_view(
    view_index: usize,
    global_tracks: &[FeatureTrack],
    landmarks: &[(usize, LandmarkTrack)],
) -> Vec<PointCorrespondence2D3D> {
    landmarks
        .iter()
        .filter_map(|(track_id, landmark)| {
            global_tracks[*track_id]
                .observations
                .iter()
                .find(|observation| observation.view_index == view_index)
                .map(|observation| {
                    PointCorrespondence2D3D::new(observation.image_point, landmark.position)
                })
        })
        .collect()
}

fn parse_ground_truth_cameras(gt_path: &Path) -> anyhow::Result<Vec<Value>> {
    let root: Value = serde_json::from_reader(fs::File::open(gt_path)?)?;
    root.get("cameras")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing cameras array in {}", gt_path.display()))
}

fn load_viff_intrinsics_from_gt(
    gt_path: &Path,
    image_size: (u32, u32),
) -> anyhow::Result<CameraIntrinsics> {
    let cameras = parse_ground_truth_cameras(gt_path)?;
    let mut fx_sum = 0.0;
    let mut fy_sum = 0.0;
    let mut cx_sum = 0.0;
    let mut cy_sum = 0.0;
    let mut cx_count = 0usize;
    let mut cy_count = 0usize;

    for camera in &cameras {
        let decomposition = decompose_projection_matrix(&parse_projection_matrix(camera)?)?;
        let intrinsics = decomposition.0;
        let fx = intrinsics[(0, 0)].abs();
        let fy = intrinsics[(1, 1)].abs();
        fx_sum += fx;
        fy_sum += fy;

        let cx = intrinsics[(0, 2)];
        let cy = intrinsics[(1, 2)];
        if (0.0..image_size.0 as f64).contains(&cx) {
            cx_sum += cx;
            cx_count += 1;
        }
        if (0.0..image_size.1 as f64).contains(&cy) {
            cy_sum += cy;
            cy_count += 1;
        }
    }

    Ok(CameraIntrinsics {
        alpha: fx_sum / cameras.len() as f64,
        beta: fy_sum / cameras.len() as f64,
        gamma: 0.0,
        u0: if cx_count > 0 {
            cx_sum / cx_count as f64
        } else {
            image_size.0 as f64 * 0.5
        },
        v0: if cy_count > 0 {
            cy_sum / cy_count as f64
        } else {
            image_size.1 as f64 * 0.5
        },
    })
}

fn parse_projection_matrix(camera: &Value) -> anyhow::Result<Matrix3x4<f64>> {
    let rows = camera
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("camera entry is not a 3x4 matrix"))?;
    let mut values = [0.0; 12];
    for (row_index, row) in rows.iter().enumerate() {
        let cols = row
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("camera row is not an array"))?;
        for col_index in 0..4 {
            values[row_index * 4 + col_index] = cols[col_index].as_f64().ok_or_else(|| {
                anyhow::anyhow!("camera[{row_index}][{col_index}] is not numeric")
            })?;
        }
    }
    Ok(Matrix3x4::from_row_slice(&values))
}

fn decompose_projection_matrix(
    projection: &Matrix3x4<f64>,
) -> anyhow::Result<(Matrix3<f64>, Matrix3<f64>, Vector3<f64>)> {
    let m = projection.fixed_columns::<3>(0).into_owned();
    let (intrinsics, rotation) = rq_decompose(&m)?;
    let camera_center = m
        .qr()
        .solve(&(-projection.column(3).into_owned()))
        .ok_or_else(|| anyhow::anyhow!("failed to solve camera center"))?;
    Ok((intrinsics, rotation, camera_center))
}

fn rq_decompose(matrix: &Matrix3<f64>) -> anyhow::Result<(Matrix3<f64>, Matrix3<f64>)> {
    let qr = flipud(matrix).transpose().qr();
    let (q1, r1) = qr.unpack();
    let mut intrinsics = flipud(&fliplr(&r1.transpose()));
    let mut rotation = flipud(&q1.transpose());

    let s0 = if intrinsics[(0, 0)] >= 0.0 { 1.0 } else { -1.0 };
    let s1 = if intrinsics[(1, 1)] >= 0.0 { 1.0 } else { -1.0 };
    let sign_fix = Matrix3::new(s0, 0.0, 0.0, 0.0, s1, 0.0, 0.0, 0.0, s0 * s1);
    intrinsics *= sign_fix;
    rotation = sign_fix * rotation;

    if rotation.determinant() < 0.0 {
        let reflect_z = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        intrinsics *= reflect_z;
        rotation = reflect_z * rotation;
    }

    let scale = intrinsics[(2, 2)].abs();
    let intrinsics = intrinsics / scale;
    let svd = rotation.svd(true, true);
    let mut u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("RQ decomposition missing U"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("RQ decomposition missing V^T"))?;
    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        u.column_mut(2).neg_mut();
        rotation = u * v_t;
    }

    Ok((intrinsics, rotation))
}

fn flipud(matrix: &Matrix3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        matrix[(2, 0)],
        matrix[(2, 1)],
        matrix[(2, 2)],
        matrix[(1, 0)],
        matrix[(1, 1)],
        matrix[(1, 2)],
        matrix[(0, 0)],
        matrix[(0, 1)],
        matrix[(0, 2)],
    )
}

fn fliplr(matrix: &Matrix3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        matrix[(0, 2)],
        matrix[(0, 1)],
        matrix[(0, 0)],
        matrix[(1, 2)],
        matrix[(1, 1)],
        matrix[(1, 0)],
        matrix[(2, 2)],
        matrix[(2, 1)],
        matrix[(2, 0)],
    )
}
