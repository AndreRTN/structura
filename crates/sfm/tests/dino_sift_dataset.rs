use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use nalgebra::{Matrix3, Matrix3x4, Vector3};
use serde_json::Value;
use structura_feature::{
    descriptor::DescriptorFeatureSet,
    matching::{LoweRatioConfig, PointMatch},
    sift::{SiftConfig, SiftExtractor, lowe_ratio_match_features},
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

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }

        match self.rank[left_root].cmp(&self.rank[right_root]) {
            std::cmp::Ordering::Less => self.parent[left_root] = right_root,
            std::cmp::Ordering::Greater => self.parent[right_root] = left_root,
            std::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] += 1;
            }
        }
    }
}

#[test]
fn reconstructs_dino_dataset_with_sift_and_gt_intrinsics() {
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
    let ba_config = dino_bundle_adjustment_config();
    let rerun_logger = SparseCloudLogger::from_env("structura_sfm_dino_sparse");
    let mut log_step = 0_i64;
    eprintln!(
        "GT intrinsics: fx={:.3} fy={:.3} cx={:.3} cy={:.3}",
        intrinsics.alpha, intrinsics.beta, intrinsics.u0, intrinsics.v0
    );
    eprintln!(
        "BA config: max_iter={} lambda_factor={:.2} min_fidelity={:.1e} tol_abs={:.1e} tol_rel={:.1e}",
        ba_config.levenberg_params.base.max_iterations,
        ba_config.levenberg_params.lambda_factor,
        ba_config.levenberg_params.min_model_fidelity,
        ba_config.levenberg_params.base.error_tol_absolute,
        ba_config.levenberg_params.base.error_tol_relative,
    );

    let feature_sets = extract_sift_features(&image_paths).unwrap();
    let sequential_pairs = build_sequential_pairs(&feature_sets, &intrinsics).unwrap();
    eprintln!(
        "built {} sequential two-view models",
        sequential_pairs.len()
    );

    let global_tracks = build_global_tracks(&feature_sets, &sequential_pairs);
    eprintln!("global tracks: {}", global_tracks.len());

    let (initial_pair, initial) = choose_initial_pair(&intrinsics, &sequential_pairs).unwrap();
    eprintln!(
        "initial pair: {} -> {} with {} inliers",
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
    eprintln!("initial landmarks: {}", landmarks.len());
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
            "view {}: {} 3D-2D correspondences",
            view_index,
            correspondences.len()
        );

        if correspondences.len() < 24 {
            continue;
        }

        let registration = match register_next_view(&intrinsics, view_index, &correspondences) {
            Ok(registration) => registration,
            Err(error) => {
                eprintln!("view {} registration failed: {error}", view_index);
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
            let track_ids = landmarks
                .iter()
                .map(|(track_id, _)| *track_id)
                .collect::<Vec<_>>();
            let current = landmarks
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
                    registered_views = optimized.views;
                    landmarks = track_ids
                        .into_iter()
                        .zip(optimized.landmarks.into_iter())
                        .collect();
                    eprintln!(
                        "BA after {} views: {} landmarks, mean reproj {:.3}",
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
                    eprintln!("bundle adjustment skipped after registration failure: {error}");
                }
            }
        }
    }

    eprintln!(
        "final reconstruction: {} views, {} landmarks",
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

    assert!(
        registered_views.len() >= 24,
        "registered only {} views",
        registered_views.len()
    );
    assert!(
        landmarks.len() >= 400,
        "triangulated only {} landmarks",
        landmarks.len()
    );
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

fn build_sequential_pairs(
    feature_sets: &[DescriptorFeatureSet],
    intrinsics: &CameraIntrinsics,
) -> anyhow::Result<Vec<SequentialPair>> {
    let mut pairs = Vec::new();

    for source_view_index in 0..feature_sets.len() - 1 {
        let target_view_index = source_view_index + 1;
        let mutual_matches = mutual_matches(
            &feature_sets[source_view_index],
            &feature_sets[target_view_index],
            LoweRatioConfig::new(0.8),
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
        let ransac = estimate_fundamental_ransac(
            &points_a,
            &points_b,
            FundamentalRansacConfig {
                iterations: 1024,
                inlier_threshold: 1.5,
                min_inlier_count: 32,
            },
        )?;
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

    Ok(pairs)
}

fn mutual_matches(
    source: &DescriptorFeatureSet,
    target: &DescriptorFeatureSet,
    config: LoweRatioConfig,
) -> anyhow::Result<Vec<PointMatch>> {
    let forward = lowe_ratio_match_features(source, target, config)?;
    let reverse = lowe_ratio_match_features(target, source, config)?;
    let reverse_pairs = reverse
        .iter()
        .map(|matched| (matched.target_index, matched.source_index))
        .collect::<HashSet<_>>();

    Ok(forward
        .into_iter()
        .filter(|matched| reverse_pairs.contains(&(matched.source_index, matched.target_index)))
        .collect())
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
        .find_map(|pair| {
            reconstruct_initial_pair(
                intrinsics,
                &InitialPairCandidate {
                    source_view_index: pair.source_view_index,
                    target_view_index: pair.target_view_index,
                    matches: pair.matches.clone(),
                    score: pair.score,
                },
                &pair.essential_matrix,
            )
            .ok()
            .map(|reconstruction| (pair, reconstruction))
        })
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
    for pair in sequential_pairs {
        for matched in &pair.matches {
            union_find.union(
                observation_ids[pair.source_view_index][matched.source_index],
                observation_ids[pair.target_view_index][matched.target_index],
            );
        }
    }

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
