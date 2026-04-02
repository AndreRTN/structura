use anyhow::{Result, anyhow};
use structura_geometry::point::ImagePoint;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoweRatioConfig {
    pub max_ratio: f32,
}

impl LoweRatioConfig {
    pub const fn new(max_ratio: f32) -> Self {
        Self { max_ratio }
    }
}

impl Default for LoweRatioConfig {
    fn default() -> Self {
        Self { max_ratio: 0.8 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMatch {
    pub source_index: usize,
    pub target_index: usize,
    pub source_point: ImagePoint,
    pub target_point: ImagePoint,
    pub distance: f32,
    pub ratio: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexMatch {
    pub source_index: usize,
    pub target_index: usize,
    pub distance: f32,
    pub ratio: f32,
}

pub fn lowe_ratio_match_by<T, U, F>(
    source: &[T],
    target: &[U],
    config: LoweRatioConfig,
    mut distance_fn: F,
) -> Result<Vec<IndexMatch>>
where
    F: FnMut(&T, &U) -> f32,
{
    if source.is_empty() {
        return Ok(Vec::new());
    }

    if target.len() < 2 {
        return Err(anyhow!(
            "lowe ratio test requires at least two target descriptors"
        ));
    }

    let matches = {
        let mut matches = source
            .iter()
            .enumerate()
            .filter_map(|(source_index, source_descriptor)| {
                let (best, second_best) =
                    best_two_candidates_by(source_descriptor, target, &mut distance_fn)?;
                let (best_target_index, best_distance) = best;
                let (_, second_best_distance) = second_best;

                (second_best_distance > f32::EPSILON)
                    .then(|| best_distance / second_best_distance)
                    .filter(|ratio| *ratio <= config.max_ratio)
                    .map(|ratio| IndexMatch {
                        source_index,
                        target_index: best_target_index,
                        distance: best_distance,
                        ratio,
                    })
            })
            .collect::<Vec<_>>();

        matches.sort_by(|left, right| left.ratio.total_cmp(&right.ratio));
        matches
    };

    Ok(deduplicate_target_matches_by_index(matches))
}

pub fn lowe_ratio_match_points(
    source: &[ImagePoint],
    target: &[ImagePoint],
    config: LoweRatioConfig,
) -> Result<Vec<PointMatch>> {
    lowe_ratio_match_by(source, target, config, |source_point, target_point| {
        squared_distance(*source_point, *target_point).sqrt()
    })
    .map(|matches| {
        matches
            .into_iter()
            .map(|matched| PointMatch {
                source_index: matched.source_index,
                target_index: matched.target_index,
                source_point: source[matched.source_index],
                target_point: target[matched.target_index],
                distance: matched.distance,
                ratio: matched.ratio,
            })
            .collect()
    })
}

fn squared_distance(a: ImagePoint, b: ImagePoint) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

fn best_two_candidates_by<T, U, F>(
    source_descriptor: &T,
    target: &[U],
    distance_fn: &mut F,
) -> Option<((usize, f32), (usize, f32))>
where
    F: FnMut(&T, &U) -> f32,
{
    target
        .iter()
        .enumerate()
        .map(|(target_index, target_descriptor)| {
            (
                target_index,
                distance_fn(source_descriptor, target_descriptor),
            )
        })
        .fold(None, |best_pair, candidate| match best_pair {
            None => Some((candidate, None)),
            Some((best, _)) if candidate.1 < best.1 => Some((candidate, Some(best))),
            Some((best, None)) => Some((best, Some(candidate))),
            Some((best, Some(second_best))) if candidate.1 < second_best.1 => {
                Some((best, Some(candidate)))
            }
            Some((best, second_best)) => Some((best, second_best)),
        })
        .and_then(|(best, second_best)| second_best.map(|second_best| (best, second_best)))
}

fn deduplicate_target_matches_by_index(matches: Vec<IndexMatch>) -> Vec<IndexMatch> {
    matches
        .into_iter()
        .fold(Vec::new(), |mut deduped, candidate| {
            if deduped
                .iter()
                .all(|existing: &IndexMatch| existing.target_index != candidate.target_index)
            {
                deduped.push(candidate);
            }
            deduped
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_apriltag::family::TagFamilyKind;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct ChessboardDescriptor {
        point: ImagePoint,
        response: f32,
        orientation: f32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct AprilTagDescriptor {
        id: u16,
        family: TagFamilyKind,
        center: ImagePoint,
    }

    fn chessboard_distance(left: &ChessboardDescriptor, right: &ChessboardDescriptor) -> f32 {
        let dx = left.point.x - right.point.x;
        let dy = left.point.y - right.point.y;
        let position = (dx * dx + dy * dy).sqrt();
        let orientation = (left.orientation - right.orientation).abs();
        let response = (left.response - right.response).abs();
        position + orientation * 10.0 + response
    }

    fn apriltag_distance(left: &AprilTagDescriptor, right: &AprilTagDescriptor) -> f32 {
        let id_penalty = if left.id == right.id { 0.0 } else { 1000.0 };
        let family_penalty = if left.family == right.family {
            0.0
        } else {
            1000.0
        };
        let dx = left.center.x - right.center.x;
        let dy = left.center.y - right.center.y;
        id_penalty + family_penalty + (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn keeps_best_ratio_matches() {
        let source = [
            ImagePoint::new(0.0, 0.0),
            ImagePoint::new(10.0, 0.0),
            ImagePoint::new(20.0, 0.0),
            ImagePoint::new(30.0, 0.0),
        ];
        let target = [
            ImagePoint::new(0.1, 0.0),
            ImagePoint::new(10.1, 0.0),
            ImagePoint::new(20.1, 0.0),
            ImagePoint::new(30.1, 0.0),
            ImagePoint::new(100.0, 100.0),
        ];

        let matches =
            lowe_ratio_match_points(&source, &target, LoweRatioConfig::default()).unwrap();

        assert_eq!(matches.len(), 4);
        assert_eq!(matches[0].target_index, 0);
    }

    #[test]
    fn rejects_ambiguous_point_matches() {
        let source = [ImagePoint::new(0.0, 0.0)];
        let target = [ImagePoint::new(1.0, 0.0), ImagePoint::new(1.1, 0.0)];

        let matches = lowe_ratio_match_points(&source, &target, LoweRatioConfig::new(0.8)).unwrap();

        assert!(matches.is_empty());
    }

    #[test]
    fn keeps_only_best_match_per_target() {
        let source = [ImagePoint::new(0.0, 0.0), ImagePoint::new(0.05, 0.0)];
        let target = [ImagePoint::new(0.0, 0.0), ImagePoint::new(10.0, 10.0)];

        let matches =
            lowe_ratio_match_points(&source, &target, LoweRatioConfig::default()).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].source_index, 0);
        assert_eq!(matches[0].target_index, 0);
    }

    #[test]
    fn matches_chessboard_descriptors_with_orientation_and_response() {
        let source = [
            ChessboardDescriptor {
                point: ImagePoint::new(10.0, 10.0),
                response: 0.95,
                orientation: 0.05,
            },
            ChessboardDescriptor {
                point: ImagePoint::new(20.0, 20.0),
                response: 0.80,
                orientation: 1.57,
            },
        ];
        let target = [
            ChessboardDescriptor {
                point: ImagePoint::new(10.2, 10.0),
                response: 0.94,
                orientation: 0.04,
            },
            ChessboardDescriptor {
                point: ImagePoint::new(20.1, 20.0),
                response: 0.79,
                orientation: 1.56,
            },
            ChessboardDescriptor {
                point: ImagePoint::new(10.3, 10.0),
                response: 0.20,
                orientation: 2.70,
            },
            ChessboardDescriptor {
                point: ImagePoint::new(20.0, 20.2),
                response: 0.10,
                orientation: 0.20,
            },
        ];

        let matches = lowe_ratio_match_by(
            &source,
            &target,
            LoweRatioConfig::default(),
            chessboard_distance,
        )
        .unwrap();

        assert_eq!(matches.len(), 2);
        assert!(
            matches
                .iter()
                .any(|entry| entry.source_index == 0 && entry.target_index == 0)
        );
        assert!(
            matches
                .iter()
                .any(|entry| entry.source_index == 1 && entry.target_index == 1)
        );
    }

    #[test]
    fn rejects_apriltag_candidates_with_wrong_family_or_id() {
        let source = [AprilTagDescriptor {
            id: 7,
            family: TagFamilyKind::Tag36H11,
            center: ImagePoint::new(40.0, 50.0),
        }];
        let target = [
            AprilTagDescriptor {
                id: 7,
                family: TagFamilyKind::Tag36H11,
                center: ImagePoint::new(40.2, 50.1),
            },
            AprilTagDescriptor {
                id: 8,
                family: TagFamilyKind::Tag36H11,
                center: ImagePoint::new(40.1, 50.0),
            },
            AprilTagDescriptor {
                id: 7,
                family: TagFamilyKind::Tag25H9,
                center: ImagePoint::new(40.1, 50.0),
            },
        ];

        let matches = lowe_ratio_match_by(
            &source,
            &target,
            LoweRatioConfig::new(0.1),
            apriltag_distance,
        )
        .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].target_index, 0);
    }
}
