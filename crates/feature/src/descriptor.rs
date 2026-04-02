use anyhow::Result;
use structura_geometry::point::ImagePoint;

use crate::matching::{LoweRatioConfig, PointMatch, lowe_ratio_match_by};

#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorFeature {
    pub point: ImagePoint,
    pub score: f32,
    pub descriptor: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DescriptorFeatureSet {
    pub features: Vec<DescriptorFeature>,
}

impl DescriptorFeatureSet {
    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn keypoints(&self) -> Vec<ImagePoint> {
        self.features.iter().map(|feature| feature.point).collect()
    }
}

pub fn lowe_ratio_match_descriptor_features(
    source: &DescriptorFeatureSet,
    target: &DescriptorFeatureSet,
    config: LoweRatioConfig,
) -> Result<Vec<PointMatch>> {
    lowe_ratio_match_by(&source.features, &target.features, config, |lhs, rhs| {
        descriptor_distance(&lhs.descriptor, &rhs.descriptor)
    })
    .map(|matches| {
        matches
            .into_iter()
            .map(|matched| PointMatch {
                source_index: matched.source_index,
                target_index: matched.target_index,
                source_point: source.features[matched.source_index].point,
                target_point: target.features[matched.target_index].point,
                distance: matched.distance,
                ratio: matched.ratio,
            })
            .collect()
    })
}

pub(crate) fn descriptor_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum::<f32>()
        .sqrt()
}
