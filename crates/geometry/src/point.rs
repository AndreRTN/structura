use nalgebra::{Point2, Point3};

pub type ImagePoint = Point2<f32>;
pub type ImagePoint64 = Point2<f64>;
pub type WorldPoint = Point3<f64>;

#[derive(Debug, Clone, PartialEq)]
pub struct PointCorrespondence2D3D {
    pub image: ImagePoint64,
    pub world: WorldPoint,
}

impl PointCorrespondence2D3D {
    pub const fn new(image: ImagePoint64, world: WorldPoint) -> Self {
        Self { image, world }
    }
}
