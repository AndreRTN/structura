#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose3 {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

impl Pose3 {
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }
}
