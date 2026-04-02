#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagePoint {
    pub x: f32,
    pub y: f32,
}

impl ImagePoint {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}
