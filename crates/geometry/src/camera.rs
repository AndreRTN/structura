use nalgebra::{Matrix3, Vector3};

#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsics {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub u0: f64,
    pub v0: f64,
}

impl CameraIntrinsics {
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.alpha, self.gamma, self.u0, 0.0, self.beta, self.v0, 0.0, 0.0, 1.0,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CameraExtrinsics {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}
