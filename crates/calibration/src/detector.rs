use anyhow::Result;
use kornia::image::{
    Image,
    allocator::{CpuAllocator, ImageAllocator},
};

pub type GrayImage<A = CpuAllocator> = Image<u8, 1, A>;

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

pub trait Detector {
    type Detection;

    fn detect<A>(&mut self, image: &GrayImage<A>) -> Result<Vec<Self::Detection>>
    where
        A: ImageAllocator;
}
