use anyhow::Result;
use kornia::image::{
    Image,
    allocator::{CpuAllocator, ImageAllocator},
};
pub use structura_geometry::point::ImagePoint;

pub type GrayImage<A = CpuAllocator> = Image<u8, 1, A>;

pub trait Detector {
    type Detection;

    fn detect<A>(&mut self, image: &GrayImage<A>) -> Result<Vec<Self::Detection>>
    where
        A: ImageAllocator;
}
