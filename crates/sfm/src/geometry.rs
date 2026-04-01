use structura_core::error::StructuraError;

pub trait GeometryEstimator {
    fn estimate(&self) -> Result<(), StructuraError>;
}
