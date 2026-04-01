use structura_core::error::StructuraError;

pub trait BundleAdjuster {
    fn optimize(&self) -> Result<(), StructuraError>;
}
