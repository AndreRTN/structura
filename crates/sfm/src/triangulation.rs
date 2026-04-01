use structura_core::error::StructuraError;

pub trait Triangulator {
    fn triangulate(&self) -> Result<(), StructuraError>;
}
