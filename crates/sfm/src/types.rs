#[derive(Debug, Clone)]
pub struct SfmInput {
    pub images: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SfmResult {
    pub registered_views: usize,
    pub points_3d: usize,
}
