mod utils;

#[path = "teapot.rs"]
mod teapot;

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct Triangle {
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
}

#[wasm_bindgen]
pub struct Teapot {
    num_rows: u32,
    num_cols: u32,
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u16>
}

impl Teapot {

}

/// Public methods, exported to JavaScript.
#[wasm_bindgen]
impl Teapot {

    pub fn numVertices(&self) -> u32 {
        32 * self.num_rows * self.num_cols
    }

    pub fn numIndices(&self) -> u32 {
        0
    }

    pub fn vertices(&mut self) -> *const [f32; 3] {
        if (self.vertices.len() == 0) {
            self.rebuild();
        }
        self.vertices.as_ptr()
    }

    pub fn normals(&mut self) -> *const [f32; 3] {
        if (self.normals.len() == 0) {
            self.rebuild();
        }
        self.normals.as_ptr()
    }

    pub fn indices(&mut self) -> *const u16 {
        if (self.indices.len() == 0) {
            self.rebuild();
        }
        self.indices.as_ptr()
    }

    pub fn new() -> Teapot {
        let num_rows = 18;
        let num_cols = 23;

        let mut vertices = Vec::new();

        let mut normals = Vec::new();

        let mut indices = Vec::with_capacity(4);
        indices.push(1);
        indices.push(2);
        indices.push(3);

        Teapot {
            num_rows,
            num_cols,
            vertices,
            normals,
            indices
        }
    }

    fn rebuild(&mut self) {

        let num_patches = 32;
        let num_rows = self.num_rows as usize;
        let num_cols = self.num_cols as usize;

        let num_vertices = num_patches * num_rows * num_cols;
        let num_indices = num_patches * 2 * 3 * (num_rows - 1) * (num_cols - 1);

        let (vertex_data, normal_data, uv_data, index_data) = teapot::create_vertices(num_rows, num_cols);

        self.vertices = Vec::with_capacity(num_vertices);
        self.normals = Vec::with_capacity(num_vertices);
        for i in 0..num_vertices {
            self.vertices.push([vertex_data[i][0], vertex_data[i][1], vertex_data[i][2]]);
            self.normals.push(normal_data[i]);
        }

        self.indices = Vec::with_capacity(num_indices);
        for i in 0..num_indices {
            self.indices.push(index_data[i] as u16);
        }
    }

    pub fn render(&self) -> String {
        let num_rows = self.num_rows as usize;
        let num_cols = self.num_cols as usize;

        //        self.to_string()
        "Fudge".to_string()
    }
}
