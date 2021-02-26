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
pub struct Teapot {
    num_rows: u32,
    num_cols: u32,
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u16>
}

/// Public methods, exported to JavaScript.
#[wasm_bindgen]
impl Teapot {

    pub fn num_vertices(&self) -> u32 {
        32 * self.num_rows * self.num_cols
    }

    pub fn num_indices(&self) -> u32 {
        32 * 2 * 3 * (self.num_rows - 1) * (self.num_cols - 1)
    }

    pub fn vertices(&mut self) -> *const [f32; 3] {
        if self.vertices.len() == 0 {
            self.rebuild();
        }
        self.vertices.as_ptr()
    }

    pub fn normals(&mut self) -> *const [f32; 3] {
        if self.normals.len() == 0 {
            self.rebuild();
        }
        self.normals.as_ptr()
    }

    pub fn indices(&mut self) -> *const u16 {
        if self.indices.len() == 0 {
            self.rebuild();
        }
        self.indices.as_ptr()
    }

    pub fn new() -> Teapot {
        let num_rows = 10;
        let num_cols = 13;

        let vertices = Vec::new();

        let normals = Vec::new();

        let indices = Vec::new();

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

        let (vertex_data, normal_data, _uv_data, index_data) = teapot::create_vertices(num_rows, num_cols);

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
}
