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
    v0: Vec<[f32; 3]>,
    v1: Vec<[f32; 3]>,
    v2: Vec<[f32; 3]>,
}

#[wasm_bindgen]
pub struct Teapot {
    num_rows: u32,
    num_cols: u32,
    triangles: Vec<Triangle>,
}

impl Teapot {

}

/// Public methods, exported to JavaScript.
#[wasm_bindgen]
impl Teapot {

    pub fn vertices(&self) -> *const Triangle {
        self.triangles.as_ptr()
    }

    pub fn new() -> Teapot {
        let num_rows = 18;
        let num_cols = 23;
        let triangles = Vec::with_capacity(12);
//        triangles.push(Triangle([0,0,0],[1,0,0],[0,1,0]));

        Teapot {
            num_rows,
            num_cols,
            triangles
        }
    }

    pub fn render(&self) -> String {
        let num_rows = self.num_rows as usize;
        let num_cols = self.num_cols as usize;

        let (vertex_data, normal_data, uv_data, index_data) = teapot::create_vertices(num_rows, num_cols);

        //        self.to_string()
        "Fudge".to_string()
    }
}
