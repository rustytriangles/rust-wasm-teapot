mod utils;

#[path = "teapot.rs"]
mod teapot;

use std::fmt;
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
}

impl Teapot {

}

/// Public methods, exported to JavaScript.
#[wasm_bindgen]
impl Teapot {

    pub fn new() -> Teapot {
        let num_rows = 18;
        let num_cols = 23;

        Teapot {
            num_rows,
            num_cols
        }
    }

    pub fn render(&self) -> String {
        self.to_string()
    }

}
