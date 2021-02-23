import { Teapot } from "rust-wasm-teapot";
import { memory } from "rust-wasm-teapot/rust_wasm_teapot_bg";

var gmod = require('./graphics');

const ctx = canvas.getContext('2d');

const renderLoop = () => {
    const rect = canvas.getBoundingClientRect();
    const gl = canvas.getContext('webgl2');
	if (!gl) {
	    console.log('Could not create context');
	    return;
	}

    // drawing code goes here
	gmod.draw(gl, frameCounter);

    requestAnimationFrame(renderLoop);
};

requestAnimationFrame(renderLoop);
