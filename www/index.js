//import { Teapot } from "rust-wasm-teapot";
//import { memory } from "rust-wasm-teapot/rust_wasm_teapot_bg";

var gmod = require('./graphics');

//const ctx = canvas.getContext('2d');

var frameCounter = 0;

const renderLoop = () => {
    const rect = canvas.getBoundingClientRect();
    const gl = canvas.getContext('webgl2');
	if (!gl) {
	    console.log('Could not create context');
	    return;
	}

    // get geomBuffers from teapot and pass them into gmod.draw

	gmod.draw(gl, frameCounter);

    requestAnimationFrame(renderLoop);
};

requestAnimationFrame(renderLoop);
