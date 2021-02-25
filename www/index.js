import { Teapot } from "rust-wasm-teapot";
import { memory } from "rust-wasm-teapot/rust_wasm_teapot_bg";

var gmod = require('./graphics');

const canvas = document.getElementById("canvas");
canvas.width = 640;
canvas.height = 480;

const teapot = Teapot.new();
let geomInfo = undefined;

var frameCounter = 0;

const renderLoop = () => {
    const rect = canvas.getBoundingClientRect();
    const gl = canvas.getContext('webgl2');
	if (!gl) {
	    console.log('Could not create context');
	    return;
	}

    if (!geomInfo) {
        const numVertices = teapot.numVertices();
        const vertex_data = teapot.vertices();
        const normal_data = teapot.normals();
        const vertices = new Float32Array(memory.buffer,
                                          vertex_data,
                                          3 * numVertices);
        const normals = new Float32Array(memory.buffer,
                                         normal_data,
                                         3 * numVertices);
        const numIndices = teapot.numIndices();
        const index_data = teapot.indices();
        const indices = new Int16Array(memory.buffer,
                                       index_data,
                                       numIndices);

        geomInfo = {
            vertices: vertices,
            numComponents: 3,
            numVertices: vertices.length/3,
            normals: normals,
            indices: indices,
            numTriangles: indices.length / 3
        };
    }

	gmod.draw(gl, geomInfo, frameCounter);
    frameCounter += 1;

    requestAnimationFrame(renderLoop);
};

requestAnimationFrame(renderLoop);
