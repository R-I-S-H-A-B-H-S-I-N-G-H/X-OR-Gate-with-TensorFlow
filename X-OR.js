var w, h;
var x_c = [];
var y_c = [];
var inp_guess = [];
var x = 0,
	y = 0;

var n, inputs, out;
var limiter = 10000;
var i = 0;
var tst_inp;
var y;
var inpdata = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];
var outdata = [[0], [1], [1], [0]];
var resolution = 20

function setup() {
	// frameRate(75);
	createCanvas(500, 500);
	w = width / resolution;
	h = height / resolution;
	n = new NeuralNetwork(2, 10, 1);
	inputs = tf.tensor2d(inpdata);
	out = tf.tensor2d(outdata);
	points();
}

function draw() {
	background(0);
	nural();
	var g_c = guess(inp_guess);
	// console.log(g_c);
	var count = 0;
	for (var y = 0; y < h; y++) {
		for (var x = 0; x < w; x++) {
			fill(g_c[count++] * 255);
			rect(x * resolution, y * resolution, resolution, resolution);
		}
	}

	
	// noLoop();
}

function nural() {
	tf.tidy(() => {
		if (n.training_complete) {
			n.train(inputs, out);
		}

		// console.log(a);
	});
}

function guess(inputs) {
	return tf.tidy(() => {
		try {
			inputs = tf.tensor2d(inputs);
			return n.model.predict(inputs).dataSync();
		} catch (e) {
			return null;
		}
	});
}

function points() {
	var c = 0;
	for (var y = 0; y < h; y++) {
		for (var x = 0; x < w; x++) {
			inp_guess[c] = [map(x, 0, w, 0, 1), map(y, 0, h, 0, 1)];
			x_c[c] = map(x, 0, w, 0, 1);
			y_c[c++] = map(y, 0, h, 0, 1);
		}
	}
}
