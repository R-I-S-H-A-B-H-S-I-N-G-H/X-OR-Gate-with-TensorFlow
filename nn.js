// const tf=require("@tensorflow/tfjs");
// import * as tf from "@tensorflow/tfjs";

class NeuralNetwork {
	constructor(input, hidden, output) {
		this.g = 0;
		this.training_complete = true;
		this.epoch_l = 1;
		this.err_limit = 0.1;
		this.err = 100;
		this.err_diff = 1;
		this.lr = 0.1;
		this.loss = "meanSquaredError";
		const optimizer = tf.train.adam(this.lr);

		const config_h = {
			units: hidden,
			inputShape: [input],
			activation: "sigmoid",
		};
		const config_o = {
			units: output,
			activation: "sigmoid",
		};

		const config_compile = {
			optimizer: optimizer,
			loss: this.loss,
		};

		this.model = tf.sequential();
		this.hidden = tf.layers.dense(config_h);
		this.output = tf.layers.dense(config_o);

		this.model.add(this.hidden);
		this.model.add(this.output);

		this.model.compile(config_compile);
	}

	// guess(inputs) {
	// 	// try {
	// 	// 	this.g =n.model.predict(inputs).dataSync();
	// 	// } catch (e) {}
	// 	// return this.g;
	// 	return n.model.predict(inputs).dataSync();
	// }

	async train(inputs, targets) {
		this.training_complete = false;
		var h = await this.model.fit(inputs, targets, {
			shuffel: true,
			epochs: 1,
			callbacks: {
				onTrainEnd: (epochs, logs) => {
					// console.log("training completed");
				},
			},
		});
		this.err_diff = this.err - h.history.loss[0];
		this.err = h.history.loss[0];
		// console.log(h.history.loss[0])
		this.training_complete = true;
	}
}
