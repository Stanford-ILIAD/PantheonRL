import * as tf from '@tensorflow/tfjs-core';

import * as Overcooked from "overcooked";
let OvercookedGame = Overcooked.OvercookedGame.OvercookedGame;
let OvercookedMDP = Overcooked.OvercookedMDP;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let [NORTH, SOUTH, EAST, WEST] = Direction.CARDINAL;
let [STAY, INTERACT] = [Direction.STAY, Action.INTERACT];
import { loadGraphModel } from '@tensorflow/tfjs-converter';


function sampleIndexFromCategorical(probas) {
	// Stolen from: https://stackoverflow.com/questions/8877249/generate-random-integers-with-probabilities
	let randomNum = Math.random();
	let accumulator = 0;
	let lastProbaIndex = probas.length - 1;

	for (var i = 0; i < lastProbaIndex; i++) {
		accumulator += probas[i];
		if (randomNum < accumulator) {
			return i;
		}
	}
	return lastProbaIndex;
}

export default function getOvercookedPolicy(model_type, layout_name, playerIndex, argmax) {
	// Returns a Promise that resolves to a policy
	if (model_type == "human") {
		return new Promise(function (resolve, reject) {
			resolve(null);
		});
	}

	const modelPromise = loadGraphModel('static/assets/' + model_type + '_' + layout_name + '_agent/model.json');

	return modelPromise.then(function (model) {
		return new Promise(function (resolve, reject) {
			resolve(function (state, done, lstm_state, game) {
				let sim_threads = model.inputs[0].shape[0];
				let [result, shape] = game.mdp.lossless_state_encoding(state, playerIndex, sim_threads);
				//console.log(result);
				//console.log(shape);
				let state_tensor = tf.tensor(result, shape);

				let action_tensor;
				let action_probs;

				// NOTE: Useful for debugging model issues
				// console.log(model) 
				if (model.inputs.length == 1) { 
					// Non-recurrent models
					action_tensor = model.execute({ "ppo_agent/ppo2_model/Ob": state_tensor });
					action_probs = action_tensor.arraySync()[0];
				} else if (model.inputs.length == 3) { 
					// Recurrent models
					let shape = [sim_threads];
					let dones = constant(0, shape);
					dones[0] = done;
					let dones_tensor = tf.tensor(dones, shape);

					let lstm_state_shape = model.inputs[2].shape;
					if (lstm_state === null) {
						lstm_state = constant(0, lstm_state_shape);
					}
					let in_lstm_state_tensor = tf.tensor(lstm_state, lstm_state_shape);
					let [out_lstm_state_tensor, action_tensor] = model.execute({ "ppo_agent/ppo2_model/Ob": state_tensor, "ppo_agent/ppo2_model/pi/lstm_state": in_lstm_state_tensor, "ppo_agent/ppo2_model/pi/lstm_mask": dones_tensor });
					action_probs = action_tensor.arraySync()[0];
					lstm_state = out_lstm_state_tensor.arraySync();
				} else {
					console.log("input was wrong size? probably the model is being saved wrong");
				}

				let action_index;
				if (argmax == true) {
					action_index = argmax(action_probs);
				}
				else {
					// will happen if argmax == false or if argmax == undefined
					action_index = sampleIndexFromCategorical(action_probs)
				}

				return [Action.INDEX_TO_ACTION[action_index], lstm_state];
			});
		});
	});
}

function constant(element, shape) {
	function helper(i) {
		let size = shape[i];
		if (i === shape.length - 1) {
			return Array(size).fill(element);
		}
		return Array(size).fill().map(() => helper(i + 1));
	}
	return helper(0);
}

function argmax(array) {
	let bestIndex = 0;
	let bestValue = array[bestIndex];
	for (let i = 1; i < array.length; i++) {
		if (array[i] > bestValue) {
			bestIndex = i;
			bestValue = array[i];
		}
	}
	return bestIndex;
}
