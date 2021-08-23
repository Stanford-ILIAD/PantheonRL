import $ from "jquery"
import _ from "lodash"

import getOvercookedPolicy from "./js/load_tf_model.js";
import OvercookedTrajectoryReplay from "./js/overcooked-replay.js";

import * as Overcooked from "overcooked"
let OvercookedMDP = Overcooked.OvercookedMDP;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let [NORTH, SOUTH, EAST, WEST] = Direction.CARDINAL;
let [STAY, INTERACT] = [Direction.STAY, Action.INTERACT];

// Parameters
let PARAMS = {
    MAIN_TRIAL_TIME: 60, //seconds
    TIMESTEP_LENGTH: 150, //milliseconds
    DELIVERY_POINTS: 20,
    PLAYER_INDEX: 1,  // Either 0 or 1
    MODEL_TYPE: 'ppo_bc'  // Either ppo_bc, ppo_sp, or pbt
};
let trajectoryPath = "static/assets/test_traj.json";
let trajectoryData; 
/***********************************
      Main trial order
************************************/


let layouts = {
    "cramped_room":[
        "XXPXX",
        "O  2O",
        "X1  X",
        "XDXSX"
    ],
    "asymmetric_advantages":[
        "XXXXXXXXX",
        "O XSXOX S",
        "X   P 1 X",
        "X2  P   X",
        "XXXDXDXXX"
    ],
    "coordination_ring":[
        "XXXPX",
        "X 1 P",
        "D2X X",
        "O   X",
        "XOSXX"
    ],
    "forced_coordination":[
        "XXXPX",
        "O X1P",
        "O2X X",
        "D X X",
        "XXXSX"
    ],
    "counter_circuit": [
        "XXXPPXXX",
        "X      X",
        "D XXXX S",
        "X2    1X",
        "XXXOOXXX"
    ], 
    "mdp_test": [
	"XXPXX", 
	"O  2O", 
	"T1  T", 
	"XDPSX"
    ]
};

let game;
function replayGame(endOfGameCallback){
    // make sure the game metadata matches what's 
    // in the trajectory file
    $("#overcooked").empty();
    if (trajectoryData != undefined) {

        replayTrajectory(trajectoryData, endOfGameCallback);
    }
    else { 
        console.log("Upload data not found")
        $.getJSON(trajectoryPath, function(trajectoryData) {
            replayTrajectory(trajectoryData, endOfGameCallback);
        }); 
    } 
}

function replayTrajectory(trajectoryDataObj, endOfGameCallback) {
    let mdp_params = trajectoryDataObj.mdp_params[0];
    game = new OvercookedTrajectoryReplay({
        container_id: "overcooked", 
        trajectory: trajectoryDataObj, 
        start_grid: layouts[mdp_params.layout_name],
        MAX_TIME : PARAMS.MAIN_TRIAL_TIME, //seconds
        cook_time: mdp_params.cook_time,
        init_orders: mdp_params.start_order_list,
        completion_callback: () => {
            console.log("Time up");
            endOfGameCallback();
        },
        DELIVERY_REWARD: PARAMS.DELIVERY_POINTS

    })
    game.init()

}


function endGame() {
    game.close();
    $("#overcooked").empty();
}

// Handler to be added to $(document) on keydown when a game is not in
// progress, that will then start the game when Enter is pressed.
function startGameOnEnter(e) {
    // Do nothing for keys other than Enter
    if (e.which !== 13) {
	return;
    }

    disableEnter();
    // Reenable enter handler when the game ends
    replayGame(enableEnter)
}

function setTrajectoryPathOnLoad(event){
    trajectoryData = JSON.parse(event.target.result);
    console.log("Uploaded data: ")
    console.log(trajectoryData)
    

}

function onChange(event) {
    var reader = new FileReader();
    reader.onload = setTrajectoryPathOnLoad;
    reader.readAsText(event.target.files[0]);
    let fileName = event.target.files[0].name;
    $("#fileInfo").html("<p>Reading trajectory info from uploaded file " + fileName +"</p>")
    if (game != undefined) {
        endGame();
    }
}

function clearFile() {
    $("#fileInput").val('');
    $("#fileInfo").html("<p>Reading trajectory info from " + trajectoryPath + "</p>")
    if (game != undefined) {
        endGame();
    }
}


function alert_data(name, family){
    alert('Name : ' + name + ', Family : ' + family);
}

function enableEnter() {
    $(document).keydown(startGameOnEnter);
    $("#fileInput").on("change", onChange); 
    if ($("#fileInfo").is(':empty')) {
        $("#fileInfo").html("<p>Reading trajectory info from " + trajectoryPath + "</p>");
    }
    $("#control").html("<p>Press enter to begin!</p>");
}

function disableEnter() {
    $(document).off("keydown");
    $("#control").html('<button id="reset" type="button" class="btn btn-primary">Reset</button> <button id="clearFile" type="button" class="btn btn-primary">Clear File</button>');
    $("#reset").click(endGame);
    $("#clearFile").click(clearFile);
}

$(document).ready(() => {
    enableEnter();
});
