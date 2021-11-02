import * as Overcooked from "overcooked"
let OvercookedGame = Overcooked.OvercookedGame.OvercookedGame;
let OvercookedMDP = Overcooked.OvercookedMDP;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let [NORTH, SOUTH, EAST, WEST] = Direction.CARDINAL;
let [STAY, INTERACT] = [Direction.STAY, Action.INTERACT];

let COOK_TIME = 20;

export default class OvercookedSinglePlayerTask {
    constructor({
        container_id,
        player_index,
        npc_policies,
        mdp_params,
        task_params,
        algo,
        start_grid,
        layout_name,
        save_trajectory = false,
        TIMESTEP = 200,
        MAX_TIME = 20, //seconds
        init_orders = null,
        completion_callback = () => { console.log("Time up") },
        timestep_callback = (data) => { },
        DELIVERY_REWARD = 20
    }) {
        //NPC policies get called at every time step
        if (typeof (npc_policies) === 'undefined') {
            // TODO maybe delete this? 
            npc_policies = {
                1:
                    (function () {
                        let action_loop = [
                            SOUTH, WEST, NORTH, EAST
                        ];
                        let ai = 0;
                        let pause = 4;
                        return (s) => {
                            let a = STAY;
                            if (ai % pause === 0) {
                                a = action_loop[ai / pause];
                            }
                            ai += 1;
                            ai = ai % (pause * action_loop.length);
                            return a
                        }
                    })()
            }
        }
        this.npc_policies = npc_policies;
        this.player_index = player_index;
        this.algo = algo;

        let player_colors = { 0: 'blue', 1: 'green' };

        this.game = new OvercookedGame({
            start_grid,
            container_id,
            assets_loc: "static/assets/",
            ANIMATION_DURATION: TIMESTEP * .9,
            tileSize: 80,
            COOK_TIME: COOK_TIME,
            explosion_time: Number.MAX_SAFE_INTEGER,
            DELIVERY_REWARD: DELIVERY_REWARD,
            player_colors: player_colors
        });
        this.init_orders = init_orders;
        if (Object.keys(npc_policies).length == 1) {
            console.log("Single human player vs agent");
            this.game_type = 'human_vs_agent';
        }
        else {
            console.log("Agent playing vs agent")
            this.game_type = 'agent_vs_agent';
        }

        this.layout_name = layout_name;
        this.TIMESTEP = TIMESTEP;
        this.MAX_TIME = MAX_TIME;
        this.time_left = MAX_TIME;
        this.cur_gameloop = 0;
        this.score = 0;
        this.completion_callback = completion_callback;
        this.timestep_callback = timestep_callback;
        this.mdp_params = mdp_params;
        this.mdp_params['cook_time'] = COOK_TIME;
        this.mdp_params['start_order_list'] = init_orders;
        this.task_params = task_params;
        this.save_trajectory = save_trajectory
        this.trajectory = {
            'ep_states': [[]],
            'ep_actions': [[]],
            'ep_rewards': [[]],
            'mdp_params': [mdp_params]
        }
    }

    init() {
        this.game.init();

        this.start_time = new Date().getTime();
        this.state = this.game.mdp.get_start_state(this.init_orders);
        this.game.drawState(this.state);
        this.joint_action = [STAY, STAY];
        // this.lstm_state = [null, null];
        this.done = 1;

        this.gameloop = setInterval(() => {
            for (const npc_index of this.npc_policies) {
                // let [npc_a, lstm_state] = this.npc_policies[npc_index](this.state, this.done, this.lstm_state[npc_index], this.game);
                
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "/predict", false); // false for synchronous
                xhr.setRequestHeader('Content-Type', 'application/json');
                xhr.send(JSON.stringify({
                    state: this.state,
                    npc_index: npc_index,
                    layout_name: this.layout_name,
                    algo: this.algo,
                    timestep: this.cur_gameloop,
                }));
                var action_idx = JSON.parse(xhr.responseText)["action"];
                let npc_a = Action.INDEX_TO_ACTION[action_idx];
                console.log(npc_a);

                // this.lstm_state[npc_index] = lstm_state;
                this.joint_action[npc_index] = npc_a;
            }

            this.joint_action_idx = [Action.ACTION_TO_INDEX[this.joint_action[0]], Action.ACTION_TO_INDEX[this.joint_action[1]]];
            let [[next_state, prob], reward] =
                this.game.mdp.get_transition_states_and_probs({
                    state: this.state,
                    joint_action: this.joint_action
                });

            // Apparently doing a Parse(Stringify(Obj)) is actually the most succinct way. 
            // to do a deep copy in JS 
            // let cleanedState = JSON.parse(JSON.stringify(this.state));
            // cleanedState['objects'] = Object.values(cleanedState['objects']);
            this.trajectory.ep_states[0].push(JSON.stringify(this.state))
            this.trajectory.ep_actions[0].push(JSON.stringify(this.joint_action_idx))
            this.trajectory.ep_rewards[0].push(reward)
            //update next round
            this.game.drawState(next_state);
            this.score += reward;
            this.game.drawScore(this.score);
            let time_elapsed = (new Date().getTime() - this.start_time) / 1000;
            this.time_left = Math.round(this.MAX_TIME - time_elapsed);
            this.game.drawTimeLeft(this.time_left);
            this.done = 0

            //record data
            this.timestep_callback({
                state: this.state,
                joint_action: this.joint_action,
                next_state: next_state,
                reward: reward,
                time_left: this.time_left,
                score: this.score,
                time_elapsed: time_elapsed,
                cur_gameloop: this.cur_gameloop,
                client_id: undefined,
                is_leader: undefined,
                partner_id: undefined,
                datetime: +new Date()
            });

            //set up next timestep
            this.state = next_state;
            this.joint_action = [STAY, STAY];
            this.cur_gameloop += 1;
            this.activate_response_listener();

            //time run out
            if (this.time_left < 0) {
                this.time_left = 0;
                this.close();
            }
        }, this.TIMESTEP);
        this.activate_response_listener();
    }

    close() {
        if (typeof (this.gameloop) !== 'undefined') {
            clearInterval(this.gameloop);
        }

        var today = new Date();
        var traj_time = (today.getMonth() + 1) + '_' + today.getDate() + '_' + today.getFullYear() + '_' + today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
        let trajectory = this.trajectory;
        let task_params = this.task_params;

        // Looks like all the internal objects are getting stored as strings rather than actual arrays or objects
        // So it looks like Bodyparser only parses the top levl keys, and keeps everything on the lower level as strings rather 
        // than processing it recursively 

        let parsed_trajectory_data = {
            "ep_states": [[]],
            "ep_rewards": [[]],
            "ep_actions": [[]],
            "mdp_params": []
        }

        parsed_trajectory_data['mdp_params'][0] = trajectory.mdp_params[0];
        ["ep_states", "ep_rewards", "ep_actions"].forEach(function (key, key_index) {
            trajectory[key][0].forEach(function (item, index) {
                parsed_trajectory_data[key][0].push(JSON.parse(item))
            })
        })

        document.getElementById('control').innerHTML = "Updating model...please wait...";
        setTimeout(() => {  
            // make dom finishes rendering

        var xhr = new XMLHttpRequest();
        xhr.open("POST", "/updatemodel", false); // false for synchronous
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(JSON.stringify({
            traj_id: traj_time + "_human=" + this.player_index,
            traj: parsed_trajectory_data,
            layout_name: this.layout_name,
            algo: this.algo,
        }));
        var status = JSON.parse(xhr.responseText);
        console.log("status: " + status);

        // let fileName = traj_time + "_" + task_params.MODEL_TYPE + "_" + task_params.PLAYER_INDEX + ".json";

        // // A way to download a json file purely through JS
        // var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(parsed_trajectory_data));
        // var dlAnchorElem = document.getElementById('downloadAnchorElem');
        // dlAnchorElem.setAttribute("href", dataStr);
        // dlAnchorElem.setAttribute("download", fileName);
        // dlAnchorElem.click();

        // Old code for saving to overcooked-demo folder
        //
        // $.ajax({url: "/save_trajectory",
        //         type: "POST", 
        //         contentType: 'application/json',
        //         data: JSON.stringify(traj_file_data),
        //         success: function(response) {
        // console.log(`Save trajectory status is ${response}`)
        // }})


        this.game.close();
        this.disable_response_listener();
        this.completion_callback();

        }, 15);
    }

    activate_response_listener() {
        $(document).on("keydown", (e) => {
            let action;
            switch (e.which) {
                case 37: // left
                    action = WEST;
                    break;

                case 38: // up
                    action = NORTH;
                    break;

                case 39: // right
                    action = EAST;
                    break;

                case 40: // down
                    action = SOUTH;
                    break;

                case 32: //space
                    action = INTERACT;
                    break;

                default: return; // exit this handler for other keys
            }
            e.preventDefault(); // prevent the default action (scroll / move caret)

            this.joint_action[this.player_index] = action;
            this.disable_response_listener();
        });
    }

    disable_response_listener() {
        $(document).off('keydown');
    }
}
