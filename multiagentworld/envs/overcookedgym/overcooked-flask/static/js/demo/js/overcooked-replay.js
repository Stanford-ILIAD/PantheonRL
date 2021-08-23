import * as Overcooked from "overcooked"
let OvercookedGame = Overcooked.OvercookedGame.OvercookedGame;
let OvercookedMDP = Overcooked.OvercookedMDP;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let PlayerState = OvercookedMDP.PlayerState; 
let OvercookedState = OvercookedMDP.OvercookedState; 
let ObjectState = OvercookedMDP.ObjectState;

let [NORTH, SOUTH, EAST, WEST] = Direction.CARDINAL;
let [STAY, INTERACT] = [Direction.STAY, Action.INTERACT];

let lookupActions = OvercookedMDP.lookupActions; 
let dictToState = OvercookedMDP.dictToState;

export default class OvercookedTrajectoryReplay{
    constructor ({
        container_id,
        trajectory,
        start_grid = [
            'XXXXXPXX',
            'O     2O',
            'T1     T',
            'XXXDPSXX'
        ],
        MAX_TIME = 1, //seconds
        cook_time=5,
        init_orders=null,
        completion_callback = () => {console.log("Time up")},
        timestep_callback = (data) => {},
        DELIVERY_REWARD = 20
    }) 
    {

    	let player_colors = {};
    	player_colors[0] = 'green';
        player_colors[1] = 'blue';

        this.game = new OvercookedGame({
            start_grid,
            container_id,
            assets_loc: "static/assets/",
            ANIMATION_DURATION: 200*.9,
            tileSize: 80,
            COOK_TIME: cook_time,
            explosion_time: Number.MAX_SAFE_INTEGER,
            DELIVERY_REWARD: DELIVERY_REWARD,
            player_colors: player_colors
        });
        this.init_orders = init_orders;
        console.log("Trajectory replay");
        this.observations = trajectory.ep_states[0];
        this.actions = trajectory.ep_actions[0];
        this.MAX_TIME = MAX_TIME;
        this.time_left = MAX_TIME;
        this.cur_gameloop = 0;
        this.score = 0;
        this.completion_callback = completion_callback;
        this.timestep_callback = timestep_callback;
        this.total_timesteps = this.observations.length - 1;
        this.paused = false;
        this.keyboard_paused = false;
        this.last_step_time = new Date().getTime();
        this.seconds_per_step = 0.5;
        this.speed_play = 0;
        this.speed_seconds_per_step = 0.1; 
    }


    init() {
        this.game.init();

        this.start_time = new Date().getTime();
	
        this.gameloop = setInterval(() => {
            if (this.cur_gameloop > this.total_timesteps) {
                this.close()
            }
            if (this.cur_gameloop < 0) {
                this.cur_gameloop = 0;
            }
            let game_loop_percentage = Math.round(100*this.cur_gameloop/this.total_timesteps);
            document.getElementById("stepSlider").value = game_loop_percentage; 

            if (this.time_left == 0) {
                this.close();
            }

            if (this.paused == false && this.keyboard_paused == false) {
                this.disable_response_listener()
                this.last_step_time = new Date().getTime();
                let state_dict = this.observations[this.cur_gameloop]
              
                this.state = dictToState(state_dict)

                this.game.drawState(this.state);
                let actions_arr = this.actions[this.cur_gameloop]; 
                this.joint_action = lookupActions(actions_arr);
                // read the two player actions out of the trajectory 
                // Do a transition and get the next state and reward.
                let  [[next_state, prob], reward] =
                    this.game.mdp.get_transition_states_and_probs({
                        state: this.state,
                        joint_action: this.joint_action
                    });

                this.time_left = this.total_timesteps - this.cur_gameloop
                this.game.drawTimeLeft(this.time_left);

                //record data
                this.timestep_callback({
                    state: this.state,
                    joint_action: this.joint_action,
                    next_state: next_state,
                    reward: reward,
                    time_left: this.time_left,
                    score: this.score,
                    time_elapsed: this.cur_gameloop,
                    cur_gameloop: this.cur_gameloop,
                    client_id: undefined,
                    is_leader: undefined,
                    partner_id: undefined,
                    datetime: +new Date()
                });
                //set up next timestep
                this.paused = true
                this.activate_response_listener();
            }
            let seconds_since_step = (new Date().getTime() - this.last_step_time)/1000; 
            if (this.keyboard_paused == false) {
                if (this.speed_play < 0 && seconds_since_step > this.speed_seconds_per_step) {
                    this.paused = false;
                    this.cur_gameloop -= 1;
                }
                else if (this.speed_play > 0 && seconds_since_step > this.speed_seconds_per_step) {
                    this.paused = false;
                    this.cur_gameloop += 1;
                }
                else if (this.speed_play == 0 && seconds_since_step > this.seconds_per_step) {
                    this.paused = false;
                    this.cur_gameloop += 1;
                }
            }
            

            //time run out
            
        }, this.TIMESTEP);
        //By default it seems like we'd want to remove the response listener 
        // But we could maybe also keep a response listener that takes in the keys Left and Right 
        // And if it gets left it regresses the state, and if it gets. right, it progresses the state
        //this.activate_response_listener();
    }

    close () {
        if (typeof(this.gameloop) !== 'undefined') {
            clearInterval(this.gameloop);
        }
        this.game.close();
        this.disable_response_listener();
        this.completion_callback();
    }

    activate_response_listener () {
        var slider = document.getElementById("stepSlider");
        let total_timesteps = this.total_timesteps; 
        let game = this; 
        slider.oninput = function() {
            let slider_percent = this.value/100.0;
            game.cur_gameloop = Math.round(slider_percent*game.total_timesteps); 
            game.paused = false;
        }
        $(document).on("keydown", (e) => {
            switch(e.which) {
            case 37: // left
                this.cur_gameloop -= 1;
                this.speed_play = -1; 
                this.paused = false; 
                break;

            case 39: // right
                this.cur_gameloop += 1;
                this.speed_play = +1; 
                this.paused = false; 
                break;

            case 32: //space
                if (this.keyboard_paused) {
                    this.keyboard_paused = false;
                    console.log("Unpausing")
                }
                else {
                    this.keyboard_paused = true;
                    console.log("Pausing")
                }
                
                break;
            default: return; // exit this handler for other keys
            }
            e.preventDefault(); // prevent the default action (scroll / move caret)
        });

	$(document).on("keyup", (e) => {
            if (e.which == 37 || e.which == 39) {
		this.speed_play = 0; 
            }
            
	}); 
    }

    disable_response_listener () {
        $(document).off('keydown');
    }
}
