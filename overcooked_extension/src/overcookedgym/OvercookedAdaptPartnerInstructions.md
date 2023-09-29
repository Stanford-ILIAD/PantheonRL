# Adaptive Partner Experiments. (**Make sure to be in the base directory of PantheonRL**)

#### Train bunch of partners
```
python3 trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --seed 10 --preset 1
python3 trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --seed 11 --preset 1
python3 trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --seed 12 --preset 1
python3 trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --seed 13 --preset 1
python3 trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --seed 14 --preset 1
```

#### Train to play against group of partners
```
python3 trainer.py OvercookedMultiEnv-v0 PPO FIXED FIXED FIXED FIXED --alt-config \
    '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-10"}' \
    '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-11"}' \
    '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-12"}' \
    '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-13"}' \
    --env-config '{"layout_name":"simple"}' --seed 20 -t 1000000 --preset 1

python3 trainer.py OvercookedMultiEnv-v0 ModularAlgorithm FIXED FIXED FIXED FIXED --ego-config '{"marginal_reg_coef": 0.5}' --alt-config \
    '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-10"}' \
    '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-11"}' \
    '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-12"}' \
    '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-13"}' \
    --env-config '{"layout_name":"simple"}' --seed 21 -t 1000000 --preset 1
```

#### Adapt to new partner
```
python3 trainer.py OvercookedMultiEnv-v0 PPO FIXED --alt-config '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-14"}' --env-config '{"layout_name":"simple"}' --seed 30 --preset 1

python3 trainer.py OvercookedMultiEnv-v0 LOAD FIXED --ego-config '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-ego-20"}' --alt-config '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-14"}' --env-config '{"layout_name":"simple"}' --seed 31 --preset 1

python3 trainer.py OvercookedMultiEnv-v0 LOAD FIXED --ego-config '{"type":"ModularAlgorithm", "location":"models/OvercookedMultiEnv-v0-simple-ModularAlgorithm-ego-21"}' --alt-config '{"type":"PPO", "location":"models/OvercookedMultiEnv-v0-simple-PPO-alt-14"}' --env-config '{"layout_name":"simple"}' --seed 32 --preset 1

```