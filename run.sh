#!/usr/bin/env bash

python main.py --title exp_afterfix_2 --batch_size 256 --gamma 0.99 --tau 1e-3 \
                --lr_actor 1e-4 --lr_critic 3e-4 --fc1_units 400 --fc2_units 300 --device 1 --port 64740 --n_episodes 2000
