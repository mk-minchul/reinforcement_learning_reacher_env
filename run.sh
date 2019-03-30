#!/usr/bin/env bash
python main.py --title default --gamma 0.99 --tau 1e-3 --use_batch_norm False --n_critic_layer 3 --device 4 --port 64738
python main.py --title use_bn --gamma 0.99 --tau 1e-3 --use_batch_norm True --n_critic_layer 3 --device 5 --port 64735
python main.py --title use_bn_n_critic_layer4 --gamma 0.99 --tau 1e-3 --use_batch_norm True --n_critic_layer 4 --device 6 --port 64736
python main.py --title use_bn_n_critic_layer4_lr_actor_5e-4 --gamma 0.99 --tau 1e-3 --use_batch_norm True --n_critic_layer 4 --lr_actor 5e-4 --device 7 --port 64737



