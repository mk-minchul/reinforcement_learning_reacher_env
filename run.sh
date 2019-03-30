#!/usr/bin/env bash
python main.py --title default --gamma 0.99 --tau 1e-3 --use_batch_norm False --n_critic_layer 3 --device 4 --port 64738
python main.py --title use_bn --gamma 0.99 --tau 1e-3 --use_batch_norm True --n_critic_layer 3 --device 5 --port 64735
python main.py --title use_bn_n_tau_1e-2 --gamma 0.99 --tau 1e-2 --use_batch_norm True --n_critic_layer 3 --device 6 --port 64736
python main.py --title use_bn_n_lr_actor_5e-4_tau_1e-2 --gamma 0.99 --tau 1e-2 --use_batch_norm True --n_critic_layer 3 --lr_actor 5e-4 --device 7 --port 64737


python main.py --title no_bn_n_tau_1e-2 --gamma 0.99 --tau 1e-2 --use_batch_norm False --n_critic_layer 3 --device 6 --port 64738
python main.py --title bn_n_tau_5e-2 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 3 --device 7 --port 64735


