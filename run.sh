#!/usr/bin/env bash
python main.py --title default --gamma 0.99 --tau 1e-3 --use_batch_norm False --n_critic_layer 3 --device 4 --port 64738
python main.py --title use_bn --gamma 0.99 --tau 1e-3 --use_batch_norm True --n_critic_layer 3 --device 5 --port 64735
python main.py --title use_bn_n_tau_1e-2 --gamma 0.99 --tau 1e-2 --use_batch_norm True --n_critic_layer 3 --device 6 --port 64736
python main.py --title use_bn_n_lr_actor_5e-4_tau_1e-2 --gamma 0.99 --tau 1e-2 --use_batch_norm True --n_critic_layer 3 --lr_actor 5e-4 --device 7 --port 64737


python main.py --title no_bn_n_tau_1e-2 --gamma 0.99 --tau 1e-2 --use_batch_norm False --n_critic_layer 3 --device 6 --port 64738
python main.py --title bn_n_tau_5e-2 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 3 --device 7 --port 64735

python main.py --title bs_256_exp1 --batch_size 256 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 4 --weight_decay 0.99 --lr_actor 1e-4 --device 4 --port 64735
python main.py --title bs_256_exp2 --batch_size 256 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 4 --lr_actor 1e-4 --device 5 --port 64736
python main.py --title bs_256_exp3 --batch_size 256 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 3 --device 6 --port 64737
python main.py --title bs_256_exp4 --batch_size 128 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 3 --weight_decay 0.99 --device 7 --port 64738


python main.py --title bs_256_exp5 --batch_size 256 --gamma 0.99 --tau 1e-2 --use_batch_norm True --n_critic_layer 4 --lr_actor 1e-4 --device 4 --port 64735
python main.py --title bs_256_exp6 --batch_size 256 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 5 --lr_actor 1e-4 --device 5 --port 64736
python main.py --title bs_256_exp7 --batch_size 256 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 4 --lr_actor 1e-3 --device 6 --port 64737
python main.py --title bs_256_exp8 --batch_size 256 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 5 --lr_actor 1e-4 --lr_critic 1e-4 --device 7 --port 64738

python main.py --title bs_256_exp9 --batch_size 256 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 4 --lr_actor 1e-4 --device 5 --port 64736 --n_episodes 4000


python main.py --title exp10 --batch_size 256 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 4 --n_actor_layer 4 \
                --lr_actor 1e-3 --lr_critic 1e-3 --fc2_units 512 --fc2_units 256 --device 4 --port 64735 --n_episodes 2000


python main.py --title exp11 --batch_size 256 --gamma 0.99 --tau 5e-2 --use_batch_norm True --n_critic_layer 4 --n_actor_layer 4 \
                --lr_actor 1e-4 --lr_critic 1e-3 --fc2_units 128 --fc2_units 64 --device 5 --port 64736 --n_episodes 2000


python main.py --title exp12 --batch_size 256 --gamma 0.99 --tau 1e-2 --use_batch_norm True --n_critic_layer 4 --n_actor_layer 4 \
                --lr_actor 1e-3 --lr_critic 1e-3 --fc2_units 512 --fc2_units 256 --device 6 --port 64737 --n_episodes 2000


python main.py --title exp13 --batch_size 256 --gamma 0.99 --tau 1e-2 --use_batch_norm True --n_critic_layer 3 --n_actor_layer 4 \
                --lr_actor 1e-4 --lr_critic 1e-3 --fc2_units 256 --fc2_units 128 --device 7 --port 64738 --n_episodes 2000

