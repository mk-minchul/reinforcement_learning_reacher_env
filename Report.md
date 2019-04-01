# Banana Navigation

### Learning Algorithm
- DDPG method
- Critic is used to approximate the maximizer over the Q values of the next state. 
- It modifies DQN such that it takes the continuous actions. 
- The actor is learning the argmax of Q

- input space: 
    - 33 dimensions
    - continuous
- action space: 
    - 4 dimension, continuous


### Problem:
Tried training under variety of settings, yet did not see it reach above 10.  

### experiment log

#### batchsize 128 experiment 1
    - batch norm : False
    - tau : 1e-3
    - n_critic_layer : 3
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/default_2019-03-30_01:10:12/scores.png)

#### batchsize 128 experiment 2 ( has batchnorm)
    - batch norm : True
    - tau : 1e-3
    - n_critic_layer : 3
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/use_bn_2019-03-30_01:13:53/scores.png)


#### batchsize 128 experiment 3 (smaller tau)
    - batch norm : True
    - tau : 1e-2
    - n_critic_layer : 3
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/use_bn_n_tau_1e-2_2019-03-30_01:36:33/scores.png)


#### batchsize 128 experiment 4 (smaller tau, bigger lr for actor)
    - batch norm : True
    - tau : 1e-2
    - n_critic_layer : 3
    - gamma : 0.99
    - lr_actor : 5e-4
    - weight_decay 1.0

![score](experiments/use_bn_n_lr_actor_5e-4_tau_1e-2_2019-03-30_01:36:47/scores.png)


#### batchsize 128 experiment 5 (no batchnorm, smaller tau)
    - batch norm : False
    - tau : 1e-2
    - n_critic_layer : 3
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/no_bn_n_tau_1e-2_2019-03-30_11:14:54/score.png)


#### batchsize 128 experiment 6 (yes batchnorm, a bit smaller tau)
    - batch norm : True
    - tau : 5e-2
    - n_critic_layer : 3
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/bn_n_tau_5e-2_2019-03-30_11:15:02/score.png)


#### batchsize 128 experiment 7 (yes batchnorm, a bit smaller tau)
    - batch norm : True
    - tau : 5e-2
    - n_critic_layer : 3
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/bn_n_tau_5e-2_2019-03-30_11:15:02/score.png)


#### batchsize 256 experiment 1 (batchnorm, weightdecay 0.99)
    - batch norm : True
    - tau : 5e-2
    - n_critic_layer : 4
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 0.99

![score](experiments/bs_256_exp1_2019-03-30_16:01:58/score.png)

#### batchsize 256 experiment 2 (batchnorm)
    - batch norm : True
    - tau : 5e-2
    - n_critic_layer : 4
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/bs_256_exp2_2019-03-30_16:02:15/score.png)


#### batchsize 256 experiment 3 (batchnorm, n critic layer 3)
    - batch norm : True
    - tau : 5e-2
    - n_critic_layer : 3
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/bs_256_exp3_2019-03-30_16:02:23/score.png)


#### batchsize 256 experiment 4 (batchnorm, n critic layer 3, weight decay)
    - batch norm : True
    - tau : 5e-2
    - n_critic_layer : 3
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 0.99

![score](experiments/bs_256_exp4_2019-03-30_16:04:38/score.png)


#### batchsize 256 experiment 5 (batchnorm, smaller tau, n critic layer 4)
    - batch norm : True
    - tau : 1e-2
    - n_critic_layer : 4
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/bs_256_exp5_2019-03-30_23:23:04/score.png)


#### batchsize 256 experiment 6 (batchnorm, smaller tau, n critic layer 5)
    - batch norm : True
    - tau : 1e-2
    - n_critic_layer : 5
    - gamma : 0.99
    - lr_actor : 1e-4
    - weight_decay 1.0

![score](experiments/bs_256_exp6_2019-03-30_23:23:22/score.png)

#### batchsize 256 experiment 7 (batchnorm, n critic layer 4, lr actor 1e-3)
    - batch norm : True
    - tau : 5e-2
    - n_critic_layer : 4
    - gamma : 0.99
    - lr_actor : 1e-3
    - weight_decay 1.0

![score](experiments/bs_256_exp7_2019-03-30_23:23:38/score.png)


#### batchsize 256 experiment 7 (batchnorm, n critic layer 5, lr actor 1e-4 lr_critic 1-e4)
    - batch norm : True
    - tau : 5e-2
    - n_critic_layer : 5
    - gamma : 0.99
    - lr_actor : 1e-4
    - lr_critic : 1e-4
    - weight_decay 1.0

![score](experiments/bs_256_exp8_2019-03-30_23:23:52/score.png)


### Ideas for Future Work
Might need to change the algorithm. Should I run more episodes?