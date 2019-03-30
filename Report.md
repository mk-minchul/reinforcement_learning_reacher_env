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

### Plot of Rewards
![Score Plot](score.png)

### Problem:
The algorithm maybe not explorative enough to learn the space. 

### Ideas for Future Work