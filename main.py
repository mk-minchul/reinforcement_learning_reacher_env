import sys
import os
# temporary fix for path
python_path = os.getcwd() + "/python"
sys.path.insert(0, python_path)

from python.unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import random
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import gym

from ddpg_agent import Agent


# from deep_rl import *

def ddpg(env, state_size, action_size, num_agents, brain_name,
         n_episodes=1000, max_t=300, print_every=100):

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)

        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):

            actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
            actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores

def main():

    env = UnityEnvironment(file_name='data/Reacher_Linux_NoVis_single/Reacher.x86_64',
                           base_port=64735, no_graphics=False, docker_training=False)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    scores = np.zeros(num_agents)  # initialize the score (for each agent)

    ddpg(env, state_size, action_size, num_agents, brain_name,
         n_episodes=1000, max_t=300, print_every=100)

    while True:
        actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))



if __name__ == '__main__':
    main()