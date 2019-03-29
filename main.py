import sys
import os
import argparse
# temporary fix for path
python_path = os.getcwd() + "/python"
sys.path.insert(0, python_path)

from python.unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("agg")

from ddpg_agent import Agent


# from deep_rl import *

def ddpg(env, state_size, action_size, num_agents, brain_name,
         n_episodes=1000, max_t=300, print_every=100, title=None,
         batch_size=128, gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0, device="cuda:0"):

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2,
                  batch_size=batch_size, gamma=gamma, tau=tau, lr_actor=lr_actor,
                  lr_critic=lr_critic, weight_decay=weight_decay, device=device)
    if title is None:
        title = "experiment"
    f = open("{}_scores.txt".format(title), "w")
    scores_deque = deque(maxlen=print_every)
    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
        states = env_info.vector_observations

        scores = np.zeros(num_agents)
        agent.reset()

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)
            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step

            if np.any(dones):
                break
        scores_deque.append(np.mean(scores))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            f.write("{},{}\n".format(i_episode, np.mean(scores_deque)))
    f.close()
    return scores

def main():
    parser = argparse.ArgumentParser(description='Options to train the model.')
    parser.add_argument("--title", type=str, default="experiment")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda:{}".format(args.device)
    env = UnityEnvironment(file_name='data/Reacher_Linux_NoVis_multiple/Reacher.x86_64',
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

    ddpg(env, state_size, action_size, num_agents, brain_name, title=args.title,
         n_episodes=5000, max_t=300, print_every=100,
         batch_size=args.batch_size, gamma=args.gamma, tau=args.tau,
         lr_actor=args.lr_actor, lr_critic=args.lr_critic, weight_decay=args.weight_decay, device=args.device)



if __name__ == '__main__':
    main()
