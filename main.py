import sys
import os
import argparse
# temporary fix for path
python_path = os.getcwd() + "/python"
sys.path.insert(0, python_path)

from python.unityagents import UnityEnvironment
import torch
from time import gmtime, strftime
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from ddpg_agent import Agent


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def ddpg(env, state_size, action_size, num_agents, brain_name,
         n_episodes=1000, max_t=1000, print_every=10, title=None,
         batch_size=128, gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0, device="cuda:0",
         fc1_units=128, fc2_units=64, n_updates=10, update_intervals=20):

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2, num_agents=num_agents,
                  batch_size=batch_size, gamma=gamma, tau=tau, lr_actor=lr_actor,
                  lr_critic=lr_critic, weight_decay=weight_decay, device=device,
                  fc1_units=fc1_units, fc2_units=fc2_units)

    # create save directory
    if title is None:
        title = "experiment"
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    title = title + "_" + current_time

    # write a new file
    os.makedirs("experiments/{}".format(title), exist_ok=True)
    f = open("experiments/{}/scores.txt".format(title), "w")
    f.close()

    scores_deque = deque(maxlen=100)
    mean_scores = []

    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
        states = env_info.vector_observations

        scores = np.zeros(num_agents)
        agent.reset()

        for t in range(max_t):
            # 1. observe states with the current policty mu theta + noise
            actions = agent.act(states)

            # 2. Execute a in the environment and observe next state (s,a,r,s',d')
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished

            # 3. save experiences to the replay buffer
            agent.remember(states, actions, rewards, next_states, dones)

            # 4. learn by sampling from the replay buffer
            # if it is time to update, for however many updates
            agent.update(n_updates, update_intervals, t)

            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step

            if np.any(dones):
                break
        scores_deque.append(np.mean(scores))
        print('\rEpisode {}\tLast 100 average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        # save score and model every print_every
        if i_episode % print_every == 0:
            f = open("experiments/{}/scores.txt".format(title), "a")
            f.write("{},{}\n".format(i_episode, np.mean(scores_deque)))
            f.close()
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            mean_scores.append(np.mean(scores_deque))
            # save if best model
            if np.mean(scores_deque) == max(mean_scores):
                torch.save(agent.actor_local.state_dict(), 'experiments/{}/checkpoint_actor.pth'.format(title))
                torch.save(agent.critic_local.state_dict(), 'experiments/{}/checkpoint_critic.pth'.format(title))

            if np.mean(scores_deque) >= 30:
                print("\rEnvironment solved with average score of 30")
                break

    return mean_scores, title

def main():
    parser = argparse.ArgumentParser(description='Options to train the model.')
    parser.add_argument("--title",                          type=str,       default="experiment")
    parser.add_argument("--batch_size",                     type=int,       default=128)
    parser.add_argument("--gamma",                          type=float,     default=0.99)
    parser.add_argument("--tau",                            type=float,     default=1e-3)
    parser.add_argument("--lr_actor",                       type=float,     default=1e-4)
    parser.add_argument("--lr_critic",                      type=float,     default=3e-4)
    parser.add_argument("--weight_decay",                   type=float,     default=0)
    parser.add_argument("--device",                         type=int,       default=0)
    parser.add_argument('--port',                           type=int,       default=64735)
    parser.add_argument('--n_episodes',                     type=int,       default=2000)
    parser.add_argument('--fc1_units',                      type=int,       default=128)
    parser.add_argument('--fc2_units',                      type=int,       default=64)
    parser.add_argument('--n_updates',                      type=int,       default=10)
    parser.add_argument('--update_intervals',               type=int,       default=20)

    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda:{}".format(args.device)
    env = UnityEnvironment(file_name='data/Reacher_Linux_NoVis_multiple{}/Reacher.x86_64'.format(args.port-64734),
                           base_port=args.port, no_graphics=False, docker_training=False)
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
    os.makedirs("experiments/",exist_ok=True)
    print("Experiment result will be saved at : experiments/{}".format(args.title))
    mean_scores, title = ddpg(env, state_size, action_size, num_agents, brain_name, title=args.title,
                              n_episodes=args.n_episodes, max_t=1000, print_every=10,
                              batch_size=args.batch_size, gamma=args.gamma, tau=args.tau,
                              lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                              weight_decay=args.weight_decay, device=args.device,
                              fc1_units=args.fc1_units, fc2_units=args.fc2_units,
                              n_updates=args.n_updates, update_intervals=args.update_intervals
                              )

    # create plot
    fig, ax = plt.subplots()
    epochs = 10 * np.arange(1, len(mean_scores)+1).astype(np.int64)
    ax.plot(epochs, mean_scores)
    ax.axhline(10, color="red")
    ax.set_xlabel("number of episodes")
    ax.set_ylabel("avg score of 100 episodes")
    ax.set_title("Averge Score Plot {}".format(args.title))
    plt.savefig("experiments/{}/score.png".format(title))

    print("done")

if __name__ == '__main__':
    main()
