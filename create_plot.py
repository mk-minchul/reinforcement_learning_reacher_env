from matplotlib import pyplot as plt
import os
def plot_avg_reward(path):

    with open(path, "r") as f:
        data = f.readlines()
        epochs = []
        scores = []
        for line in data[1:]:
            epoch, score = line.strip().split(",")
            epochs.append(int(epoch)*100)
            scores.append(float(score))

    fig, ax = plt.subplots()
    ax.plot(epochs, scores)
    ax.axhline(10, color="red")
    ax.set_xlabel("number of episodes")
    ax.set_ylabel("avg score of 100 episodes")

    ax.set_title("Averge Score Plot for Reacher Env")
    plt.savefig("score.png")

if __name__ == '__main__':
    path = "gamma_099_tau_1e-3_scores.txt"
    plot_avg_reward(path)

