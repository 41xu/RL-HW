import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit(object):
    def __init__(self, k):
        self.k = k
        self.action_value = np.zeros(k)
        self.optimal = 0

    def pull(self, action):
        return 0, True

    def clc(self):
        self.action_value = np.zeros(k)
        self.optimal = 0


class GaussMAB(MultiArmedBandit):
    def __init__(self, k=15, mu=8, sigma=1):
        super(GaussMAB, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.clc()

    def pull(self, action):
        return np.random.normal(self.action_value[action]), action == self.optimal

    def clc(self):
        self.action_value = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_value)


class Agent(object):
    def __init__(self, bandit, polity, prior=0, gamma=None):
        self.k = bandit.k
        self.prior = prior
        self.value = self.prior * np.ones(self.k)
        self.action_attempts = np.zeros(self.k)
        self.polity = polity
        self.last_action = None
        self.gamma = gamma

    def choose(self):
        action = self.polity.choose(self)
        self.last_action = action
        return action

    def clc(self):
        self.value[:] = self.prior
        self.last_action = None

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1
        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action]
        else:
            g = self.gamma
        q = self.value[self.last_action]
        self.value[self.last_action] += g * (reward - q)


class EGreedy(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value))
        else:
            action = np.argmax(agent.value)
            check = np.where(agent.value == agent.value[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)


def epsilon_greedy(agents, bandit, trials=100, epochs=100):
    score = np.zeros((trials, len(agents)))
    optimal = np.zeros((trials, len(agents)))
    for _ in range(epochs):
        bandit.clc()
        for agent in agents:
            agent.clc()
        for t in range(trials):
            for i, agent in enumerate(agents):
                action = agent.choose()
                reward, flag = bandit.pull(action)
                agent.observe(reward)
                score[t, i] += reward
                if flag:
                    optimal[t, i] += 1
            # print(score)
            # print("*" * 20)
            # print(optimal)
    return score / epochs, optimal / epochs


if __name__ == '__main__':
    k = 15
    epsilon = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    trials = [500, 1000, 2000]
    bandit = GaussMAB(k)
    polity = [EGreedy(e) for e in epsilon]
    agents = [Agent(bandit, p) for p in polity]
    for trial in trials:
        score, optimal = epsilon_greedy(agents, bandit, trials=trial, epochs=trial//2)
        legend = ["\u03B5=" + str(e) for e in epsilon]
        label = "\u03B5-greedy"
        plt.plot(score)
        plt.legend(legend)
        plt.title(label)
        plt.ylabel("Average Reward")
        plt.savefig("hw1-{}.png".format(trial))
        plt.show()
