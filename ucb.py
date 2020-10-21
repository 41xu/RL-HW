import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit(object):
    def __init__(self, k):
        self.k = k
        self.action_value = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True

    def clc(self):
        self.action_value = np.zeros(self.k)
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
        self.t = 0

    def choose(self):
        action = self.polity.choose(self)
        self.last_action = action
        return action

    def clc(self):
        self.value[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1
        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action]
        else:
            g = self.gamma
        q = self.value[self.last_action]
        self.value[self.last_action] += g * (reward - q)
        self.t += 1


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


class UCB(object):
    def __init__(self, c):
        self.c = c

    def choose(self, agent):
        exploration = np.log(agent.t + 1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1 / self.c)
        q = agent.value + exploration
        action = np.argmax(q)
        if len(np.where(q == q[action])[0]) == 1:
            return action
        else:
            return np.random.choice(np.where(q == q[action])[0])


def ucb(agents, bandit, trials=100, epochs=100):
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
    return score / epochs, optimal / epochs


if __name__ == '__main__':
    k = 15
    bandit = GaussMAB(k)
    pol_ucb = UCB(2)
    pol_e = EGreedy(0.1)
    agents = [Agent(bandit, pol_ucb), Agent(bandit,pol_e)]
    trial = 2000
    score, optimal = ucb(agents, bandit, trials=trial, epochs=trial // 2)
    label = "ucb & \u03B5-greedy"
    plt.plot(score)
    plt.title(label)
    plt.legend(["ucb=2","\u03B5=0.1"])
    plt.ylabel("Average Reward")
    plt.savefig("hw2")
    plt.show()
