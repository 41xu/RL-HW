import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit(object):
    def __init__(self, k):
        self.k = k
        self.action_value = np.zeros(k)
        self.optimal = 0

    def pull(self, action):
        return 0, True


class GaussMAB(MultiArmedBandit):
    def __init__(self, k, mu=0, sigma=1):
        super(GaussMAB, self).__init__(k)
        self.mu = mu
        self.sigma = sigma

    def pull(self, action):
        return np.random.normal(self.action_value[action]), action == self.optimal


class Agent(object):
    def __init__(self, bandit, polity, prior=0):
        self.k = bandit.k
        self.prior = prior
        self.value = self.prior * np.ones(self.k)
        self.polity = polity
        self.last_action = None

    def choose(self):
        action = self.polity.choose(self)
        self.last_action = action
        return action


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


def epsilon_greedy(k=15, epsilon=0.1, trials=100, experiments=1):
    score = np.zeros(trials)
    optimal = np.zeros(trials)
    bandit = GaussMAB(k)
    polity = EGreedy(epsilon)
    agent = Agent(bandit, polity)
    for _ in range(experiments):
        for t in range(trials):
            action = agent.choose()
            reward, flag = bandit.pull(action)
            score[t] += reward
            if flag:
                optimal[t] += 1
        print(score)
        print("*"*20)
        print(optimal)
    return score / experiments, optimal / experiments


if __name__ == '__main__':
    k = 15
    score,optimal=epsilon_greedy(k=k,epsilon=0.1,trials=1000,experiments=500)
    # 可视化部分
    plt.plot(score)
    plt.show()
