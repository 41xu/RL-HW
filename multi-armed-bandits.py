import numpy as np


class MultiArmedBandit(object):
    def __init__(self, k):
        self.k = k
        self.action_value = np.zeros(k)


class GaussMAB(MultiArmedBandit):
    def __init__(self, k, mu=0, sigma=1):
        super(GaussMAB, self).__init__(k)
        self.mu = mu
        self.sigma = sigma


class Agent(object):
    def __init__(self, bandit,prior=0):
        self.k = bandit.k
        self.prior=prior
        self.value=self.prior*np.ones(self.k)



class EGreedy(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose(self,agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value))
        else:
            action=np.argmax(agent.value)
            check=np.where(agent.value==agent.value[action])[0]
            if len(check)==1:
                return action
            else:
                return np.random.choice(check)


narm = 15
bandit = GaussMAB(narm)


def epsilon_greedy(k=narm, epsilon=0.1,trials=100,agents=[]):
    score=np.zeros((trials,len(agents)))
    optimal=np.zeros(score.shape)
    for ephco in range(trials):
        for i,agent in enumerate(agents):
            action=agent


