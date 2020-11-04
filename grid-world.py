import numpy as np

NVAL = 5  # value窗口大小5*5
ACTIONS = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
A_POS, A_PRIME_POS = [0, 1], [4, 1]
B_POS, B_PRIME_POS = [0, 3], [2, 3]
R = 0.9
P = 0.25
TER = 1e-4


class Grid(object):
    def __init__(self, nval=5):
        self.nval = nval
        self.VALUE= np.zeros((self.nval, self.nval))
        self.CUR = np.zeros(self.VALUE.shape)

    def bellman(self):
        while True:
            self.CUR=np.zeros(self.VALUE.shape)
            for i in range(self.nval):
                for j in range(self.nval):
                    for act in ACTIONS:
                        [ni, nj], reward = self.action([i, j], act)
                        # print(ni,nj,reward)
                        # print(self.VALUE)
                        # print("*"*10)
                        self.CUR[i, j] += P * (reward + R * self.VALUE[ni, nj])
            if np.sum(np.abs(self.VALUE - self.CUR)) < TER:
                break
            self.VALUE=self.CUR
        return self.VALUE

    def action(self, state, act):
        if state == A_POS:
            return A_PRIME_POS, 10
        elif state == B_POS:
            return B_PRIME_POS, 5
        next_state = state + act
        if 0 <= next_state[0]< NVAL and 0 <= next_state[1] < NVAL:
            reward = 0.
        else:
            reward = -1.
            next_state = state
        return next_state, reward
grid=Grid(NVAL)
value=grid.bellman()
print(value)