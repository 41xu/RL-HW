import numpy as np
import matplotlib.pyplot as plt


def RandomWalkTD0(alpha=0.1, gamma=1, values=np.zeros(7)):
    nodes = ['L', 'A', 'B', 'C', 'D', 'E', 'R']
    pos = 3  # C
    node = nodes[pos]
    flag = True

    while flag:
        direction = np.random.choice(['l', 'r'])
        new_pos = pos - 1 if direction == 'l' else pos + 1
        reward = 0.
        if nodes[new_pos] in ('L', 'R'):
            reward = 1. if nodes[new_pos] == 'R' else 0.
        else:
            reward += gamma * values[new_pos]
        values[pos] += alpha * (reward - values[pos])
        pos = new_pos
        node = nodes[pos]
        if node in ('L', 'R'):
            flag = False
    return values


def RandomWalkMC(gamma=1, values=np.zeros(7)):
    nodes = ['L', 'A', 'B', 'C', 'D', 'E', 'R']
    pos = 3
    node = nodes[pos]
    flag = True
    i, N, G = 0, 0, 0
    while flag:
        direction = np.random.choice(['l', 'r'])
        new_pos = pos - 1 if direction == 'l' else pos + 1
        reward = 0.
        if nodes[new_pos] in ('L', 'R'):
            reward = 1. if nodes[new_pos] == 'R' else 0.
        else:
            reward += gamma * values[new_pos]
        G += np.power(gamma, i) * reward
        i += 1
        N += 1
        pos = new_pos
        node = nodes[pos]
        if node in ('L', 'R'):
            flag = False
    values[pos] += (G - values[pos]) * 1. / N
    return values


if __name__ == '__main__':
    walks = 200
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15]
    actual_statue_values = np.linspace(0, 1, 7)
    print(actual_statue_values)
    actual_statue_values[0] = 0
    actual_statue_values[-1] = 0
    plt.figure(figsize=[10, 6])
    for alpha in alphas:
        errors = []
        for i in range(walks):
            values = RandomWalkTD0(alpha=alpha)
            rmse = np.sqrt(np.mean(np.power(values - actual_statue_values, 2)))
            errors.append(rmse)
        plt.plot(errors, label ="aplha={}".format(alpha))
    errors=[]
    for i in range(walks):
        values=RandomWalkMC(gamma=0.01)
        rmse=np.sqrt(np.mean(np.power(values-actual_statue_values,2)))
        errors.append(rmse)
    plt.plot(errors,label="MC")
    plt.xlabel("Episodes",size=14)
    plt.ylabel("RMSE",size=14)
    plt.legend(loc="upper")
    plt.savefig('TD_MC.png')
    plt.show()
    errors=[]

