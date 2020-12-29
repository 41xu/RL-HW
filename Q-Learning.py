"""
agent: "o", target: rightmost location
"""
import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6  # number of states, 1 dimension is the length
ACTION = ['left', 'right']
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 13
FRESH_TIME = 0.3  # fresh time for one move
ENV=['-']*(N_STATES-1)+['T']

def q_table_init(n_states, actions):
    """
    :param n_states: number of states
    :param actions: action list
    :return:
    """
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return table


def choose_action(state,q_table):
    state_actions=q_table.iloc[state,:]
    print(state_actions)
    if np.random.uniform()>EPSILON or (state_actions==0).all(): # none actions
        action=np.random.choice(ACTION)
    else:
        action=state_actions.idxmax()
    return action


def get_feedback(S,A):
    if A=='right':
        if S==N_STATES-2:
            S_='terminal'
            R=1 # reward
        else:
            S_=S+1
            R=0
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R


def update(S,episode,cnt):
    if S=='terminal':
        print("Episode %s: total steps = %s"%(episode+1,cnt))
        time.sleep(2)
        print(' '*10)
    else:
        ENV[S]='o' # state update
        tmp=''.join(ENV)
        print(tmp)
        time.sleep(FRESH_TIME)


def rl():
    # main loop
    QTABLE=q_table_init(N_STATES,ACTION)
    for episode in range(MAX_EPISODES):
        cnt=0
        S=0
        flag=False
        update(S,episode,cnt)
        while not flag:
            A=choose_action(S,QTABLE)
            S_,R=get_feedback(S,A)
            pred=QTABLE.loc[S,A]
            if S_!='terminal':
                target=R+GAMMA*QTABLE.iloc[S_,:].max()
            else:
                target=R
                flag=True
            QTABLE.loc[S,A]+=ALPHA*(target-pred)
            S=S_
            cnt+=1
            update(S,episode,cnt)
    return QTABLE


if __name__=='__main__':
    QTABLE=rl()
    print(QTABLE)


