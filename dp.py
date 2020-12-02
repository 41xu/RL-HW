import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

# 每个场的车容量
MAX_CARS = 20
# 每晚最多移动的车数
MAX_MOVE_OF_CARS = 5
# A场租车请求的平均值
RENTAL_REQUEST_FIRST_LOC = 3
# B场租车请求的平均值
RENTAL_REQUEST_SECOND_LOC = 4
# A场还车请求的平均值
RETURNS_FIRST_LOC = 3
# B场还车请求的平均值
RETURNS_SECOND_LOC = 2
# 收益折扣
DISCOUNT = 0.9
# 租车收益
RENTAL_CREDIT = 10
# 移车支出
MOVE_CAR_COST = 2
# （移动车辆）动作空间：【-5，5】
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)
# 租车还车的数量满足一个poisson分布，限制由泊松分布产生的请求数大于POISSON_UPPER_BOUND时其概率压缩至0
POISSON_UPPER_BOUND = 11
# 存储每个（n,lamda）对应的泊松概率
poisson_cache = dict()

def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam # 定义唯一key值，除了索引没有实际价值
    if key not in poisson_cache:
        # 计算泊松概率，这里输入为n与lambda，输出泊松分布的概率质量函数，并保存到poisson_cache中
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

def expected_return(state, action, state_value, constant_returned_cars):
    # initailize total return
    returns = 0.0

    # 移动车辆产生负收益
    returns -= MOVE_CAR_COST * abs(action)

    # 移动后的车辆总数不能超过20
    NUM_OF_CARS_FIRST_LOC = min(state[0] - action, MAX_CARS)
    NUM_OF_CARS_SECOND_LOC = min(state[1] + action, MAX_CARS)

    # 遍历两地全部的可能概率下（<11）租车请求数目
    for rental_request_first_loc in range(POISSON_UPPER_BOUND):
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            # prob为两地租车请求的联合概率,概率为泊松分布
            # 即：1地请求租车rental_request_first_loc量且2地请求租车rental_request_second_loc量
            prob = poisson_probability(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) * \
                   poisson_probability(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            # 两地原本的车的数量
            num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOC
            num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC

            # 有效的租车数目必须小于等于该地原有的车辆数目
            valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
            valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

            # 计算回报，更新两地车辆数目变动
            reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT
            num_of_cars_first_loc -= valid_rental_first_loc
            num_of_cars_second_loc -= valid_rental_second_loc

            # 如果还车数目为泊松分布的均值
            if constant_returned_cars:
                # 两地的还车数目均为泊松分布均值
                returned_cars_first_loc = RETURNS_FIRST_LOC
                returned_cars_second_loc = RETURNS_SECOND_LOC
                # 还车后总数不能超过车场容量
                num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                # 核心：
                # 策略评估：V(s) = p(s',r|s,π(s))[r + γV(s')]
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])

            # 否则计算所有泊松概率分布下的还车空间
            else:
                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(
                            returned_cars_first_loc, RETURNS_FIRST_LOC) * poisson_probability(returned_cars_second_loc, RETURNS_SECOND_LOC)
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        # 联合概率为【还车概率】*【租车概率】
                        prob_ = prob_return * prob
                        returns += prob_ * (reward + DISCOUNT *
                                            state_value[num_of_cars_first_loc_, num_of_cars_second_loc_])
    return returns
def iter_value(constant_returned_cars=True):
    # 初始化价值函数为0
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    # 初始化策略为0
    policy = np.zeros(value.shape, dtype=np.int)
    # 设置迭代参数
    iterations = 0

    while True:
        # policy evaluation (in-place) 策略评估（in-place）
        # 未改进前，第一轮policy全为0，即[0，0，0...]
        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    # 更新V（s）
                    new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)
                    # in-place操作
                    value[i, j] = new_state_value
            # 比较V_old(s)、V(s)，收敛后退出循环
            max_value_change = abs(old_value - value).max()
            # print('max value change {}'.format(max_value_change))
            if max_value_change < 1e-4:
                break

        # policy improvement
        # 在上一部分可以看到，策略policy全都是0，如不进行策略改进，其必然不会收敛到实际最优策略。
        # 所以需要如下策略改进
        policy_stable = True
        # i、j分别为两地现有车辆总数
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                action_returns = []
                # actions为全部的动作空间，即[-5、-4...4、5]
                for action in actions:
                    if 0 <= action <= i or -j <= action <= 0:
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-np.inf)
                # 找出产生最大动作价值的动作
                new_action = actions[np.argmax(action_returns)]
                # 更新策略
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:
                    policy_stable = False

        if policy_stable:
            print("stable value: ",value)
            break


if __name__ == '__main__':
    iter_value()