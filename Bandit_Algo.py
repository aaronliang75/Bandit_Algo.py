import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

stock_t_data = pd.read_csv('./RL_A1_Data/hs300_stock_t_params.csv', index_col=0)


class Bandit:
    def __init__(self, n_assets=200, epsilon=0.01, UCB_param=0.00001, params_data=stock_t_data):
        self.k = n_assets
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param

        self.epsilon = epsilon

        self.params_data = params_data

        self.q_true = None
        self.q_estimation = None
        self.action_count = None
        self.best_action = None

    def true_val_generator(self):
        """

        :return: a value array containing q_true for each 'run'
        """
        params_data = self.params_data
        reward_arr = []
        for i in range(params_data.shape[0]):
            params = params_data.iloc[i].values
            t_rv = stats.t.rvs(loc=params[0], scale=params[1], df=params[2], size=1)[0]
            reward_arr.append(t_rv)

        return np.array(reward_arr)

    def reset(self):
        """

        :return:
        (1) get q_true
        (2) initialize q_estimation = 0, action_count= 0
        (3) determine best_action based on q_true
        """
        # real reward for selecting each assets
        # generated from the fitted distribution based on historical data
        # params is a list of distribution parameters, [loc,scale,df]
        self.q_true = self.true_val_generator()

        # estimation for each action
        self.q_estimation = np.zeros(self.k)

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    def reward_deviation(self, action):
        """

        :param action: one arm
        :return: add some noise to reward and try to simulate bandit game
        """
        params_data = self.params_data
        t_param = params_data.iloc[action]
        reward_dev = stats.t.rvs(loc=t_param[0], scale=t_param[1], df=t_param[2], size=1)[0]

        return reward_dev

    def act(self):
        """

        :return: choose the best action based on q_estimation
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                             self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    def step(self, action):
        """

        :param action: one arm
        :return:
        (1) make q_estimation get close to q_true
        (2) return the reward for one epoch
        """
        # generate the reward under N(real reward, 1)
        reward = self.reward_deviation(action) + self.q_true[action]

        self.time += 1
        self.action_count[action] += 1

        # update estimation using sample averages
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]

        return reward


def simulate(runs, time, bandit):
    """

    :param runs: learning by averaging and to decide whether the algo converge
    :param time: training epochs for 1 run
    :param bandit: one instance for bandit class
    :return:
    (1) mean_best_action_counts: the number of times best actions been chosen
    (2) mean_rewards: average of reward by runs
    (3) policy: to log chosen actions
    """
    rewards = np.zeros((runs, time))
    policy = np.zeros((runs, time))
    best_action_counts = np.zeros(rewards.shape)

    for r in tqdm(range(runs)):
        bandit.reset()
        for t in range(time):
            action = bandit.act()
            # let q_estimate converge to q_true
            reward = bandit.step(action)
            rewards[r, t] = reward
            # save daily return w.r.t action
            # best_action is fixed once initializing q_true
            if action == bandit.best_action:
                best_action_counts[r, t] = 1
                # the number_code of stocks
                # the reason why we add one here because policy is initialized to be 0-vector
                # we want to differentiate initial value with the number_code of stock
                policy[r, t] = action + 1

    mean_best_action_counts = best_action_counts.mean(axis=0)
    mean_rewards = rewards.mean(axis=0)
    policy = pd.DataFrame(policy)
    return mean_best_action_counts, mean_rewards, policy

"""
Execution...
"""

classic_bandit = Bandit(n_assets=stock_t_data.shape[0], epsilon=0.01, UCB_param=0.0002, params_data=stock_t_data)

best_actions, avg_reward, policy = simulate(runs=20, time=100, bandit=classic_bandit)

# Best stock to invest
policy.to_csv('./Policy.csv')
n_action_arr = []

for i in range(policy.shape[0]):
    n_action = policy.iloc[i].dropna()
    n_action_arr.append(np.argmax(np.bincount(n_action)))


n_action_df = pd.DataFrame(n_action_arr, columns=['stock_number'])
n_action_df.loc[:, 'sign'] = 1
n_action_df = n_action_df.groupby('stock_number')['sign'].count().sort_values(ascending=False).to_frame()
n_action_df = n_action_df[n_action_df['sign'] != 0]
print("Best Stock to Invest: #", n_action_df.index[0])
n_action_df.to_csv('./n_action_df.csv')

plt.figure(figsize=(20, 10))
plt.plot(avg_reward, lw=3)
plt.xlabel('Epochs')
plt.ylabel('Average Reward')
plt.title('Average Reward / Daily Return in Bandit Algorithm', fontsize=14)
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(best_actions, lw=3)
plt.xlabel('Epochs')
plt.ylabel('% Optimal Action')
plt.title('Optimal Action (%) in Bandit Algorithm', fontsize=15)
plt.show()
