import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class HybridModel:
    def __init__(self, k, classifier, gamma = 0.9, alpha = 0.05,
                 start_epsilon = 1, min_epsilon = 0.05, epsilon_decay_rate = 0.99, 
                 min_epochs = 300, max_epochs = 600):
        self.k = k
        self.classifier = classifier

        self.alpha = alpha
        self.visit_count = np.zeros((k, 3))
        self.gamma = gamma

        self.start_epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

        self.min_epochs = min_epochs
        self.max_epochs = max_epochs

        self.kmeans_scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters = k, n_init = 20)

        self.Q = np.zeros((k, 3))

    # K-means
    def fit_kmeans(self, stock_prices, min_score = 0.2):
        X = np.vstack(stock_prices["state"])
        X_scaled = self.kmeans_scaler.fit_transform(X)

        score = 0
        while score < min_score:
            state_clusters = self.kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, state_clusters)
        return state_clusters

    def kmeans_predict(self, stock_prices):
        X = np.vstack(stock_prices["state"])
        X_scaled = self.kmeans_scaler.transform(X)

        state_clusters = self.kmeans.predict(X_scaled)
        return state_clusters

    def plot_state_clusters(self, stock_prices, state_clusters):
        pca = PCA(n_components = 2)
        X = self.kmeans_scaler.transform(stock_prices["state"])
        X = pca.fit_transform(X)

        plt.scatter(X[:, 0], X[:, 1], c = state_clusters, cmap = 'tab20', s = 5)
        plt.title("Price states clustered using K-means")
        plt.show()

    # K-means + GaussianNB Classifier + SARSA
    def train(self, stock_prices, show_log = True):
        self.Q = np.zeros((self.k, 3))
        trade_actions = self.classifier.predict(stock_prices)

        state_clusters = self.fit_kmeans(stock_prices)[20:-1]
        open_prices = stock_prices["open"].values[20:]

        current_epsilon = self.start_epsilon
        past_fifteen_rewards = deque()

        for i in range(self.max_epochs):
            # State initialization
            position = 0
            entry_price = 0
            total_reward = 0

            s_prev = state_clusters[0]

            allowed_actions = [2]
            if trade_actions[0] == 0:
                allowed_actions.append(0)

            if np.random.rand() < current_epsilon:
                action = np.random.choice(allowed_actions)
            else:
                action = allowed_actions[np.argmax(self.Q[s_prev][allowed_actions])]

            # Training loop
            # Trade starts at index 21 and ends at T - 1
            for t in range(1, len(open_prices)):
                price_t = open_prices[t]

                # Stop loss
                if position == 1 and (price_t - entry_price) / entry_price <= -0.1:
                    action = 1

                # Force sell on last trading day
                if t == len(open_prices) - 1:
                    action = 1

                # Compute reward
                reward = 0
                if position == 0 and action == 0:
                    position = 1
                    entry_price = price_t
                if position == 1 and action == 1:
                    position = 0
                    returns = (price_t - entry_price) / entry_price
                    reward = returns - 0.002
                total_reward += reward

                # Select action for tomorrow based on information up to the closing price state and trade position of today
                # Action is selected before Q-value update (SARSA)
                if t < len(open_prices) - 1:
                    s_t = state_clusters[t]
                    next_allowed = [2]
                    if position == 0 and trade_actions[t] == 0:
                        next_allowed.append(0)
                    elif position == 1 and trade_actions[t] == 1:
                        next_allowed.append(1)

                if np.random.rand() < current_epsilon:
                    next_action = np.random.choice(next_allowed)
                else:
                    next_action = next_allowed[np.argmax(self.Q[s_t][next_allowed])]

                alpha = 1 / (1 + self.visit_count[s_prev, action])
                self.Q[s_prev, action] += alpha * (reward + self.gamma * self.Q[s_t, next_action] - self.Q[s_prev, action])
                self.visit_count[s_prev, action] += 1

                s_prev = s_t
                action = next_action
            
            if len(past_fifteen_rewards) == 15 and i + 1 > self.min_epochs:
                rewards_array = np.array(past_fifteen_rewards)

                rolling_stds = []
                # 6 rolling windows of size 10
                for j in range(15 - 10 + 1):
                    rolling_stds.append(np.std(rewards_array[j:j + 10]))

                if all(std < 0.05 for std in rolling_stds):
                    if show_log:
                        print(f"Epoch {i + 1}: Total reward = {round(total_reward, 3)}, Past Ten Total Rewards Std = {rolling_stds[-1]}")
                        print(f"Model stabilizes at epoch {i + 1}")    
                    return True
                if show_log:
                    print(f"Epoch {i + 1}: Total reward = {round(total_reward, 3)}, Past Ten Total Rewards Std = {rolling_stds[-1]}")

            past_fifteen_rewards.append(total_reward)
            if len(past_fifteen_rewards) > 15:
                past_fifteen_rewards.popleft()

            # Reduce exploration on next epoch
            current_epsilon = max(current_epsilon * self.epsilon_decay_rate, self.min_epsilon)
        
            if i == self.max_epochs:
                if show_log:
                    print("Failed to stabilize")
                return False

    def get_action(self, state, action_space):
        return action_space[np.argmax(self.Q[state][action_space])]