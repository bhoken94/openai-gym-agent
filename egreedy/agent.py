import torch
import torch.nn.functional as F
from experience_replay import ExperienceReplayMemory
from network import Network
import numpy as np


class Agent:
    def __init__(self, num_input, num_action, gamma, lr, batch_size, epsilon, eps_min=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = eps_min
        self.epsilon_dec = eps_dec
        self.batch_size = batch_size
        self.lr = lr
        self.network = Network(num_input, num_action, lr)
        self.reward_window = []
        self.memory = ExperienceReplayMemory(capacity=100000, input_dims=[num_input])

    def select_action(self, observation):
        # e-greedy
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation])
            actions = self.network.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.network.num_action)
        return action

    def store_transition(self, new_state, action, reward, last_state, done):
        self.memory.push(last_state, action, reward, new_state, done)

    # Funzione che prende in entrata i batch degli stati, azioni e reward
    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return
        self.network.optimizer.zero_grad()  # resetto i gradienti dell'optimizer
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch, new_state_batch, action_batch, reward_batch, done_batch = self.memory.sample(self.batch_size)
        q_state = self.network.forward(state_batch)[
            batch_index, action_batch]  # ritorna i q value per quel batch_action
        q_state_next = self.network.forward(new_state_batch)
        # q_next_state = self.network.forward(new_state_batch).detach().max(1)[0]  # ritorna il max dei q value ritornati
        q_state_next[done_batch] = 0.0
        q_target = self.gamma * torch.max(q_state_next, dim=1)[0] + reward_batch
        td_loss = F.smooth_l1_loss(q_state, q_target)  # calcolo il loss
        td_loss.backward()  # aggiorno i gradienti
        self.network.optimizer.step()  # aggiorno le weights
        # e-greedy
        self.epsilon = self.epsilon - self.epsilon_dec \
            if self.epsilon > self.epsilon_min else self.epsilon_min
