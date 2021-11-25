import torch
import torch.nn.functional as F
from experience_replay import ExperienceReplayMemory
from network import Network
import numpy as np


class Agent:
    def __init__(self, num_input, num_action, gamma, lr, batch_size, t):
        self.gamma = gamma
        self.t = t
        self.batch_size = batch_size
        self.lr = lr
        self.network = Network(num_input, num_action, lr)
        self.reward_window = []
        self.memory = ExperienceReplayMemory(capacity=100000, input_dims=[num_input])

    def select_action(self, observation):
        # Softmax
        state = torch.tensor([observation])
        probs = F.softmax(self.network.forward(state) * self.t, dim=1)  # 100 regola il dilemma Exloration vs Exploitation. Più basso = più esplorazione
        action = probs.multinomial(num_samples=1).item()  # fa una estrazione dalle probabilità calcolate sopra
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
        q_state = self.network.forward(state_batch)[batch_index, action_batch]  # ritorna i q value per quel batch_action
        # Per prendere il max si può fare anche cosi
        # q_next_state = self.network.forward(new_state_batch).detach().max(1)[0]
        q_next_state = self.network.forward(new_state_batch)
        q_next_state[done_batch] = 0.0
        q_target = self.gamma * torch.max(q_next_state, dim=1)[0] + reward_batch
        td_loss = F.smooth_l1_loss(q_state, q_target)  # calcolo il loss
        td_loss.backward()  # aggiorno i gradienti
        self.network.optimizer.step()  # aggiorno le weights
