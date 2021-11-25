import numpy as np
import torch


class ExperienceReplayMemory(object):
    def __init__(self, capacity, input_dims):
        self.capacity = capacity
        self.state_memory = np.zeros((self.capacity, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.capacity, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.capacity, dtype=np.int32)
        self.reward_memory = np.zeros(self.capacity, dtype=np.float32)
        self.done_memory = np.zeros(self.capacity, dtype=np.bool)
        self.memory_counter = 0

    def push(self, last_state, action, reward, new_state, done):
        index = self.memory_counter % self.capacity
        self.state_memory[index] = last_state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.done_memory[index] = done

        self.memory_counter += 1

    # Metodo per ritornare un batch di transizioni dalla memoria per aggiornare la rete
    def sample(self, batch_size):
        max_mem = min(self.memory_counter, self.capacity)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        state_batch = torch.tensor(self.state_memory[batch])
        new_state_batch = torch.tensor(self.new_state_memory[batch])
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(self.reward_memory[batch])
        terminal_batch = torch.tensor(self.done_memory[batch])
        return state_batch, new_state_batch, action_batch, reward_batch, terminal_batch
