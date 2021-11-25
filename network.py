import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):

    def __init__(self, input_size, num_action, lr):
        super(Network, self).__init__()
        self.input_size = input_size
        self.num_action = num_action
        self.fullConnection1 = nn.Linear(input_size, 264)
        self.fullConnection2 = nn.Linear(264, 264)
        self.fullConnection3 = nn.Linear(264, num_action)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    # Implementiamo la forward propagation dove mandiamo l'input agli hidden layer
    def forward(self, state):
        x = F.relu(self.fullConnection1(state))
        x = F.relu(self.fullConnection2(x))
        output = self.fullConnection3(x)
        return output
