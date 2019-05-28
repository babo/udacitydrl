import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 4 * state_size)
        self.fc2 = nn.Linear(4 * state_size, 4 * action_size)
        self.fc3 = nn.Linear(4 * action_size, 1)
        self.fc4 = nn.Linear(4 * action_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        a = self.fc4(state)
        state = self.fc3(state) + (a - a.mean(1, keepdim=True))
        return state
