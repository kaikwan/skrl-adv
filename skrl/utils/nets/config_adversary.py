import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_positions=6):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_positions * 3)
        self.num_clutter_objects = num_positions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Clamp output
        x = torch.clamp(x, -1, 1)
        assert all(x >= -1) and all(x <= 1), f"Output not clamped: {x}"

        return x
    
    def sample(self):
        # input = torch.randn(size=(self.input_dim,))
        # return self.forward(input)
        return torch.randn(size=(self.num_clutter_objects * 3,)) * 2 - 1
