import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, sequence_length=128, num_envs=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) # 36, 
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.num_envs = num_envs
        self.hidden_size = hidden_size
        self.training = False

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"lstm": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.netlstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

    def compute(self, inputs):
        states = inputs["states"] # num_envs * rollouts / minibatch, observation_size
        terminated = inputs.get("terminated", None) # num_envs * rollouts / minibatch 
        hidden_states = inputs["lstm"][0] # num_layers, num_envs * rollouts / minibatch, hidden_size
        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length, Hin=observation_size
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.lstm(rnn_input[:,i0:i1,:], hidden_states) # rnn_input = (N, i0:i1, Hin), hidden_states = (D * num_layers, N, Hout)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.lstm(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.lstm(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.fc(rnn_output), {"lstm": [hidden_states]}
