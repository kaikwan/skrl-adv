import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.spaces.torch import unflatten_tensorized_space

# define shared model (stochastic and deterministic models) using mixins
class SharedRNN(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", initial_log_std=0,
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=128):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")
        
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(
                                #  nn.LazyLinear(out_features=128),
                                #  nn.ELU(),
                                 nn.LazyLinear(out_features=64),
                                 nn.ELU())

        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=initial_log_std), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)


    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)


    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs, role=""):
        if role == "policy":
            states = inputs["states"]
            terminated = inputs.get("terminated", None)
            hidden_states = inputs["rnn"][0]

            # training
            if self.training:
                rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
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
                        rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                        hidden_states[:, (terminated[:,i1-1]), :] = 0
                        rnn_outputs.append(rnn_output)

                    rnn_output = torch.cat(rnn_outputs, dim=1)
                # no need to reset the RNN state in the sequence
                else:
                    rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
            # rollout
            else:
                rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

            # flatten the RNN output
            rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

            return self.policy_layer(self.net(rnn_output)), self.log_std_parameter, {"rnn": [hidden_states]}
        elif role == "value":
            states = inputs["states"]
            terminated = inputs.get("terminated", None)
            hidden_states = inputs["rnn"][0]

            # training
            if self.training:
                rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

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
                        rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                        hidden_states[:, (terminated[:,i1-1]), :] = 0
                        rnn_outputs.append(rnn_output)

                    rnn_output = torch.cat(rnn_outputs, dim=1)
                # no need to reset the RNN state in the sequence
                else:
                    rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
            # rollout
            else:
                rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

            # flatten the RNN output
            rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

            return self.value_layer(self.net(rnn_output)), {"rnn": [hidden_states]}