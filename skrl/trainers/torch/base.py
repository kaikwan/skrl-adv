from typing import List, Optional, Union

import atexit
import sys
import tqdm

import torch
import torch.nn as nn

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.agents.torch.ppo import PPO
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory


# TODO: find better way to import this without hard coding it here
# although it seems like this is hard coded at the top of the files for SKRL demos too
# define the shared model
class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        # shared layers/network
        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU())

        # separated layers ("policy")
        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # separated layer ("value")
        self.value_layer = nn.Linear(32, 1)

        # Shared value is 0
        self._shared_output = None

    # override the .act(...) method to disambiguate its call
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    # forward the input to compute model output according to the specified role
    def compute(self, inputs, role):
        if role == "policy":
            # save shared layers/network output to perform a single forward-pass
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            # use saved shared layers/network output to perform a single forward-pass, if it was saved
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None  # reset saved shared output to prevent the use of erroneous data in subsequent steps
            return self.value_layer(shared_output), {}

def generate_equally_spaced_scopes(num_envs: int, num_simultaneous_agents: int) -> List[int]:
    """Generate a list of equally spaced scopes for the agents

    :param num_envs: Number of environments
    :type num_envs: int
    :param num_simultaneous_agents: Number of simultaneous agents
    :type num_simultaneous_agents: int

    :raises ValueError: If the number of simultaneous agents is greater than the number of environments

    :return: List of equally spaced scopes
    :rtype: List[int]
    """
    scopes = [int(num_envs / num_simultaneous_agents)] * num_simultaneous_agents
    if sum(scopes):
        scopes[-1] += num_envs - sum(scopes)
    else:
        raise ValueError(
            f"The number of simultaneous agents ({num_simultaneous_agents}) is greater than the number of environments ({num_envs})"
        )
    return scopes


class Trainer:
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope if agents_scope is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")
        self.stochastic_evaluation = self.cfg.get("stochastic_evaluation", False)

        self.initial_timestep = 0

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # set positioning strategy (have local copy for ease of access)
        self.positioning_strategy = env._env.env.cfg.positioning_strategy
        self.adversary_active = self.positioning_strategy == "pure_adversary"

        # setup adversary
        num_inputs = 12 # arbitrary number of inputs, is a noise vector to condition on
        num_outputs = (self.env._env.env.num_clutter_objects + 1) * 3 # clutter + main object
        adversary_shared_model = SharedModel(num_inputs, num_outputs, device=env.device)
        models = { # TODO: not sure if this is supposed to be shared
            "policy": adversary_shared_model,
            "value": adversary_shared_model
        }
        adversary_cfg = {
            "rollouts": 1
        }

        self.adversary = PPO(
            models=models,
            device=env.device,
            observation_space=num_inputs,
            action_space=num_outputs,
            memory=RandomMemory(num_envs=self.env.num_envs, memory_size=adversary_cfg["rollouts"], device=env.device),
            cfg=adversary_cfg
        )
        self.adversary.init()

        # register environment closing if configured
        if self.close_environment_at_exit:

            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.torch.is_distributed:
            if config.torch.rank:
                self.disable_progressbar = True

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # parallel agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    logger.warning("The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError(
                            f"The number of agents ({len(self.agents)}) is greater than the number of parallelizable environments ({self.env.num_envs})"
                        )
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError(
                        f"The number of agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.agents_scope)})"
                    )
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError(
                        f"The scopes ({sum(self.agents_scope)}) don't cover the number of parallelizable environments ({self.env.num_envs})"
                    )
                # generate agents' scopes
                index = 0
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        ADVERSARY_ACTION_SPACE = self.env._env.env.adversary_action.shape[-1]
        prev_rand_state = torch.randn((self.env.num_envs, 12), device=self.env.device)
        prev_adversary_action = torch.zeros((self.env.num_envs, ADVERSARY_ACTION_SPACE), device=self.env.device)
        for i in range(self.env.num_envs):
            rand_state = torch.randn(12, device=self.env.device)

            # Sample adversary action
            if self.positioning_strategy == "domain_rand":
                adversary_action = torch.rand(ADVERSARY_ACTION_SPACE, device=self.env.device) * 2 - 1
            elif self.positioning_strategy == "domain_rand_restricted":
                adversary_action = torch.rand(ADVERSARY_ACTION_SPACE, device=self.env.device) * 1.75 - 1
            elif self.positioning_strategy == "pure_adversary":
                adversary_action = self.adversary.act(rand_state, timestep=0, timesteps=self.timesteps)[0]
            else:
                raise ValueError(f"Invalid positioning strategy: {self.positioning_strategy}")

            # Update tracking values and environment values
            self.env._env.env.adversary_action[i] = adversary_action
            prev_rand_state[i] = rand_state
            prev_adversary_action[i] = adversary_action
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            self.adversary.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            states = next_states
            if timestep == 0 or terminated.any() or truncated.any():
                with torch.no_grad():
                    # loop through all envs that need to be reset and update their adversary action
                    reset_env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                    curr_rand_state = torch.randn((self.env.num_envs, 12), device=self.env.device)
                    curr_adversary_action = torch.zeros((self.env.num_envs, ADVERSARY_ACTION_SPACE), device=self.env.device)
                    for i in reset_env_ids:
                        rand_state = torch.randn(12, device=self.env.device)

                        # Sample adversary action
                        if self.positioning_strategy == "domain_rand":
                            adversary_action = torch.rand(ADVERSARY_ACTION_SPACE, device=self.env.device) * 2 - 1
                        elif self.positioning_strategy == "domain_rand_restricted":
                            adversary_action = torch.rand(ADVERSARY_ACTION_SPACE, device=self.env.device) * 1.75 - 1
                        elif self.positioning_strategy == "pure_adversary":
                            adversary_action = self.adversary.act(rand_state, timestep=timestep, timesteps=self.timesteps)[0]
                        else:
                            raise ValueError(f"Invalid positioning strategy: {self.positioning_strategy}")

                        # Update tracking values and environment values
                        self.env._env.env.adversary_action[i] = adversary_action
                        curr_rand_state[i] = rand_state
                        curr_adversary_action[i] = adversary_action
                
                    if timestep > 0 and self.adversary_active:
                        # TODO: unsure about assignment of states and next_states
                        self.adversary.record_transition(
                            states=prev_rand_state,
                            actions=prev_adversary_action,
                            rewards=(-1 * rewards),
                            next_states=prev_rand_state,
                            terminated=terminated,
                            truncated=truncated,
                            infos=infos,
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )
                        prev_rand_state = curr_rand_state
                        prev_adversary_action = curr_adversary_action

                if timestep > 0 and self.adversary_active:
                    self.adversary.post_interaction(timestep=timestep, timesteps=self.timesteps)
                        

    def single_agent_eval(self) -> None:
        """Evaluate agent

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                actions = outputs[0] if self.stochastic_evaluation else outputs[-1].get("mean_actions", outputs[0])

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def multi_agent_train(self) -> None:
        """Train multi-agents

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states

    def multi_agent_eval(self) -> None:
        """Evaluate multi-agents

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                actions = (
                    outputs[0]
                    if self.stochastic_evaluation
                    else {k: outputs[-1][k].get("mean_actions", outputs[0][k]) for k in outputs[-1]}
                )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states
