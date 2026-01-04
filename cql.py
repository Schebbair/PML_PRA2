from collections import defaultdict
import random
from typing import List, DefaultDict

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim

class CQL:
    """
    Centralized Q-Learning Agent
    """
    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        epsilon: float = 1.0,
        **kwargs,
    ):
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.joint_n_acts = np.prod(self.n_acts)

        #just a q_table
        self.q_table: List[DefaultDict] = defaultdict(lambda: 0)

    #converts list of actions [a1,a2] to a single integer
    def joint_index(self, actions: List[int]) -> int:
        idx = 0
        multiplier = 1
        for i in reversed(range(self.num_agents)):
            idx += actions[i] * multiplier
            multiplier *= self.n_acts[i]
        return idx
    
    #converts integer back to list of actions
    def actions_from_idx(self, index: int) -> List[int]:
        actions = []
        for i in reversed(range(self.num_agents)):
            actions.append(index % self.n_acts[i])
            index //= self.n_acts[i]
        return list(reversed(actions))
    
    def act(self, obss) -> List[int]:
        """
        Implement the epsilon-greedy action selection here for stateless task

        **IMPLEMENT THIS FUNCTION**

        :param obss (List): list of observations for each agent
        :return (List[int]): index of selected action for each agent
        """
        joint_obs = tuple(obss)

        if random.random() < self.epsilon:
            joint_action_idx = random.randint(0, self.joint_n_acts -1)
        else:
                
            q_values = []

            for idx in range(self.joint_n_acts):
                value = self.q_table[str((joint_obs, idx))]
                q_values.append(value)

            joint_action_idx = int(np.argmax(q_values))
        
        return self.actions_from_idx(joint_action_idx)

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """
        Updates the Q-tables based on agents' experience

        **IMPLEMENT THIS FUNCTION**

        :param obss (List[np.ndarray]): list of observations for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray]): list of observations after taking the action for each agent
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (List[float]): updated Q-values for current actions of each agent
        """
        ### PUT YOUR CODE HERE ###

        joint_obs = tuple(obss)
        joint_next_obs = tuple(n_obss)
        joint_action_idx = self.joint_index(actions)

        total_reward = sum(rewards)

        key = str((joint_obs, joint_action_idx))
        current_q = self.q_table[key]

        if done:
            max_next_q = 0.0
        else:

            next_q = []
            
            for idx in range(self.joint_n_acts):
                value = self.q_table[str((joint_next_obs, idx))]
                next_q.append(value)
            
            max_next_q = max(next_q)
        
        td_target = total_reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error
                
        self.q_table[key] = new_q

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99
