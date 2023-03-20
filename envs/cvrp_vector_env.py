import gym
import numpy as np
from gym import spaces

from .vrp_data import VRPDataset


def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)


def dist(loc1, loc2):
    return ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5


class CVRPVectorEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.max_nodes = 50
        self.capacity_limit = 40
        self.n_traj = 50
        # if eval_data==True, load from 'test' set, the '0'th data
        self.eval_data = False
        self.eval_partition = "test"
        self.eval_data_idx = 0
        self.demand_limit = 10
        assign_env_config(self, kwargs)

        obs_dict = {"observations": spaces.Box(low=0, high=1, shape=(self.max_nodes, 2))}
        obs_dict["depot"] = spaces.Box(low=0, high=1, shape=(2,))
        obs_dict["demand"] = spaces.Box(low=0, high=1, shape=(self.max_nodes,))
        obs_dict["action_mask"] = spaces.MultiBinary(
            [self.n_traj, self.max_nodes + 1]
        )  # 1: OK, 0: cannot go
        obs_dict["last_node_idx"] = spaces.MultiDiscrete([self.max_nodes + 1] * self.n_traj)
        obs_dict["current_load"] = spaces.Box(low=0, high=1, shape=(self.n_traj,))

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.max_nodes + 1] * self.n_traj)
        self.reward_space = None

        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def _STEP(self, action):

        self._go_to(action)  # Go to node 'action', modify the reward
        self.num_steps += 1
        self.state = self._update_state()

        # need to revisit the first node after visited all other nodes
        self.done = (action == 0) & self.is_all_visited()

        return self.state, self.reward, self.done, self.info

    # Euclidean cost function
    def cost(self, loc1, loc2):
        return dist(loc1, loc2)

    def is_all_visited(self):
        # assumes no repetition in the first `max_nodes` steps
        return self.visited[:, 1:].all(axis=1)

    def _update_state(self):
        obs = {"observations": self.nodes[1:]}  # n x 2 array
        obs["depot"] = self.nodes[0]
        obs["action_mask"] = self._update_mask()
        obs["demand"] = self.demands
        obs["last_node_idx"] = self.last
        obs["current_load"] = self.load
        return obs

    def _update_mask(self):
        # Only allow to visit unvisited nodes
        action_mask = ~self.visited

        # can only visit depot when last node is not depot or all visited
        action_mask[:, 0] |= self.last != 0
        action_mask[:, 0] |= self.is_all_visited()

        # not allow visit nodes with higher demand than capacity
        action_mask[:, 1:] &= self.demands <= (
            self.load.reshape(-1, 1) + 1e-5
        )  # to handle the floating point subtraction precision

        return action_mask

    def _RESET(self):
        self.visited = np.zeros((self.n_traj, self.max_nodes + 1), dtype=bool)
        self.visited[:, 0] = True
        self.num_steps = 0
        self.last = np.zeros(self.n_traj, dtype=int)  # idx of the last elem
        self.load = np.ones(self.n_traj, dtype=float)  # current load

        if self.eval_data:
            self._load_orders()
        else:
            self._generate_orders()
        self.state = self._update_state()
        self.info = {}
        self.done = np.array([False] * self.n_traj)
        return self.state

    def _load_orders(self):
        data = VRPDataset[self.eval_partition, self.max_nodes, self.eval_data_idx]
        self.nodes = np.concatenate((data["depot"][None, ...], data["loc"]))
        self.demands = data["demand"]
        self.demands_with_depot = self.demands.copy()

    def _generate_orders(self):
        self.nodes = np.random.rand(self.max_nodes + 1, 2)
        self.demands = (
            np.random.randint(low=1, high=self.demand_limit, size=self.max_nodes)
            / self.capacity_limit
        )
        self.demands_with_depot = self.demands.copy()

    def _go_to(self, destination):
        dest_node = self.nodes[destination]
        dist = self.cost(dest_node, self.nodes[self.last])
        self.last = destination
        self.load[destination == 0] = 1
        self.load[destination > 0] -= self.demands[destination[destination > 0] - 1]
        self.demands_with_depot[destination[destination > 0] - 1] = 0
        self.visited[np.arange(self.n_traj), destination] = True
        self.reward = -dist

    def step(self, action):
        # return last state after done,
        # for the sake of PPO's abuse of ff on done observation
        # see https://github.com/opendilab/DI-engine/issues/497
        # Not needed for CleanRL
        # if self.done.all():
        #     return self.state, self.reward, self.done, self.info

        return self._STEP(action)

    def reset(self):
        return self._RESET()
