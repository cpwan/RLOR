import gym
import numpy as np
from gym import spaces

from .tsp_data import TSPDataset


def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)


def dist(loc1, loc2):
    return ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5


class TSPVectorEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.max_nodes = 50
        self.n_traj = 50
        # if eval_data==True, load from 'test' set, the '0'th data
        self.eval_data = False
        self.eval_partition = "test"
        self.eval_data_idx = 0
        assign_env_config(self, kwargs)

        obs_dict = {"observations": spaces.Box(low=0, high=1, shape=(self.max_nodes, 2))}
        obs_dict["action_mask"] = spaces.MultiBinary(
            [self.n_traj, self.max_nodes]
        )  # 1: OK, 0: cannot go
        obs_dict["first_node_idx"] = spaces.MultiDiscrete([self.max_nodes] * self.n_traj)
        obs_dict["last_node_idx"] = spaces.MultiDiscrete([self.max_nodes] * self.n_traj)
        obs_dict["is_initial_action"] = spaces.Discrete(1)

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.max_nodes] * self.n_traj)
        self.reward_space = None

        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.visited = np.zeros((self.n_traj, self.max_nodes), dtype=bool)
        self.num_steps = 0
        self.last = np.zeros(self.n_traj, dtype=int)  # idx of the first elem
        self.first = np.zeros(self.n_traj, dtype=int)  # idx of the first elem

        if self.eval_data:
            self._load_orders()
        else:
            self._generate_orders()
        self.state = self._update_state()
        self.info = {}
        self.done = False
        return self.state

    def _load_orders(self):
        self.nodes = np.array(TSPDataset[self.eval_partition, self.max_nodes, self.eval_data_idx])

    def _generate_orders(self):
        self.nodes = np.random.rand(self.max_nodes, 2)

    def step(self, action):

        self._go_to(action)  # Go to node 'action', modify the reward
        self.num_steps += 1
        self.state = self._update_state()

        # need to revisit the first node after visited all other nodes
        self.done = (action == self.first) & self.is_all_visited()

        return self.state, self.reward, self.done, self.info

    # Euclidean cost function
    def cost(self, loc1, loc2):
        return dist(loc1, loc2)

    def is_all_visited(self):
        # assumes no repetition in the first `max_nodes` steps
        return self.visited[:, :].all(axis=1)

    def _go_to(self, destination):
        dest_node = self.nodes[destination]
        if self.num_steps != 0:
            dist = self.cost(dest_node, self.nodes[self.last])
        else:
            dist = np.zeros(self.n_traj)
            self.first = destination

        self.last = destination

        self.visited[np.arange(self.n_traj), destination] = True
        self.reward = -dist

    def _update_state(self):
        obs = {"observations": self.nodes}  # n x 2 array
        obs["action_mask"] = self._update_mask()
        obs["first_node_idx"] = self.first
        obs["last_node_idx"] = self.last
        obs["is_initial_action"] = self.num_steps == 0
        return obs

    def _update_mask(self):
        # Only allow to visit unvisited nodes
        action_mask = ~self.visited
        # can only visit first node when all nodes have been visited
        action_mask[np.arange(self.n_traj), self.first] |= self.is_all_visited()
        return action_mask
