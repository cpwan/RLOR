"""
Problem specific node embedding for static feature.
"""

import torch
import torch.nn as nn


def AutoEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "tsp": TSPEmbedding,
        "cvrp": VRPEmbedding,
        "sdvrp": VRPEmbedding,
        "pctsp": PCTSPEmbedding,
        "op": OPEmbedding,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding


class TSPEmbedding(nn.Module):
    """
    Embedding for the traveling salesman problem.

    Args:
        embedding_dim: dimension of output
    Inputs: input
        * **input** : [batch, n_customer, 2]
    Outputs: out
        * **out** : [batch, n_customer, embedding_dim]
    """

    def __init__(self, embedding_dim):
        super(TSPEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.context_dim = 2 * embedding_dim  # Embedding of first and last node

        self.init_embed = nn.Linear(node_dim, embedding_dim)

    def forward(self, input):
        out = self.init_embed(input)
        return out


class VRPEmbedding(nn.Module):
    """
    Embedding for the capacitated vehicle routing problem.
    The shape of tensors in ``input`` is summarized as following:

    +-----------+-------------------------+
    | key       | size of tensor          |
    +===========+=========================+
    | 'loc'     | [batch, n_customer, 2]  |
    +-----------+-------------------------+
    | 'depot'   | [batch, 2]              |
    +-----------+-------------------------+
    | 'demand'  | [batch, n_customer, 1]  |
    +-----------+-------------------------+

    Args:
        embedding_dim: dimension of output
    Inputs: input
        * **input** : dict of ['loc', 'depot', 'demand']
    Outputs: out
        * **out** : [batch, n_customer+1, embedding_dim]
    """

    def __init__(self, embedding_dim):
        super(VRPEmbedding, self).__init__()
        node_dim = 3  # x, y, demand

        self.context_dim = embedding_dim + 1  # Embedding of last node + remaining_capacity

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding

    def forward(self, input):  # dict of 'loc', 'demand', 'depot'
        # batch, 1, 2 -> batch, 1, embedding_dim
        depot_embedding = self.init_embed_depot(input["depot"])[:, None, :]
        # [batch, n_customer, 2, batch, n_customer, 1]  -> batch, n_customer, embedding_dim
        node_embeddings = self.init_embed(
            torch.cat((input["loc"], input["demand"][:, :, None]), -1)
        )
        # batch, n_customer+1, embedding_dim
        out = torch.cat((depot_embedding, node_embeddings), 1)
        return out


class PCTSPEmbedding(nn.Module):
    """
    Embedding for the prize collecting traveling salesman problem.
    The shape of tensors in ``input`` is summarized as following:

    +------------------------+-------------------------+
    | key                    | size of tensor          |
    +========================+=========================+
    | 'loc'                  | [batch, n_customer, 2]  |
    +------------------------+-------------------------+
    | 'depot'                | [batch, 2]              |
    +------------------------+-------------------------+
    | 'deterministic_prize'  | [batch, n_customer, 1]  |
    +------------------------+-------------------------+
    | 'penalty'              | [batch, n_customer, 1]  |
    +------------------------+-------------------------+

    Args:
        embedding_dim: dimension of output
    Inputs: input
        * **input** : dict of ['loc', 'depot', 'deterministic_prize', 'penalty']
    Outputs: out
        * **out** : [batch, n_customer+1, embedding_dim]
    """

    def __init__(self, embedding_dim):
        super(PCTSPEmbedding, self).__init__()
        node_dim = 4  # x, y, prize, penalty
        self.context_dim = embedding_dim + 1  # Embedding of last node + remaining prize to collect

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding

    def forward(self, input):  # dict of 'loc', 'deterministic_prize', 'penalty', 'depot'
        # batch, 1, 2 -> batch, 1, embedding_dim
        depot_embedding = self.init_embed_depot(input["depot"])[:, None, :]
        # [batch, n_customer, 2, batch, n_customer, 1, batch, n_customer, 1]  -> batch, n_customer, embedding_dim
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    input["loc"],
                    input["deterministic_prize"][:, :, None],
                    input["penalty"][:, :, None],
                ),
                -1,
            )
        )
        # batch, n_customer+1, embedding_dim
        out = torch.cat((depot_embedding, node_embeddings), 1)
        return out


class OPEmbedding(nn.Module):
    """
    Embedding for the orienteering problem.
    The shape of tensors in ``input`` is summarized as following:

    +----------+-------------------------+
    | key      | size of tensor          |
    +==========+=========================+
    | 'loc'    | [batch, n_customer, 2]  |
    +----------+-------------------------+
    | 'depot'  | [batch, 2]              |
    +----------+-------------------------+
    | 'prize'  | [batch, n_customer, 1]  |
    +----------+-------------------------+

    Args:
        embedding_dim: dimension of output
    Inputs: input
        * **input** : dict of ['loc', 'depot', 'prize']
    Outputs: out
        * **out** : [batch, n_customer+1, embedding_dim]
    """

    def __init__(self, embedding_dim):
        super(OPEmbedding, self).__init__()
        node_dim = 3  # x, y, prize
        self.context_dim = embedding_dim + 1  # Embedding of last node + remaining prize to collect

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding

    def forward(self, input):  # dict of 'loc', 'prize', 'depot'
        # batch, 1, 2 -> batch, 1, embedding_dim
        depot_embedding = self.init_embed_depot(input["depot"])[:, None, :]
        # [batch, n_customer, 2, batch, n_customer, 1, batch, n_customer, 1]  -> batch, n_customer, embedding_dim
        node_embeddings = self.init_embed(
            torch.cat((input["loc"], input["prize"][:, :, None]), -1)
        )
        # batch, n_customer+1, embedding_dim
        out = torch.cat((depot_embedding, node_embeddings), 1)
        return out
