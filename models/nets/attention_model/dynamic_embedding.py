"""
Problem specific node embedding for dynamic feature.
"""

import torch.nn as nn


def AutoDynamicEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "tsp": NonDyanmicEmbedding,
        "cvrp": NonDyanmicEmbedding,
        "sdvrp": SDVRPDynamicEmbedding,
        "pctsp": NonDyanmicEmbedding,
        "op": NonDyanmicEmbedding,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding


class SDVRPDynamicEmbedding(nn.Module):
    """
    Embedding for dynamic node feature for the split delivery vehicle routing problem.

    It is implemented as a linear projection of the demands left in each node.

    Args:
        embedding_dim: dimension of output
    Inputs: state
        * **state** : a class that provide ``state.demands_with_depot`` tensor
    Outputs: glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
        * **glimpse_key_dynamic** : [batch, graph_size, embedding_dim]
        * **glimpse_val_dynamic** : [batch, graph_size, embedding_dim]
        * **logit_key_dynamic** : [batch, graph_size, embedding_dim]

    """

    def __init__(self, embedding_dim):
        super(SDVRPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(1, 3 * embedding_dim, bias=False)

    def forward(self, state):
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            state.demands_with_depot[:, 0, :, None].clone()
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class NonDyanmicEmbedding(nn.Module):
    """
    Embedding for problems that do not have any dynamic node feature.

    It is implemented as simply returning zeros.

    Args:
        embedding_dim: dimension of output
    Inputs: state
        * **state** : not used, just for consistency
    Outputs: glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
        * **glimpse_key_dynamic** : [batch, graph_size, embedding_dim]
        * **glimpse_val_dynamic** : [batch, graph_size, embedding_dim]
        * **logit_key_dynamic** : [batch, graph_size, embedding_dim]

    """

    def __init__(self, embedding_dim):
        super(NonDyanmicEmbedding, self).__init__()

    def forward(self, state):
        return 0, 0, 0
