"""
Problem specific global embedding for global context.
"""

import torch
from torch import nn


def AutoContext(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "tsp": TSPContext,
        "cvrp": VRPContext,
        "sdvrp": VRPContext,
        "pctsp": PCTSPContext,
        "op": OPContext,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding


def _gather_by_index(source, index):
    """
    target[i,1,:] = source[i,index[i],:]
    Inputs:
        source: [B x H x D]
        index: [B x 1] or [B]
    Outpus:
        target: [B x 1 x D]
    """

    target = torch.gather(source, 1, index.unsqueeze(-1).expand(-1, -1, source.size(-1)))
    return target


class PrevNodeContext(nn.Module):
    """
    Abstract class for Context.
    Any subclass, by default, will return a concatenation of

    +---------------------+-----------------+
    | prev_node_embedding | state_embedding |
    +---------------------+-----------------+

    The ``prev_node_embedding`` is the node embedding of the last visited node.
    It is obtained by ``_prev_node_embedding`` method.
    It requires ``state.get_current_node()`` to provide the index of the last visited node.

    The ``state_embedding`` is the global context we want to include, such as the remaining capacity in VRP.
    It is obtained by ``_state_embedding`` method.
    It is not implemented. The subclass of this abstract class needs to implement this method.

    Args:
        problem: an object defining the settings of the environment
        context_dim: the dimension of the output
    Inputs: embeddings, state
        * **embeddings** : [batch x graph size x embed dim]
        * **state**: An object providing observations in the environment. \
                    Needs to supply ``state.get_current_node()``
    Outputs: context_embedding
        * **context_embedding**: [batch x 1 x context_dim]

    """

    def __init__(self, context_dim):
        super(PrevNodeContext, self).__init__()
        self.context_dim = context_dim

    def _prev_node_embedding(self, embeddings, state):
        current_node = state.get_current_node()
        prev_node_embedding = _gather_by_index(embeddings, current_node)
        return prev_node_embedding

    def _state_embedding(self, embeddings, state):
        raise NotImplementedError("Please implement the embedding for your own problem.")

    def forward(self, embeddings, state):
        prev_node_embedding = self._prev_node_embedding(embeddings, state)
        state_embedding = self._state_embedding(embeddings, state)
        # Embedding of previous node + remaining capacity
        context_embedding = torch.cat((prev_node_embedding, state_embedding), -1)
        return context_embedding


class TSPContext(PrevNodeContext):
    """
    Context node embedding for traveling salesman problem.
    Return a concatenation of

    +------------------------+---------------------------+
    | first node's embedding | previous node's embedding |
    +------------------------+---------------------------+

    .. note::
        Subclass of :class:`.PrevNodeContext`. The argument, inputs, outputs follow the same specification.

        In addition to supplying  ``state.get_current_node()`` for the index of the previous visited node.
        The input ``state`` needs to supply ``state.first_a`` for the index of the first visited node.

    .. warning::
        The official implementation concatenates the context with [first node, prev node].
        However, if we follow the paper closely, it should instead be [prev node, first node].
        Please check ``forward_code`` and ``forward_paper`` for the different implementations.
        We follow the official implementation in this class.
    """

    def __init__(self, context_dim):
        super(TSPContext, self).__init__(context_dim)
        self.W_placeholder = nn.Parameter(torch.Tensor(self.context_dim).uniform_(-1, 1))

    def _state_embedding(self, embeddings, state):
        first_node = state.first_a
        state_embedding = _gather_by_index(embeddings, first_node)
        return state_embedding

    def forward_paper(self, embeddings, state):
        batch_size = embeddings.size(0)
        if state.i.item() == 0:
            context_embedding = self.W_placeholder[None, None, :].expand(
                batch_size, 1, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = super().forward(embeddings, state)
        return context_embedding

    def forward_code(self, embeddings, state):
        batch_size = embeddings.size(0)
        if state.i.item() == 0:
            context_embedding = self.W_placeholder[None, None, :].expand(
                batch_size, 1, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = _gather_by_index(
                embeddings, torch.cat([state.first_a, state.get_current_node()], -1)
            ).view(batch_size, 1, -1)
        return context_embedding

    def forward_vectorized(self, embeddings, state):
        n_queries = state.states["first_node_idx"].shape[-1]
        batch_size = embeddings.size(0)
        out_shape = (batch_size, n_queries, self.context_dim)

        switch = state.is_initial_action  # tensor, 1 if is initial action
        switch = switch[:, None, None].expand(out_shape)  # mask for each data

        # only used for the first action
        placeholder_embedding = self.W_placeholder[None, None, :].expand(out_shape)
        # used after first action
        indexes = torch.stack([state.first_a, state.get_current_node()], -1).flatten(-2)
        normal_embedding = _gather_by_index(embeddings, indexes).view(out_shape)

        context_embedding = switch * placeholder_embedding + (~switch) * normal_embedding
        return context_embedding

    def forward(self, embeddings, state):
        return self.forward_vectorized(embeddings, state)


class VRPContext(PrevNodeContext):
    """
    Context node embedding for capacitated vehicle routing problem.
    Return a concatenation of

    +---------------------------+--------------------+
    | previous node's embedding | remaining capacity |
    +---------------------------+--------------------+

    .. note::
        Subclass of :class:`.PrevNodeContext`. The argument, inputs, outputs follow the same specification.

        In addition to supplying  ``state.get_current_node()`` for the index of the previous visited node.
        The input ``state`` needs to supply ``state.VEHICLE_CAPACITY`` and ``state.used_capacity``
        for calculating the remaining capcacity.
    """

    def __init__(self, context_dim):
        super(VRPContext, self).__init__(context_dim)

    def _state_embedding(self, embeddings, state):
        state_embedding = state.VEHICLE_CAPACITY - state.used_capacity[:, :, None]
        return state_embedding


class PCTSPContext(PrevNodeContext):
    """
    Context node embedding for prize collecting traveling salesman problem.
    Return a concatenation of

    +---------------------------+----------------------------+
    | previous node's embedding | remaining prize to collect |
    +---------------------------+----------------------------+

    .. note::
        Subclass of :class:`.PrevNodeContext`. The argument, inputs, outputs follow the same specification.

        In addition to supplying  ``state.get_current_node()`` for the index of the previous visited node.
        The input ``state`` needs to supply ``state.get_remaining_prize_to_collect()``.
    """

    def __init__(self, context_dim):
        super(PCTSPContext, self).__init__(context_dim)

    def _state_embedding(self, embeddings, state):
        state_embedding = state.get_remaining_prize_to_collect()[:, :, None]
        return state_embedding


class OPContext(PrevNodeContext):
    """
    Context node embedding for orienteering problem.
    Return a concatenation of

    +---------------------------+---------------------------------+
    | previous node's embedding | remaining tour length to travel |
    +---------------------------+---------------------------------+

    .. note::
        Subclass of :class:`.PrevNodeContext`. The argument, inputs, outputs follow the same specification.

        In addition to supplying  ``state.get_current_node()`` for the index of the previous visited node.
        The input ``state`` needs to supply ``state.get_remaining_length()``.
    """

    def __init__(self, context_dim):
        super(OPContext, self).__init__(context_dim)

    def _state_embedding(self, embeddings, state):
        state_embedding = state.get_remaining_length()[:, :, None]
        return state_embedding
