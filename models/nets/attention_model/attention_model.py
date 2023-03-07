from torch import nn

from ...nets.attention_model.decoder import Decoder
from ...nets.attention_model.embedding import AutoEmbedding
from ...nets.attention_model.encoder import GraphAttentionEncoder


class AttentionModel(nn.Module):
    r"""
    The Attention Model from Kool et al.,2019.
    `Link to paper <https://arxiv.org/abs/1803.08475>`_.

    For an instance :math:`s`,

    .. math::
        \log(p_\theta(\pi|s)),\pi = \mathrm{AttentionModel}(s)

    The following is executed:

    .. math::
        \begin{aligned}
        \pmb{x} &= \mathrm{Embedding}(s)                                      \\
        \pmb{h} &=  \mathrm{Encoder}(\pmb{x})                           \\
        \{\log(\pmb{p}_t)\},\pi &=  \mathrm{Decoder}(s, \pmb{h}) \\
        \log(p_\theta(\pi|s)) &= \sum\nolimits_t\log(\pmb{p}_{t,\pi_t})
        \end{aligned}
    where :math:`\pmb{h}_i` is the node embedding for each node :math:`i` in the graph.

    In a nutshell, :math:`\mathrm{Embedding}` is a linear projection.
    The :math:`\mathrm{Encoder}` is a transformer.
    The :math:`\mathrm{Decoder}` uses (multi-head) attentions.
    The policy (sequence of action) :math:`\pi` is decoded autoregressively.
    The log-likelihood :math:`\log(p_\theta(\pi|s))` of the policy is also returned.

    .. seealso::
        The definition of :math:`\mathrm{Embedding}`, :math:`\mathrm{Encoder}`, and
        :math:`\mathrm{Decoder}` can be found in the
        :mod:`.embedding`, :mod:`.encoder`, :mod:`.decoder` modules.

    Args:
        embedding_dim : the dimension of the embedded inputs
        hidden_dim : the dimension of the hidden state of the encoder
        problem : an object defining the state of the problem
        n_encode_layers: number of encoder layers
        tanh_clipping : the clipping scale for the decoder
    Inputs: inputs
        * **inputs**: problem instance :math:`s`. [batch, graph_size, input_dim]
    Outputs: ll, pi
        * **ll**: :math:`\log(p_\theta(\pi|s))` the log-likelihood of the policy. [batch, T]
        * **pi**: the policy generated :math:`\pi`. [batch, T]
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        n_encode_layers=2,
        tanh_clipping=10.0,
        normalization="batch",
        n_heads=8,
    ):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.n_heads = n_heads

        self.problem = problem

        self.embedding = AutoEmbedding(problem.NAME, {"embedding_dim": embedding_dim})
        step_context_dim = self.embedding.context_dim

        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
        )
        self.decoder = Decoder(embedding_dim, step_context_dim, n_heads, problem, tanh_clipping)

    def set_decode_type(self, decode_type):
        self.decoder.set_decode_type(decode_type)

    def forward(self, input):
        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding)
        _log_p, pi = self.decoder(input, encoded_inputs)
        ll = self._calc_log_likelihood(_log_p, pi)
        return ll, pi

    def _calc_log_likelihood(self, _log_p, pi):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)
