import torch
from torch import nn
import math

from .mlp import MLP


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self,
                 hidden_size,
                 attention_head_size,
                 dropout,
                 bias=True,
                 relu=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.relu = relu
    
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = (
            attention_scores / math.sqrt(self.attention_head_size)
        )

        if self.relu:
            # relu(Q*K.T/sqrt(head_size))*V
            attention_probs = nn.functional.relu(attention_scores)
        else:
            # softmax(Q*K.T/sqrt(head_size))*V
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config, relu=False):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number 
        # of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]

        self.relu = relu
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                hidden_size=self.hidden_size,
                attention_head_size=self.attention_head_size,
                dropout=config["attention_probs_dropout_prob"],
                bias=self.qkv_bias,
                relu=self.relu
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat(
            [attention_output for attention_output, _ in attention_outputs],
            dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack(
                [attention_probs for _, attention_probs in attention_outputs],
                dim=1)
            return (attention_output, attention_probs)


class RandomFeaturesAttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the PerformerMultiHeadAttention module.
    """

    def __init__(
        self,
        hidden_size,
        attention_head_size, num_attention_heads,
        dropout, bias=True, m=16
    ):
        super().__init__()
        # x --> (batch_size, hidden_size, num_patches)
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(
            hidden_size * num_attention_heads,
            hidden_size * num_attention_heads
        )
        # number of random features
        self.m = m
        self.w = nn.Parameter(
            torch.randn(self.m, attention_head_size), requires_grad=False
        )

    def prm_exp(self, x):
        # ==== positive random features for gaussian kernels ====
        # x = (batch_size, num_patches, hidden_size)
        # w = (m, hidden_size)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum("bti,mi->btm", x, self.w)
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def forward(self, x):
        kp, qp = (
            self.prm_exp(self.key(x)),
            self.prm_exp(self.query(x)),
        )  # (B, T, m), (B, T, m)
        D = torch.einsum("bti,bi->bt", qp, kp.sum(dim=1)).unsqueeze(
            dim=2
        )  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum("bin,bim->bnm", self.value(x), kp)  # (B, hidden_size, m)
        attention_probs = kptv
        attention_output = torch.einsum("bti,bni->btn", qp, kptv) / D.repeat(
            1, 1, self.attention_head_size
        )  # (B, T, hidden_size)/Diag
        return (attention_output, attention_probs)


class RandomFeaturesMultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config, m=16):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = RandomFeaturesAttentionHead(
                self.hidden_size,
                self.attention_head_size,
                self.num_attention_heads,
                self.qkv_bias,
                config["attention_probs_dropout_prob"],
                m=m
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat(
            [attention_output for attention_output, _ in attention_outputs], dim=-1
        )
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # print(f'size of attention_output {attention_output.shape}')
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack(
                [attention_probs for _, attention_probs in attention_outputs], dim=1
            )
            return (attention_output, attention_probs)

class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config, performer=False, relu=False, m=16):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        self.m = m
        self.relu = relu
        if self.use_faster_attention:
            # self.attention = FasterMultiHeadAttention(config)
            raise NotImplementedError
        else:
            if performer:
                if not m:
                    raise ValueError('m should not be None.')
                self.attention = RandomFeaturesMultiHeadAttention(config, m=self.m)
            else:
                self.attention = MultiHeadAttention(config, relu=self.relu)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(
                self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention 
        # probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)