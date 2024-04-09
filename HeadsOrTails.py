import torch
import torch.nn as nn
import math

class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)

class tanh_FAVOR_AttentionHead(nn.Module):
  """
  A single attention head.
  This module is used in the MultiHeadAttention module.
  """
  def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
      super().__init__()
      self.hidden_size = hidden_size
      self.attention_head_size = attention_head_size
      # Create the query, key, and value projection layers
      self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
      self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
      self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

      self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
      # Project the input into query, key, and value
      # The same input is used to generate the query, key, and value,
      # so it's usually called self-attention.
      # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
      query = self.query(x)
      key = self.key(x)
      value = self.value(x)
      # Calculate the attention scores
      # favor+ attention: tanh(Q*K.T)*V
      attention_scores = torch.matmul(query, key.transpose(-1, -2))
      attention_scores = torch.tanh(attention_scores)
      attention_probs = nn.functional.softmax(attention_scores, dim=-1)
      attention_probs = self.dropout(attention_probs)
      # Calculate the attention output
      attention_output = torch.matmul(attention_probs, value)
      return (attention_output, attention_probs)

class ReLU_FAVOR_AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # favor+ attention with ReLU: ReLU(Q*K.T)*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = torch.relu(attention_scores)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)

class SM_AP_RF_AttentionHead(nn.Module):
    def __init__(self, hidden_size, input_size, dropout, num_random_features=32, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_random_features = num_random_features
        
        # Initialize random feature matrix for approximate softmax
        self.random_features = nn.Parameter(torch.randn(num_random_features, input_size) / math.sqrt(input_size))
        
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, input_size, bias=bias)
        self.key = nn.Linear(hidden_size, input_size, bias=bias)
        self.value = nn.Linear(hidden_size, input_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, input_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Calculate the attention scores using random features approximation
        # Compute random features for query and key
        query_random_features = torch.matmul(query, self.random_features[:self.num_random_features])
        key_random_features = torch.matmul(key, self.random_features[:self.num_random_features])
        
        # Compute attention scores
        attention_scores = torch.matmul(query_random_features, key_random_features.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.input_size)
        
        # Apply softmax to obtain attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)

