from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    # nn.Linear (input size, output size)
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

  def transform(self, x, linear_layer):
    """
    x = hidden_states: [bs, seq_len, hidden_state] A 3D tensor representing the input sequences. Its dimensions are [batch size, sequence length, hidden state size].
    linear_layer = nn.Linear(config.hidden_size, self.all_head_size)
    """
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    # When you create an instance of nn.Linear, it initializes two parameters: a weight matrix and a bias vector.
    # When you pass the input tensor x to linear_layer, it multiplies x by the weight matrix and then adds the bias vector.
    """
    [bs, seq_len, hidden_state] * [hidden_size & all_head_size] 
    = [bs, seq_len, all_head_size]
    """
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by splitting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    """
    proj = [bs, seq_len, all_head_size]
    since self.all_head_size = self.num_attention_heads * self.attention_head_size
    """
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    # there are 4 entry, 0,1,2,3, this means swap 2nd and third
    proj = proj.transpose(1, 2)
    """
    result dim = [bs, num_attention_heads, seq_len, attention_head_size]
    bs x num_attention_heads x seq_len x attention_head_size (4D)
    """
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    key,query,value = [bs, num_attention_heads, seq_len, attention_head_size]
    """
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiplying query and key
    # to get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    """
    key.transpose(-1, -2) swaps the last two dimensions of the key tensor.
    [bs, num_attention_heads, seq_len, attention_head_size] * [bs, num_attention_heads, attention_head_size, seq_len] = 
    [bs, num_attention_heads, seq_len, seq_len]
    """
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th token, given by the i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 
    attention_scores = attention_scores + attention_mask

    # normalize the scores
    attention_probs = F.softmax(attention_scores, dim=-1)
    # multiply the attention scores to the value and get back V'
    # attn_output = [bs, num_attention_heads, seq_len, attention_head_size]
    attn_output = torch.matmul(attention_probs, value)
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
    attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, self.all_head_size)

    return attn_output

  def attention(self, key, query, value, attention_mask):
    # Calculate attention scores
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / (self.attention_head_size ** 0.5)

    # Apply the attention mask
    attention_scores = attention_scores + attention_mask

    # Normalize the scores
    attention_probs = F.softmax(attention_scores, dim=-1)

    # Multiply the attention scores with the value tensor to get the new "value" tensor
    context_layer = torch.matmul(attention_probs, value)

    # Concatenate multi-heads
    context_layer = context_layer.transpose(1, 2).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state] A 3D tensor representing the input sequences. Its dimensions are [batch size, sequence length, hidden state size].
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # dimensions of all *_layers are [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # multi-head attention
    self.self_attention = BertSelfAttention(config)
    # add-norm
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # another add-norm
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    this function is applied after the multi-head attention layer or the feed forward layer
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Dense transformation of the output
    output = dense_layer(output)
    # Dropout
    """
    Dropout is a regularization technique where randomly selected neurons are ignored during training, which helps in preventing overfitting.
    """
    output = dropout(output)
    # Add (residual connection)
    output = output + input
    # Normalize
    """
    for each individual sample in the batch, it normalizes the features (or activations) so that they have a mean of 0 and a standard deviation of 1.
    This normalization is done independently for each sample, and it's applied at every layer of the network.
    """
    return ln_layer(output)


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the input and output of the multi-head attention layer
    3. a feed forward layer
    4. a add-norm that takes the input and output of the feed forward layer
    """
    # multi-head attention w/ self.self_attention
    attention_output = self.self_attention(hidden_states, attention_mask)

    # add-norm layer
    attention_output = self.add_norm(hidden_states, attention_output,
                                     self.attention_dense, self.attention_dropout,
                                     self.attention_layer_norm)

    # feed forward
    interm_output = self.interm_af(self.interm_dense(attention_output))

    # another add-norm layer
    layer_output = self.add_norm(attention_output, interm_output,
                                 self.out_dense, self.out_dropout,
                                 self.out_layer_norm)

    return layer_output


class BertModel(BertPreTrainedModel):
  """
  the BERT model returns the final embeddings for each token in a sentence
  it consists of
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # get word embedding from self.word_embedding
    inputs_embeds = self.word_embedding(input_ids)

    # get position index and position embedding from self.pos_embedding
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)

    # get token type ids. since we are not considering token types, this is just a placeholder
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # add these three embeddings together
    embeds = inputs_embeds + tk_type_embeds + pos_embeds

    # apply layer norm and dropout
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)

    return embeds

  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
