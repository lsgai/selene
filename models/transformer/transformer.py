from __future__ import absolute_import, division, print_function, unicode_literals
import json
import logging
import math
import os
import sys
from io import open

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertConfig
from pytorch_transformers.modeling_bert import *
from pytorch_transformers.tokenization_bert import BertTokenizer


class BertEmbeddingsDNA(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings.
  """
  def __init__(self, config):
    super(BertEmbeddingsDNA, self).__init__()
    self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
    # label should not need to have ordering ?
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    self.config = config

    if self.config.aa_type_emb:
      print ('\n\nturn on the token-type style embed.\n\n')
      ## okay to say 4 groups + 1 extra , we need special token to map to all 0, so CLS SEP PAD --> group 0
      ## 20 major amino acids --> 4 major groups
      ## or... we have mutation/not --> 2 major groups. set not mutation = 0 as base case
      ## we did not see experiment with AA type greatly improve outcome

      ## !! notice that padding_idx=0 will not be 0 because of initialization MUST MANUAL RESET 0
      self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

    # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
    # any TensorFlow checkpoint file
    self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, input_ids, token_type_ids=None, position_ids=None):
    seq_length = input_ids.size(1)
    if position_ids is None:
      position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
      position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    # if token_type_ids is None:
    #   token_type_ids = torch.zeros_like(input_ids)

    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)

    if self.config.aa_type_emb:
      # @token_type_ids is batch x aa_len x domain_type --> output batch x aa_len x domain_type x dim
      token_type_embeddings = self.token_type_embeddings(token_type_ids)
      ## must sum over domain (additive effect)
      token_type_embeddings = torch.sum(token_type_embeddings,dim=2) # get batch x aa_len x dim
      embeddings = words_embeddings + position_embeddings  + token_type_embeddings

    else:
      embeddings = words_embeddings + position_embeddings  # + token_type_embeddings

    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


class BertEmbeddingsLabel(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings.
  """
  def __init__(self, config):
    super(BertEmbeddingsLabel, self).__init__()

    self.config = config

    self.word_embeddings = nn.Embedding(config.label_size, config.hidden_size) ## , padding_idx=0
    # label should not need to have ordering ?
    # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

    # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
    # any TensorFlow checkpoint file
    if self.config.scale_label_vec: ## if we freeze, then we will not use any layer norm. let's try using the vectors as they are.
      self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    ## should always drop to avoid overfit
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, input_ids, token_type_ids=None, position_ids=None):
    # seq_length = input_ids.size(1)
    # if position_ids is None:
    #   position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    #   position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    # if token_type_ids is None:
    #   token_type_ids = torch.zeros_like(input_ids)

    embeddings = self.word_embeddings(input_ids)
    # position_embeddings = self.position_embeddings(position_ids)
    # token_type_embeddings = self.token_type_embeddings(token_type_ids)

    # embeddings = words_embeddings # + position_embeddings + token_type_embeddings
    if self.config.scale_label_vec:
      embeddings = self.LayerNorm(embeddings)

    ## should always drop to avoid overfit
    embeddings = self.dropout(embeddings)

    return embeddings


class BertModel2Emb(BertPreTrainedModel):
  r"""
  Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
    **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
      Sequence of hidden-states at the output of the last layer of the model.
    **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
      Last layer hidden-state of the first token of the sequence (classification token)
      further processed by a Linear layer and a Tanh activation function. The Linear
      layer weights are trained from the next sentence prediction (classification)
      objective during Bert pretraining. This output is usually *not* a good summary
      of the semantic content of the input, you're often better with averaging or pooling
      the sequence of hidden-states for the whole input sequence.
    **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
      list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
      of shape ``(batch_size, sequence_length, hidden_size)``:
      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    **attentions**: (`optional`, returned when ``config.output_attentions=True``)
      list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

  Examples::

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

  """
  def __init__(self, config):
    super(BertModel2Emb, self).__init__(config)

    self.embeddings = BertEmbeddingsDNA(config)
    self.embeddings_label = BertEmbeddingsLabel(config) ## label takes its own emb layer
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config)

    self.init_weights()

  def _resize_token_embeddings(self, new_num_tokens):
    old_embeddings = self.embeddings.word_embeddings
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    self.embeddings.word_embeddings = new_embeddings
    return self.embeddings.word_embeddings

  def _resize_label_embeddings(self, new_num_tokens):
    old_embeddings = self.embeddings_label.word_embeddings
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    self.embeddings_label.word_embeddings = new_embeddings
    return self.embeddings_label.word_embeddings

  def _prune_heads(self, heads_to_prune):
    """ Prunes heads of the model.
      heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
      See base class PreTrainedModel
    """
    for layer, heads in heads_to_prune.items():
      self.encoder.layer[layer].attention.prune_heads(heads)

  def resize_label_embeddings(self, new_num_tokens=None):
    """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
    Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

    Arguments:

      new_num_tokens: (`optional`) int:
        New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
        If not provided or None: does nothing and just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.

    Return: ``torch.nn.Embeddings``
      Pointer to the input tokens Embeddings Module of the model
    """
    base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
    model_embeds = base_model._resize_label_embeddings(new_num_tokens)
    if new_num_tokens is None:
      return model_embeds

    # Update base model and current model config
    self.config.label_size = new_num_tokens
    base_model.label_size = new_num_tokens

    # Tie weights again if needed
    if hasattr(self, 'tie_weights'):
      self.tie_weights()

    return model_embeds

  def forward(self, input_ids, input_ids_aa, label_index_id, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

    ##!! to avoid a lot of re-structuring, let's define @input_ids=>protein_vector from interaction network
    ## assume @input_ids is batch x 1 x dim, each batch is a protein so it has 1 vector

    # if attention_mask is None:
    #   attention_mask = torch.ones_like(input_ids) ## probably don't need this very much. if we pass in mask and token_type, which we always do for batch mode
    # # if token_type_ids is None:
    # #   token_type_ids = torch.zeros_like(input_ids)

    # # We create a 3D attention mask from a 2D tensor mask.
    # # Sizes are [batch_size, 1, 1, to_seq_length]
    # # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # # this attention mask is more simple than the triangular masking of causal attention
    # # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # # masked positions, this operation will create a tensor which is 0.0 for
    # # positions we want to attend and -10000.0 for masked positions.
    # # Since we are adding it to the raw scores before the softmax, this is
    # # effectively the same as removing these entirely.
    # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
    # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    if head_mask is not None:
      if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
      elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
      head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
      head_mask = [None] * self.config.num_hidden_layers

    ## need to split the @input_ids into AA side and label side, @input_ids_aa @label_index_id

    ## COMMENT if @x is already 1-hot embedding, we can just use it.
    embedding_output = self.embeddings(input_ids_aa, position_ids=position_ids, token_type_ids=token_type_ids)
    embedding_output_label = self.embeddings_label(label_index_id, position_ids=None, token_type_ids=None)

    # concat into the original embedding
    if self.config.ppi_front:
      ## masking may vary, because some proteins don't have vec emb
      embedding_output = torch.cat([input_ids,embedding_output,embedding_output_label], dim=1) ## we add protein_vector as variable @input_ids
    else:
      ## COMMENT @input_ids_aa should probably be changed later
      embedding_output = torch.cat([embedding_output,embedding_output_label], dim=1) ## @embedding_output is batch x num_aa x dim so append @embedding_output_label to dim=1 (basically adding more words to @embedding_output)
      print ('embedding_output')
      print (embedding_output.shape)

    # @embedding_output is just some type of embedding, the @encoder will apply attention weights
    encoder_outputs = self.encoder(embedding_output,
                                   attention_mask=None, ## @extended_attention_mask must mask using the entire set of sequence + label input
                                   head_mask=head_mask)

    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class TokenClassificationBase (BertPreTrainedModel):

  ## !! we change this to do 1-hot prediction
  ## take in K labels so we have vector of 1-hot length K
  ## for each label, we get a vector output from BERT, then we predict 0/1

  def __init__(self, config_name, sequence_length, n_genomic_features):

    ## create config object base on path name. bert needs config object
    self.config = BertConfig.from_pretrained(config_name)

    super(TokenClassificationBase, self).__init__(self.config)

    self.num_labels = n_genomic_features ## about 919 for histone marks

    self.bert = BertModel2Emb(self.config)
    self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    self.classifier = nn.Sequential(nn.Linear(self.config.hidden_size, 1),
                                    nn.Sigmoid())

    self.init_weights() # https://github.com/lonePatient/Bert-Multi-Label-Text-Classification/issues/19

    ## all batch has same length. so we can determine a lot of varibles and define them once.
    self.label_range = torch.LongTensor ( np.arange(self.num_labels) ).unsqueeze(0).cuda() ## 1 x num_label
    self.attention_mask_label = torch.LongTensor ( np.concatenate ( (np.zeros(sequence_length),np.ones(self.num_labels)) ) ).unsqueeze(0).cuda() ## 1 x seq_len + num_label

  def forward(self, x):
    ## COMMENT: must take only x=batch x 4 x 1000 because @selene pipeline requires only this input

    ## COMMENT convert @x into word-indexing style. so we want @x = [[1,1,2,2,...], [3,3,4,4,...]] --> batch x seq_len

    real_batch_size = x.shape[0]
    print ('see x before')
    print (x[0])
    x = np.reshape(np.where(x==1)[1], (real_batch_size,1000)) ## get back normal indexing 

    print ('see x after')
    print (x.shape)
    print (x[0])
    
    ## trick: input_ids=None so we keep same consistency
    ## default @x is DNA + label --> so it is already an embedding
    ## @label_index_id can be determined ahead of time

    label_index_id = self.label_range.expand(real_batch_size,-1) ## batch x num_label ... 1 row for 1 ob in batch
    attention_mask_label = self.attention_mask_label.expand(real_batch_size,-1)

    ## COMMENT use @x as a fixed embedding
    outputs = self.bert(None, x, label_index_id, position_ids=None, token_type_ids=None)

    sequence_output = outputs[0] ## last layer. ## last layer outputs is batch_num x len_sent x dim
    sequence_output = self.dropout(sequence_output)

    print ('outputs')
    print (outputs[0].shape)

    logits = self.classifier(sequence_output)

    print ('logits')
    print (logits.shape)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

    ## must extract only the label side (i.e. 2nd sentence)
    ## @attention_mask is used to avoid padding in the @active_loss
    ## so we need to only create another @attention_mask_label to pay attention to labels only

    # Only keep active parts of the loss
    if attention_mask_label is not None: ## change @attention_mask --> @attention_mask_label
      print ('attention_mask_label')
      print (attention_mask_label)
      active_loss = attention_mask_label.view(-1) == 1 # attention_mask_label.view(-1) --> same as flatten
      print (active_loss.shape)
      print ('logits.view(-1)')
      print (logits.view(-1).shape)
      active_logits = logits.view(-1)[active_loss] ## remove  self.num_labels

    outputs = (active_logits,) + outputs

    return outputs[1]  # scores, (hidden_states), (attentions) ## remove  (loss),


def criterion():
  return nn.BCELoss()

def get_optimizer(lr):
  return (torch.optim.Adam, {"lr": lr})

