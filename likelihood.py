import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs

__all__ = [
    "crf_sequence_score", "crf_log_norm", "crf_log_likelihood",
    "crf_unary_score", "crf_binary_score", "CrfForwardRnnCell"
]

def _lengths_to_masks(lengths, max_length):
  tiled_ranges = array_ops.tile(
      array_ops.expand_dims(math_ops.range(max_length), 0),
      [array_ops.shape(lengths)[0], 1])
  lengths = array_ops.expand_dims(lengths, 1)
  masks = math_ops.to_float(
      math_ops.to_int64(tiled_ranges) < math_ops.to_int64(lengths))
  return masks


def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
  unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
  binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                   transition_params)
  sequence_scores = unary_scores + binary_scores
  return sequence_scores


def crf_log_norm(inputs, sequence_lengths, transition_params):
  first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
  first_input = array_ops.squeeze(first_input, [1])
  rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])

  forward_cell = CrfForwardRnnCell(transition_params)
  _, alphas = rnn.dynamic_rnn(
      cell=forward_cell,
      inputs=rest_of_input,
      sequence_length=sequence_lengths - 1,
      initial_state=first_input,
      dtype=dtypes.float32)
  log_norm = math_ops.reduce_logsumexp(alphas, [1])
  return log_norm


def crf_log_likelihood(inputs,
                       tag_indices,
                       sequence_lengths,
                       transition_params=None):
       # log_likelihood是对数似然函数，transition_params是转移概率矩阵
            # crf_log_likelihood{inputs:[batch_size,max_seq_length,num_tags],
            # tag_indices:[batchsize,max_seq_length],
            # sequence_lengths:[real_seq_length]
            # transition_params: A [num_tags, num_tags] transition matrix
            # log_likelihood: A scalar containing the log-likelihood of the given sequence of tag indices.
  num_tags = inputs.get_shape()[2].value

  if transition_params is None:
    transition_params = vs.get_variable("transitions", [num_tags, num_tags])

  sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths,
                                       transition_params)
  log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

  log_likelihood = sequence_scores - log_norm
  return log_likelihood, transition_params


def crf_unary_score(tag_indices, sequence_lengths, inputs):
  batch_size = array_ops.shape(inputs)[0]
  max_seq_len = array_ops.shape(inputs)[1]
  num_tags = array_ops.shape(inputs)[2]

  flattened_inputs = array_ops.reshape(inputs, [-1])

  offsets = array_ops.expand_dims(
      math_ops.range(batch_size) * max_seq_len * num_tags, 1)
  offsets += array_ops.expand_dims(math_ops.range(max_seq_len) * num_tags, 0)
  flattened_tag_indices = array_ops.reshape(offsets + tag_indices, [-1])

  unary_scores = array_ops.reshape(
      array_ops.gather(flattened_inputs, flattened_tag_indices),
      [batch_size, max_seq_len])

  masks = _lengths_to_masks(sequence_lengths, array_ops.shape(tag_indices)[1])

  unary_scores = math_ops.reduce_sum(unary_scores * masks, 1)
  return unary_scores


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
  num_tags = transition_params.get_shape()[0]
  num_transitions = array_ops.shape(tag_indices)[1] - 1

  start_tag_indices = array_ops.slice(tag_indices, [0, 0],
                                      [-1, num_transitions])
  end_tag_indices = array_ops.slice(tag_indices, [0, 1], [-1, num_transitions])

  flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
  flattened_transition_params = array_ops.reshape(transition_params, [-1])

  binary_scores = array_ops.gather(flattened_transition_params,
                                   flattened_transition_indices)

  masks = _lengths_to_masks(sequence_lengths, array_ops.shape(tag_indices)[1])
  truncated_masks = array_ops.slice(masks, [0, 1], [-1, -1])
  binary_scores = math_ops.reduce_sum(binary_scores * truncated_masks, 1)
  return binary_scores


class CrfForwardRnnCell(rnn_cell.RNNCell):
  def __init__(self, transition_params):
    self._transition_params = array_ops.expand_dims(transition_params, 0)
    self._num_tags = transition_params.get_shape()[0].value

  @property
  def state_size(self):
    return self._num_tags

  @property
  def output_size(self):
    return self._num_tags

  def __call__(self, inputs, state, scope=None):
    state = array_ops.expand_dims(state, 2)
    transition_scores = state + self._transition_params
    new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])
    return new_alphas, new_alphas