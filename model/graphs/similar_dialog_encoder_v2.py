from bert_model.modeling_base import *
from model.basic_encoder import BasicEncoder
from model.esim_attention import ESIMAttention
from utils import sequence_feature

import pickle
import tensorflow as tf

class Model(object):
  def __init__(self, hparams, bert_config=None, bert_model:BertModel=None, label_id_phs=None,
               input_phs=None, length_phs=None, knowledge_phs=None, similar_phs=None):

    self.hparams = hparams
    self.bert_config = bert_config
    self.bert_model = bert_model
    self.label_id_phs = label_id_phs
    self.length_phs = length_phs
    self.knowledge_token_phs = knowledge_phs
    self.similar_phs = similar_phs

    self._get_pretrained_variables()

    self.encoder = BasicEncoder(self.hparams, self.hparams.dropout_keep_prob)

    # dialog_label_ids, knowledge_label_ids = label_id_phs
    # knowledge_ids_ph, knowledge_mask_ph, knowledge_seg_ids_ph = knowledge_phs
    # dialog_len_ph, response_len_ph, knowledge_len_ph = length_phs
  def _get_pretrained_variables(self):
    self.bert_pretrained_word_embeddings = self.bert_model.embedding_table

  def _bert_sentences_split(self, bert_sequence_output, max_seq_len_a, max_seq_len_b):
    dialog_bert_outputs, response_bert_outputs = tf.split(bert_sequence_output, [max_seq_len_a, max_seq_len_b], axis=1)

    return dialog_bert_outputs, response_bert_outputs

  def _similar_dialog_lstm(self, similar_dialog_input_ph, similar_dialog_len_ph):
    """
		:param similar_dialog_input_ph: [batch, top_n, max_sequence_len]
		:param similar_dialog_len_ph: [batch, top_n]
		:return:
		"""
    input_shape = get_shape_list(similar_dialog_input_ph, expected_rank=3)
    batch_size = input_shape[0]
    top_n = input_shape[1]
    max_seq_len = input_shape[2]

    # batch, top_n, max_seq_len, 768
    similar_dialog_embedded = tf.nn.embedding_lookup(self.bert_pretrained_word_embeddings, similar_dialog_input_ph)
    # reshape inputs, length
    similar_dialog_embedded = tf.reshape(similar_dialog_embedded, shape=[-1, max_seq_len, self.hparams.embedding_dim])
    similar_dialog_len = tf.reshape(similar_dialog_len_ph, shape=[-1])
    similar_dialog_out = self.encoder.lstm_encoder(similar_dialog_embedded, similar_dialog_len, "similar_dialog_lstm")
    similar_dialog_out = tf.reshape(similar_dialog_out, shape=[batch_size, top_n, max_seq_len,
                                                               self.hparams.rnn_hidden_dim * 2])

    return similar_dialog_out

  def build_graph(self):
    # dialog_cls : [batch, 768]
    # knowledge_bilstm_out : [batch, 5, 1536]
    bert_seq_out = self.bert_model.get_sequence_output()
    dialog_bert_outputs, response_bert_outputs = \
      self._bert_sentences_split(bert_seq_out, self.hparams.dialog_max_seq_length, self.hparams.response_max_seq_length)
    # bert_cls_out = self.bert_model.get_pooled_output()

    similar_input_ids_ph, similar_input_mask_ph, similar_len_ph = self.similar_phs
    # batch, top_n, max_seq_out, rnn_hidden_dim * 2
    similar_dialogs_lstm_outputs = self._similar_dialog_lstm(similar_input_ids_ph, similar_len_ph)
    unstacked_similar_dialog_lstm_out = tf.unstack(similar_dialogs_lstm_outputs, self.hparams.top_n, axis=1)
    unstacked_similar_dialog_len = tf.unstack(similar_len_ph, self.hparams.top_n, axis=1)

    # response_bert_outputs : batch, 40, 768
    # similar_dilaog_lstm_outputs : batch, top_n, 320, 512
    dialog_len_ph, response_len_ph, _ = self.length_phs #[batch]
    response_len = response_len_ph - 1
    response_lstm_outputs = self.encoder.lstm_encoder(response_bert_outputs, response_len, name="response_lstm")
    response_fw, response_bw = sequence_feature(response_lstm_outputs, response_len)
    response_concat = tf.concat([response_fw, response_bw], axis=-1)
    esim_att_out_l =[]
    for each_dialog_out, each_dialog_len in zip(unstacked_similar_dialog_lstm_out, unstacked_similar_dialog_len):
      # batch, 320, rnn_hidden_dim*2 -> each_dialog_out
      esim_att = ESIMAttention(self.hparams, self.hparams.dropout_keep_prob,
                               text_a=response_lstm_outputs, text_a_len=response_len,
                               text_b=each_dialog_out, text_b_len=each_dialog_len)
      # batch, rnn_hidden_dim * 2 : 256 * 2 = 512
      esim_att_out_l.append(esim_att.text_a_att_outs)

    # batch, rnn_hidden_dim * 2 : (total top_n : 3)
    mlp_layers = []
    for each_att_out in esim_att_out_l:
      concat_features = tf.concat([response_concat, each_att_out], axis=-1)
      layer_input = concat_features
      for i in range(3):
        dense_out = tf.layers.dense(
          inputs=layer_input,
          units=768,
          activation=tf.nn.relu,
          kernel_initializer=create_initializer(0.02),
          name="mlp_%d" % i
        )
        dense_out = tf.nn.dropout(dense_out, self.hparams.dropout_keep_prob)
        layer_input = dense_out
      mlp_layers.append(layer_input)
    # element-wise summation
    output_layer = tf.add_n(mlp_layers, "mlp_layers_add_n")

    logits, loss_op = self._final_output_layer(output_layer)

    return logits, loss_op


  def _final_output_layer(self, final_input_layer):

    dialog_label_ids, knowledge_label_ids = self.label_id_phs
    if self.hparams.loss_type == "sigmoid": logits_units = 1
    else: logits_units = 2

    logits = tf.layers.dense(
      inputs=final_input_layer,
      units=logits_units,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      name="logits"
    )

    if self.hparams.loss_type == "sigmoid":
      logits = tf.squeeze(logits, axis=-1)
      loss_op = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,labels=tf.cast(dialog_label_ids, tf.float32),name="binary_cross_entropy")
    else:
      loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,labels=dialog_label_ids, name="cross_entropy")

    loss_op = tf.reduce_mean(loss_op, name="cross_entropy_mean")

    return logits, loss_op