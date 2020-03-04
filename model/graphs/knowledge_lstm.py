from bert_model.modeling_base import *
from model.basic_encoder import BasicEncoder
from utils import sequence_feature

import pickle
import tensorflow as tf

class Model(object):
  def __init__(self, hparams, bert_config=None, bert_model=None, input_phs=None, label_id_phs=None,
               length_phs=None, knowledge_phs=None, similar_phs=None):
    self.hparams = hparams
    self.bert_config = bert_config
    self.bert_model = bert_model
    self.label_id_phs = label_id_phs
    self.length_phs = length_phs
    self.knowledge_token_phs = knowledge_phs

    self.encoder = BasicEncoder(self.hparams, self.hparams.dropout_keep_prob)


    # dialog_label_ids, knowledge_label_ids = label_id_phs
    # knowledge_ids_ph, knowledge_mask_ph, knowledge_seg_ids_ph = knowledge_phs
    # dialog_len_ph, response_len_ph, knowledge_len_ph = length_phs

  def build_graph(self):
    # dialog_cls : [batch, 768]
    # knowledge_bilstm_out : [batch, 5, 1536]
    bert_dialog_cls = self.bert_model.get_pooled_output()
    knowledge_bilstm_out = self._bert_pretrained_knowledge(self.knowledge_token_phs, self.length_phs[2])

    knowledge_labels = self.label_id_phs[1]
    # knowledge_exist_len : [batch_size]
    knowledge_exist_len = tf.reduce_sum(knowledge_labels, axis=-1)
    # knowledge_exist_len = tf.Print(knowledge_exist_len, [knowledge_exist_len], message="knowledge_exist_len_sum", summarize=16)

    # knowledge_labels = tf.tile(tf.expand_dims(self.label_id_phs[1], axis=-1), multiples=[1, 1, tf.shape(knowledge_bilstm_out)[2]])
    # knowledge_bilstm_out = tf.multiply(knowledge_bilstm_out, tf.cast(knowledge_labels,tf.float32))

    # [batch, 5] [[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]]
    tiled_bert_dialog_cls = tf.tile(tf.expand_dims(bert_dialog_cls, axis=1),[1, self.hparams.top_n, 1])
    dialog_knowledge_concat = tf.concat([tiled_bert_dialog_cls, knowledge_bilstm_out], axis=-1)

    lstm_outputs = self.encoder.lstm_encoder(dialog_knowledge_concat, knowledge_exist_len,
                                             "dialog_cls_knowledge_lstm", rnn_hidden_dim=256)
    features_fw, features_bw = sequence_feature(lstm_outputs, knowledge_exist_len, sep_pos=False)

    lstm_hidden_outputs = tf.concat([features_fw, features_bw], axis=-1)
    filtered_lstm_hidden_outputs = tf.multiply(tf.cast(tf.expand_dims(knowledge_exist_len, axis=-1), tf.float32),
                                               lstm_hidden_outputs)
    # filtered_lstm_hidden_outputs = tf.Print(filtered_lstm_hidden_outputs, [filtered_lstm_hidden_outputs], message="lstm_hidden_outputs", summarize=512)

    # batch, 768
    # concat_outputs = tf.concat(lstm_hidden_outputs, axis=-1)
    dialog_cls_projection = tf.layers.dense(
      name="dialog_cls_projection",
      inputs=bert_dialog_cls,
      units=512,
      kernel_initializer=create_initializer(initializer_range=0.02))
    dialog_knowledge_cls_projection = tf.layers.dense(
      name="dialog_knowledge_cls_projection",
      inputs=lstm_hidden_outputs,
      units=512,
      activation=tf.nn.relu,
      kernel_initializer=create_initializer(initializer_range=0.02))

    output_layer = tf.where(tf.equal(tf.reduce_sum(filtered_lstm_hidden_outputs, axis=-1), 0.),
                            dialog_cls_projection, dialog_knowledge_cls_projection)

    self.test1 = tf.equal(output_layer, dialog_cls_projection)
    self.test2 = tf.equal(output_layer, dialog_knowledge_cls_projection)
    self.test_sum = tf.cast(self.test1, tf.int32) + tf.cast(self.test2, tf.int32)

    logits, loss_op = self._final_output_layer(output_layer)

    return logits, loss_op

  def _bert_pretrained_knowledge(self, knowledge_tokens_ph, knowledge_lengths_ph):
    # knowledge_phs : [batch, top_n, max_seq_len, 768]
    input_shape = get_shape_list(knowledge_tokens_ph, expected_rank=4)
    batch_size = input_shape[0]
    top_n = input_shape[1]
    knowledge_max_seq_len = input_shape[2]
    embedding_dim = input_shape[3]

    print(input_shape)
    knowledge_tokens_embeddded = tf.reshape(knowledge_tokens_ph, shape=[-1, knowledge_max_seq_len, embedding_dim])
    knowledge_lengths = tf.reshape(knowledge_lengths_ph, shape=[-1])

    print(knowledge_tokens_embeddded)
    print(knowledge_lengths)
    knowledge_lstm_outputs = self.encoder.lstm_encoder(knowledge_tokens_embeddded, knowledge_lengths, "knowledge_lstm")
    knowledge_fw, knowledge_bw = sequence_feature(knowledge_lstm_outputs, knowledge_lengths, sep_pos=True)
    knowledge_concat_features = tf.concat([knowledge_fw, knowledge_bw], axis=-1)
    knowledge_concat_features = \
      tf.reshape(knowledge_concat_features, shape=[batch_size, top_n, self.hparams.rnn_hidden_dim*2])
    # [batch, top_n, 1536]

    return knowledge_concat_features

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