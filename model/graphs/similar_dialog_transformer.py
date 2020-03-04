from bert_model.modeling_base import *
from model.basic_encoder import BasicEncoder
from model.esim_attention import ESIMAttention
from utils import sequence_feature

import pickle
import tensorflow as tf

class Model(object):
  def __init__(self, hparams, bert_config=None, bert_model:BertModel=None, input_phs=None,
               label_id_phs=None, length_phs=None, knowledge_phs=None, similar_phs=None):
    self.hparams = hparams
    self.bert_hidden_dropout_prob = 0.1
    if self.hparams.do_evaluate:
      self.bert_hidden_dropout_prob = 0.0
    self.bert_config = bert_config
    self.bert_model = bert_model
    self.input_phs = input_phs
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
    # [vocab_size, embedding_dim] e.g: 30522,768
    self.word_embeddings = self.bert_model.embedding_table
    # [max_seq_len(512), embedding_dim] e.g: 512, 768
    self.full_position_embeddings = self.bert_model.full_position_embeddings
  def _bert_sentences_split(self, input, max_seq_len_a, max_seq_len_b):
    dialog_bert_outputs, response_bert_outputs = tf.split(input, [max_seq_len_a, max_seq_len_b], axis=1)

    return dialog_bert_outputs, response_bert_outputs

  def _embedding_processing(self, input_tensor):
    with tf.variable_scope("embedding_processing", reuse=tf.AUTO_REUSE):
      input_shape = get_shape_list(input_tensor, expected_rank=3)
      batch_size = input_shape[0]
      seq_length = input_shape[1]
      width = input_shape[2]

      output = input_tensor

      position_embeddings = tf.slice(self.full_position_embeddings, [0, 0], [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width]) # 1, seq_len, width
      position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
      output += position_embeddings

      output = layer_norm_and_dropout(output, self.bert_hidden_dropout_prob)

      return output

  def _similar_dialog_self_attention_layer(self, from_ids, to_mask, embedding_output, num_hidden_layers, var_name=""):
    with tf.variable_scope(var_name, reuse=tf.AUTO_REUSE):
      # from_ids : batch, from_seq_len
      # to_mask : batch, to_seq_len

      attention_mask = create_attention_mask_from_input_mask(from_ids, to_mask)

      # Run the stacked transformer.
      # `sequence_output` shape = [batch_size, seq_length, hidden_size].
      all_encoder_layers = transformer_model(
        from_tensor=embedding_output,
        to_tensor=embedding_output,
        attention_mask=attention_mask,
        hidden_size=self.bert_config.hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=6,
        intermediate_size=self.bert_config.intermediate_size,
        intermediate_act_fn=get_activation("gelu"),
        hidden_dropout_prob=self.bert_hidden_dropout_prob,
        attention_probs_dropout_prob=self.bert_hidden_dropout_prob,
        initializer_range=0.02,
        do_return_all_layers=True,
        hparams=self.hparams)

    return all_encoder_layers[-1]

  def _similar_dialog_cross_attention_layer(self, from_ids, to_mask, from_embedding_out, to_embedding_out, var_name=""):
    with tf.variable_scope(var_name, reuse=tf.AUTO_REUSE):
      attention_mask = create_attention_mask_from_input_mask(from_ids, to_mask)
      all_encoder_layers = transformer_model(
        from_tensor=from_embedding_out,
        to_tensor=to_embedding_out,
        attention_mask=attention_mask,
        hidden_size=self.bert_config.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=6,
        intermediate_size=self.bert_config.intermediate_size,
        intermediate_act_fn=get_activation("gelu"),
        hidden_dropout_prob=self.bert_hidden_dropout_prob,
        attention_probs_dropout_prob=self.bert_hidden_dropout_prob,
        initializer_range=0.02,
        do_return_all_layers=True,
        hparams=self.hparams,
        att_type="cross")

    return all_encoder_layers[-1]

  def _similar_dialog_transformer(self, similar_input_ids_ph, similar_input_mask_ph):
    """
		:param similar_dialog_input_ph: [batch, top_n, max_sequence_len]
		:param similar_dialog_len_ph: [batch, top_n]
		:return:
		"""
    input_shape = get_shape_list(similar_input_ids_ph, expected_rank=3)
    batch_size = input_shape[0]
    top_n = input_shape[1]
    max_seq_len = input_shape[2]

    # batch, top_n, max_seq_len, 768
    similar_dialog_embedded = tf.nn.embedding_lookup(self.word_embeddings, similar_input_ids_ph)
    unstacked_similar_ids = tf.unstack(similar_input_ids_ph, axis=1)
    unstacked_similar_mask = tf.unstack(similar_input_mask_ph, axis=1)
    unstacked_similar_emb = tf.unstack(similar_dialog_embedded, axis=1)

    input_ids_ph, input_mask_ph = self.input_phs
    dialog_id, response_id = \
      tf.split(input_ids_ph, [self.hparams.dialog_max_seq_length, self.hparams.response_max_seq_length], axis=1)
    response_emb = tf.nn.embedding_lookup(self.word_embeddings, response_id)
    response_emb_out = self._embedding_processing(response_emb)
    dialog_mask, response_mask = \
      tf.split(input_mask_ph, [self.hparams.dialog_max_seq_length, self.hparams.response_max_seq_length], axis=1)

    concat_outs = []
    for sim_id, sim_mask, sim_emb in zip(unstacked_similar_ids, unstacked_similar_mask, unstacked_similar_emb):
      embedding_out = self._embedding_processing(sim_emb) # positional_embedding processing

      self_dialog_out = self._similar_dialog_self_attention_layer(sim_id, sim_mask, embedding_out, 3, var_name="similar_dialog_self")
      self_response_out = self._similar_dialog_self_attention_layer(response_id, response_mask, response_emb_out, 3, var_name="similar_dialog_self")
      dialog_cross_out = self._similar_dialog_cross_attention_layer(sim_id, response_mask, self_dialog_out, self_response_out, var_name="similar_dialog_cross")
      response_cross_out = self._similar_dialog_cross_attention_layer(response_id, sim_mask, self_response_out, self_dialog_out, var_name="similar_dialog_cross")

      # batch, 320, 768 * 4
      dialog_concat = tf.concat([self_dialog_out, dialog_cross_out,
                                 self_dialog_out - dialog_cross_out,
                                 self_dialog_out * dialog_cross_out], axis=-1)
      dialog_concat = tf.layers.dense(
        inputs=dialog_concat,
        units=768,
        kernel_initializer=create_initializer(0.02),
        reuse=tf.AUTO_REUSE,
        name="dilaog_concat_projection"
      )

      dialog_aggregation = self._similar_dialog_self_attention_layer(sim_id, sim_mask, dialog_concat, 1, var_name="dialog_aggregation")

      # batch, 40, 768 * 4
      response_concat = tf.concat([self_response_out, response_cross_out,
                                   self_response_out - response_cross_out,
                                   self_response_out * response_cross_out], axis=-1)

      response_concat = tf.layers.dense(
        inputs=response_concat,
        units=768,
        kernel_initializer=create_initializer(0.02),
        reuse=tf.AUTO_REUSE,
        name="response_concat_projection"
      )
      response_aggregation = self._similar_dialog_self_attention_layer(response_id, response_mask, response_concat, 1, var_name="response_aggregation")

      # [batch, 768]
      dialog_max = tf.reduce_max(dialog_aggregation, axis=1)
      response_max = tf.reduce_max(response_aggregation, axis=1)
      concat_out = tf.concat([dialog_max, response_max], axis=1)

      concat_outs.append(concat_out)

    return concat_outs

  def build_graph(self):
    bert_cls_out = self.bert_model.get_pooled_output()

    similar_input_ids_ph, similar_input_mask_ph, similar_len_ph = self.similar_phs
    self_cross_att_outs = self._similar_dialog_transformer(similar_input_ids_ph, similar_input_mask_ph)
    # batch, rnn_hidden_dim * 2 : (total top_n : 3)
    mlp_layers = []
    for each_att_out in self_cross_att_outs:
      concat_features = tf.concat([bert_cls_out, each_att_out], axis=-1)
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

    feed_forward_out = tf.layers.dense(
      inputs=final_input_layer,
      units=64,
      activation=tf.nn.relu,
      kernel_initializer=create_initializer(0.02),
      name="feed_forward_outputs"
    )

    logits = tf.layers.dense(
      inputs=feed_forward_out,
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