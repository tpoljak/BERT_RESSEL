from bert_model.modeling_base import *

import tensorflow as tf

class Model(object):
	def __init__(self, hparams, bert_config=None, bert_model=None, input_phs=None, label_id_phs=None,
	             length_phs=None, knowledge_phs=None, similar_phs=None):
		self.hparams = hparams
		self.bert_config = bert_config
		self.bert_model = bert_model
		self.label_id_phs = label_id_phs
		self.length_phs = length_phs
		self.knowledge_phs = knowledge_phs

		# dialog_label_ids, knowledge_label_ids = label_id_phs
		# dialog_len_ph, response_len_ph, knowledge_len_ph = length_phs
		# knowledge_ids_ph, knowledge_mask_ph, knowledge_seg_ids_ph = knowledge_phs

	def build_graph(self):
		bert_dialog_cls = self.bert_model.get_pooled_output()
		bert_knowledge_cls, _ = self._bert_knowledge(self.knowledge_phs)
		# attention_score : [batch, 3]
		with tf.variable_scope("knowledge_attention"):
			knowledge_att_score, knowledge_att_output = self._dialog_knowledge_attention(bert_dialog_cls, bert_knowledge_cls)
			knowledge_loss_op = self._knowledge_loss(knowledge_att_score, self.label_id_phs[1])

			# [batch, 768 * 2]
			output_layer = tf.concat([bert_dialog_cls, knowledge_att_output], axis=-1)
			output_layer = tf.layers.dense(
				inputs=output_layer,
				units=128,
				activation=tf.nn.selu,
				kernel_initializer=tf.keras.initializers.he_normal(seed=7777),
				name="logits_feed_forward"
			)
			logits, loss_op = self._final_output_layer(output_layer, knowledge_loss_op)
		return logits, loss_op

	def _bert_knowledge(self, knowledge_phs):
		top_n = self.hparams.top_n
		if not self.hparams.do_evaluate:
			is_training = True
		else:
			is_training = False
		use_one_hot_embeddings = False

		knowledge_ids_ph, knowledge_mask_ph, knowledge_seg_ids_ph = knowledge_phs
		unstacked_knowledge_ids = tf.unstack(knowledge_ids_ph, top_n, axis=1)
		unstacked_knoweldge_mask = tf.unstack(knowledge_mask_ph, top_n, axis=1)
		unstacked_knowledge_seg_ids = tf.unstack(knowledge_seg_ids_ph, top_n, axis=1)

		bert_knowledges = []
		for i in range(top_n):
			bert_knowledges.append(BertModel(
				config=self.bert_config,
				is_training=is_training,
				input_ids=unstacked_knowledge_ids[i],
				input_mask=unstacked_knoweldge_mask[i],
				token_type_ids=unstacked_knowledge_seg_ids[i],
				use_one_hot_embeddings=use_one_hot_embeddings,
				scope='bert',
				hparams=self.hparams
			))

		bert_knowledge_cls = []
		bert_knowledge_seq_outputs = []
		for i in range(top_n):
			bert_knowledge_cls.append(bert_knowledges[i].get_pooled_output())
			bert_knowledge_seq_outputs.append(bert_knowledges[i].get_sequence_output())

		bert_knowledge_cls = tf.stack(bert_knowledge_cls, axis=1, name="bert_knowledge_cls")
		bert_knowledge_seq_outputs = tf.stack(bert_knowledge_seq_outputs, axis=1, name="bert_knowledge_seq_out")

		return bert_knowledge_cls, bert_knowledge_seq_outputs


	def _dialog_knowledge_attention(self, bert_dialog_cls, bert_knowledge_cls):
		# [batch, 768]
		expanded_dialog_cls = tf.expand_dims(bert_dialog_cls, axis=1)
		# [batch, 1, 768] * [batch, 768, 3] : batch, 1, 3
		attention_score = tf.matmul(expanded_dialog_cls, tf.transpose(bert_knowledge_cls, perm=[0,2,1]))
		attention_score = tf.nn.sigmoid(tf.squeeze(attention_score, axis=1))
		# batch, 3, 1 batch, 3, 768 -> batch, 768
		attended_result = tf.reduce_sum(
			tf.multiply(tf.expand_dims(attention_score, axis=-1), bert_knowledge_cls), axis=1)

		return attention_score, attended_result

	def _knowledge_loss(self, knowledge_att_score, knowledge_label_ids):
		"""
		:param knowledge_attention_score: [batch, 3]
		:param knowledge_label_ids: [batch, 3] : sigmoid
		:return:
		"""
		knowledge_loss_op = tf.losses.log_loss(
			labels=tf.cast(knowledge_label_ids, tf.float32), predictions=knowledge_att_score)
		return knowledge_loss_op

	def _bert_sentences_split(self, bert_sequence_output, max_seq_len_a, max_seq_len_b):
		"""
		:param bert_sequence_output: [batch, max_seq_len(300), 768]
		[CLS] Dialog [SEP] [PAD] ... Response [SEP] [PAD] ...
		:param max_seq_len_a: 280
		:param max_seq_len_b: 40
		:return: dialog_bert_outputs : [batch, max_seq_len_a, 768] | resposne_bert_outputs : [bert, max_seq_len_b, 768]
		"""
		dialog_bert_outputs, response_bert_outputs = tf.split(bert_sequence_output, [max_seq_len_a, max_seq_len_b], axis=1)

		return dialog_bert_outputs, response_bert_outputs

	def _final_output_layer(self, final_input_layer, knowledge_loss_op):

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

		main_loss_op = tf.reduce_mean(loss_op, name="cross_entropy_mean")
		loss_op = main_loss_op + knowledge_loss_op

		return logits, loss_op