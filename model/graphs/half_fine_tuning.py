from bert_model.modeling_base import *

import tensorflow as tf

class Model(object):
	def __init__(self, hparams, bert_config=None, bert_model=None, input_phs=None, label_id_phs=None,
	             length_phs=None, knowledge_phs=None, similar_phs=None):
		self.hparams = hparams
		self.bert_config = bert_config
		self.bert_model = bert_model
		self.label_id_phs = label_id_phs
		self.length_phs=length_phs[1] # response
		self.knowledge_phs = knowledge_phs

	def build_graph(self):
		pooled_output = self.bert_model.get_pooled_output()
		logits, loss_op = self._final_output_layer(pooled_output, self.label_id_phs)

		sequence_output = self.bert_model.all_encoder_layers[-1] # batch, 32

		# sequence_output = tf.unstack(sequence_output, num=4, axis=0)
		# sequence_output = tf.concat(sequence_output, axis=-1)
		# print(sequence_output.shape)
		# sequence_mask = tf.sequence_mask(self.length_phs, maxlen=320,dtype=tf.float32) # batch, 320
		# sequence_output = sequence_output * tf.expand_dims(sequence_mask, axis=-1)

		# sentence_argmax = tf.argmax(sequence_output, axis=1) # batch, 768

		# return logits, loss_op, sequence_output, sentence_argmax
		return logits, sequence_output


	def _final_output_layer(self, final_input_layer, label_ids):
		dialog_label_ids, knowledge_label_ids = label_ids
		output_layer = final_input_layer

		# if not self.hparams.do_evaluate:
		# 	# I.e., 0.1 dropout
		# 	print("output_layer dropout!! : 0.9")
		# 	output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

		if self.hparams.loss_type == "sigmoid":
			logits_units = 1
		else:
			logits_units = 2

		logits = tf.layers.dense(
			inputs=output_layer,
			units=logits_units,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
			name="logits"
		)

		if self.hparams.loss_type == "sigmoid":
			logits = tf.squeeze(logits, axis=-1)
			loss_op = tf.nn.sigmoid_cross_entropy_with_logits(
				logits=logits, labels=tf.cast(dialog_label_ids, tf.float32), name="binary_cross_entropy")
		else:
			loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits, labels=dialog_label_ids, name="cross_entropy")
		loss_op = tf.reduce_mean(loss_op, name="cross_entropy_mean")

		return logits, loss_op