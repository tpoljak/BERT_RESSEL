import tensorflow as tf

from utils import stack_bidirectional_rnn, sequence_feature
from model.basic_encoder import BasicEncoder

class ESIMAttention(object):
	def __init__(self, hparams, dropout_keep_prob, text_a, text_a_len, text_b, text_b_len, only_text_a=True):
		self.hparams = hparams
		self.dropout_keep_prob = dropout_keep_prob

		self.matching_encoder = BasicEncoder(self.hparams, self.dropout_keep_prob)
		m_text_a, m_text_b = self._attention_matching_layer(text_a, text_b)

		m_fw_text_a_state, m_bw_text_a_state = self._matching_aggregation_a_layer(m_text_a, text_a_len)
		self.text_a_att_outs = tf.concat([m_fw_text_a_state, m_bw_text_a_state], axis=-1)

	def _attention_text_b(self, similarity_matrix, text_a):
		"""
		:param similarity_matrix: [batch_size, max_text_b_len, max_text_a_len]
		:param text_a: [batch_size, max_text_a_len, hidden_dim]
		:return:
		"""
		attention_weight_text_a = \
			tf.where(tf.equal(similarity_matrix, 0.), similarity_matrix, tf.nn.softmax(similarity_matrix))

		attended_text_b = tf.matmul(attention_weight_text_a, text_a)

		return attended_text_b

	def _attention_text_a(self, similarity_matrix, text_b):
		"""
		:param similarity_matrix: [batch_size, max_text_b_len, max_text_a_len]
		:param text_b: [batch_size, max_text_b_len, hidden_dim]
		:return: attend_text_a
		"""
		sim_trans_mat = tf.transpose(similarity_matrix, perm=[0, 2, 1])
		attention_weight_text_b = \
			tf.where(tf.equal(sim_trans_mat, 0.), sim_trans_mat, tf.nn.softmax(sim_trans_mat))

		attended_text_a = tf.matmul(attention_weight_text_b, text_b)

		return attended_text_a

	def _similarity_matrix(self, text_a, text_b):
		"""
		Dot attention : text_a, text_b bert
		:param text_a: [batch, max_text_a_len, 768]
		:param text_b: [batch, max_text_b_len, 768]
		:return: similarity_matrix #[batch, max_text_b_len, max_text_a_len]
		"""
		similarity_matrix = tf.matmul(text_b, tf.transpose(text_a, perm=[0, 2, 1]))

		return similarity_matrix

	def _attention_matching_layer(self, text_a, text_b):
		similarity = self._similarity_matrix(text_a, text_b)
		# shape: [batch, max_text_a_len, 768]
		attended_text_a = self._attention_text_a(similarity, text_b)
		# shape: [batch, max_text_b_len, 768]
		attended_text_b = self._attention_text_b(similarity, text_a)

		m_text_a = tf.concat(axis=-1, values=[text_a, attended_text_a,
																					text_a - attended_text_a,
																					tf.multiply(text_a, attended_text_a)])

		m_text_b = tf.concat(axis=-1, values=[text_b, attended_text_b,
																					text_b - attended_text_b,
																					tf.multiply(text_b, attended_text_b)])

		return m_text_a, m_text_b

	def _matching_aggregation_a_layer(self, m_text_a, text_a_len):
		"""text_a_matching"""
		m_text_a_lstm_outputs = self.matching_encoder.lstm_encoder(m_text_a, text_a_len, name="text_a_matching")
		m_text_a_max = tf.reduce_max(m_text_a_lstm_outputs, axis=1)
		m_fw_text_a_state, m_bw_text_a_state = sequence_feature(m_text_a_lstm_outputs, text_a_len)

		return m_fw_text_a_state, m_bw_text_a_state

	def _matching_aggregation_b_layer(self, m_text_b, text_b_len):
		"""text_b_matching"""
		m_text_b_lstm_outputs = self.matching_encoder.lstm_encoder(m_text_b, text_b_len, name="text_b_matching")
		m_text_b_max = tf.reduce_max(m_text_b_lstm_outputs, axis=1)
		m_fw_text_b_state, m_bw_text_b_state = sequence_feature(m_text_b_lstm_outputs, text_b_len)

		return m_fw_text_b_state, m_bw_text_b_state