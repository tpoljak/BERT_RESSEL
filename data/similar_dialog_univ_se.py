import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import pickle
import time
from _datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from data.data_utils import get_dialog_dataset

class SentenceEmbedder:
	ENCODER_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
	# @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
	batch_size = 512

	_g = tf.Graph()
	with _g.as_default():
		_input_ph = tf.placeholder(dtype=tf.string, shape=[None])
		_embedder = hub.Module(ENCODER_MODULE_URL)
		_embedding_op = _embedder(_input_ph)
		_init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

		_vocab_op = _g.get_tensor_by_name("module/Embeddings_en_words:0")
	_g.finalize()

	tf.logging.set_verbosity(tf.logging.ERROR)
	_sess_config = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement=False
	)
	_sess_config.gpu_options.allow_growth = True
	_session = tf.Session(config=_sess_config, graph=_g)
	_session.run(_init_op)

	vocab = _session.run(_vocab_op)
	vocab = set([v.decode("utf-8") for v in vocab])

	def __init__(self):
		raise ValueError("This is a static class.")

	@staticmethod
	def embed(sentences) -> np.ndarray:
		buffer = []
		num_steps = int(math.ceil(len(sentences) / SentenceEmbedder.batch_size))
		print("num_steps", num_steps)
		for step in range(num_steps):
			start = step * SentenceEmbedder.batch_size
			end = min((step + 1) * SentenceEmbedder.batch_size, len(sentences))
			embeddings = SentenceEmbedder._session.run(
				SentenceEmbedder._embedding_op,
				feed_dict={SentenceEmbedder._input_ph: sentences[start:end]})
			buffer.append(embeddings)
			if (step+1) % 1500 == 0:
				print(step)

		sentence_embeddings = np.concatenate(buffer, axis=0)
		assert len(sentences) == sentence_embeddings.shape[0]
		assert len(sentence_embeddings.shape) == 2

		return sentence_embeddings

def make_dialog_sentence_embeddings():
	orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
	data_type = ["train", "valid", "test"]
	print("make_universial_sentence_encoder")

	dialog_context_l = []
	for t in data_type:
		dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t, is_eot=True)
		print(len(dialog_data_l))

		for idx, dialog_data in enumerate(dialog_data_l):
			if dialog_data[2] != 1:
				continue

			utterances = dialog_data[0]
			dialog_context = ""
			for utt in utterances:
				dialog_context += utt

			dialog_context = dialog_context.strip()
			dialog_context_l.append(dialog_context)

		print("%s data load finished! " % t, len(dialog_context_l))
	print("Universial Sentence Encoder(Dialog_Embeddings)!")

	print("total length", len(dialog_context_l)) #TODO:should be 600,000
	dialog_context_embeddings = SentenceEmbedder.embed(dialog_context_l)
	print(dialog_context_embeddings.shape)
	dialog_se_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/dialog_sentence_encoding.pickle"

	with open(dialog_se_path, "wb") as fw_handle:
		pickle.dump(dialog_context_embeddings, fw_handle)

def make_response_sentence_embeddings():
	orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
	data_type = ["train"]
	print("make_universial_sentence_encoder")
	for t in data_type:
		dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t)
		print(len(dialog_data_l))

		response_l = []
		response_dict = dict()
		current_dialog_pos = 0
		for idx, dialog_data in enumerate(dialog_data_l):
			if dialog_data[2] == 0: continue
			response = dialog_data[1]
			response_l.extend(response)
			response_dict[current_dialog_pos] = response[0]

		print("response total num : ", len(response_l))
		response_embeddings = SentenceEmbedder.embed(response_l)
		print(response_embeddings.shape)
		response_se_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/response_sentence_encoding.pickle"
		with open(response_se_path, "wb") as fw_rse_handle:
			pickle.dump(response_embeddings, fw_rse_handle)

def make_utterance_sentence_embeddings():
	orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
	data_type = ["train"]
	print("make_universial_sentence_encoding by dialog utterances")

	for t in data_type:
		dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t)
		print(dialog_data_l[0])

		utt_set = set()
		utt_dict = dict()
		current_dialog_pos = 0
		for idx, dialog_data in enumerate(dialog_data_l):
			if t == "train": num = 2
			else: num = 10
			if idx % num != 0: continue
			current_dialog_pos += 1
			utterances = dialog_data[0]
			response = dialog_data[1]
			utterances.extend(response)

			utt_set.update(utterances)
			for utt in utterances:
				try:
					if current_dialog_pos in utt_dict[utt]: continue
					utt_dict[utt].append(current_dialog_pos)
				except KeyError:
					utt_dict[utt] = [current_dialog_pos]

		# utterance_index, utterance(sentence), utterance_dictionary

		print("total number of dialogues: %d" % current_dialog_pos)
		print("total number of utterances : %d" % len(utt_set))
		print("total number of utterances_dict :", len(utt_dict) )
		utt_l = list(utt_set)
		utt_embeddings = SentenceEmbedder.embed(utt_l)
		print(utt_embeddings.shape)
		utterances_se_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/utterance_sentence_encoding.pickle"

		with open(utterances_se_path, "wb") as fw_use_handle:
			temp_utt = []
			temp_utt_emb = []
			for idx, (utt, utt_emb) in enumerate(zip(utt_l, utt_embeddings)):
				temp_utt.append(utt)
				temp_utt_emb.append(utt_emb)
				if (idx + 1) % 100000 == 0:
					pickle.dump([temp_utt, temp_utt_emb], fw_use_handle)
					temp_utt = []
					temp_utt_emb = []
			pickle.dump([temp_utt, temp_utt_emb], fw_use_handle)

		utterances_dict_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/utteranace_dict.pickle"
		with open(utterances_dict_path, "wb") as fw_dict_handle:
			pickle.dump(utt_dict, fw_dict_handle)

		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
		print(timestamp, "completes!")

def get_dialog_response_list():
	orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
	data_type = ["train", "valid", "test"]

	dialog_l = []
	response_l = []
	for t in data_type:
		dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t)
		for idx, dialog_data in enumerate(dialog_data_l):
			if dialog_data[2] != 1: continue
			dialog_l.append(dialog_data[0])
			response_l.extend(dialog_data[1])

		print("%s data load finished! " % t, len(dialog_l))
	print("total data length", len(dialog_l))

	return dialog_l, response_l

class SentenceCosineSim(object):
	def __init__(self, curr_embeddings, target_embeddings):
		self.curr_batch_size = 6000
		self.target_batch_size = 100000
		self.sort_batch_size = 10
		self.curr_embeddings = curr_embeddings
		self.target_embeddings = target_embeddings
		print(curr_embeddings.shape)
		print(target_embeddings.shape)
		self.curr_utt_ph = tf.placeholder(tf.float32, shape=[None, 512], name="curr_ph")
		self.target_utt_ph = tf.placeholder(tf.float32, shape=[None, 512], name="target_ph")
		self.target_sim_ph = tf.placeholder(tf.float32, shape=[None, None], name="target_sim_ph")

	def pairwise_calculation(self, end_idx=100):
		self.end_idx = end_idx
		config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False,
		)
		config.gpu_options.allow_growth = True

		sess = tf.Session(config=config)
		self._consine_sim(self.curr_utt_ph, self.target_utt_ph)
		self._sort_cosine_sim_outputs(self.target_sim_ph)

		total_curr_num = len(self.curr_embeddings)
		print("total_curr_num : ", total_curr_num)

		total_buffer = []
		total_score_buffer = []
		total_args_buffer = []
		curr_num_steps = math.ceil(total_curr_num / self.curr_batch_size)
		print("curr_num_steps", curr_num_steps)
		for curr_step in range(curr_num_steps):
			print('-'*200)
			print("remained_curr_step : ", (curr_num_steps - curr_step))
			# curr_utt should be the current utterances
			curr_start = curr_step * self.curr_batch_size
			curr_end = min((curr_step + 1) * self.curr_batch_size, total_curr_num)
			total_target_num = len(self.target_embeddings)
			target_num_steps = math.ceil(total_target_num / self.target_batch_size)

			buffer_sim = []
			buffer_target_score = []
			buffer_target_args = []
			# print("target_num_steps", target_num_steps)
			for target_step in range(target_num_steps):
				target_start = target_step * self.target_batch_size
				target_end = min((target_step + 1) * self.target_batch_size, total_target_num)
				#[500, 1000]
				sim_val, = sess.run([self.angular_distance],
													 feed_dict={self.curr_utt_ph:self.curr_embeddings[curr_start:curr_end],
																			self.target_utt_ph:self.target_embeddings[target_start:target_end]})
				buffer_sim.append(sim_val)
				print(target_num_steps - target_step, " : ", np.array(sim_val).shape)

			total_target_score = np.concatenate(buffer_sim, axis=1)
			# print("total_target_score", np.array(total_target_score).shape)
			sort_num_steps = math.ceil(np.shape(total_target_score)[0]/self.sort_batch_size)
			# print("sort_steps", sort_num_steps)
			for sort_step in range(sort_num_steps):
				sort_start = sort_step * self.sort_batch_size
				sort_end = min((sort_step + 1) * self.sort_batch_size, np.shape(total_target_score)[0])

				sorted_score_val, sorted_args_val = sess.run([self.sorted_sim_score, self.sorted_sim_args],
																										 feed_dict={self.target_sim_ph:total_target_score[sort_start:sort_end]})
				# print(sort_num_steps - sort_step, " : ", np.array(sorted_score_val).shape)
				buffer_target_score.append(sorted_score_val)
				buffer_target_args.append(sorted_args_val)
			total_target_score = np.concatenate(buffer_target_score, axis=0)
			total_target_args = np.concatenate(buffer_target_args, axis=0)

			total_score_buffer.append(total_target_score)
			total_args_buffer.append(total_target_args)

			print(total_target_score.shape, total_target_args.shape)
			print("total_buffer_len", len(total_score_buffer))
			# if curr_step > 2: break

		# 500000, 100, 2(score, arg)
		tot_score_val = np.concatenate(total_score_buffer, axis=0)
		tot_args_val = np.concatenate(total_args_buffer, axis=0)
		print(tot_score_val.shape, tot_args_val.shape)

		return tot_score_val, tot_args_val

	def _consine_sim(self, curr_dialog, target_dialog):
		"""
		:param curr_dialog: [batch, 512]
		:param target_dialog: [5000, 512]
		:return:
		"""
		dot_prod = tf.matmul(curr_dialog, tf.transpose(target_dialog, perm=[1,0]))
		l2_norm = tf.matmul(tf.expand_dims(tf.linalg.norm(curr_dialog, ord=2, axis=-1), axis=-1),
												tf.expand_dims(tf.linalg.norm(target_dialog, ord=2, axis=-1), axis=0))
		# self.angular_distance = 1 - tf.math.acos(tf.divide(dot_prod, l2_norm) / math.pi)
		# [5000, 100000]
		cosine_sim = tf.math.minimum(tf.divide(dot_prod, l2_norm), 1.0)
		ones = tf.expand_dims(tf.tile(tf.constant([1.0], dtype=tf.float32), multiples=[tf.shape(dot_prod)[0]]), axis=-1)
		pi = tf.expand_dims(tf.tile(tf.constant([math.pi], dtype=tf.float32), multiples=[tf.shape(dot_prod)[0]]), axis=-1)
		self.angular_distance = ones - tf.divide(tf.math.acos(cosine_sim), pi)
		# self._sort_cosine_sim_outputs(self.angular_distance)

	def _sort_cosine_sim_outputs(self, sim_outputs):
		print(sim_outputs.shape)
		self.sorted_sim_score = tf.slice(tf.contrib.framework.sort(sim_outputs, axis=-1, direction="DESCENDING"),
																		 [0,0], [tf.shape(sim_outputs)[0], self.end_idx])
		self.sorted_sim_args = tf.slice(tf.contrib.framework.argsort(sim_outputs, axis=-1, direction="DESCENDING"),
																		[0,0], [tf.shape(sim_outputs)[0], self.end_idx])

class DialogCosineSim(object):
	def __init__(self, dialog_context_embeddings):
		self.batch_size = 250000
		self.dialog_context_embeddings = dialog_context_embeddings
		self.curr_dialog_ph = tf.placeholder(tf.float32, shape=[512], name="curr_dialog_ph")
		self.target_dialog_ph = tf.placeholder(tf.float32, shape=[None, 512], name="target_dialog_ph")

	def pairwise_calculation(self):
		config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False,
		)
		config.gpu_options.allow_growth = True

		sess = tf.Session(config=config)
		self._consine_sim(self.curr_dialog_ph, self.target_dialog_ph)

		total_dialog_num = len(self.dialog_context_embeddings)
		target_dialog = self.dialog_context_embeddings #[500000, 512]
		for i in range(np.shape(self.dialog_context_embeddings)[0]):
			curr_dialog = self.dialog_context_embeddings[i]

			buffer = []
			num_steps = math.ceil(total_dialog_num / self.batch_size)
			# print("num_steps", num_steps)
			for step in range(num_steps):
				start = step * self.batch_size
				end = min((step + 1) * self.batch_size, len(target_dialog))
				angular_distance_val, = sess.run([self.angular_distance],
																				 feed_dict={self.curr_dialog_ph:curr_dialog,
																										self.target_dialog_ph:target_dialog[start:end]})
				buffer.append(angular_distance_val)
			dialog_sim_val = np.concatenate(buffer, axis=0)
			# print(np.stack([np.argsort(-dialog_cosine_sim_val)[0:100], -np.sort(-dialog_cosine_sim_val)[0:100]], axis=1))
			print(i, sorted(range(len(dialog_sim_val)), key=lambda x: dialog_sim_val[x], reverse=True)[0:100])
			print(i, sorted(dialog_sim_val, reverse=True)[0:100])
			print('-'*200)
		assert "pairwise calculation"

	def _consine_sim(self, curr_dialog, target_dialog):
		"""
		:param curr_dialog: [batch, 768]
		:param target_dialog: [batch, 768]
		:return:
		"""
		tiled_curr_dialog = tf.tile(tf.expand_dims(curr_dialog, axis=0), multiples=[tf.shape(target_dialog)[0], 1])
		dot_prod = tf.reduce_sum(tf.multiply(tiled_curr_dialog, target_dialog), axis=-1)
		l2_norm = tf.multiply(tf.linalg.norm(tiled_curr_dialog, ord=2, axis=-1), tf.linalg.norm(target_dialog, ord=2, axis=-1))
		# self.angular_distance = 1 - tf.math.acos(tf.divide(dot_prod, l2_norm) / math.pi)
		self.angular_distance = tf.divide(dot_prod, l2_norm)

if __name__ == '__main__':
		# make_dialog_sentence_embeddings()

		dialog_l, response_l = get_dialog_response_list()
		similar_dialog_sort_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_sort_1000.pickle"
		if not os.path.exists(similar_dialog_sort_path):
			dialog_se_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/dialog_sentence_encoding.pickle"
			with open(dialog_se_path, "rb") as fr_dse_handle:
				dialog_emb = pickle.load(fr_dse_handle)  # total 500000
			print(np.array(dialog_emb).shape)

			scs_dialog = SentenceCosineSim(dialog_emb, dialog_emb[0:500000])
			tot_score_val, tot_args_val = scs_dialog.pairwise_calculation(1000)

			with open(similar_dialog_sort_path, "wb") as fw_sds_handle:
				pickle.dump([tot_score_val, tot_args_val], fw_sds_handle)
				print("similar_dialog_sort file save completes")
		else:
			with open(similar_dialog_sort_path, "rb") as fr_sds_handle:
				tot_score_val, tot_args_val= pickle.load(fr_sds_handle)
				print("similar_dialog_sort file load completes")

		dialog_dict = dict()
		for idx, (score, arg) in enumerate(zip(tot_score_val, tot_args_val)):
			dialog_dict[idx] = dict()
			for sub_idx, (sub_score, sub_arg) in enumerate(zip(score, arg)):
				dialog_dict[idx][sub_idx] = {"dialog_str":dialog_l[sub_arg], "dialog_score":sub_score,
																		 "dialog_arg":sub_arg, "response_str":response_l[sub_arg]}
