import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.insert(0, "/home/taesun/taesun_workspace/bert_knowledge")
print(sys.path)
import logging
import time
from datetime import datetime

from model import model_params

from utils import *

from data_process import *
from bert_model import tokenization, optimization, modeling_base

class BertKnowledgeModel(object):
	def __init__(self, hparams):
		self.hparams = hparams
		self.bert_config = modeling_base.BertConfig.from_json_file(self.hparams.bert_config_dir)
		self.tokenizer = tokenization.FullTokenizer(self.hparams.vocab_dir, self.hparams.do_lower_case)

		self._make_data_processor()

	def _make_data_processor(self):
		train_data_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_knowledge_multi_turn"
		t = "train"
		ubuntu_data = []
		with open(os.path.join(train_data_path, "bert_train.pickle"), "rb") as frb_handle:
			index = 0
			while True:
				if index != 0 and index % 100000 == 0:
					print("%d data has been loaded now" % index)
				try:
					index += 1
					ubuntu_data.append(pickle.load(frb_handle))

				except EOFError:
					print("%s data loading is finished!" % t)
					break

		def _create_train_data_examples(ubuntu_data):
			examples = []
			count = 0
			print("total length of ubuntu dialog data")
			for i, dialog_data in enumerate(ubuntu_data):
				if i % 2 != 0:
					continue
				count += 1
				guid = "train_dialog-%d" % count
				text_a = tokenization.convert_to_unicode(dialog_data[0])
				examples.append(InputExample(guid=guid, text_a=text_a))
			print("knowledge description data creation is finished! %d" % (len(examples)))

			return examples

		self.dialog_train_examples = _create_train_data_examples(ubuntu_data)

	def _get_bert_single_batch_data(self, curr_index, batch_size):
		input_ids = []
		input_mask = []
		segment_ids = []
		input_lengths = []

		example = self.dialog_train_examples
		example_string = []
		example_id = []
		for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
			feature = convert_single_example(curr_index * batch_size + index, each_example, None,
																			 self.hparams.max_seq_length, self.tokenizer, "dialog")

			input_ids.append(feature.input_ids)
			input_mask.append(feature.input_mask)
			segment_ids.append(feature.segment_ids)
			input_lengths.append(feature.input_length)
			example_string.append(each_example.text_a)
			example_id.append(each_example.guid)

		return [input_ids, input_mask, segment_ids, input_lengths], example_string, example_id

	def _make_placeholders(self):
		self.input_ids_ph = \
			tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="input_ids_ph")
		self.input_mask_ph = \
			tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="input_mask_ph")
		self.segment_ids_ph = \
			tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="knowledge_seg_ids_ph")

	def _build_train_graph(self):
		with tf.device('/gpu:%d' % self.hparams.gpu_num[0]):
			bert_model = modeling_base.BertModel(
				config=self.bert_config,
				is_training=False,
				input_ids=self.input_ids_ph,
				input_mask=self.input_mask_ph,
				token_type_ids=self.segment_ids_ph,
				use_one_hot_embeddings=False,
				scope='bert',
				hparams=self.hparams
			)
			self.pooled_outputs = bert_model.get_pooled_output()

	def _make_feed_dict(self, batch_data):
		feed_dict = {}

		input_ids, input_mask, input_segment_ids, input_lengths = batch_data
		feed_dict[self.input_ids_ph] = input_ids
		feed_dict[self.input_mask_ph] = input_mask
		feed_dict[self.segment_ids_ph] = input_segment_ids

		return feed_dict

	def train(self, pretrained_file):
		config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False,
		)
		config.gpu_options.allow_growth = True

		self.sess = tf.Session(config=config)
		self._make_placeholders()
		self._build_train_graph()

		# Tensorboard
		saver = tf.train.Saver()
		saver.restore(self.sess, pretrained_file)
		print("Restoring Session from checkpoint complete!")

		total_num = len(self.dialog_train_examples)
		print(total_num)
		batch_data_size = 300
		total_train_num = math.ceil(total_num/batch_data_size)
		print(total_train_num)

		train_dialog_dict = dict()
		index = 0
		for i in range(total_train_num):
			batch_data, example_string, example_id = self._get_bert_single_batch_data(i, batch_data_size)
			cls_val, = self.sess.run([self.pooled_outputs], feed_dict=self._make_feed_dict(batch_data))
			# print(index, ": ", np.array(cls_val).shape)
			for each_example_id, each_example, cls in zip(example_id, example_string, cls_val):
				train_dialog_dict[each_example_id] = (each_example, cls)
				index += 1
				# print(each_example_id," : ", train_dialog_dict[each_example_id][0])
			print(index, ": ", np.array(cls_val).shape)
			if index == 30000: break

		# assert index == total_num

		with open("./train_bert_cls_temp.pickle", "wb") as fw_handle:
			pickle.dump(train_dialog_dict, fw_handle)
			print("train bert cls encodings save complete! Total length of the train dialog", len(train_dialog_dict))

		if self.sess is not None:
			self.sess.close()

PARAMS_MAP = {
		"base": model_params.BASE_PARAMS,
		"dialog" : model_params.MODEL_DIALOG_PARAMS,
		"adapter" : model_params.MODEL_ADAPTER_PARAMS,
		"test" : model_params.MODEL_TRAIN_TEST_PARAMS,
		"knowledge_lstm" : model_params.MODEL_KNOWLEDGE_LSTM,
		"eval_base": model_params.EVAL_PARAMS,
		"eval_dialog": model_params.EVAL_DIALOG_PARAMS,
		"eval_adapter": model_params.EVAL_ADAPTER_PARAMS,
		"eval_test" : model_params.EVAL_TRAIN_TEST,
}

def train_model():
	hparams = PARAMS_MAP["base"]
	hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
	model = BertKnowledgeModel(hparams)
	model.train(pretrained_file=hparams.init_checkpoint)

def get_dialog_cls_token():
	dialog_pickle_path = "./train_bert_cls_temp.pickle"

	manual_dict = dict()
	with open(dialog_pickle_path, "rb") as fr_handle:
		print("Train dialog cls token encoding loading... it will take a few minutes...")
		train_dialog_dict = pickle.load(fr_handle)
		print("Train dialog loading is finished")

	return train_dialog_dict

def find_similar_dialog():
	train_dialog_dict = get_dialog_cls_token()
	sample_string, sample_cls_emb = train_dialog_dict["train_dialog-1"]
	similar_score_value_dict = dict()
	for idx, dialog_guid in enumerate(train_dialog_dict.keys()):
		if idx == 0: continue
		similar_score_value_dict[dialog_guid] = np.divide(np.sum(np.multiply(sample_cls_emb, train_dialog_dict[dialog_guid][1])),
		                                                  np.multiply(np.linalg.norm(sample_cls_emb,2), np.linalg.norm(train_dialog_dict[dialog_guid][1])))
		if idx % 1000 == 0: print(similar_score_value_dict[dialog_guid])
	print(len(similar_score_value_dict))

	sorted_sim_score = sorted(similar_score_value_dict.items(), lambda x:x[0], reverse=True)


if __name__ == '__main__':
	# train_model()
	find_similar_dialog()