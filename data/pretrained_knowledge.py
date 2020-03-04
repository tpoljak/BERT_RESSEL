import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.insert(0, "/home/taesun/taesun_workspace/projects/bert_knowledge")
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
		self.ubuntu_manual_dict = dict()
		knowledge_dir = "./knowledge/ubuntu_manual_knowledge.txt"
		with open(knowledge_dir, "r", encoding="utf-8") as f_knowledge_handle:
			for knowledge in f_knowledge_handle:
				knowledge_split = knowledge.strip().split("\t")
				print(knowledge_split)
				knowledge_name = knowledge_split[0]
				knowledge_description = knowledge_split[1]
				self.ubuntu_manual_dict[knowledge_name] = knowledge_description
		print(len(self.ubuntu_manual_dict))
		def _create_knowledge_examples(ubuntu_manual_dict):
			examples = []
			knowledge_name_list = []
			for i, knowledge_name in enumerate(ubuntu_manual_dict.keys()):
				guid = "knowledge-%d" %  (i + 1)
				text_a = tokenization.convert_to_unicode(knowledge_name + " : " + ubuntu_manual_dict[knowledge_name])
				knowledge_name_list.append(knowledge_name)
				examples.append(InputExample(guid=guid, text_a=text_a))
			print("knowledge description data creation is finished! %d" % (len(examples)))

			return examples, knowledge_name_list

		self.knowledge_description, self.knowledge_name_list = _create_knowledge_examples(self.ubuntu_manual_dict)

	def _get_bert_knowledge_batch_data(self, curr_index, batch_size):
		input_ids = []
		input_mask = []
		segment_ids = []
		input_lengths = []

		example = self.knowledge_description

		for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
			# print(each_example.text_a)
			feature = convert_single_example(curr_index * batch_size + index, each_example, None,
																			 self.hparams.knowledge_max_seq_length, self.tokenizer, "knowledge")

			input_ids.append(feature.input_ids)
			input_mask.append(feature.input_mask)
			segment_ids.append(feature.segment_ids)
			input_lengths.append(feature.input_length)

		return [input_ids, input_mask, segment_ids, input_lengths], \
		       self.knowledge_name_list[curr_index * batch_size:batch_size * (curr_index + 1)]

	def _make_placeholders(self):
		self.knowledge_ids_ph = \
			tf.placeholder(tf.int32, shape=[None, self.hparams.knowledge_max_seq_length], name="knowledge_ids_ph")
		self.knowledge_mask_ph = \
			tf.placeholder(tf.int32, shape=[None, self.hparams.knowledge_max_seq_length], name="knowledge_mask_ph")
		self.knowledge_seg_ids_ph = \
			tf.placeholder(tf.int32, shape=[None, self.hparams.knowledge_max_seq_length], name="knowledge_seg_ids_ph")

	def _build_train_graph(self):
		with tf.device('/gpu:%d' % self.hparams.gpu_num[0]):
			bert_model = modeling_base.BertModel(
				config=self.bert_config,
				is_training=False,
				input_ids=self.knowledge_ids_ph,
				input_mask=self.knowledge_mask_ph,
				token_type_ids=self.knowledge_seg_ids_ph,
				use_one_hot_embeddings=False,
				scope='bert',
				hparams=self.hparams
			)
			self.bert_sequence_outputs = bert_model.get_sequence_output()

	def _make_feed_dict(self, knowledge_data):
		feed_dict = {}

		knowledge_ids, knowledge_mask, knowledge_seg_ids, knowledge_lengths = knowledge_data
		feed_dict[self.knowledge_ids_ph] = knowledge_ids
		feed_dict[self.knowledge_mask_ph] = knowledge_mask
		feed_dict[self.knowledge_seg_ids_ph] = knowledge_seg_ids

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

		total_knowledge_num = len(self.ubuntu_manual_dict)
		print(total_knowledge_num)
		total_train_num = math.ceil(total_knowledge_num/self.hparams.train_batch_size)
		print(total_train_num)

		index = 0
		with open("./knowledge/knowledge_bert_sequence_outputs.pickle", "wb") as fw_knowledge_handle:
			for i in range(total_train_num):
				knowledge_batch_data, knowledge_batch_name_list = self._get_bert_knowledge_batch_data(i, self.hparams.train_batch_size)
				sequence_val, = self.sess.run([self.bert_sequence_outputs], feed_dict=self._make_feed_dict(knowledge_batch_data))
				for name, sequence, knowledge_length in zip(knowledge_batch_name_list, sequence_val, knowledge_batch_data[-1]):
					pickle.dump([name, sequence, knowledge_length], fw_knowledge_handle)
					index += 1
					print(name, index, sequence.shape, knowledge_length)

			assert index == total_knowledge_num

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

def get_knowledge_sequence_outputs():
	knowledge_pickle_path = "./knowledge/knowledge_bert_sequence_outputs.pickle"

	manual_dict = dict()
	with open(knowledge_pickle_path,"rb") as fr_knowledge_handle:
		while True:
			try:
				ubuntu_manual, sequence_outputs, knowledge_length = pickle.load(fr_knowledge_handle)
				manual_dict[ubuntu_manual] = (sequence_outputs, knowledge_length)
			except EOFError:
				print("knowledge description loading is finished" , len(manual_dict))
				break

	return manual_dict

if __name__ == '__main__':
	train_model()
	# get_knowledge_sequence_outputs()