import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
import time
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from collections import OrderedDict
import pickle


class Callback(CallbackAny2Vec):
	'''Callback to print loss after each epoch.'''

	def __init__(self):
		self.epoch = 0
		self.start_time = time.time()
		self.loss = 0
		self.current_step = 0
		print("Model Train has been started!")

	def on_batch_begin(self, model: Word2Vec):
		self.current_step += 10000

	def on_epoch_end(self, model):
		current_loss = model.get_latest_training_loss()
		print('Loss after epoch %d: %f, %.3f %.2f(sec)' % (self.epoch, self.loss,
		                                                   (current_loss - self.loss) / self.current_step,
		                                                   time.time() - self.start_time))
		print("end epoch ", self.current_step)
		self.current_step = 0
		self.loss = current_loss
		self.epoch += 1
		self.start_time = time.time()
		if self.epoch % 25 == 0:
			model.save(
				"/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/word2vec_model_300_epoch_%s" % self.epoch)


init_checkpoint = "/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/bert_model.ckpt"
vocab_path = "/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/vocab.txt"
vocab_dict = OrderedDict()
with open(vocab_path, "r") as fr_handle:
	idx = 0
	for line in fr_handle:
		vocab_dict[line.strip()] = idx
		idx += 1
	print(idx)

need_to_initialize_vars = []
vars_in_checkpoint = tf.train.list_variables(init_checkpoint)
checkpoint_vars = []
for var_name, _ in vars_in_checkpoint:
	checkpoint_vars.append(var_name)
	print(var_name)

embedding_table = tf.get_variable(
	name="bert/embeddings/word_embeddings",
	shape=[30522, 768],
	initializer=tf.initializers.truncated_normal(0.02),
	trainable=False
)

var_dict = dict()
for var in tf.global_variables():
	print(var)
	var_dict[var.name[:-2]] = var

domain_embedding_table = tf.get_variable(
	initializer=embedding_table,
	trainable=True,
	name="domain_embeddings"
)

with open("/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/word2vec_300_embeddings", "rb") as frb_handle:
	word2vec_embeddings = pickle.load(frb_handle)
	print(word2vec_embeddings.shape)
word_embeddings_init = tf.constant(word2vec_embeddings, dtype=tf.float32)
word2vec_table = tf.get_variable(
	name="word2vec_embeddings",
	initializer=word_embeddings_init,
	trainable=True,
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_dict)
saver.restore(sess, init_checkpoint)
embedding_table_val, word2vec_table_val = sess.run([embedding_table, word2vec_table])
sess.close()

word2vec_table_val = np.array(word2vec_table_val)
print(word2vec_table_val[0])
print(word2vec_embeddings[0])
print(word2vec_table_val.shape)
print(np.sum(np.equal(word2vec_embeddings, word2vec_table_val)))

# embedding_dict = OrderedDict()
# bert_word_embedding = np.array(embedding_table_val)
# print(bert_word_embedding.shape)
# # print(embedding_table_val[0])
# for (vocab, embedding) in zip(vocab_dict.keys(), embedding_table_val):
# 	embedding_dict[vocab] = embedding
#
# word2vec_word_embedding = list()
# model = Word2Vec.load("/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/word2vec_model_300_epoch_100")
# print(len(model.wv.vocab))
# print(len(vocab_dict))
# for idx, vocab in enumerate(vocab_dict.keys()):
# 	if vocab in model.wv.vocab:
# 		word2vec_word_embedding.append(model.wv[vocab])
# 	else:
# 		word2vec_word_embedding.append(list(np.random.normal(-1, 1, 300)))
#
# word2vec_word_embedding = np.array(word2vec_word_embedding)
# print(word2vec_word_embedding.shape)
#
# print(embedding_dict['computer'])
# print(word2vec_word_embedding[vocab_dict['computer']])



# with open("/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/word2vec_300_embeddings", "wb") as fwb_handle:
# 	pickle.dump(word2vec_word_embedding, fwb_handle)

	# print(model.most_similar(positive=['problem']))
# domain_embedding_table = tf.assign([v for v in tf.global_variables() if v.name == "domain_embeddings:0"][0],
#                                    embedding_table_val)
#
# embedding_table_val, domain_embedding_table_val = sess.run([embedding_table, domain_embedding_table])
# print(embedding_table_val)
# print(domain_embedding_table_val)