from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import random

import time
import pickle
import joblib

ubuntu_tokenized_data_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/tokenized_train_data_320_eot"
# advising_tokenized_data_path = "/mnt/raid5/taesun/data/Advising/bert_pretrain/tokenized_train_data_320_eot"
vocab_path = "/mnt/raid5/taesun/"
# tokenized_documents_list
with open(ubuntu_tokenized_data_path, "rb") as document_fr_handle:
	word2vec_tok_doc = []
	tokenized_documents = pickle.load(document_fr_handle)
	for doc in tokenized_documents:
		processed_doc = []
		for utt in doc:
			processed_doc.extend(utt)
			print(utt)
		time.sleep(10)
		print('-'*200)

		word2vec_tok_doc.append(processed_doc)
	print("Data loading has been completed! %d"% len(tokenized_documents))


# if __name__ == '__main__':
# for i in range(50):
# 	random.shuffle(word2vec_tok_doc)
embedding_model = Word2Vec(word2vec_tok_doc, size=300, window=5, min_count=1, workers=8, iter=100, sg=1,
													 compute_loss=True, callbacks=[Callback()], batch_words=10000)

# model = Word2Vec.load("/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/word2vec_model_300_epoch_100")

# print(model.most_similar(positive=['problem']))