import os
import re
import math
import nltk

class TFIDF(object):
	def __init__(self):
		self.documents_tf ={}
		self.corpus_df = {}
		self.docs_tf_idf = {}
		nltk.download('stopwords')
		stopwords = nltk.corpus.stopwords.words('english')

	def remove_string_special_char(self, string):
		stripped = re.sub('[^\w\s]', '', string)
		stripped = re.sub('_', '', stripped)
		stripped = re.sub('\s+', ' ', stripped)
		stripped = stripped.strip()

		return stripped

	def add_document(self, doc_name, document):
		# print(nltk.word_tokenize(document))
		document = self.remove_string_special_char(document)
		words = nltk.word_tokenize(document)

		# Build dictionary
		dictionary = {}
		document_words = set()
		for w in words:
			dictionary[w] = dictionary.get(w, 0.0) + 1.0
			document_words.add(w)

		for w in document_words:
			self.corpus_df[w] = self.corpus_df.get(w, 0.0) + 1.0

		self.documents_tf[doc_name] = dictionary

	def get_tf_idf(self, document_key, total_num_docs):
		# Get inverse document frequency
		tf_idf_document = {}
		document_dict = self.documents_tf[document_key]
		# print(document_dict)
		for word in document_dict:
			tf = document_dict[word]
			df = self.corpus_df[word]
			# print(tf, ":", df, ":", len(self.documents_tf))
			tf_idf_document[word] = tf * math.log10(total_num_docs / df)
		self.docs_tf_idf[document_key] = tf_idf_document

def get_dialog_dataset(data_path):
	dialog_data_l = []
	candidates_pool = set()
	current_dialog = []
	with open(data_path, "r", encoding='utf-8') as fr_handle:
		index = 0
		while True:
			index += 1
			line = fr_handle.readline().strip().split('\t')
			if line == ['']:
				print("End of dataset")
				break

			if index % 10000 == 0:
				print(index)

			ground_truth_flag = int(line[0])

			dialog_context = line[1:-1]
			response_candidate = line[-1].strip()
			candidates_pool.add(response_candidate)
			# print(line)

			processed_data = process_dialog(dialog_context, response_candidate, ground_truth_flag)
			dialog_data_l.append(processed_data)

		return dialog_data_l, candidates_pool


def process_dialog(dialog_context, response_candidate, ground_truth_flag):
	# dialog -> utterance + " \t "
	for i, utterance in enumerate(dialog_context):
		# if utterance == dialog_context[-1]:
		# 	dialog_context[i] = utterance.strip()
		# 	break
		# dialog_context[i] = utterance.strip() + " __eot__ "
		dialog_context[i] = utterance.strip()

	processed_data = [dialog_context, [response_candidate], ground_truth_flag]

	return processed_data

def make_utterance_pool():
	orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
	data_type = ["train"]

	utt_dict = dict()
	curr_pos = 0
	for t in data_type:
		dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t)
		for idx, dialog_data in enumerate(dialog_data_l):
			curr_pos += 1
			utterances = dialog_data[0]
			response = dialog_data[1]

			for utt in utterances:
				try:
					utt_dict[utt].append(idx)
				except KeyError:
					utt_dict[utt] = [idx]

			if curr_pos % 100000 == 0: print(len(utt_dict))
			if curr_pos == 1000000: exit()

if __name__ == '__main__':
	make_utterance_pool()