import os
import pickle
import time
from collections import OrderedDict

import nltk

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
		#dialog -> utterance + " \t "
		for i, utterance in enumerate(dialog_context):
				if utterance == dialog_context[-1]:
						dialog_context[i] = utterance.strip()
						break
				dialog_context[i] = utterance.strip() + " eot "
				# dialog_context[i] = utterance.strip()

		processed_data = [dialog_context, [response_candidate], ground_truth_flag]

		return processed_data

def get_stat_knowledge_relevance():
	def filtering_common_words_from_knowledge():
		def get_ubuntu_man_dict():
			knowledge_path = "./knowledge/ubuntu_manual_knowledge.txt"
			ubuntu_knowledge_dict = dict()

			with open(knowledge_path, "r", encoding="utf-8") as f_handle:
				for line in f_handle:
					ubuntu_man = line.strip().split("\t")
					if len(ubuntu_man) == 2:
						ubuntu_knowledge_dict[ubuntu_man[0]] = ubuntu_man[1]
			# print(ubuntu_knowledge_dict["lsmod"])
			# ubuntu_knowledge_dict["lsmod"] = "lsmod is a trivial program which nicely formats the contents of the /proc/modules," \
			#                                  " showing what kernel modules are currently loaded."
			# print(sorted(ubuntu_knowledge_dict))
			# with open(knowledge_path, "w", encoding="utf-8") as f_handle:
			#   for manual_key in sorted(ubuntu_knowledge_dict):
			#     f_handle.write(manual_key + "\t" + ubuntu_knowledge_dict[manual_key] + "\n")

			return ubuntu_knowledge_dict

		ubuntu_knowledge_dict = get_ubuntu_man_dict()
		common_words_path = "./knowledge/google-10000-english.txt"
		common_words = set()
		with open(common_words_path, "r", encoding="utf-8") as f_handle:
			for idx, line in enumerate(f_handle):
				if idx == 1000: break
				common_words.add(line.strip())
		count = 0
		original_knowledge_num = len(ubuntu_knowledge_dict)
		for word in common_words:
			if word in ubuntu_knowledge_dict:
				del ubuntu_knowledge_dict[word]
				print(word)
				count += 1
		print("total ubuntu knowledge which is deleted : %d(out of %d)" % (count, original_knowledge_num))

		return ubuntu_knowledge_dict

	print("Loading tf_idf score, total dialog sentence list...")

	ubuntu_knowledge_dict = filtering_common_words_from_knowledge()

	orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
	data_type = ["train"]

	for t in data_type:
		dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t)

		dialog_token_common = 0
		common_dialog_num = 0
		dialog_description_common = 0
		common_description_num = 0

		for idx, dialog_data in enumerate(dialog_data_l):
			dialog_knowledge_token_set = set()
			response_knowledge_token_set = set()

			each_dialog_description_common = 0
			utterances = dialog_data[0]
			response = dialog_data[1][0]
			label = str(dialog_data[2])

			for utt in utterances:
				for token in nltk.word_tokenize(utt.strip()):
					if token in ubuntu_knowledge_dict.keys():
						dialog_knowledge_token_set.add(token)

			for token in nltk.word_tokenize(response):
				if token in ubuntu_knowledge_dict.keys():
					response_knowledge_token_set.add(token)

			if len(dialog_knowledge_token_set.intersection(response_knowledge_token_set)) > 0:
				common_dialog_num += 1

			dialog_token_common+= len(dialog_knowledge_token_set.intersection(response_knowledge_token_set))

			for dialog_knowledge in dialog_knowledge_token_set:
				for token in nltk.word_tokenize(ubuntu_knowledge_dict[dialog_knowledge]):
					if token in response_knowledge_token_set:
						each_dialog_description_common += 1

			if each_dialog_description_common > 0:
				common_description_num += 1
			dialog_description_common += each_dialog_description_common

			if (idx + 1) % 10000 == 0:
				print(idx + 1, "dialog_token_common : ", dialog_token_common, "common_dialog_num : ", common_dialog_num,
				      "dialog_description_common : ", dialog_description_common, "common_description_num", common_description_num)


if __name__ == '__main__':
	get_stat_knowledge_relevance()