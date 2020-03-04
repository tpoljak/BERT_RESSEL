import os
import pickle
import nltk
import math
import re
import time
import numpy as np
from multiprocessing import Process, Lock, Value

from data.data_utils import get_dialog_dataset

def get_dialog_response_list():
	orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
	data_type = ["train", "valid", "test"]

	dialog_l = []
	response_l = []
	tot_dialog_data_l = []
	for t in data_type:
		dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t)
		for idx, dialog_data in enumerate(dialog_data_l):
			if dialog_data[2] != 1: continue
			tot_dialog_data_l.append(dialog_data)
			dialog_l.append(dialog_data[0])
			response_l.extend(dialog_data[1])

		print("%s data load finished! " % t, len(tot_dialog_data_l))
	# print("total data length", len(dialog_l))

	return tot_dialog_data_l, dialog_l, response_l

def get_similar_dialog():
	# tf_idf = TFIDF()
	file_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data"
	tf_idf_path = os.path.join(file_dir, "ubuntu_tf_idf.pickle")
	print("Loading tf_idf score, total dialog sentence list...")

	if os.path.exists(tf_idf_path):
		print(tf_idf_path, "exists!")

	with open(tf_idf_path, "rb") as tf_idf_fr_handle:
		docs_tf_idf = pickle.load(tf_idf_fr_handle)
		print("tf_idf pickle load complete!")

	similar_dialog_sort_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_sort_1000.pickle"
	with open(similar_dialog_sort_path, "rb") as fr_sds_handle:
		# tot_score_val : (600000,100) tot_args_val : (600000,100)
		tot_score_val, tot_args_val = pickle.load(fr_sds_handle)
		print("similar_dialog_sort file load completes")

	similar_dialog_info_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_info_top"
	# when the answer is true
	fw_sim_dialog_info_handle = None
	if not os.path.exists(similar_dialog_info_path):
		fw_sim_dialog_info_handle = open(similar_dialog_info_path, "wb")

	tot_dialog_data_l, dialog_l, response_l = get_dialog_response_list()
	dialog_dict = dict()
	curr_pos = 0
	print(len(tot_dialog_data_l))

	for idx, dialog_data in enumerate(tot_dialog_data_l):
		if dialog_data[2] != 1: continue
		if idx < 500000: t = "train"
		elif idx >= 500000 and idx < 550000: t = "valid"
		else: t = "test"

		dialog_dict[curr_pos] = dict()
		fixed_tf_idf_dict = docs_tf_idf["%s-%d" % (t, curr_pos)]
		fixed_set = set(fixed_tf_idf_dict.keys())
		fixed_utt_set = set(dialog_data[0])
		# print(fixed_utt_set)
		curr_sub_pos = 0
		tf_idf_score_dict = dict()
		overlap_set = fixed_utt_set
		for sub_idx, (sub_score, sub_arg) in enumerate(zip(tot_score_val[curr_pos], tot_args_val[curr_pos])):
			target_utterance_set = set(dialog_l[sub_arg])

			if len(fixed_utt_set.intersection(target_utterance_set)) >= 2:
				# print(sub_idx, ": intersect - %d[th] dialog!" % sub_arg)
				continue
			if len(overlap_set.intersection(target_utterance_set)) >= 2:
				continue
			curr_sub_pos += 1
			target_tf_idf_dict = docs_tf_idf["%s-%d" % (t, sub_arg)]
			target_set = set(target_tf_idf_dict.keys())
			mixed_set = target_set.union(fixed_set)
			overlap_set = overlap_set.union(target_utterance_set)

			fixed_arr = []
			target_arr = []
			for word_token in list(mixed_set):
				if word_token in fixed_set:
					fixed_arr.append(fixed_tf_idf_dict[word_token])
				else:
					fixed_arr.append(0)
				if word_token in target_set:
					target_arr.append(target_tf_idf_dict[word_token])
				else:
					target_arr.append(0)

			fixed_arr = np.array(fixed_arr)
			# print(target_set)
			target_arr = np.array(target_arr)
			cosine_sim = np.divide(np.sum(np.multiply(fixed_arr, target_arr)),
														 np.multiply(np.linalg.norm(fixed_arr, ord=2), np.linalg.norm(target_arr, ord=2)))
			dialog_dict[curr_pos][curr_sub_pos] = {"dialog_str": dialog_l[sub_arg], "dialog_score": sub_score,
																						 "dialog_arg": sub_arg, "response_str": response_l[sub_arg],
																						 "tf_idf_score": cosine_sim}
			tf_idf_score_dict[curr_sub_pos] = cosine_sim
		pickle.dump(dialog_dict[curr_pos], fw_sim_dialog_info_handle)
		sorted_tf_idf_score = sorted(tf_idf_score_dict.items(), key=lambda x:x[1], reverse=True)

		print(curr_pos, "[th] :", sorted_tf_idf_score[0:3])

		print('-'*200)
		curr_pos += 1
		fw_sim_dialog_info_handle.close()

def get_similar_dialog_info_multi_process(tot_dialog_data_l, docs_tf_idf, tot_score_val, tot_args_val,
                                          original_curr_pos:Value, lock:Lock()):

	print(os.getpid(), ": process has been made!")
	similar_dialog_info_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_info_top/%s" % os.getpid()
	fw_sim_dialog_info_handle = open(similar_dialog_info_path, "wb")

	while True:
		with lock:
			if len(tot_dialog_data_l) <= original_curr_pos.value:
				print(os.getpid(), " : process_finished")
				break
			curr_pos = original_curr_pos.value
			if curr_pos == 0: start_time = time.time()
			original_curr_pos.value += 1

		# shared resource : curr_pos
		if tot_dialog_data_l[curr_pos][2] == 1:

			if curr_pos < 500000: t = "train"
			elif curr_pos >= 500000 and curr_pos < 550000: t = "valid"
			else: t = "test"

			dialog_dict = dict()
			dialog_dict[curr_pos] = dict()
			tf_idf_score_dict = dict()

			fixed_tf_idf_dict = docs_tf_idf["%s-%d" % (t, curr_pos)]
			fixed_set = set(fixed_tf_idf_dict.keys())
			fixed_utt_set = set(tot_dialog_data_l[curr_pos][0])
			curr_sub_pos = 0
			overlap_set = fixed_utt_set
			for sub_idx, (sub_score, sub_arg) in enumerate(zip(tot_score_val[curr_pos], tot_args_val[curr_pos])):
				target_utterance_set = set(tot_dialog_data_l[sub_arg][0])

				if len(fixed_utt_set.intersection(target_utterance_set)) >= 2:
					# print(sub_idx, ": intersect - %d[th] dialog!" % sub_arg)
					continue
				if len(overlap_set.intersection(target_utterance_set)) >= 2:
					continue
				curr_sub_pos += 1
				target_tf_idf_dict = docs_tf_idf["train-%d" % sub_arg]
				target_set = set(target_tf_idf_dict.keys())
				mixed_set = target_set.union(fixed_set)
				overlap_set = overlap_set.union(target_utterance_set)

				fixed_arr = []
				target_arr = []
				for word_token in list(mixed_set):
					if word_token in fixed_set:
						fixed_arr.append(fixed_tf_idf_dict[word_token])
					else:
						fixed_arr.append(0)
					if word_token in target_set:
						target_arr.append(target_tf_idf_dict[word_token])
					else:
						target_arr.append(0)

				fixed_arr = np.array(fixed_arr)
				# print(target_set)
				target_arr = np.array(target_arr)
				cosine_sim = np.divide(np.sum(np.multiply(fixed_arr, target_arr)),
															 np.multiply(np.linalg.norm(fixed_arr, ord=2), np.linalg.norm(target_arr, ord=2)))

				dialog_dict[curr_pos][curr_sub_pos] = {"dialog_arg" : sub_arg, "dialog_score": sub_score, "tf_idf_score": cosine_sim}
				tf_idf_score_dict[curr_sub_pos] = cosine_sim

				#TODO:Need to fix... saving dialog_dict, tf_idf_dialog_sort_info_dict
			sorted_tf_idf_score = sorted(tf_idf_score_dict.items(), key=lambda x: x[1], reverse=True)
			sorted_dialog_dict = dict()
			sorted_dialog_dict[curr_pos] = dict()
			for sort_idx, (key, tf_idf_score) in enumerate(sorted_tf_idf_score[0:10]):
				sorted_dialog_dict[curr_pos][sort_idx] = dialog_dict[curr_pos][key]

			pickle.dump(sorted_dialog_dict, fw_sim_dialog_info_handle)
			if (curr_pos + 1) % 1000 == 0:
				print(os.getpid(), ":", curr_pos + 1, "[th] :", sorted_tf_idf_score[0:10])
				print('-' * 200)

	fw_sim_dialog_info_handle.close()
	"""
	similar_dialog_info_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_info_top/%s" % os.getpid()
	with open(similar_dialog_info_path, "wb") as fw_sim_dialog_info_handle:
		pickle.dump([dialog_dict, tf_idf_dialog_sort_info_dict], fw_sim_dialog_info_handle)
		print(similar_dialog_info_path, "save complete!")
	"""

def make_tf_idf_sort_dict_multi_process():
	file_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data"
	tf_idf_path = os.path.join(file_dir, "ubuntu_tf_idf.pickle")
	print("Loading tf_idf score, total dialog sentence list...")

	if os.path.exists(tf_idf_path):
		print(tf_idf_path, "exists!")

	with open(tf_idf_path, "rb") as tf_idf_fr_handle:
		docs_tf_idf = pickle.load(tf_idf_fr_handle)
		print("tf_idf pickle load complete!")

	similar_dialog_sort_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_sort_1000.pickle"
	with open(similar_dialog_sort_path, "rb") as fr_sds_handle:
		# tot_score_val : (600000,100) tot_args_val : (600000,100)
		tot_score_val, tot_args_val = pickle.load(fr_sds_handle)
		print("similar_dialog_sort file load completes")

	tot_dialog_data_l, dialog_l, response_l = get_dialog_response_list()

	dialog_dict = dict()
	tf_idf_dialog_sort_info_dict = dict()
	lock = Lock()

	curr_pos = Value('i', 0)
	procs = [Process(target=get_similar_dialog_info_multi_process,
	                 args=(tot_dialog_data_l, docs_tf_idf, tot_score_val, tot_args_val, curr_pos, lock))
	         for i in range(10)]

	for proc in procs: proc.start()
	for proc in procs: proc.join()

if __name__ == '__main__':
	# get_similar_dialog()
	path_l = os.listdir("/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_info_top/")

	print(path_l)
	tot_sorted_dialog_dict = dict()
	for path in path_l:
		each_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_info_top/%s" % path
		with open(each_path, "rb") as frb_handle:
			while True:
				try:
					each_sorted_dialog_dict = pickle.load(frb_handle)
					tot_sorted_dialog_dict.update(each_sorted_dialog_dict)
				except EOFError:
					print(each_path, " - file loading and dictionary update complete!", len(tot_sorted_dialog_dict))
					break
	print(len(tot_sorted_dialog_dict))

	tot_similar_dialog_dict_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_info_top_10.pickle"
	with open(tot_similar_dialog_dict_path, "wb") as fwb_handle:
		pickle.dump(tot_sorted_dialog_dict, fwb_handle)
		print(tot_similar_dialog_dict_path, "total similar dialog dictionary pickle dump completes!")
