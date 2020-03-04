import ijson
import random
import pickle

def create_dialogue_iter(filename, type="train", test_answer_path=None):
	with open(filename, "r", encoding="utf-8") as fr_handle:
		fr_test_answer_handle = None
		if type == "test":
			fr_test_answer_handle = open(test_answer_path, "r", encoding="utf-8")

			print("%s has opend!" % test_answer_path)

		index = 0
		json_data = ijson.items(fr_handle, "item")
		for entry in json_data:
			index += 1
			dataset = process_dialog(entry, type=type, fr_test_answer_handle=fr_test_answer_handle)

			yield dataset

	if type == "test":
		fr_test_answer_handle.close()

def process_dialog(dialog, type="train", fr_test_answer_handle=None, negative_sample_num=1):

	example_id = dialog['example-id']
	utterances = dialog['messages-so-far']

	dialog_context = ""
	for utt in utterances:
		dialog_context += utt['utterance'] + " eot "
	dialog_context = dialog_context.strip()

	# test
	if type != "test":
		correct_answer = dialog['options-for-correct-answers'][0]
		target_id = correct_answer['candidate-id']
		response = correct_answer['utterance']

	else:
		test_answers = fr_test_answer_handle.readline().split("\t")
		assert int(example_id) == int(test_answers[0])

		if len(test_answers[1].split(",")) > 1:
			target_id = int(test_answers[1].split(",")[0])
			print("Duplication ! - There are two answers.. regard first value as an answer", target_id)
		else:
			target_id = int(test_answers[1])


	candidates_l = []
	for i, utterance in enumerate(dialog['options-for-next']):
		# if type == "test":
		# 	if utterance['candidate-id'] == int(test_answers[1].split(",")[1]): continue
		if utterance['candidate-id'] == target_id:
			if type == "test": response = utterance['utterance']
			continue
		candidates_l.append(utterance) # should be 99 examples

	if type != "test":
		negative_samples = random.sample(candidates_l, negative_sample_num)
		assert len(negative_samples) == negative_sample_num
	else:
		negative_samples = candidates_l
		assert len(negative_samples) == 99

	dataset = [[dialog_context, response, 1]]
	for neg in negative_samples:
		dataset.append([dialog_context, neg["utterance"].strip(), 0])

	return dataset

def do_lower_case(inputs):
	split_inputs = inputs.strip().split(" ")
	for idx, sentence in enumerate(split_inputs):
		split_inputs[idx] = sentence.lower()
	lower_outputs = " ".join(split_inputs)

	return lower_outputs

def read_data_file(path=None):
	with open(path, "rb") as frb_handle:
		train_data = pickle.load(frb_handle)
		print(train_data[0:1])

def make_bert_train_pickle():
	input_file_path = "advising.scenario-1.train.json"
	dialog_iter = create_dialogue_iter(input_file_path, type="train")
	total_data = []
	idx = 0

	bert_file_path = "/mnt/raid5/taesun/data/Advising/bert_train_sub_eot.pickle"
	with open(bert_file_path, "wb") as fwb_handle:
		while True:
			data = next(dialog_iter, None)
			if data is None:
				print("data loading successfully completes!, %d" % len(total_data))
				break
			for dialog_triplet in data:
				pickle.dump(dialog_triplet, fwb_handle)
				idx += 1
			if idx % 1000 == 0:
				print("train data", idx, ": data has been loaded")
		print("train data total %d" % idx)

def make_bert_test_pickle():
	input_file_path = "test.blind.scenario0-1_case2.json"
	test_ground_truth_path ="advising.scenario-1.test.json.answers-case2.tsv"

	dialog_iter = create_dialogue_iter(input_file_path, type="test", test_answer_path=test_ground_truth_path)
	total_data = []
	idx = 0
	bert_file_path = "/mnt/raid5/taesun/data/Advising/bert_test_sub_eot.pickle"
	with open(bert_file_path, "wb") as fwb_handle:
		while True:
			data = next(dialog_iter, None)
			if data is None:
				print("data loading successfully completes!, %d" % idx)
				break
			for dialog_triplet in data:
				pickle.dump(dialog_triplet, fwb_handle)
				idx += 1
			if idx % 1000 == 0:
				print("test data", idx, ": data has been loaded")
		print("test data total %d" % idx)

def make_pre_training_data():
	input_file_path = "advising.scenario-1.train.json"
	pretrain_txt_path = "/mnt/raid5/taesun/data/Advising/bert_pretrain/bert_advising_pretrain.txt"
	dialog_iter = create_dialogue_iter(input_file_path, type="train")
	idx = 0
	with open(pretrain_txt_path, "w", encoding="utf-8") as fw_handle:
		while True:
			data = next(dialog_iter, None)
			if data is None:
				print("data loading successfully completes!, %d" % idx)
				break

			for dialog_triplet in data:
				if dialog_triplet[2] == 0: continue
				dialog_context = dialog_triplet[0]
				response = dialog_triplet[1]
				# print(dialog_triplet)
				for utt in dialog_context.split("eot"):
					if len(utt.strip()) == 0: continue
					fw_handle.write(utt.strip() + " eot\n")
				fw_handle.write(response.strip() + "\n")
				fw_handle.write("\n")
				idx += 1
			if idx % 1000 == 0:
				print("total ground_truth train data", idx, ": data has been loaded")
		print("train data total %d" % idx)

if __name__ == '__main__':
	make_bert_train_pickle()
	make_bert_test_pickle()
	# make_pre_training_data()