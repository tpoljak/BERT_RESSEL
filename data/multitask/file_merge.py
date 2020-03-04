import os


def pretrain_corpus_merge():
	advising_path = "/mnt/raid5/taesun/data/Advising/bert_pretrain/bert_advising_pretrain.txt"
	ubuntu_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/bert_ubuntu_pretrain.txt"
	paths = [advising_path, ubuntu_path]
	for t in paths:
		with open(t, "r", encoding='utf-8') as f_handle:
			count = 0
			dialog = ""
			dialog_l = []
			for line in f_handle:
				dialog += line
				if line == "\n":
					count += 1
					dialog_l.append(dialog)
					dialog = ""
			print(count)

if __name__ == '__main__':
  pretrain_corpus_merge()