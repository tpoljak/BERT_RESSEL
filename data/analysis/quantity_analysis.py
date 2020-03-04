import os
import pickle
import time

turn_num = 4
#TODO: read pickle (find examples)
data_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_dialog_turn_num/bert_test_eot_%s.pickle" % turn_num

def read_pickle(data_dir):
  ubuntu_data = []
  with open(data_dir, "rb") as frb_handle:
    index = 0
    while True:
      if index != 0 and index % 100000 == 0:
        print("%d data has been loaded now" % index)
      try:
        index += 1
        ubuntu_data.append(pickle.load(frb_handle))

      except EOFError:
        print("%s data loading is finished!" % data_dir)
        break

  return ubuntu_data

def create_examples(inputs):
  """Creates examples for the training and dev sets."""
  examples = []
  with open("./bert_test_eot_%s.txt" % turn_num, "w") as fw_handle:

    for (i, dialog_data) in enumerate(inputs):
      if dialog_data[2] == "0":
        fw_handle.write("incorrect : " + dialog_data[1] + '\n')
        continue
      fw_handle.write("=" * 200 + '\n')
      fw_handle.write(str(i) + "-> dialog : " + dialog_data[0] + '\n')
      fw_handle.write("correct : " + dialog_data[1] + '\n')

  print("complete!")

if __name__ == '__main__':
  ubuntu_data = read_pickle(data_dir)
  create_examples(ubuntu_data)