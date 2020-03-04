from bert_model import tokenization

import os
import re
import math
import pickle
import numpy as np
import random
import nltk

# from bert_data_process import convert_single_example, convert_separate_example

def load_glove_vocab():
    """
    Args:
        filename: path to the glove vectors hparams.glove_path
    """
    glove_path = "/mnt/raid5/shared/word_embeddings/glove.42B.300d.txt"
    print("Loading glove vocab...")
    vocab = set()
    with open(glove_path, "r", encoding='utf-8') as f_handle:
        for line in f_handle:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("Getting Glove Vocabulary is done. %d tokens" % len(vocab))

    return vocab

def export_trimmed_glove_vectors():
    """
     Saves glove vectors in numpy array
     Args:
         vocab: dictionary vocab[word] = index
         glove_filename: a path to a glove file
         trimmed_filename: a path where to store a matrix in npy
         dim: (int) dimension of embeddings
     """
    data_vocab, word2id = read_vocab_file()
    glove_vocab = load_glove_vocab()

    common_vocab = set(data_vocab) & glove_vocab
    print("%d vocabulary is in GLoVe" % len(common_vocab))

    embeddings = np.random.uniform(low=-1.0, high=1.0, size=(len(word2id), 300))
    print(embeddings.shape)

    glove_path = "/mnt/raid5/shared/word_embeddings/glove.42B.300d.txt"
    trimmed_glove_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/glove_ubuntu.42B.300d.trimmed.npz"


    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]

            if len(embedding) < 2:
                continue

            if word in word2id:
                # print(word)
                word_idx = word2id[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_glove_path, embeddings=embeddings)

def make_vocab_file():
    path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"

    data_type = ["train", "valid", "test"]
    total_vocab = set()
    print(data_type)

    # train, valid, test
    for t in data_type:
        vocab = set()
        dialog_data_l = get_dialog_dataset(path % t)
        print("getting vocabulary from %s dataset" % t)
        print("data_path : ", (path % t))
        index = 0
        for dialog_data in dialog_data_l:
            index += 1
            if index % 10000 == 0:
                print(index, " : ", len(vocab))
            #utterances
            for utterance in dialog_data[0]:
                tokenized_utterance = nltk.word_tokenize(utterance)
                for i, word in enumerate(tokenized_utterance):
                    tokenized_utterance[i] = word.lower()
                vocab.update(tokenized_utterance)

            #response
            tokenized_response = nltk.word_tokenize(dialog_data[1][0])
            for i, word in enumerate(tokenized_response):
                tokenized_response[i] = word.lower()
            vocab.update(tokenized_response)

        print("# %s vocab : " % t, len(vocab))

        total_vocab = total_vocab | vocab

    print("# train, valid, test vocab : ", len(total_vocab))

    with open(path % "ubuntu_vocab", "w", encoding="utf-8") as fw_handle:
        total_vocab = list(total_vocab)
        for word in total_vocab:
            fw_handle.write(word)

def get_dialog_dataset(data_path, is_eot = None):

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

            processed_data = process_dialog(dialog_context, response_candidate, ground_truth_flag, is_eot=is_eot)
            dialog_data_l.append(processed_data)

        return dialog_data_l, candidates_pool

def process_dialog(dialog_context, response_candidate, ground_truth_flag, is_eot = None):
    #dialog -> utterance + " \t "
    for i, utterance in enumerate(dialog_context):

        if is_eot is not None:
          dialog_context[i] = utterance.strip() + " [EOT] "
        else:
          dialog_context[i] = utterance.strip()

    processed_data = [dialog_context, [response_candidate], ground_truth_flag]

    return processed_data

def read_vocab_file():
    path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/ubuntu_vocab.txt"

    word2id = dict()
    data_vocab = []
    with open(path, "r", encoding='utf-8') as fr_handle:
        for i, word in enumerate(fr_handle):

            word2id[word[:-1]] = i + 1
            data_vocab.append(word[:-1])

        word2id["__pad__"] = 0
        word2id["__unk__"] = len(word2id)

        data_vocab.insert(0, "__pad__")
        data_vocab.append("__unk__")

        # print("vocab.txt has %d vocabulary" % len(word2id))
        # print("__pad__ and __unk__ is appended")

    return data_vocab, word2id

def make_data_pickle():
    path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
    pickle_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.pickle"
    data_type = ["train","valid","test"]
    print(data_type)

    # train, valid, test
    for t in data_type:
        with open(pickle_path % t, "wb") as fw_handle:
            dialog_data_l = get_dialog_dataset(path % t)
            print("make pickle dataset of %s" % t)
            print("data_path : ", (path % t))
            index = 0
            for dialog_data in dialog_data_l:

                if index % 10 == 0:
                    print(index + 1, "[th] : ", dialog_data[2])

                # utterances
                tokenized_utterances_l = []
                for utterance in dialog_data[0]:
                    tokenized_utterance = nltk.word_tokenize(utterance)
                    for i, word in enumerate(tokenized_utterance):
                        tokenized_utterance[i] = word.lower()

                    tokenized_utterances_l.append(tokenized_utterance)
                # response
                tokenized_response = nltk.word_tokenize(dialog_data[1][0])
                # print([tokenized_utterances_l, tokenized_response, dialog_data[2]])

                pickle.dump([tokenized_utterances_l, tokenized_response, dialog_data[2]], fw_handle)
                index += 1

def make_bert_multi_turn_data_pickle(num_negative_samples=5):
  orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
  data_path = "bert_%s.pickle"
  file_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_multi_turn_negative_%s" % num_negative_samples
  data_type = ["train", "valid", "test"]

  if not os.path.exists(file_dir):
    os.makedirs(file_dir)

  for t in data_type:
    print(t + " data is loading now...")
    curr_idx = 0
    with open(os.path.join(file_dir, data_path % t), "wb") as fw_handle:
      dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t)
      print(len(dialog_data_l))
      print("candidates_pool", len(candidates_pool))
      current_ground_truth = ""
      print(dialog_data_l[0])
      for idx, dialog_data in enumerate(dialog_data_l):
        utterances = dialog_data[0]
        response = dialog_data[1][0]
        label = str(dialog_data[2])

        dialog_context = ""
        for utt in utterances:
          dialog_context += utt
        dialog_context = dialog_context.strip()

        if t in ["test", "valid"]:
          pickle.dump([dialog_context, response, label], fw_handle)
          curr_idx += 1
          continue

        # pos : neg ==> 1 : 1
        if num_negative_samples == 1:
          pickle.dump([dialog_context, response, label], fw_handle)
          curr_idx += 1

        else:
          if label == "1":
            current_ground_truth = response
            pickle.dump([dialog_context, response, label], fw_handle)
            curr_idx += 1

          # negative sample
          if label == "0":
            for post_idx in range(1, num_negative_samples + 1):
              try:
                neg_sample = dialog_data_l[idx + 2 * post_idx][1][0]
              except IndexError:
                print(idx, ":", idx + 2 * post_idx, "index Error")
                neg_sample = random.sample(candidates_pool.difference(response, current_ground_truth), 1)[0]
              finally:
                pickle.dump([dialog_context, neg_sample, label], fw_handle)
                curr_idx += 1

        if curr_idx % 10000 == 0:
          print(str(curr_idx) + " data has been saved now...")
          print(dialog_context)

      print(t + " data pickle save complete")

def get_tf_idf_ubuntu_dialog_knowledge_token():
  def make_ubuntu_man_dict():
    knowledge_path = "./knowledge/ubuntu_manual_knowledge.txt"
    ubuntu_knowledge_dict = dict()

    with open(knowledge_path, "r", encoding="utf-8") as f_handle:
      for line in f_handle:
        ubuntu_man = line.strip().split("\t")
        if len(ubuntu_man) == 2:
          ubuntu_knowledge_dict[ubuntu_man[0]] = ubuntu_man[1]

    return ubuntu_knowledge_dict

  ubuntu_knowledge_dict = make_ubuntu_man_dict()
  stopwords = nltk.corpus.stopwords.words('english')
  for common_word in stopwords:
    if common_word in ubuntu_knowledge_dict.keys():
      ubuntu_knowledge_dict.pop(common_word)

  orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
  # 1000000 500000 500000
  data_type = ["train", "valid", "test"]
  total_dialog_sentence_list = []
  for t in data_type:
    dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t, is_eot=True)
    total_dialog_sentence_list.extend(dialog_data_l)

  # add dialog and calculate term frequency (train, valid, test) # 2000000
  print("total dialog length : ", len(total_dialog_sentence_list))
  tf_idf = TFIDF()

  total_dilaog_len = 0
  for idx, dialog_data in enumerate(total_dialog_sentence_list):
    if idx < 1000000:
      num = 2
      t = "train"
    else:
      num = 10
      if idx < 1500000: t = "valid"
      else: t = "test"

    if idx % num != 0: continue
    else:
      utterances = dialog_data[0]
      dialog_context = ""
      for utt in utterances:
        dialog_context += utt
      dialog_context = dialog_context.strip()

      tf_idf.add_document("%s-%d" % (t, total_dilaog_len), dialog_context)
      total_dilaog_len += 1

      if total_dilaog_len % 10000 == 0:
        print("tf_idf calculation : ", total_dilaog_len)

  for i in range(total_dilaog_len):
    if i < 500000: t = "train"
    elif i >= 500000 and i < 550000: t = "valid"
    else: t = "test"

    tf_idf.get_tf_idf("%s-%d" % (t, i), total_dilaog_len)

  return tf_idf.docs_tf_idf

def make_bert_multi_turn_knowledge_pickle():

  none_count = 0
  orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
  data_path = "bert_%s.pickle"
  file_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_knowledge_multi_turn_test"
  tf_idf_path = os.path.join(file_dir,"ubuntu_tf_idf.pickle")
  data_type = ["train", "valid", "test"]

  if not os.path.exists(file_dir):
    os.makedirs(file_dir)

  if os.path.exists(tf_idf_path):
    print("Loading tf_idf score, total dialog sentence list...")
    with open(tf_idf_path,"rb") as tf_idf_fr_handle:
      total_dialog_sentence_list, tf_idf, ubuntu_knowledge_dict = pickle.load(tf_idf_fr_handle)
  else:
    total_dialog_sentence_list, tf_idf, ubuntu_knowledge_dict = get_tf_idf_ubuntu_dialog_knowledge_token()
    with open(tf_idf_path, "wb") as tf_idf_fw_handle:
      pickle.dump([total_dialog_sentence_list, tf_idf, ubuntu_knowledge_dict], tf_idf_fw_handle)

  def filtering_common_words_from_knowledge(knowledge_dict):
    common_words_path = "./knowledge/google-10000-english.txt"
    common_words = set()
    with open(common_words_path, "r", encoding="utf-8") as f_handle:
      for idx, line in enumerate(f_handle):
        if idx == 1000: break
        common_words.add(line.strip())
    count = 0
    original_knowledge_num = len(knowledge_dict)
    deleted_knowledge = []
    for word in common_words:
      if word in knowledge_dict:
        deleted_knowledge.append(word)
        del knowledge_dict[word]
        count += 1
    print("total ubuntu knowledge which is deleted : %d(out of %d)" % (count, original_knowledge_num))
    print(deleted_knowledge)
    return knowledge_dict

  print("original knowledge dictionary", len(ubuntu_knowledge_dict))
  ubuntu_knowledge_dict = filtering_common_words_from_knowledge(ubuntu_knowledge_dict)
  print("filtered knowledge dictionary",len(ubuntu_knowledge_dict))

  knowledge_tf_idf_sorted = dict()
  total_dilaog_len = 0
  curr_idx = 0
  ubuntu_knowledge_used_tokens = set()

  fw_knowledge_pickle = None
  for idx, dialog_data in enumerate(total_dialog_sentence_list):
    if idx < 1000000:
      t = "train"
      if idx == 0:
        fw_knowledge_pickle = open(os.path.join(file_dir, data_path % t), "wb")
      num = 2
    else:
      num = 10
      if idx < 1500000:
        t = "valid"
        if idx == 1000000:
          fw_knowledge_pickle.close()
          fw_knowledge_pickle = open(os.path.join(file_dir, data_path % t), "wb")
      else:
        t = "test"
        if idx == 1500000:
          fw_knowledge_pickle.close()
          fw_knowledge_pickle = open(os.path.join(file_dir, data_path % t), "wb")

    if idx % num == 0:
      total_dilaog_len += 1
      tf_idf.get_tf_idf("%s-%d" % (t, total_dilaog_len), len(total_dialog_sentence_list))
      knowledge_tf_idf_sorted = sorted(tf_idf.docs_tf_idf["%s-%d" % (t, total_dilaog_len)].items(),
                                       key=lambda x: x[1], reverse=True)
      print(knowledge_tf_idf_sorted)
      exit()

    utterances = dialog_data[0]
    response = dialog_data[1][0]
    label = str(dialog_data[2])

    dialog_context = ""
    temp = dict()
    for utt in utterances:
      for token in nltk.word_tokenize(utt.strip()):
        if token in ubuntu_knowledge_dict.keys():
          for tf_token, tf_score in knowledge_tf_idf_sorted:
            if token == tf_token:
              temp[token] = tf_score


      dialog_context += utt
    dialog_context = dialog_context.strip()

    dialog_knowledge = ""
    if len(temp) == 0:
      dialog_knowledge = "none"
      none_count += 1
    else:
      # print(sorted(temp.items(), key=lambda x: x[1], reverse=True))
      # print(sorted(temp.items(), key=lambda x: x[1], reverse=True)[0])
      # print(dialog_context)
      if len(temp) <= 3 : max_man_len = len(temp)
      else: max_man_len = 3
      for token, tf_score in sorted(temp.items(), key=lambda x: x[1], reverse=True)[0:max_man_len]:
        ubuntu_knowledge_used_tokens.add(token)
        dialog_knowledge += token + " : " + ubuntu_knowledge_dict[token] + " eok "

    # print([dialog_context, response, label, dialog_knowledge])
    pickle.dump([dialog_context, response, label, dialog_knowledge], fw_knowledge_pickle)

    curr_idx += 1

    if curr_idx % 10000 == 0:
      print([dialog_context, response, label, dialog_knowledge])
      print("ubuntu_knowledge_exist : ", len(ubuntu_knowledge_used_tokens))
      print("no knowledge : ", none_count)
      print(str(curr_idx) + " data has been saved now...")

  print("total ubuntu knowledge which is utilized : ", len(ubuntu_knowledge_used_tokens))
  print("no knowledge : ", none_count)
  fw_knowledge_pickle.close()

def make_bert_knowledge_top_n(top_n=3):
  """
  :param top_n:  pickle data format = [dialog_data, response, label, [knowledge top_n], knowledge_labels [1,1,0]]
  :return:
  """

  none_count = 0
  data_path = "bert_%s.pickle"
  file_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_knowledge_multi_turn"
  knowledge_top_n_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_knowledge_top_%s" % top_n
  tf_idf_path = os.path.join(file_dir,"ubuntu_tf_idf.pickle")

  if not os.path.exists(knowledge_top_n_dir):
    os.makedirs(knowledge_top_n_dir)

  if os.path.exists(tf_idf_path):
    print("Loading tf_idf score, total dialog sentence list...")
    with open(tf_idf_path,"rb") as tf_idf_fr_handle:
      total_dialog_sentence_list, tf_idf, ubuntu_knowledge_dict = pickle.load(tf_idf_fr_handle)
  else:
    total_dialog_sentence_list, tf_idf, ubuntu_knowledge_dict = get_tf_idf_ubuntu_dialog_knowledge_token()
    with open(tf_idf_path, "wb") as tf_idf_fw_handle:
      pickle.dump([total_dialog_sentence_list, tf_idf, ubuntu_knowledge_dict], tf_idf_fw_handle)

  def filtering_common_words_from_knowledge(knowledge_dict):
    common_words_path = "./knowledge/google-10000-english.txt"
    common_words = set()
    with open(common_words_path, "r", encoding="utf-8") as f_handle:
      for idx, line in enumerate(f_handle):
        if idx == 1000: break
        common_words.add(line.strip())
    count = 0
    original_knowledge_num = len(knowledge_dict)
    deleted_knowledge = []
    for word in common_words:
      if word in knowledge_dict:
        deleted_knowledge.append(word)
        del knowledge_dict[word]
        count += 1
    print("total ubuntu knowledge which is deleted : %d(out of %d)" % (count, original_knowledge_num))
    print(deleted_knowledge)
    return knowledge_dict

  ubuntu_knowledge_dict = filtering_common_words_from_knowledge(ubuntu_knowledge_dict)
  print(len(ubuntu_knowledge_dict))

  knowledge_tf_idf_sorted = dict()
  total_dilaog_len = 0
  curr_idx = 0
  ubuntu_knowledge_used_tokens = set()

  fw_knowledge_pickle = None
  for idx, dialog_data in enumerate(total_dialog_sentence_list):
    if idx < 1000000:
      num = 2
      t = "train"
      if idx == 0:
        fw_knowledge_pickle = open(os.path.join(knowledge_top_n_dir, data_path % t), "wb")

    else:
      num = 10
      if idx < 1500000:
        t = "valid"
        if idx == 1000000:
          fw_knowledge_pickle.close()
          fw_knowledge_pickle = open(os.path.join(knowledge_top_n_dir, data_path % t), "wb")
      else:
        t = "test"
        if idx == 1500000:
          fw_knowledge_pickle.close()
          fw_knowledge_pickle = open(os.path.join(knowledge_top_n_dir, data_path % t), "wb")

    if idx % num == 0:
      total_dilaog_len += 1
      tf_idf.get_tf_idf("%s-%d" % (t, total_dilaog_len), len(total_dialog_sentence_list))
      knowledge_tf_idf_sorted = sorted(tf_idf.docs_tf_idf["%s-%d" % (t, total_dilaog_len)].items(),
                                       key=lambda x: x[1], reverse=True)

    utterances = dialog_data[0]
    response = dialog_data[1][0]
    label = str(dialog_data[2])

    dialog_context = ""
    temp_manual_tokens = dict()
    for utt in utterances:
      for token in nltk.word_tokenize(utt.strip()):
        if token in ubuntu_knowledge_dict.keys():
          for tf_token, tf_score in knowledge_tf_idf_sorted:
            if token == tf_token:
              temp_manual_tokens[token] = tf_score

      dialog_context += utt
    dialog_context = dialog_context.strip()

    dialog_knowledge = ""
    num_manual = len(temp_manual_tokens)
    ranked_knowledge_list = []
    ranked_knowledge_token_list = []
    ranked_knowledge_labels = []

    # print(num_manual)

    if num_manual > 0:
      dialog_ranked_manual = sorted(temp_manual_tokens.items(), key=lambda x: x[1], reverse=True)[0:num_manual]
      for token, tf_score in dialog_ranked_manual:
        # print(token, ":", ubuntu_knowledge_dict[token])
        ranked_knowledge_list.append(token)
        ranked_knowledge_labels.append(1)
        ubuntu_knowledge_used_tokens.add(token)

    if num_manual > top_n:
      ranked_knowledge_list = ranked_knowledge_list[0:top_n]
      ranked_knowledge_labels = ranked_knowledge_labels[0:top_n]
      num_manual = top_n

    for idx in range(top_n - num_manual):
      # when the knowledge doesn't exist... select random knowledge
      token = random.sample(list(ubuntu_knowledge_dict),1)
      token = token[0]
      ranked_knowledge_list.append(token)
      ranked_knowledge_labels.append(0)
    if num_manual == 0:
      none_count += 1

    assert len(ranked_knowledge_list) == top_n
    assert len(ranked_knowledge_labels) == top_n

    # print([dialog_context, response, label, dialog_knowledge])
    pickle.dump([dialog_context, response, label, ranked_knowledge_list, ranked_knowledge_labels], fw_knowledge_pickle)

    curr_idx += 1
    if curr_idx % 10000 == 0:
      print([dialog_context, response, label, ranked_knowledge_list, ranked_knowledge_labels])
      print("ubuntu_knowledge_exist : ", len(ubuntu_knowledge_used_tokens), "No knowledge token in dialog : %d" % none_count)
      print(str(curr_idx) + " data has been saved now...")

  print("total ubuntu knowledge which is utilized : ", len(ubuntu_knowledge_used_tokens))
  fw_knowledge_pickle.close()

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


  data_path = "bert_%s.pickle"
  file_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_knowledge_multi_turn"
  knowledge_top_n_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_knowledge_top_5"
  tf_idf_path = os.path.join(file_dir, "ubuntu_tf_idf.pickle")

  print("Loading tf_idf score, total dialog sentence list...")
  with open(tf_idf_path, "rb") as tf_idf_fr_handle:
    total_dialog_sentence_list, tf_idf, _ = pickle.load(tf_idf_fr_handle)

  ubuntu_knowledge_dict = filtering_common_words_from_knowledge()
  knowledge_tf_idf_sorted = dict()
  total_dilaog_len = 0
  curr_idx = 0
  ubuntu_knowledge_used_tokens = set()
  count = 0
  general_token_count = 0
  knowledge_rates = 0
  no_knowledge_dialog = 0

  fw_knowledge_pickle = None
  for idx, dialog_data in enumerate(total_dialog_sentence_list):
    if idx < 1000000:
      num = 2
      t = "train"
    else:
      if idx < 1500000:
        t = "valid"
      else:
        t = "test"
      num = 10

    if idx % num == 0:
      total_dilaog_len += 1
      tf_idf.get_tf_idf("%s-%d" % (t, total_dilaog_len), len(total_dialog_sentence_list))
      knowledge_tf_idf_sorted = sorted(tf_idf.docs_tf_idf["%s-%d" % (t, total_dilaog_len)].items(),
                                       key=lambda x: x[1], reverse=True)
    utterances = dialog_data[0]
    response = dialog_data[1][0]
    label = str(dialog_data[2])

    dialog_context = ""
    temp = dict()
    for utt in utterances:
      for token in nltk.word_tokenize(utt.strip()):
        # print(token)
        if token in ubuntu_knowledge_dict.keys():
          for tf_token, tf_score in knowledge_tf_idf_sorted:
            if token == tf_token:
              temp[token] = tf_score

      dialog_context += utt
    dialog_context = dialog_context.strip()

    dialog_ranked_manual = sorted(temp.items(), key=lambda x: x[1], reverse=True)[0:5]
    dialog_manual_description = list()
    dialog_manual_token_list = list()
    dialog_manual_list = list()
    for man_key, description in dialog_ranked_manual:
      dialog_manual_description.append(man_key + " : " + ubuntu_knowledge_dict[man_key])
      # dialog_manual_token_list.extend(nltk.word_tokenize(man_key + ":" + ubuntu_knowledge_dict[man_key]))
      dialog_manual_list.append(man_key)
    # print(type(dialog_ranked_manual))
    # print(dialog_ranked_manual)
    # print(dialog_manual_list)
    general_token = set()
    tf_idf_token = set()
    if idx % num == 0:
      # print(idx,  " : ",response, "manual_token_num : ", len(dialog_manual_description))
      print(idx)
      for token in nltk.word_tokenize(response):
        if token in ubuntu_knowledge_dict:
          ubuntu_knowledge_used_tokens.add(token)
          print("token:", token)
          general_token.add(token)
          if token in dialog_manual_list:
            print("tf_idf token", token)
            tf_idf_token.add(token)
      if len(general_token) == 0 : no_knowledge_dialog += 1
      if len(tf_idf_token) > 0:
        count += 1
        knowledge_rates += len(tf_idf_token)/len(general_token)

      if len(general_token) > 0:
        general_token_count += 1

      print(idx, ": count", float(count / ((idx + 2)/2)),
            "general_token_count", float(general_token_count / ((idx + 2)/2)),
            "general/tf_idf", float(knowledge_rates/count),
            "used_tokens", len(ubuntu_knowledge_used_tokens),
            "no_knowledge_dialog", no_knowledge_dialog / ((idx+2)/2))

      print("-"*200)

    if idx == 1000000 - 1:
      print("count", float(count/((idx + 2)/2)))
      break

def get_tf_idf_vocab():
  data_path = "bert_%s.pickle"
  file_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_knowledge_multi_turn"
  # knowledge_top_n_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_knowledge_top_%s" % top_n
  tf_idf_path = os.path.join(file_dir, "ubuntu_tf_idf.pickle")

  if not os.path.exists(file_dir):
    os.makedirs(file_dir)

  if os.path.exists(tf_idf_path):
    print("Loading tf_idf score, total dialog sentence list...")
    with open(tf_idf_path, "rb") as tf_idf_fr_handle:
      total_dialog_sentence_list, tf_idf, ubuntu_knowledge_dict = pickle.load(tf_idf_fr_handle)
  else:
    total_dialog_sentence_list, tf_idf, ubuntu_knowledge_dict = get_tf_idf_ubuntu_dialog_knowledge_token()
    with open(tf_idf_path, "wb") as tf_idf_fw_handle:
      pickle.dump([total_dialog_sentence_list, tf_idf, ubuntu_knowledge_dict], tf_idf_fw_handle)

  total_dilaog_len = 0
  ubuntu_word_token_set = set()
  for idx, dialog_data  in enumerate(total_dialog_sentence_list):
    if idx < 1000000:
      num = 2
      t = "train"
    else:
      num = 10
      if idx < 1500000:
        t = "valid"
      else:
        t = "test"

    if idx % num == 0:
      total_dilaog_len += 1
      tf_idf.get_tf_idf("%s-%d" % (t, total_dilaog_len), len(total_dialog_sentence_list))
      for word_token, score in tf_idf.docs_tf_idf["%s-%d" % (t, total_dilaog_len)].items():
        ubuntu_word_token_set.add(word_token)
        # knowledge_tf_idf_sorted = sorted(tf_idf.docs_tf_idf["%s-%d" % ("train", idx+1)].items(),
        #                                  key=lambda x: x[1], reverse=True)

    if (idx + 1) % 10000 == 0:
      print("%d[th] :" % (idx + 1), len(ubuntu_word_token_set))


def make_bert_single_turn_data_pickle():
    # dialog last utterance max_length : 50, response max_length : 50 (?) total : 100 // performance comparison
    # single_turn : no effective
    orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
    data_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_single_turn_%s.pickle"
    data_type = ["train", "valid", "test"]

    # preprocessing max_seq_len
    # train, valid, test
    for t in data_type:
      print(t)
      with open(data_path % t, "wb") as fw_handle:
        dialog_data_l = get_dialog_dataset(orig_path % t)
        print(len(dialog_data_l))

        for dialog_data in dialog_data_l:
          last_utterance = dialog_data[0][-1]
          response = dialog_data[1][0]
          label = str(dialog_data[2])

          pickle.dump([last_utterance, response, label], fw_handle)

def make_bert_utterance_data_file(dialog_len=10):

    orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
    data_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_ubuntu_data_%s"
    data_path = "bert_%s.pickle"
    data_length_path = "bert_%s_length.txt"
    data_type = ["train", "valid", "test"]

    if not os.path.isdir(data_dir % dialog_len):
        os.makedirs(data_dir % dialog_len)

        for t in data_type:
            print(data_type)
            with open(os.path.join(data_dir % dialog_len, data_path % t), "wb") as fw_handle:
                with open(os.path.join(data_dir % dialog_len, data_length_path % t), "w", encoding='utf-8') as fw_len_handle:
                    dialog_data_l = get_dialog_dataset(orig_path % t)
                    print(len(dialog_data_l))

                    if t == "train":
                        for i in range(30):
                            print(i + 1, "th shuffling has finished!")
                            random.shuffle(dialog_data_l)
                        print("Shuffling Process is done! Total dialog context : %d" % len(dialog_data_l))

                    for index, dialog_data in enumerate(dialog_data_l):
                        utterances = dialog_data[0]
                        response = dialog_data[1][0]
                        label = str(dialog_data[2])

                        if len(utterances) > dialog_len:
                            utterances = utterances[-dialog_len:]
                            fw_len_handle.write(str(dialog_len) + "\n")

                        elif len(utterances) == dialog_len:
                            fw_len_handle.write(str(dialog_len) + "\n")

                        for utt in utterances:
                            pickle.dump([utt, response, label], fw_handle)

                        if len(utterances) < dialog_len:
                            fw_len_handle.write(str(len(utterances)) + "\n")
                            for i in range(dialog_len-len(utterances)):
                                pickle.dump(["[PAD]", "[PAD]", str(0)], fw_handle)

                        if (index + 1) % 10000 == 0:
                            print(index + 1, "th data has been saved")

def make_bert_pretrain_data_file():
    #line by line for training data
    original_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
    write_data_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/original/bert_advising_pretrain_with_eot.txt"

    # only train
    data_type = ["train"]

    # preprocessing max_seq_len
    # train, valid, test

    for t in data_type:
        with open(write_data_path, "w", encoding="utf-8") as fw_handle:
            dialog_data_l, candidates_pool = get_dialog_dataset(original_path % t, is_eot=True)
            print(len(dialog_data_l))

            sentence_num = 0
            for i, dialog_data in enumerate(dialog_data_l):
                # negative label
                if dialog_data[2] == 0:
                    continue
                utterances = dialog_data[0]
                response = dialog_data[1][0]

                for utt in utterances:
                    fw_handle.write(utt.strip() + "\n")
                    sentence_num += 1
                fw_handle.write(response.strip() + "\n")
                fw_handle.write("\n")
                sentence_num += 1

                if (i+1) % 10000 == 0:
                    print(i+1,"th data has been saved | # sentences", sentence_num)

def make_similar_dialog_data_pickle(top_n=3):
  orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
  data_path = "bert_%s.pickle"
  file_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_top_%s/" % top_n
  data_type = ["train", "valid", "test"]

  def _get_similar_dialogs():
    """
    path_l = os.listdir("/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_info_top/")
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
    """

    path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_info_top_10.pickle"
    with open(path, "rb") as frb_handle:
      print(path, "data loading... it takes a few seconds!")
      tot_sorted_dialog_dict = pickle.load(frb_handle)
    print("total similar dialog data loading completes! : ", len(tot_sorted_dialog_dict))
    return tot_sorted_dialog_dict

  tot_sorted_dialog_dict = _get_similar_dialogs()
  assert len(tot_sorted_dialog_dict) == 600000

  def _get_dialog_response_list():
    orig_path = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/%s.txt"
    data_type = ["train", "valid", "test"]

    tot_dialog_data_l = []
    for t in data_type:
      dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t)
      for idx, dialog_data in enumerate(dialog_data_l):
        if dialog_data[2] != 1: continue
        tot_dialog_data_l.append(dialog_data)
      print("%s data load finished! " % t, len(tot_dialog_data_l))

    return tot_dialog_data_l

  pure_dialog_data_l = _get_dialog_response_list()

  if not os.path.exists(file_dir):
    os.makedirs(file_dir)

  curr_pos = 0
  for t in data_type:
    print(t + " data is loading now...")
    with open(os.path.join(file_dir, data_path % t), "wb") as fw_handle:
      dialog_data_l, candidates_pool = get_dialog_dataset(orig_path % t)

      curr_similar_dialog = []
      for idx, dialog_data in enumerate(dialog_data_l):
        utterances = dialog_data[0]
        response = dialog_data[1][0]
        label = str(dialog_data[2])
        if label == "1": curr_similar_dialog = []

        dialog_context = ""
        for utt in utterances:
          dialog_context += utt.strip() + " eot "
        dialog_context = dialog_context.strip()

        if label == "1":
          # print('-'*200)
          # print(utterances, response)
          for i in range(top_n):
            # print('*'*200)
            # print(tot_sorted_dialog_dict[curr_pos][i])
            # dialog_args, dialog_score, tf_idf_score -> pure dialog_data_l
            similar_dialog_data = pure_dialog_data_l[tot_sorted_dialog_dict[curr_pos][i]["dialog_arg"]]
            #TODO:Dialog should be changed as a string with " eot "
            similar_dialog_context = ""
            for sim_utt in similar_dialog_data[0]:
              similar_dialog_context += sim_utt.strip() + " eot "
            similar_dialog_context += similar_dialog_data[1][0].strip() # response for similar dialog
            # print(similar_dialog_context.strip())
            curr_similar_dialog.append(similar_dialog_context.strip())

          curr_pos += 1
        # [dialog_context, response, label] -> pickle
        pickle.dump([dialog_context, response, label, curr_similar_dialog], fw_handle)
        assert len(curr_similar_dialog) == top_n
        if (idx + 1) % 10000 == 0:
          print(str(curr_pos) + " data has been saved now...")
      print(t + " data pickle save complete")

class DataStat(object):
  def __init__(self):
    self.data_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_multi_turn_negative_1/bert_%s.pickle" % \
                    "train"
    self.bert_vocab_file = "/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/vocab.txt"
    self.data_l = []

    self.load_data_dir()
    self.tokenizer = tokenization.FullTokenizer(self.bert_vocab_file, True)

    self.max_dialog_context = 0
    self.max_response = 0
    self.max_utterance = 0

    self.avg_dialog_context = 0
    self.avg_response = 0
    self.avg_utterance = 0

    self.min_dialog_context = 10000
    self.min_response = 10000
    self.min_utterance = 10000

    self.get_sentence_statistics()

    print("="*200)
    print("Final Stat Info")
    print("avg_dialog_context", self.avg_dialog_context)
    print("avg_response", self.avg_response)

    print("max_dialog_context", self.max_dialog_context)
    print("max_response", self.max_response)

    print("min_dialog_context", self.min_dialog_context)
    print("min_response", self.min_response)

  def load_data_dir(self):

    with open(self.data_dir, "rb") as frb_handle:
      index = 0
      while True:
        if index != 0 and index % 100000 == 0:
          print("%d data has been loaded now" % index)
        try:
          index += 1
          self.data_l.append(pickle.load(frb_handle))

        except EOFError:
          print("%s data loading is finished!" % self.data_dir)
          break

  def get_sentence_statistics(self):
    """Creates examples for the training and dev sets."""
    dialog_examples = []
    response_examples = []

    for (i, dialog_data) in enumerate(self.data_l):
      text_dilaog_context = tokenization.convert_to_unicode(dialog_data[0])
      text_response = tokenization.convert_to_unicode(dialog_data[1])
      label = tokenization.convert_to_unicode(dialog_data[2])

      tokens_dialog_context = self.tokenizer.tokenize(text_dilaog_context)
      tokens_response = self.tokenizer.tokenize(text_response)

      tok_dialog_len = len(tokens_dialog_context)
      tok_response_len = len(tokens_response)
      # print(i, ":", tok_dialog_len)

      self.avg_dialog_context += tok_dialog_len
      self.avg_response += tok_response_len

      if self.max_dialog_context < tok_dialog_len:
        self.max_dialog_context = tok_dialog_len

      if self.max_response < tok_response_len:
        self.max_response = tok_response_len

      if self.min_dialog_context > tok_dialog_len:
        self.min_dialog_context = tok_dialog_len

      if self.min_response > tok_response_len:
        self.min_response = tok_response_len

      if (i + 1) % 1000 == 0:
        print(i + 1, "th text stat info")
        print("avg_dialog_context", self.avg_dialog_context / (i+1))
        print("avg_response", self.avg_response / (i+1))

        print("max_dialog_context", self.max_dialog_context)
        print("max_response", self.max_response)

        print("min_dialog_context", self.min_dialog_context)
        print("min_response", self.min_response)
        print('-'*200)
    self.avg_dialog_context /= len(self.data_l)
    self.avg_response /= len(self.data_l)

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

if __name__ == '__main__':
  # get_stat_knowledge_relevance()
  # make_bert_multi_turn_knowledge_pickle()
  # make_bert_knowledge_top_n(5)

  # make_bert_multi_turn_data_pickle(1)
  # data_stat = DataStat()
  # get_tf_idf_vocab()
  # bert_append_sep()

  """
  file_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/"
  tf_idf_path = os.path.join(file_dir, "ubuntu_tf_idf.pickle")
  docs_tf_idf = get_tf_idf_ubuntu_dialog_knowledge_token()
  with open(tf_idf_path, "wb") as tf_idf_fw_handle:
    pickle.dump(docs_tf_idf, tf_idf_fw_handle) # docs_tf_idf : dict
  """
  # make_similar_dialog_data_pickle(3)
  make_bert_pretrain_data_file()