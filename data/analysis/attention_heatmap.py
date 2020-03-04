import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from bert_model import tokenization
from analysis.merge_subtokens import merge_subtokens

# idx = 1420
# idx = 16460
# idx = 17310
# idx = 18200
# idx = 26400
# idx = 27030
# idx = 29540

idx = 11660

def heatmap(score_mat, response, dialog):
  score_mat = pd.DataFrame(score_mat)
  indices = [i for i, x in enumerate(dialog) if x == "[EOT]"]
  # 0:40
  # 40:45
  # 45:69
  # print(" ".join(dialog))
  score_mat.columns = dialog
  score_mat.index = response
  print(score_mat)
  first_score = score_mat.iloc[:, 40:45]
  print(first_score)
  print("-"*200)
  plt.figure(figsize=(2.4, 3.6))
  sns.set(font_scale=0.8)
  sns.heatmap(first_score, xticklabels=True, yticklabels=True, cmap=sns.cm.rocket_r, cbar=False)
  plt.xticks(rotation=90)
  plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
  plt.savefig('att_heatmap_%d_2.pdf' % idx, dpi=1000,)
  # second_score = score_mat[:][40:45]
  # print(second_score)
  # plt.figure(figsize=(3, 3))
  # sns.heatmap(second_score, xticklabels=True, yticklabels=True, cmap=sns.cm.rocket_r, cbar=False)
  # plt.savefig('att_heatmap_%d_2.pdf' % idx, dpi=1000,)
  # third_score = score_mat[:][45:69]
  # print(third_score)
  # plt.figure(figsize=(7, 3))
  # sns.heatmap(third_score, xticklabels=True, yticklabels=True, cmap=sns.cm.rocket_r, cbar=False)
  # plt.savefig('att_heatmap_%d_3.pdf' % idx, dpi=1000,)
  plt.show()

#-----------------------------------------------------------------------------------------------------

def sentence_heatmap(score_sent_mat, dialog, response):
  hm_sent_mat = softmax((np.max(score_sent_mat, axis=0)*25), dim=-1)
  print(response)
  print(list(hm_sent_mat))

def softmax(x, dim=-1):
  """Compute softmax values for each sets of scores in x."""
  exp_x = np.exp(x)
  sum_exp_x = np.sum(exp_x, axis=dim)
  sf = exp_x / np.expand_dims(sum_exp_x, axis=dim)

  return sf
if __name__ == '__main__':
  tokenizer = tokenization.FullTokenizer("/mnt/raid5/shared/bert/tensorflow/uncased_L-12_H-768_A-12/vocab.txt", True)

  with open("./attention_score_%s.pickle" % idx, "rb") as frb_handle:
    dialog, response, raw_dialog, raw_response, sequence_rep = pickle.load(frb_handle)  # dialog_len, response_len

  dialog_len = len(dialog)
  response_len = len(response)
  dialog_rep = np.array(sequence_rep[0:dialog_len])  # 24, 768
  response_rep = np.array(sequence_rep[280:280 + response_len])  # 40, 768

  dialog_merged_embeddings = merge_subtokens([" ".join(raw_dialog)], tokenizer, np.expand_dims(dialog_rep,0), is_cls=True)
  response_merged_embeddings = merge_subtokens([" ".join(raw_response)], tokenizer, np.expand_dims(response_rep,0))

  # 24, 40
  dialog_sentence, response_sentence = [], []
  dialog_sent_vec, response_sent_vec = [], []
  for dialog_original, dialog_vec in dialog_merged_embeddings:
    dialog_sentence.append(dialog_original)
    dialog_sent_vec.append(dialog_vec)

  for response_original, resposne_vec in response_merged_embeddings:
    response_sentence.append(response_original)
    response_sent_vec.append(resposne_vec)
  print(dialog_sentence)
  print(response_sentence)
  print(' '.join(dialog_sentence))
  print(' '.join(response_sentence))
  # print(dialog_sent)

  # response_sent_vec = np.transpose(response_sent_vec, (1, 0))
  #
  # score_mat = np.matmul(dialog_sent_vec, response_sent_vec)
  # score_mat = score_mat / 768
  # score_mat = np.transpose(score_mat, (1,0))
  # heatmap(score_mat, response_sentence, dialog_sentence)
  # sentence_heatmap(score_mat, dialog_sentence, response_sentence)

  # dialog_rep
  # dialog(tok)
  # raw_dialog
  #
  # print(dialog + response)
  # print(["[CLS]"] + raw_dialog + ["[SEP]"] + raw_response + ["[SEP]"])

# sentence_heatmap(score_mat, dialog, response)