import numpy as np

sents = "modprobe loads a kernel module lsmod lists the loaded modules" # text_a

def merge_subtokens(sents, tokenizer, sents_bert_embeddings,merge_subtokens=True, merge_strategy='first', is_cls=False):
  # sents : encoding
  sents_tokenized = [tokenizer.tokenize(s) for s in sents]
  # print(np.array(sents).shape)
  # print(np.array(sents_bert_embeddings).shape)
  # print(np.array(sents_tokenized).shape)
  # print(sents)
  # print(sents_tokenized)

  sents_encodings = []
  for sent_tokens, sent_vecs in zip(sents_tokenized, sents_bert_embeddings):

    sent_encodings = []
    sent_vecs = sent_vecs[1:-1] if is_cls else sent_vecs[0:-1] # ignoring [CLS] and [SEP]

    for token, vec in zip(sent_tokens, sent_vecs):
      # layers_vecs = np.split(np.array(vec), 4) # due to -pooling_layer -4 -3 -2 -1
      # layers_sum = np.array(layers_vecs, dtype=np.float32).sum(axis=0)
      sent_encodings.append((token, vec))
    sents_encodings.append(sent_encodings)

  redundant_tokens_count = 0
  if merge_subtokens:
    sents_encodings_merged = []
    for sent, sent_encodings in zip(sents, sents_encodings):

      sent_tokens_vecs = []
      for token in sent.split():  # these are preprocessed tokens
        token_vecs = []
        for subtoken in tokenizer.tokenize(token):
          if len(sent_encodings) == 0:  # sent may be longer than max_seq_len
            redundant_tokens_count += 1
            # print(subtoken)

            # print('ERROR: seq too long ?')
            break

          encoded_token, encoded_vec = sent_encodings.pop(0)
          assert subtoken == encoded_token
          token_vecs.append(encoded_vec)

        token_vec = np.zeros(768)
        if len(token_vecs) == 0:
          pass
        elif merge_strategy == 'first':
          token_vec = np.array(token_vecs[0])
        elif merge_strategy == 'sum':
          token_vec = np.array(token_vecs).sum(axis=0)
        elif merge_strategy == 'mean':
          token_vec = np.array(token_vecs).mean(axis=0)

        sent_tokens_vecs.append((token, token_vec))
      sents_encodings_merged.append(sent_tokens_vecs)
    # print(redundant_tokens_count)
    if redundant_tokens_count:
      sent_encodings = sents_encodings_merged[0][0:-redundant_tokens_count]
    else:
      sent_encodings = sents_encodings_merged[0]

  return sent_encodings