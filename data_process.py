from bert_model import tokenization

import os
import csv
import random
import pickle
import re
import collections
import numpy as np
import tensorflow as tf

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class KnowledgeExample(object):
  def __init__(self, bert_seq_out, length, label):
    self.bert_seq_out = bert_seq_out
    self.length = length
    self.label = label

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               input_length,
               position_ids=None,
               is_real_example=True,
               raw_dialog=None,
               raw_response=None):

    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.position_ids = position_ids
    self.label_id = label_id
    self.input_length = input_length
    self.is_real_example = is_real_example
    self.raw_dialog = raw_dialog
    self.raw_response=raw_response


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class UbuntuProcessor(DataProcessor):
    # dialog_data_l :[utterances, response, label
    def __init__(self, hparams):
        self.hparams = hparams
        self.pickle_dir = hparams.pickle_dir
        self.do_train_knowledge = hparams.do_train_knowledge
        self.training_shuffle_num = hparams.training_shuffle_num
        self.do_similar_dialog = hparams.do_similar_dialog

    def get_train_examples(self, data_dir, is_separate = False, is_knowledge_separable=False, is_bert_pretrained=True):
        """See base class."""
        print(self.pickle_dir)
        ubuntu_train_data = self._read_pickle(os.path.join(data_dir, self.pickle_dir % "test"), shuffle=True)
        if is_separate:
          self.train_dialog, self.train_response = self._create_seperate_examples(ubuntu_train_data, "test")
          return self.train_dialog, self.train_response

        else:
          self.train_knowledge = None
          self.train_similar_dialog = None
          self.train_example = self._create_examples(ubuntu_train_data, "test")

          if self.do_similar_dialog:
            self.train_similar_dialog = self._create_similar_examples(ubuntu_train_data, "train_similar")
          if self.do_train_knowledge and not is_knowledge_separable:
            self.train_knowledge = self._create_knowledge_examples(ubuntu_train_data, "train_knowledge")
          if is_knowledge_separable:
            self.train_knowledge = self._create_knowledge_separate_examples(ubuntu_train_data, "train_knowledge")
          if self.do_train_knowledge and is_bert_pretrained:
            self.train_knowledge = self._create_knowledge_bert_seq_examples(ubuntu_train_data, "train_knowledge")

          return self.train_example, self.train_knowledge, self.train_similar_dialog

    def get_dev_examples(self, data_dir, shuffle=False, is_separate = False, is_knowledge_separable=False, is_bert_pretrained=True):
        """See base class."""
        self.valid_example, self.valid_knowledge, self.valid_similar_dialog = None, None, None
        if not os.path.exists(os.path.join(data_dir, self.pickle_dir % "valid")):
          return None, None, None
        ubuntu_valid_data = self._read_pickle(os.path.join(data_dir, self.pickle_dir % "valid"))
        if is_separate:
          self.valid_dialog, self.valid_response = self._create_seperate_examples(ubuntu_valid_data, "valid")
          return self.valid_dialog, self.valid_response

        else:
          self.valid_knowledge = None
          self.valid_similar_dialog = None
          self.valid_example = self._create_examples(ubuntu_valid_data, "valid")

          if self.do_similar_dialog:
            self.valid_similar_dialog = self._create_similar_examples(ubuntu_valid_data, "valid_similar")
          if self.do_train_knowledge and not is_knowledge_separable:
            self.valid_knowledge = self._create_knowledge_examples(ubuntu_valid_data, "valid_knowledge")
          if is_knowledge_separable:
            self.valid_knowledge = self._create_knowledge_separate_examples(ubuntu_valid_data, "valid_knowledge")
          if self.do_train_knowledge and is_bert_pretrained:
            self.valid_knowledge = self._create_knowledge_bert_seq_examples(ubuntu_valid_data, "valid_knowledge")

          return self.valid_example, self.valid_knowledge, self.valid_similar_dialog

    def get_test_examples(self, data_dir, shuffle=False, is_separate = False, is_knowledge_separable=False, is_bert_pretrained=True):
        """See base class."""
        ubuntu_test_data = self._read_pickle(os.path.join(data_dir, self.pickle_dir % "test"))
        if is_separate:
          self.test_dialog, self.test_response = self._create_seperate_examples(ubuntu_test_data, "test")
          return self.test_dialog, self.test_response

        else:
          self.test_knowledge = None
          self.test_similar_dialog = None
          self.test_example =  self._create_examples(ubuntu_test_data, "test")

          if self.do_similar_dialog:
            self.test_similar_dialog = self._create_similar_examples(ubuntu_test_data, "test_similar")
          if self.do_train_knowledge and not is_knowledge_separable:
            self.test_knowledge = self._create_knowledge_examples(ubuntu_test_data, "test_knowledge")
          if is_knowledge_separable:
            self.test_knowledge = self._create_knowledge_separate_examples(ubuntu_test_data, "test_knowledge")
          if self.do_train_knowledge and is_bert_pretrained:
            self.test_knowledge = self._create_knowledge_bert_seq_examples(ubuntu_test_data, "test_knowledge")

          return self.test_example, self.test_knowledge, self.test_similar_dialog

    def get_labels(self):
        """See base class."""
        self.label_list = ["0", "1"]
        return self.label_list

    def get_lengths(self, data_len_dir):
        set_type = ["train","valid","test"]
        self.train_lengths, self.valid_lengths, self.test_lengths = [], [], []

        lengths = {
            "train" : self.train_lengths,
            "valid" : self.valid_lengths,
            "test" : self.test_lengths
        }

        for t in set_type:
            t_path = "bert_" + t + "_length.txt"
            with open(os.path.join(data_len_dir, t_path) , "r", encoding='utf-8') as fr_handle:
                for length in fr_handle:
                    lengths[t].append(length.strip())
            print("%s data lengths : [%d] load complete!" % (t, len(lengths[t])))

        return self.train_lengths, self.valid_lengths, self.test_lengths

    def _read_pickle(self, data_dir, shuffle=False):
        print("[Reading %s]" % data_dir)
        with open(data_dir, "rb") as frb_handle:
            ubuntu_data = []
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


            if "train" in data_dir and self.hparams.less_data_rate < 1.0:
              num_ubuntu_data = len(ubuntu_data) * self.hparams.less_data_rate
              print("less data! : %d" % num_ubuntu_data)
              ubuntu_data = ubuntu_data[0:int(num_ubuntu_data)]

            if shuffle and self.training_shuffle_num > 1:
                ubuntu_data = self.data_shuffling(ubuntu_data, self.training_shuffle_num)

            return ubuntu_data

    def data_shuffling(self, inputs, shuffle_num):
        for i in range(shuffle_num):
            # print(i + 1, "th shuffling has finished!")
            random.shuffle(inputs)
        print("Shuffling Process is done! Total dialog context : %d" % len(inputs))

        return inputs

    def data_process_feature(self, hparams, tokenizer):

        self.max_seq_length = hparams.max_seq_length
        self.dialog_max_seq_length = hparams.dialog_max_seq_length
        self.response_max_seq_length = hparams.response_max_seq_length
        self.knowledge_max_seq_length = hparams.knowledge_max_seq_length
        self.tokenizer = tokenizer

        print(self.max_seq_length)
        print(self.dialog_max_seq_length)
        print(self.response_max_seq_length)

    def _create_examples(self, inputs, set_type):
      """Creates examples for the training and dev sets."""
      examples = []

      for (i, dialog_data) in enumerate(inputs):

        guid = "%s-%d" % (set_type, i + 1)
        text_a = tokenization.convert_to_unicode(dialog_data[0])
        text_b = tokenization.convert_to_unicode(dialog_data[1])
        label = tokenization.convert_to_unicode(str(dialog_data[2]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      print("%s data creation is finished! %d" % (set_type, len(examples)))

      return examples

    def _create_similar_examples(self, inputs, set_type):
      examples = []

      for (i, dialog_data) in enumerate(inputs):
        sub_examples = []
        for sub_i, each_similar_dialog in enumerate(dialog_data[3]):
          guid = "%s-%d-%d" % (set_type, i + 1, sub_i + 1)
          text_a = tokenization.convert_to_unicode(each_similar_dialog)
          sub_examples.append(InputExample(guid=guid, text_a=text_a))
        examples.append(sub_examples)
      print("%s data creation is finished! %d" % (set_type, len(examples)))
      return examples

    def _create_knowledge_examples(self, inputs, set_type):

      examples = []
      for (i, dialog_data) in enumerate(inputs):
        guid = "%s-%d" % (set_type, i + 1)
        text_a = tokenization.convert_to_unicode(dialog_data[3])

        examples.append(
          InputExample(guid=guid, text_a=text_a))
      print("%s data creation is finished! %d" % (set_type, len(examples)))
      return examples

    def _create_knowledge_bert_seq_examples(self, inputs, set_type):
      def get_knowledge_sequence_outputs():
        knowledge_pickle_path = "/home/taesun/taesun_workspace/bert_knowledge/data/knowledge/knowledge_bert_sequence_outputs.pickle"
        manual_dict = dict()
        with open(knowledge_pickle_path, "rb") as fr_knowledge_handle:
          while True:
            try:
              ubuntu_manual, sequence_outputs, knowledge_length = pickle.load(fr_knowledge_handle)
              manual_dict[ubuntu_manual] = (sequence_outputs, knowledge_length)
            except EOFError:
              print("knowledge description loading is finished", len(manual_dict))
              break

        return manual_dict

      knowledge_dict = get_knowledge_sequence_outputs()
      examples = []
      for (i, dialog_data) in enumerate(inputs):
        # it will be a knowledge tokens(e.g. 0alias, sudo)
        # dialog_data[3] : [sudo, lsmod, sudo, ...], [1,1,1,0,0]

        each_bert_seq_out = []
        each_knowledge_length = []
        each_knowledge_label = []
        for knowlege_name, knowledge_label in zip(dialog_data[3], dialog_data[4]):
          # knowledge bert sequence outputs : [80, 768]
          each_bert_seq_out.append(knowledge_dict[knowlege_name][0])
          # knowledge each length
          each_knowledge_length.append(knowledge_dict[knowlege_name][1])
          # knowledge label
          each_knowledge_label.append(knowledge_label)
        examples.append(
          KnowledgeExample(bert_seq_out=each_bert_seq_out, length=each_knowledge_length, label=each_knowledge_label))

      print("%s knowledge data creation is finished! %d" % (set_type, len(examples)))
      return examples

    def _create_knowledge_separate_examples(self, inputs, set_type):
      knowledge_num = self.hparams.top_n
      examples = []
      for (i, dialog_data) in enumerate(inputs):
        separate_examples = []
        text_b = tokenization.convert_to_unicode(dialog_data[1])
        for j, (knowledge, knowledge_label) in enumerate(zip(dialog_data[3], dialog_data[4])):
          guid = "%s-%d-%d" % (set_type, i + 1, j + 1)
          text_a = tokenization.convert_to_unicode(knowledge)
          label = tokenization.convert_to_unicode(str(knowledge_label))
          separate_examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        examples.append(separate_examples)

      print("%s data creation is finished! %d" % (set_type, len(examples)))
      return examples

    def _create_seperate_examples(self, inputs, set_type):
      dialog_examples = []
      response_examples = []

      for (i, dialog_data) in enumerate(inputs):
        guid = "%s-%d" % (set_type, i + 1)
        dialog = tokenization.convert_to_unicode(dialog_data[0])
        response = tokenization.convert_to_unicode(dialog_data[1])
        label = tokenization.convert_to_unicode(dialog_data[2])
        dialog_examples.append(
          InputExample(guid=guid, text_a=dialog, label=label))
        response_examples.append(
          InputExample(guid=guid, text_a=response, label=label)
        )

      print("%s data creation is finished! %d" % (set_type, len(dialog_examples)))

      return dialog_examples, response_examples

    def get_bert_knowledge_batch_data(self, curr_index, batch_size, set_type="train"):
      input_ids = []
      input_mask = []
      segment_ids = []
      input_lengths = []

      examples = {
        "train": self.train_knowledge,
        "valid": self.valid_knowledge,
        "test": self.test_knowledge
      }
      example = examples[set_type]

      for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
        feature = convert_single_example(curr_index * batch_size + index, each_example, None,
                                         self.knowledge_max_seq_length, self.tokenizer, "knowledge")

        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)
        input_lengths.append(feature.input_length)

      return [input_ids, input_mask, segment_ids, input_lengths]

    def get_bert_pretrained_knowledge_top_n_batch_data(self, curr_index, batch_size, set_type="train"):
      knowledge_tokens = []
      knowledge_lengths = []
      knowledge_label_ids = []

      examples = {
        "train": self.train_knowledge,
        "valid": self.valid_knowledge,
        "test": self.test_knowledge
      }
      example = examples[set_type]

      if example is None:
        return None

      for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
        knowledge_tokens.append(each_example.bert_seq_out)
        knowledge_lengths.append(each_example.length)
        knowledge_label_ids.append(each_example.label)

      return [knowledge_tokens, knowledge_lengths, knowledge_label_ids]

    def get_bert_knowledge_top_n_batch_data(self, curr_index, batch_size, set_type="train"):
      input_ids = []
      input_mask = []
      segment_ids = []
      input_lengths = []
      label_ids = []

      examples = {
        "train": self.train_knowledge,
        "valid": self.valid_knowledge,
        "test": self.test_knowledge
      }
      example = examples[set_type]
      if example is None:
        return None

      # input_ids: [batch_size, top_n, max_seq_len]
      for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
        temp_input_ids = []
        temp_input_mask = []
        temp_segment_ids = []
        temp_input_lengths = []
        temp_label_ids = []
        for j, each_knowledge in enumerate(each_example):
          feature = convert_single_example(curr_index * batch_size + index, each_knowledge, self.label_list,
                                           self.knowledge_max_seq_length, self.tokenizer, "knowledge")

          temp_input_ids.append(feature.input_ids)
          temp_input_mask.append(feature.input_mask)
          temp_segment_ids.append(feature.segment_ids)
          temp_input_lengths.append(feature.input_length)
          temp_label_ids.append(feature.label_id)

        input_ids.append(temp_input_ids)
        input_mask.append(temp_input_mask)
        segment_ids.append(temp_segment_ids)
        # input_length : knowledge_length
        input_lengths.append(temp_input_lengths)
        label_ids.append(temp_label_ids)

      return [input_ids, input_mask, segment_ids, input_lengths, label_ids]

    def get_similar_dialog_batch_data(self, curr_index, batch_size, set_type="train"):
      input_ids = []
      input_mask = []
      input_lengths = []

      examples = {
        "train": self.train_similar_dialog,
        "valid": self.valid_similar_dialog,
        "test": self.test_similar_dialog
      }
      example = examples[set_type]
      if example is None:
        return None

      # input_ids: [batch_size, top_n, max_seq_len]
      for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
        temp_input_ids = []
        temp_input_mask = []
        temp_input_lengths = []

        for j, each_dialog in enumerate(each_example):
          feature = _conver_single_example_wihtout_cls_sep(each_dialog, self.max_seq_length, self.tokenizer)

          temp_input_ids.append(feature.input_ids)
          temp_input_mask.append(feature.input_mask)
          temp_input_lengths.append(feature.input_length)

        input_ids.append(temp_input_ids)
        input_mask.append(temp_input_mask)
        input_lengths.append(temp_input_lengths)

      return [input_ids, input_mask, input_lengths]

    def get_analysis_bert_data(self, examples):

      input_ids = []
      input_mask = []
      segment_ids = []
      position_ids = []
      text_a_lengths = []
      text_b_lengths = []
      label_ids = []

      for index, each_example in enumerate(examples):
        print(each_example.text_a)
        feature = convert_analysis_example(each_example, self.max_seq_length, self.tokenizer)

        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)
        position_ids.append(feature.position_ids)
        text_a_lengths.append(feature.input_length)
        text_b_lengths.append(feature.input_length)
        label_ids.append(feature.label_id)

      return [input_ids, input_mask, segment_ids, (text_a_lengths, text_b_lengths), label_ids, position_ids]

    def get_bert_batch_data(self, curr_index, batch_size, set_type="train"):
      input_ids = []
      input_mask = []
      segment_ids = []
      label_ids = []
      text_a_lengths = []
      text_b_lengths = []
      is_real_examples = []
      position_ids = []
      raw_dialogs = []
      raw_responses = []

      examples = {
          "train": self.train_example,
          "valid": self.valid_example,
          "test": self.test_example
      }
      example = examples[set_type]
      sentence_examples = []
      for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
        sentence_examples.append(each_example)

        feature = convert_separate_example(
          curr_index * batch_size + index,
          each_example, self.label_list, self.max_seq_length,
          self.dialog_max_seq_length, self.response_max_seq_length, self.tokenizer
        )
        raw_dialogs.append(each_example.text_a)
        raw_responses.append(each_example.text_b)
        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)
        label_ids.append(feature.label_id)
        text_a_lengths.append(feature.input_length[0])
        text_b_lengths.append(feature.input_length[1])
        position_ids.append(feature.position_ids)
        is_real_examples.append(feature.is_real_example)

      return [input_ids, input_mask, segment_ids, (text_a_lengths, text_b_lengths), label_ids, position_ids, (raw_dialogs,raw_responses)]

    def get_seperate_batch_data(self, curr_index, batch_size, set_type="train"):
      def get_batch_data(max_seq_length, example, sentence_type="dialog"):
        input_ids = []
        input_mask = []
        segment_ids = []
        label_ids = []
        input_lengths = []
        is_real_examples = []

        for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
          feature = convert_single_example(curr_index * batch_size + index,
                                           each_example, self.label_list, max_seq_length, self.tokenizer, sentence_type)

          input_ids.append(feature.input_ids)
          input_mask.append(feature.input_mask)
          segment_ids.append(feature.segment_ids)
          label_ids.append(feature.label_id)
          input_lengths.append(feature.input_length)
          is_real_examples.append(feature.is_real_example)

        return [input_ids, input_mask, segment_ids, input_lengths, label_ids]

      examples = {
        "train": [self.train_dialog, self.train_response],
        # "valid": [self.valid_dialog, self.valid_response],
        "test": [self.test_dialog, self.test_response]
      }

      dialog_example, response_example = examples[set_type]

      dialog_batch = get_batch_data(self.dialog_max_seq_length, dialog_example, "dialog")
      response_batch = get_batch_data(self.response_max_seq_length, response_example, "response")

      return dialog_batch, response_batch


class KodiProcessor(DataProcessor):
  """Processor for the KoDi data set"""

  def get_train_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv("../train.tsv"), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self):
    return self._create_examples(
        self._read_tsv("../test.tsv"), "test")

  def get_labels(self):
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(
         InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    print("%d %s data creation is finished!" % (len(examples), set_type))
    return examples


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    print("example", example)

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
    print("file_based_input_fn_builder : input_fn", d)

    return d

  return input_fn

""" make tf.records """
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""
  tf.logging.info("Convert a set of `InputExample`s to a TFRecord file.")

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()

def ubuntu_sep_token_append(tokens_a):

  tokens_a = " ".join(tokens_a)

  modified_tokens_a = list()
  sep_split_tokens_a = tokens_a.strip().split(" e ##ot ")

  for index, utt in enumerate(sep_split_tokens_a):
    if utt != sep_split_tokens_a[-1]:
      utt += " [SEP] "
    modified_tokens_a.extend(utt.strip().split(" "))

  return modified_tokens_a

def dialog_position_id(tokens_a, position_ids, reverse=True):
  tokens_a = " ".join(tokens_a)
  tokens_a_split_by_utt = tokens_a.strip().split(" e ##ot ")

  position_id_list = list(range(len(tokens_a_split_by_utt)))
  for index, utt in enumerate(tokens_a_split_by_utt):
    utt_tokens = utt.strip().split(" ")
    if index != len(tokens_a_split_by_utt) - 1:
      utt_tokens.extend(["e","##ot"])

    for _ in range(len(utt_tokens)):
      if not reverse:
        position_ids.append(index)
      else:
        position_ids.append(position_id_list[-(index + 1)])
  # last_position_id = positions_ids[-1]

  return position_ids

def convert_analysis_example(example, max_seq_length=320, tokenizer=None):

  tokens_a = tokenizer.tokenize(example.text_a) # 280
  print(tokens_a)

  tokens = []
  segment_ids = []
  input_mask = []
  position_ids = []
  zero_position_id = 0

  # tokens.append("[CLS]")
  # segment_ids.append(0)
  # input_mask.append(1)
  # position_ids.append(zero_position_id)

  # text_a
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
    input_mask.append(1)
    position_ids.append(zero_position_id)

  # tokens.append("[SEP]")
  # segment_ids.append(0)
  # input_mask.append(1)
  # position_ids.append(zero_position_id)
  print(len(tokens))

  #text_a padding
  while len(tokens) < max_seq_length:
    tokens.append("[PAD]")
    segment_ids.append(0)
    input_mask.append(0)
    position_ids.append(zero_position_id)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(position_ids) == max_seq_length

  tf.logging.info("*** Example ***")
  tf.logging.info("guid: %s" % (example.guid))
  tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))

  feature = InputFeatures(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    label_id=0,
    input_length=len(tokens_a),
    position_ids=position_ids,
    is_real_example=True)

  return feature


def convert_separate_example(ex_index, example, label_list, max_seq_length,
                             max_seq_a, max_seq_b, tokenizer, sentence_type="dialog"):
  if isinstance(example, PaddingInputExample):
    return InputFeatures(
      input_ids=[0] * max_seq_length,
      input_mask=[0] * max_seq_length,
      segment_ids=[0] * max_seq_length,
      label_id=0,
      input_length=0,
      position_ids=[100] * max_seq_length,
      is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a) # 280
  tokens_b = tokenizer.tokenize(example.text_b) # 40

  # #TODO:tokens_a : how many turns in a dialog
  # from collections import Counter
  # dialog_counter = Counter(tokens_a)
  # print(dialog_counter["[EOT]"])

  # 278 + [CLS] [SEP] : 280
  while len(tokens_a) + 2 > max_seq_a:
    if sentence_type == "dialog":
      tokens_a.pop(0)
    else:
      tokens_a.pop()

  # 39 + [SEP] : 40
  while len(tokens_b) + 1 > max_seq_b:
    tokens_b.pop()

  tokens = []
  segment_ids = []
  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = []
  input_lengths = []
  position_ids = []
  zero_position_id = 200

  tokens.append("[CLS]")
  segment_ids.append(0)
  input_mask.append(1)
  position_ids.append(zero_position_id)

  # position_ids = dialog_position_id(tokens_a, position_ids, reverse=True)
  # last_position_id = position_ids[-1] + 1

  # text_a
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
    input_mask.append(1)
    position_ids.append(zero_position_id)

  # assert len(position_ids) == len(tokens)

  tokens.append("[SEP]")
  segment_ids.append(0)
  input_mask.append(1)
  input_lengths.append(len(tokens))
  position_ids.append(zero_position_id)

  #text_a padding
  while len(tokens) < max_seq_a:
    tokens.append("[PAD]")
    segment_ids.append(0)
    input_mask.append(0)
    position_ids.append(zero_position_id)

  total_tokens_a = len(tokens)
  # text_b
  for token in tokens_b:
    tokens.append(token)
    segment_ids.append(1)
    input_mask.append(1)
    # for response position(should be the last position in a dialog context)
    # position_ids.append(last_position_id)
    position_ids.append(0)

  tokens.append("[SEP]")
  segment_ids.append(1)
  input_mask.append(1)
  input_lengths.append(len(tokens) - total_tokens_a)
  position_ids.append(zero_position_id)

  # text_b padding
  while len(tokens) < max_seq_length:
    tokens.append("[PAD]")
    segment_ids.append(1)
    input_mask.append(0)
    position_ids.append(zero_position_id)
  # print(tokens)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(position_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 1:
      print("*** Example ***")
      print("guid: %s" % (example.guid))
      print("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
      print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      print("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    label_id=label_id,
    input_length=input_lengths,
    position_ids=position_ids,
    is_real_example=True)

  return feature

def _conver_single_example_wihtout_cls_sep(example, max_seq_length, tokenizer):
  if isinstance(example, PaddingInputExample):
    return InputFeatures(
      input_ids=[0] * max_seq_length,
      input_mask=[0] * max_seq_length,
      segment_ids=[0] * max_seq_length,
      label_id=0,
      input_length=0,
      is_real_example=False)

  tokens_a = tokenizer.tokenize(example.text_a)
  if len(tokens_a) > max_seq_length:
    input_length = max_seq_length
    tokens_a = tokens_a[0:(max_seq_length)]
  else:
    input_length = len(tokens_a)

  tokens = []
  segment_ids = []

  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  feature = InputFeatures(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    label_id=None,
    input_length=input_length,
    is_real_example=True)

  return feature


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, sentence_type="dialog"):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        input_length=0,
        is_real_example=False)

  label_id = None
  if label_list:
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i
    label_id = label_map[example.label]

  input_length = 0
  tokens_a = tokenizer.tokenize(example.text_a)

  # if dataset == "ubuntu":
  #   tokens_a = ubuntu_sep_token_append(tokens_a)

  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      input_length = max_seq_length
      if sentence_type == "dialog":
        tokens_a = tokens_a[-(max_seq_length - 2):]
      else:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    else:
      input_length = len(tokens_a) + 2

  tokens = []
  segment_ids = []

  tokens.append("[CLS]")
  segment_ids.append(0)

  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)

    sentence_length = len(tokens)

  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)

    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if ex_index < 3:
      print("*** Example ***")
      print("guid: %s" % (example.guid))
      print("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      input_length = input_length,
      is_real_example=True)

  return feature

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop(0)
      # tokens_a.pop()

    else:
      tokens_b.pop()

def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1
    print("num_examples",len(examples))

  return examples