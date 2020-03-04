import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging
from datetime import datetime
import time

from utils import *
from evaluate_utils import *

from data_process import *
from bert_model import tokenization, optimization, modeling_base
from analysis.merge_subtokens import *

class Model(object):
	def __init__(self, hparams):
		self.hparams = hparams
		self._logger = logging.getLogger(__name__)
		self.train_setup_vars = dict()
		self.train_setup_vars["on_training"] = False
		self.train_setup_vars["do_evaluate"] = False
		self.train_setup_vars["is_train_continue"] = False
		self.bert_config = modeling_base.BertConfig.from_json_file(self.hparams.bert_config_dir)

		self._make_data_processor()

	def _make_data_processor(self):
		processors = {
			"ubuntu": UbuntuProcessor,
		}

		data_dir = self.hparams.data_dir
		self.processor = processors[self.hparams.task_name](self.hparams)
		self.train_examples, self.train_knowledge_examples, self.train_similar_examples = \
			self.processor.get_train_examples(data_dir)
		self.valid_examples, self.valid_knowledge_examples, self.valid_similar_examples = \
			self.processor.get_dev_examples(data_dir)
		self.test_examples, self.test_knowledge_examples, self.test_similar_examples = \
			self.processor.get_test_examples(data_dir)
		self.label_list = self.processor.get_labels()

		self.tokenizer = tokenization.FullTokenizer(self.hparams.vocab_dir, self.hparams.do_lower_case)
		self.processor.data_process_feature(self.hparams, self.tokenizer)

		self.num_train_steps = int(
			len(self.train_examples) / self.hparams.train_batch_size * self.hparams.num_epochs)
		self.warmup_proportion = 0.1
		self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)

	def _make_placeholders(self):
		top_n = self.hparams.top_n
		knowledge_response_max_len = self.hparams.knowledge_max_seq_length
		# Bert Init Checkpoint
		self.input_ids_ph = tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="input_ids_ph")
		self.input_mask_ph = tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="input_mask_ph")
		self.segment_ids_ph = tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="segment_ids_ph")
		self.dialog_position_ids_ph = tf.placeholder(tf.int32, shape=[None, self.hparams.max_seq_length], name="dialog_position_ids_ph")

		self.dialog_len_ph = tf.placeholder(tf.int32, shape=[None], name="dialog_len_ph")
		self.response_len_ph = tf.placeholder(tf.int32, shape=[None], name="response_len_ph")

		self.knowledge_tokens_ph = \
			tf.placeholder(tf.float32, shape=[None, top_n, self.hparams.knowledge_max_seq_length, 768], name="knowledge_tokens_ph")
		self.knowledge_len_ph = tf.placeholder(tf.int32, shape=[None, top_n], name="knowledge_len_ph")

		self.similar_input_ids_ph = tf.placeholder(tf.int32, shape=[None, top_n, self.hparams.max_seq_length], name="similar_input_ids_ph")
		self.similar_input_mask_ph = tf.placeholder(tf.int32, shape=[None, top_n, self.hparams.max_seq_length], name="similar_input_mask_ph")
		self.similar_input_len_ph = tf.placeholder(tf.int32, shape=[None, top_n], name="similar_dialog_len_ph")

		self.label_ids_ph = tf.placeholder(tf.int32, shape=[None], name="label_ids_ph")
		self.knowledge_label_ids_ph = tf.placeholder(tf.int32, shape=[None, top_n], name="knowledge_label_ids_ph")

		self.is_real_examples_ph = tf.placeholder(tf.float32, shape=[None], name="is_real_examples")
		self.dropout_keep_prob_ph = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")

	def _select_train_variables(self):
		# not keep training(loop) | training from pretrained_file(checkpoint)
		if not self.train_setup_vars["on_training"]:
			self.need_to_initialize_vars = []
			vars_in_checkpoint = tf.train.list_variables(self.hparams.init_checkpoint)
			checkpoint_vars = []
			for var_name, _ in vars_in_checkpoint:
				checkpoint_vars.append(var_name)
				# print(var_name)

			var_dict = dict()
			for var in tf.global_variables():
				if var.name[:-2] not in checkpoint_vars:

					print("not included variable : ", var.name)
					self.need_to_initialize_vars.append(var)
					continue

				print(var)
				var_dict[var.name[:-2]] = var

			if not self.train_setup_vars["is_train_continue"] and not self.train_setup_vars["do_evaluate"]:
				saver = tf.train.Saver(var_dict)
				saver.restore(self.sess, self.hparams.init_checkpoint)

			# all bert pretrained var names
			self.pretrained_all_var_names = []
			# half of the transformer layers which will not be trained during the fine-tuning training
			self.pretrained_not_train_var_names = []
			if len(self.hparams.train_transformer_layer) == 0:
				print("not train transformer_layer", "-"*50)
				for var in tf.trainable_variables():
					if var not in self.need_to_initialize_vars:
						self.pretrained_not_train_var_names.append(var)
						self.pretrained_all_var_names.append(var)
			else:
				for var in tf.trainable_variables():
					self.pretrained_all_var_names.append(var)
					print(var)
					var_name_split = var.name.split("/")
					if len(var_name_split) > 1:
						if var_name_split[1] == "encoder":
							layer_num = int(var_name_split[2].split("_")[-1])
							if layer_num not in self.hparams.train_transformer_layer \
									and var not in self.need_to_initialize_vars:
								self.pretrained_not_train_var_names.append(var)

	def _build_train_graph(self):
		gpu_num = len(self.hparams.gpu_num)
		if gpu_num > 1:print("-" * 10, "Using %d Multi-GPU" % gpu_num, "-" * 10)
		else:print("-" * 10, "Using Single-GPU", "-" * 10)

		if not self.train_setup_vars["do_evaluate"]: is_training = True
		else:is_training = False
		use_one_hot_embeddings = False

		input_ids_ph = tf.split(self.input_ids_ph, gpu_num, 0)
		input_mask_ph = tf.split(self.input_mask_ph, gpu_num, 0)
		segment_ids_ph = tf.split(self.segment_ids_ph, gpu_num, 0)
		dialog_position_ids_ph = tf.split(self.dialog_position_ids_ph, gpu_num, 0)

		dialog_len_ph = tf.split(self.dialog_len_ph, gpu_num, 0)
		response_len_ph = tf.split(self.response_len_ph, gpu_num, 0)

		knowledge_tokens_ph = tf.split(self.knowledge_tokens_ph, gpu_num, 0)
		knowledge_len_ph = tf.split(self.knowledge_len_ph, gpu_num, 0)
		knowledge_label_ids_ph = tf.split(self.knowledge_label_ids_ph, gpu_num, 0)

		similar_input_ids_ph = tf.split(self.similar_input_ids_ph, gpu_num, 0)
		similar_input_mask_ph = tf.split(self.similar_input_mask_ph, gpu_num, 0)
		similar_len_ph = tf.split(self.similar_input_len_ph, gpu_num, 0)

		label_ids_ph = tf.split(self.label_ids_ph, gpu_num, 0)
		# is_real_examples_ph = tf.split(self.is_real_examples_ph, gpu_num, 0)

		tower_grads = []
		tot_losses = []
		tot_logits = []
		tot_labels = []
		tot_outputs = []
		tot_argmax = []

		tvars = []
		for i, gpu_id in enumerate(self.hparams.gpu_num):
			with tf.device('/gpu:%d' % gpu_id):
				with tf.variable_scope('', reuse=tf.AUTO_REUSE):
					print("bert_graph_multi_gpu :", gpu_id)

					if self.hparams.do_dialog_state_embedding: each_dialog_position_ids = dialog_position_ids_ph[i]
					else: each_dialog_position_ids = None

					bert_model = modeling_base.BertModel(
						config=self.bert_config,
						is_training=is_training,
						input_ids=input_ids_ph[i],
						input_mask=input_mask_ph[i],
						token_type_ids=segment_ids_ph[i],
						dialog_position_ids=each_dialog_position_ids,
						use_adapter_layer=self.hparams.do_adapter_layer,
						scope='bert',
						hparams=self.hparams
					)

					input_phs = (input_ids_ph[i], input_mask_ph[i])
					length_phs = (dialog_len_ph[i], response_len_ph[i], knowledge_len_ph[i])
					knowledge_phs = knowledge_tokens_ph[i]
					similar_phs = None
					if self.hparams.do_similar_dialog:
						similar_phs = (similar_input_ids_ph[i], similar_input_mask_ph[i], similar_len_ph[i])
					label_ids_phs = (label_ids_ph[i], knowledge_label_ids_ph[i])
					created_model = self.hparams.graph.Model(self.hparams, self.bert_config, bert_model, input_phs,
																									 label_ids_phs, length_phs, knowledge_phs, similar_phs)
					logits, loss_op, seq_outputs, sentence_argmax = created_model.build_graph()

					tot_losses.append(loss_op)
					tot_logits.append(logits)
					tot_labels.append(label_ids_ph[i])
					tot_outputs.append(seq_outputs)
					tot_argmax.append(sentence_argmax)

					if i == 0:

						self._select_train_variables()
						if self.hparams.do_adam_weight_optimizer:
							self.optimizer, self.global_step = optimization.create_optimizer(
								loss_op, self.hparams.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)
						else:
							self.optimizer = tf.train.AdamOptimizer(self.hparams.learning_rate)
							self.global_step = tf.Variable(0, name="global_step", trainable=False)

					if not self.hparams.do_train_bert:
						if i == 0:
							for var in tf.trainable_variables():
								if var not in self.pretrained_not_train_var_names:
									tvars.append(var)
					else:
						tvars = tf.trainable_variables()

					if self.hparams.do_adam_weight_optimizer:
						# This is how the model was pre-trained.
						grads = tf.gradients(loss_op, tvars)
						(grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
						tower_grads.append(zip(grads, tvars))
					else:
						grads = self.optimizer.compute_gradients(loss_op, var_list=tvars)
						tower_grads.append(grads)
					tf.get_variable_scope().reuse_variables()

		avg_grads = average_gradients(tower_grads)
		self.loss_op = tf.divide(tf.add_n(tot_losses), gpu_num)
		self.logits = tf.concat(tot_logits, axis=0)
		self.sequence_outputs = seq_outputs
		self.sentence_argmax = sentence_argmax
		tot_labels = tf.concat(tot_labels, axis=0)

		with tf.variable_scope('', reuse=tf.AUTO_REUSE):
			self.train_op = self.optimizer.apply_gradients(avg_grads, self.global_step)
			# new_global_step = self.global_step + 1
			# self.train_op = tf.group(self.train_op, [self.global_step.assign(new_global_step)])

		if self.hparams.loss_type == "sigmoid":
			correct_pred = tf.equal(tf.round(tf.nn.sigmoid(self.logits)), tf.cast(self.label_ids_ph, tf.float32))
			self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		else:
			eval = tf.nn.in_top_k(self.logits, self.label_ids_ph, 1)
			correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))
			self.accuracy = tf.divide(correct_count, tf.shape(self.label_ids_ph)[0])
			self.confidence = tf.nn.softmax(self.logits, axis=-1)

		if not self.train_setup_vars["do_evaluate"] and not self.train_setup_vars["on_training"] \
				and not self.train_setup_vars["is_train_continue"]:
			self._initialize_uninitialized_variables()

	def _initialize_uninitialized_variables(self):
		uninitialized_vars = []
		self._logger.info("Initializing Updated Variables...")
		for var in tf.global_variables():
			if var in self.pretrained_all_var_names and var not in self.need_to_initialize_vars:
				print("Pretrained Initialized Variables / ", var)
				continue
			uninitialized_vars.append(var)
			print("Update Initialization / ", var)
		init_new_vars_op = tf.variables_initializer(uninitialized_vars)
		self.sess.run(init_new_vars_op)

		total_parameters = 0
		for variable in self.optimizer.variables():
			# shape is an array of tf.Dimension
			shape = variable.get_shape()
			# print(shape)
			# print(len(shape))
			variable_parameters = 1
			for dim in shape:
				# print(dim)
				variable_parameters *= dim.value
			# print(variable_parameters)
			total_parameters += variable_parameters
		print("total_parameters", total_parameters)

	def _make_feed_dict(self, batch_data, dropout_keep_prob):
		feed_dict = {}
		dialog_data, knowledge_data, similar_dialog_data = batch_data
		input_ids, input_mask, segment_ids, (dialog_lengths, response_lengths), label_ids, dialog_position_ids, _ = dialog_data

		feed_dict[self.input_ids_ph] = input_ids
		feed_dict[self.input_mask_ph] = input_mask
		feed_dict[self.segment_ids_ph] = segment_ids
		feed_dict[self.dialog_position_ids_ph] = dialog_position_ids

		feed_dict[self.dialog_len_ph] = dialog_lengths
		feed_dict[self.response_len_ph] = response_lengths

		feed_dict[self.label_ids_ph] = label_ids
		feed_dict[self.dropout_keep_prob_ph] = dropout_keep_prob

		if knowledge_data is not None:
			knowledge_tokens, knowledge_lengths, knowledge_label_ids = knowledge_data
			feed_dict[self.knowledge_tokens_ph] = knowledge_tokens
			feed_dict[self.knowledge_len_ph] = knowledge_lengths
			feed_dict[self.knowledge_label_ids_ph] = knowledge_label_ids

		if similar_dialog_data is not None:
			similar_dialog_inputs, similar_dialog_mask, similar_dialog_lengths = similar_dialog_data
			feed_dict[self.similar_input_ids_ph] = similar_dialog_inputs
			feed_dict[self.similar_input_mask_ph] = similar_dialog_mask
			feed_dict[self.similar_input_len_ph] = similar_dialog_lengths

		return feed_dict

	def train(self, pretrained_file=None):
		config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False,
		)
		config.gpu_options.allow_growth = True

		if pretrained_file:
			# self.train_setup_vars["on_training"] = True
			self.train_setup_vars["is_train_continue"] = True

		self.sess = tf.Session(config=config)
		self._make_placeholders()
		self._build_train_graph()

		# Tensorboard
		saver = tf.train.Saver(max_to_keep=10)
		if pretrained_file:
			self.sess.run(tf.global_variables_initializer())
			saver.restore(self.sess, pretrained_file)
			self._logger.info("Restoring Session from checkpoint complete!")

		self.tensorboard_summary = TensorBoardSummaryWriter(self.hparams.root_dir, self.sess, self.sess.graph)
		# merge = tf.summary.merge_all()
		step_loss_mean, step_accuracy_mean = 0, 0
		total_data_len = int(math.ceil(len(self.train_examples) / self.hparams.train_batch_size))
		self._logger.info("Batch iteration per epoch is %d" % total_data_len)

		start_time = datetime.now().strftime('%H:%M:%S')
		self._logger.info("Start train model at %s" % start_time)
		for epoch_completed in range(self.hparams.num_epochs):
			start_time = time.time()
			if epoch_completed > 0 and self.hparams.training_shuffle_num > 0:
				self.train_examples, self.train_knowledge_examples, self.train_similar_examples = \
					self.processor.get_train_examples(self.hparams.data_dir)

			for i in range(total_data_len):
				dialog_data = self.processor.get_bert_batch_data(i, self.hparams.train_batch_size, "train")
				knowledge_batch_data = self.processor.get_bert_pretrained_knowledge_top_n_batch_data(i, self.hparams.train_batch_size, "train")
				similar_dialog_data = self.processor.get_similar_dialog_batch_data(i, self.hparams.train_batch_size, "train")
				batch_data = (dialog_data, knowledge_batch_data, similar_dialog_data)

				accuracy_val, loss_val, global_step_val, _ = self.sess.run(
					[self.accuracy,
					 self.loss_op,
					 self.global_step,
					 self.train_op],
					feed_dict=self._make_feed_dict(batch_data, self.hparams.dropout_keep_prob)
				)
				step_loss_mean += loss_val
				step_accuracy_mean += accuracy_val

				if global_step_val % self.hparams.tensorboard_step == 0:
					step_loss_mean /= self.hparams.tensorboard_step
					step_accuracy_mean /= self.hparams.tensorboard_step

					self.tensorboard_summary.add_summary("train/cross_entropy", step_loss_mean, global_step_val)
					self.tensorboard_summary.add_summary("train/accuracy", step_accuracy_mean, global_step_val)
					# self.tensorboard_summary.add_tensor_summary(tf_summary_val, global_step_val)

					self._logger.info("[Step %d][%d th] loss: %.4f, accuracy: %.2f%%  (%.2f seconds)" % (
						global_step_val,
						i + 1,
						step_loss_mean,
						step_accuracy_mean * 100,
						time.time() - start_time))

					step_loss_mean, step_accuracy_mean = 0, 0
					start_time = time.time()

				if global_step_val % self.hparams.evaluate_step == 0:
					self.train_setup_vars["on_training"] = True
					save_path = self.model_evaluate_save(saver, global_step_val)
					self.train_setup_vars["do_evaluate"]  = False

					# tf.reset_default_graph()
					# self.sess = tf.Session(config=config)
					# self._make_placeholders()
					# self._build_train_graph()
					#
					# saver = tf.train.Saver()
					# saver.restore(self.sess, save_path)
					continue

				if global_step_val % self.hparams.save_step == 0:
					self._logger.info("Saving Model...[Step %d]" % global_step_val)
					self.model_save(saver, global_step_val)

			self._logger.info("End of epoch %d." % (epoch_completed + 1))
		self.tensorboard_summary.close()

		if self.sess is not None:
			self.sess.close()

	def model_save(self, saver, global_step_val):
		save_path = saver.save(self.sess, os.path.join(self.hparams.root_dir, "model.ckpt"), global_step=global_step_val)
		self._logger.info("Model saved at : %s" % save_path)

	def model_evaluate_save(self, saver, global_step_val):

		save_path = saver.save(self.sess, os.path.join(self.hparams.root_dir, "model.ckpt"), global_step=global_step_val)
		self._logger.info("Model saved at : %s" % save_path)
		self._run_evaluate(data_type="test")

		return save_path

	def _run_evaluate(self, data_type="test"):
		k_list = self.hparams.recall_k_list
		total_examples = 0
		total_correct = np.zeros([len(k_list)], dtype=np.int32)
		total_mrr = 0

		index = 0
		total_data_len = int(math.ceil(len(self.test_examples) / self.hparams.eval_batch_size))
		self._logger.info("Evaluation batch iteration per epoch is %d" % total_data_len)

		if self.train_setup_vars["do_evaluate"]:
			print("Evaluation batch iteration per epoch is %d" % total_data_len)

		for i in range(total_data_len):
			dialog_data = self.processor.get_bert_batch_data(i, self.hparams.eval_batch_size, data_type)
			knowledge_batch_data = self.processor.get_bert_pretrained_knowledge_top_n_batch_data(i, self.hparams.eval_batch_size, data_type)
			similar_dialog_data = self.processor.get_similar_dialog_batch_data(i, self.hparams.eval_batch_size, data_type)
			batch_data = (dialog_data, knowledge_batch_data, similar_dialog_data)

			if self.hparams.loss_type == "sigmoid":
				logits_val = self.sess.run([self.logits], feed_dict=self._make_feed_dict(batch_data, 1.0))
				pred_score = logits_val[0]
			else:
				confidence_val = self.sess.run([self.confidence], feed_dict=self._make_feed_dict(batch_data, 1.0))
				pred_score = confidence_val[0][:, 1]
			#TODO:dialog[3][0] : dialog_lengths, dilaog[3][1] : response_lengths

			rank_by_pred = calculate_candidates_ranking(pred_score, dialog_data[4], self.hparams.evaluate_candidates_num)
			num_correct, pos_index = logits_recall_at_k(rank_by_pred, k_list)

			total_mrr += logits_mean_reciprocal_rank(rank_by_pred)

			total_correct = np.add(total_correct, num_correct)
			index += 1
			# print(index)
			total_examples = index * rank_by_pred.shape[0]

			recall_result = ""
			if index % self.hparams.evaluate_print_step == 0:
				for i, k in enumerate(k_list):
					recall_val = "%.2f" % float((total_correct[i] / total_examples) * 100)
					self.tensorboard_summary.add_summary(
						"recall_test/recall_%s" % k, recall_val, index)

				avg_mrr = "%.4f" % float(total_mrr / total_examples)
				self.tensorboard_summary.add_summary(
					"mrr_test/mean_reciprocal_rank", avg_mrr, index)

				for i in range(len(k_list)):
					recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % (
							(total_correct[i] / total_examples) * 100)
				else:
					print("%d[th] | %s | MRR : %.3f" % (index, recall_result, float(total_mrr / total_examples)))
				self._logger.info("%d[th] | %s | MRR : %.3f" % (index, recall_result, float(total_mrr / total_examples)))

		avg_mrr = float(total_mrr / total_examples)
		recall_result = ""

		for i in range(len(k_list)):
			recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % ((total_correct[i] / total_examples) * 100)
		self._logger.info(recall_result)
		self._logger.info("MRR: %.4f" % avg_mrr)

		return k_list, (total_correct / total_examples) * 100, avg_mrr

	def analysis_run_evaluate(self, eval_split="test"):
		k_list = self.hparams.recall_k_list
		total_examples = 0
		total_correct = np.zeros([len(k_list)], dtype=np.int32)
		total_mrr = 0

		index = 0
		total_data_len = int(math.ceil(len(self.test_examples) / self.hparams.eval_batch_size))
		self._logger.info("Evaluation batch iteration per epoch is %d" % total_data_len)

		if self.train_setup_vars["do_evaluate"]:
			print("Evaluation batch iteration per epoch is %d" % total_data_len)


		print_idx = 11660
		curr_idx = (print_idx // 250)
		for i in range(total_data_len):
			if i < curr_idx: continue
			dialog_data = self.processor.get_bert_batch_data(i, self.hparams.eval_batch_size, eval_split)
			# knowledge_batch_data = self.processor.get_bert_pretrained_knowledge_top_n_batch_data(i, self.hparams.eval_batch_size, eval_split)
			# similar_dialog_data = self.processor.get_similar_dialog_batch_data(i, self.hparams.eval_batch_size, eval_split)
			batch_data = (dialog_data, None, None)

			logits_val, seq_out_val = self.sess.run([self.logits, self.seq_outputs], feed_dict=self._make_feed_dict(batch_data, 1.0))
			pred_score = logits_val
			sequence_outs = seq_out_val

			if i == curr_idx:
				index = print_idx - 250 * curr_idx
				print(i*250 + index)
				print("curr_index", index)
				input_ids, input_mask, segment_ids, (dialog_lengths, response_lengths), label_ids, dialog_position_ids, (raw_dialogs, raw_responses) = dialog_data
				print(dialog_lengths[index], response_lengths[index])

				dialog_toks = self.tokenizer.convert_ids_to_tokens(input_ids[index][0:dialog_lengths[index]])
				response_toks = self.tokenizer.convert_ids_to_tokens(input_ids[index][280: 280 + response_lengths[index]])

				raw_dialog = raw_dialogs[index].split(" ")
				raw_response = raw_responses[index].split(" ")

				print(raw_dialog)
				print(raw_response)

				with open("./analysis/attention_score_%d.pickle" % (i*250 + index), "wb") as fw_handle:
					print(sequence_outs[index].shape)
					pickle.dump([dialog_toks, response_toks, raw_dialog, raw_response, np.array(sequence_outs[index])], fw_handle)

			# pred_score : batch*candidates_num
			input_ids = dialog_data[0]

			rank_by_pred = calculate_candidates_ranking(pred_score, dialog_data[4], self.hparams.evaluate_candidates_num)
			num_correct, pos_index = logits_recall_at_k(rank_by_pred, k_list)
			print(np.array(logits_val).shape)
			# print(num_correct[0])

			total_mrr += logits_mean_reciprocal_rank(rank_by_pred)
			total_correct = np.add(total_correct, num_correct)
			index += 1
			# print(index)
			total_examples = index * rank_by_pred.shape[0]
			# print(float(total_mrr / total_examples))
			# print(str(i), "/", str(total_data_len))
			# exit()
			exit()

		avg_mrr = float(total_mrr / (len(self.test_examples) / 10))
		recall_result = ""
		print(self.hparams.pickle_dir % "test")

		for i in range(len(k_list)):
			recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % ((total_correct[i] / (len(self.test_examples) / 10)) * 100)

		with open("/home/taesun/taesun_projects/ResSel/analysis/dialogue_turn.txt", "a") as fr_handle:
			fr_handle.write(self.hparams.pickle_dir % "test" + "\n" + recall_result + "\n" + "MRR: %.4f" % avg_mrr + "\n")

		print(recall_result)
		print("MRR: %.4f" % avg_mrr)

		return k_list, (total_correct / total_examples) * 100, avg_mrr


	def quantity_analysis_run_evaluate(self):

		examples = ["have you tried the regular way to start with do you run ubuntu with gnome",
								"probably ca n't do that on the xbox can you get a command prompt on it at all try rhythmbox",
								" afroken check out xrandr as the control mechanism for video ..",
								"certainly sudo apt-get install zsh zsh-doc zsh-lovers"]

		self._logger.info("Evaluation batch iteration per epoch is %d" % len(examples))
		if self.train_setup_vars["do_evaluate"]:
			print("Evaluation batch iteration per epoch is %d" % len(examples))
		dialog_examples = []
		for (i, dialog_data) in enumerate(examples):
			text_a = tokenization.convert_to_unicode(dialog_data)
			dialog_examples.append(InputExample(guid=i, text_a=text_a))
		feed_dict = {}
		[input_ids, input_mask, segment_ids, (text_a_lengths, text_b_lengths), label_ids, position_ids] = \
			self.processor.get_analysis_bert_data(dialog_examples)

		feed_dict[self.input_ids_ph] = input_ids
		feed_dict[self.input_mask_ph] = input_mask
		feed_dict[self.segment_ids_ph] = segment_ids
		feed_dict[self.dropout_keep_prob_ph] = 1.0

		feed_dict[self.dialog_position_ids_ph] = position_ids
		feed_dict[self.dialog_len_ph] = text_a_lengths
		feed_dict[self.response_len_ph] = text_b_lengths
		feed_dict[self.label_ids_ph] = label_ids

		sequence_out_val = self.sess.run([self.seq_outputs], feed_dict=feed_dict) # batch, 768 * 4
		print(np.array(sequence_out_val).shape)
		seq_merged_embeddings = merge_subtokens(examples, self.tokenizer, sequence_out_val[0])

		# sents_argmax = []
		# for merged_emb in seq_merged_embeddings:
		# 	sent_vecs = []
		# 	for tok, vec in merged_emb:
		# 		sent_vecs.append(vec)
		# 	sent_vecs = np.array(sent_vecs)
		# 	print(sent_vecs.shape)
		# 	sent_argmax = np.argmax(sent_vecs, axis=0)
		# 	sents_argmax.append(sent_argmax)
		#
		#
		# from collections import Counter
		#
		# for i in range(len(sents_argmax)):
		# 	print(examples[i].split())
		# 	counter_dict = Counter(sents_argmax[i])
		# 	for counter_key in counter_dict.keys():
		# 		counter_dict[counter_key]= round(counter_dict[counter_key] / len(sents_argmax[i]) , 3)
		#
		# # softmax_sum = sum([np.exp(counter_dict[counter_key]) for counter_key in counter_dict.keys()])
		# # for counter_key in counter_dict.keys():
		# # 	counter_dict[counter_key] = round(np.exp(counter_dict[counter_key]) / softmax_sum, 3)
		#
		# 	val_list = []
		# 	for key, val in sorted(counter_dict.items(), reverse=False):
		# 		val_list.append(val)
		# 	print(val_list)

	def analysis_evaluate(self, saved_file:str):
		# saved_file example
		config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False,
		)
		config.gpu_options.allow_growth = True
		self.train_setup_vars["do_evaluate"] = True
		self.train_setup_vars["is_train_continue"] = True

		tf.reset_default_graph()
		self.sess = tf.Session(config=config)
		self._make_placeholders()
		# self._build_train_graph()

		bert_model = modeling_base.BertModel(
			config=self.bert_config,
			is_training=False,
			input_ids=self.input_ids_ph,
			input_mask=self.input_mask_ph,
			token_type_ids=self.segment_ids_ph,
			dialog_position_ids=None,
			use_adapter_layer=False,
			scope='bert',
			hparams=self.hparams
		)
		input_phs = (self.input_ids_ph, self.input_mask_ph)
		length_phs = (self.dialog_len_ph, self.response_len_ph, self.knowledge_len_ph)
		label_ids_phs = (self.label_ids_ph, self.knowledge_label_ids_ph)

		created_model = self.hparams.graph.Model(self.hparams, self.bert_config, bert_model, input_phs,
																						 label_ids_phs, length_phs, None, None)
		self.logits, self.seq_outputs = created_model.build_graph()

		self.sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		saver.restore(self.sess, saved_file)

		self._logger.info("Evaluation Step - Test")
		self.analysis_run_evaluate()

	def evaluate(self, saved_file: str, eval_model_num=1):
		# saved_file example
		config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False,
		)
		config.gpu_options.allow_growth = True
		self.train_setup_vars["do_evaluate"] = True
		self.train_setup_vars["is_train_continue"] = True

		tf.reset_default_graph()
		self.sess = tf.Session(config=config)
		self._make_placeholders()
		self._build_train_graph()
		self.sess.run(tf.global_variables_initializer())

		self._logger.info("Evaluation Step - Test")
		self._run_evaluate("test")

		eval_out_path = os.path.join(self.hparams.root_dir + "%s/" % (saved_file.strip().split("/")[-2]))
		if not os.path.exists(eval_out_path):
			os.makedirs(eval_out_path)

		self.eval_out_fw_handle = open(os.path.join(eval_out_path + "eval_result.txt"), "w", encoding="utf-8")

		for i in range(eval_model_num):
			saver = tf.train.Saver()
			saved_file_global_step = int(saved_file.strip().split("-")[-1])
			print("Global_step : ", saved_file_global_step)
			saver.restore(self.sess, saved_file)

			root_dir = os.path.join(self.hparams.root_dir, "%s/" % (saved_file.strip().split("/")[-2] + "-" +
																															str(saved_file_global_step)))
			self.tensorboard_summary = TensorBoardSummaryWriter(root_dir, self.sess, self.sess.graph)

			print("Evaluate Session %s has been restored!" % saved_file)
			# self._logger.info("Evaluation Step - Valid")
			# k_list, recall_res, avg_mrr = self._run_evaluate("valid")
			# for i, k in senumerate(k_list):
			#   self.tensorboard_summary.add_summary(
			#     "recall_valid/recall_%s" % k, float(recall_res[i]), saved_file_global_step)
			#   self.tensorboard_summary.add_summary(
			#     "mrr_valid/mean_reciprocal_error", float(avg_mrr), saved_file_global_step)

			self._logger.info("Evaluation Step - Test")
			self._run_evaluate("test")

			saved_filed_split = saved_file.strip().split("-")
			saved_filed_split[-1] = str(saved_file_global_step + self.hparams.next_saved_step)
			saved_file = "-".join(saved_filed_split).strip()
			print(saved_file)
			self.tensorboard_summary.close()
			tf.reset_default_graph()

		self.eval_out_fw_handle.close()