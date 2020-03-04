from model.graphs import half_fine_tuning, knowledge_top_n, knowledge_lstm, similar_dialog_encoder, similar_dialog_encoder_v2, \
	similar_dialog_transformer
from collections import defaultdict
import math

BASE_PARAMS = defaultdict(
	# lambda: None,  # Set default value to None.

	# GPU params
	gpu_num = [0,1],

	# Input params
	train_batch_size=32,
	eval_batch_size=500,
	max_seq_length=320,
	dialog_max_seq_length=280,
	response_max_seq_length=40,
	knowledge_max_seq_length=80,

	# Training params
	learning_rate=0.00003,
	loss_type="sigmoid",
	less_data_rate=1.0,
	training_shuffle_num=50,
	dropout_keep_prob=0.75,
	num_epochs=3,
	tensorboard_step=100,
	embedding_dim=768,
	rnn_hidden_dim=768,
	rnn_depth=1,

	# Training setup params
	do_lower_case=True,
	do_train_bert=True,
	is_train_continue=False,
	on_training=False,
	do_train_knowledge=False,
	do_similar_dialog=False,
	do_evaluate=False,
	do_transformer_residual=False,
	do_adam_weight_optimizer=False,
	use_one_hot_embeddings=False,
	use_domain_embeddings=False,
	use_word2vec_embeddings=False,

	# Train Model Config
	task_name="ubuntu",
	do_dialog_state_embedding=False,
	do_adapter_layer=False,
	graph=half_fine_tuning,

	# Need to change to train...(e.g.data dir, config dir, vocab dir, etc.)
	init_checkpoint="/mnt/raid5/shared/bert/tensorflow/uncased_L-12_H-768_A-12/bert_model.ckpt",
	# init_checkpoint="/mnt/raid5/taesun/data/ResSel/ubuntu_v1/bert_ubuntu_pretrain_320_mlm_eot_model/model.ckpt-200000",
	bert_config_dir="/mnt/raid5/shared/bert/tensorflow/uncased_L-12_H-768_A-12/bert_config.json",
	vocab_dir="/mnt/raid5/shared/bert/tensorflow/uncased_L-12_H-768_A-12/vocab.txt",
	root_dir="/mnt/raid5/taesun/thesis/bert_rs/ubuntu/bert_post_mlm_nsp_eot_base/",
	data_dir="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_multi_turn_negative_1/",
	pickle_dir="bert_%s_sub_eot.pickle",

	# Others
	recall_k_list=[1, 2, 5, 10],
	train_transformer_layer=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
	top_n=5,
	evaluate_step=15625,
	save_step=int(15625),
	evaluate_print_step=250,

	evaluate_candidates_num=10,
	next_saved_step=6250,
)

MODEL_POST_BASE = BASE_PARAMS.copy()
MODEL_POST_BASE.update(
	gpu_num=[0],
	train_batch_size=16,
	eval_batch_size=250,
	evaluate_candidates_num=10,
	next_saved_step=31250,
	learning_rate=1e-05,
	data_dir="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_dialog_turn_num/",
	# init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot/model.ckpt-130000",
	# init_checkpoint="/mnt/raid5/taesun/data/ResSel/ubuntu_v1/bert_ubuntu_pretrain_320_mlm_nsp_eot_model/model.ckpt-200000",
	init_checkpoint="/mnt/raid5/shared/bert/tensorflow/uncased_L-12_H-768_A-12/bert_model.ckpt",
	root_dir="/mnt/raid5/taesun/thesis/bert_rs/ubuntu/bert_post_mlm_nsp_eot_base/",
  pickle_dir = "bert_%s_eot_3.pickle",
	do_train_bert = False,
	evaluate_step = 31250,
	save_step=int(31250),
	evaluate_print_step=1,
	train_transformer_layer=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
)

MODEL_POST_FT = BASE_PARAMS.copy()
MODEL_POST_FT.update(
	init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot_domain_post/model.ckpt-200000",
	root_dir="/mnt/raid5/taesun/bert/ubuntu/conference/bert_post_ft/",
	train_transformer_layer=[12],
	do_train_bert=False,
)

MODEL_POST_DA = BASE_PARAMS.copy()
MODEL_POST_DA.update(
	data_dir="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_multi_turn_negative_4/",
	init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot_domain_post/model.ckpt-200000",
	root_dir="/mnt/raid5/taesun/bert/ubuntu/conference/bert_post_ft/",
	train_transformer_layer=[6, 7, 8, 9, 10, 11],
	do_train_bert=False,
	save_step=int(2500000/128),
	num_epochs=3,
)

MODEL_DOMAIN = BASE_PARAMS.copy()
MODEL_DOMAIN.update(
	original_init_checkpoint="/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/bert_model.ckpt",
	init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot_domain_post/model.ckpt-200000",
	root_dir = "/mnt/raid5/taesun/bert/ubuntu/conference/bert_domain_weighted_sum/",
	use_domain_embeddings=True,
	do_adapter_layer=False,
	# train_transformer_layer=[6, 7, 8, 9, 10, 11],
	# do_train_bert=False,
)

MODEL_DOMAIN_ADAPTER = BASE_PARAMS.copy()
MODEL_DOMAIN_ADAPTER.update(
	init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot_domain/model.ckpt-200000",
	root_dir="/mnt/raid5/taesun/bert/ubuntu/conference/bert_domain_adapter/",
	use_domain_embeddings=True,
	do_adapter_layer=True,
	do_train_bert=False,
	train_transformer_layer=[],
	num_epochs=20,
)

"""
ADVISING
"""

MODEL_ADVISING_BASE = BASE_PARAMS.copy()
MODEL_ADVISING_BASE.update(
	gpu_num=[0],
	data_dir="/mnt/raid5/taesun/data/Advising/",
	recall_k_list=[1, 2, 5, 10, 50, 100],
	init_checkpoint="/mnt/raid5/shared/bert/tensorflow/uncased_L-12_H-768_A-12/bert_model.ckpt",
	root_dir="/mnt/raid5/taesun/thesis/bert_rs/advising/bert_base",
	pickle_dir="bert_%s_sub_eot.pickle",

	evaluate_candidates_num=100,
	next_saved_step=6250,
	num_epochs=3,
	learning_rate=0.00003,
	evaluate_step=6250,
	save_step=int(6250),
	do_train_bert=True,
	train_batch_size=16,
	eval_batch_size=200,
	evaluate_print_step=250
)

MODEL_ADVISING_POST = MODEL_ADVISING_BASE.copy()
MODEL_ADVISING_POST.update(
	gpu_num=[0],
	root_dir="/mnt/raid5/taesun/thesis/bert_rs/advising/post_mlm_nsp_eot",
	init_checkpoint="/mnt/raid5/taesun/data/ResSel/advising/bert_advising_pretrain_320_mlm_nsp_eot_model/model.ckpt-100000",
	# init_checkpoint="/mnt/raid5/taesun/data/Advising/advising_pretrain_output_320_eot/model.ckpt-100000",
	data_dir="/mnt/raid5/taesun/data/Advising/",
	pickle_dir="bert_%s_eot.pickle",
	num_epochs=1,
	learning_rate=1e-05,
	evaluate_step=6250,
	save_step=int(6250),
	do_train_bert=False,
	train_batch_size=16,
	eval_batch_size=200,
	evaluate_print_step=250,
	train_transformer_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
)

MODEL_ADVISING_POST_FT = MODEL_ADVISING_BASE.copy()
MODEL_ADVISING_POST_FT.update(
	root_dir="/mnt/raid5/taesun/bert/advising/conference/bert_post_vft/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	train_transformer_layer=[6, 7, 8, 9, 10, 11],
	do_train_bert=False,
	num_epochs=3,
	learning_rate=1e-05,
	evaluate_step=3125,
	save_step=int(6250),
	train_batch_size=32,
	eval_batch_size=400,
	evaluate_print_step=250
)

MODEL_ADVISING_POST_DA = MODEL_ADVISING_BASE.copy()
MODEL_ADVISING_POST_DA.update(
	data_dir="/mnt/raid5/taesun/data/Advising/data_augmentation",
	root_dir="/mnt/raid5/taesun/bert/advising/conference/bert_post_da/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	train_transformer_layer=[6, 7, 8, 9, 10, 11],
	do_train_bert=False,
	save_step=6250,
	num_epochs=3,
)

MODEL_ADVISING_POST_ADAPTER = MODEL_ADVISING_BASE.copy()
MODEL_ADVISING_POST_ADAPTER.update(
	root_dir="/mnt/raid5/taesun/bert/advising/conference/bert_post_adapter/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	use_domain_embeddings=False,
	do_adapter_layer=True,
)

MODEL_ADVISING_DOMAIN = MODEL_ADVISING_BASE.copy()
MODEL_ADVISING_DOMAIN.update(
	root_dir="/mnt/raid5/taesun/bert/advising/conference/bert_domain_weighted_sum/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	original_init_checkpoint="/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/bert_model.ckpt",
	use_domain_embeddings=True,
	do_adapter_layer=False,
	# train_transformer_layer=[4, 5, 6, 7, 8, 9, 10, 11],
	# do_train_bert=False,
	num_epochs=20,
)

MODEL_ADVISING_DOMAIN_ADAPTER = MODEL_ADVISING_BASE.copy()
MODEL_ADVISING_DOMAIN_ADAPTER.update(
	root_dir="/mnt/raid5/taesun/bert/advising/conference/bert_domain_post_adapter/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	original_init_checkpoint="/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/bert_model.ckpt",
	use_domain_embeddings=True,
	do_adapter_layer=True,
	train_transformer_layer=[4, 5, 6, 7, 8, 9, 10, 11],
	do_train_bert=False,
	num_epochs=20,
)

MODEL_NEGATIVE_BASE = BASE_PARAMS.copy()
MODEL_NEGATIVE_BASE.update(
	data_dir="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/multi_turn_negative_4",
	root_dir="/mnt/raid5/taesun/bert/ubuntu/bert_negative_base/runs/",
	save_step=int(2500000 / 64),
	training_shuffle_num=50
)

"""eval_params"""
EVAL_PARAMS = BASE_PARAMS.copy()
EVAL_PARAMS.update(
	gpu_num = [0,1],
	eval_batch_size=500,
	training_shuffle_num=0,
	do_evaluate=True,
	root_dir="/mnt/raid5/taesun/bert/ubuntu/bert_evaluate_tensorboard/bert_base",
	do_adam_weight_optimizer=False,
	dropout_keep_prob=1.0,
	evaluate_candidates_num=10,
	next_saved_step=31250,
	evaluate_print_step=100
)

EVAL_POST = EVAL_PARAMS.copy()
EVAL_POST.update(
	gpu_num = [0],
	eval_batch_size=100,
	root_dir="/mnt/raid5/taesun/bert/ubuntu/bert_evaluate_tensorboard/bert_post",
	init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot_domain_post/model.ckpt-200000",
	use_domain_embeddings=False,
	do_adapter_layer=False,
	next_saved_step=31250,
	evaluate_print_step=500,
)

EVAL_POST_FT = EVAL_PARAMS.copy()
EVAL_POST_FT.update(
	gpu_num = [0],
	eval_batch_size=50,
	root_dir="/mnt/raid5/taesun/bert/ubuntu/bert_evaluate_tensorboard/bert_post_ft",
	init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot_domain_post/model.ckpt-200000",
	use_domain_embeddings=False,
	do_adapter_layer=False,
	next_saved_step=15625,
	evaluate_print_step=2000,
	train_transformer_layer=[12],
	# train_transformer_layer=[2,3,4,5,6,7,8,9,10,11],
	# server 1 : 10, 11
	do_train_bert=False,
)

EVAL_POST_DA = EVAL_PARAMS.copy()
EVAL_POST_DA.update(
	gpu_num = [0],
	eval_batch_size=100,
	root_dir="/mnt/raid5/taesun/bert/ubuntu/bert_evaluate_tensorboard/bert_post_da",
	init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot_domain_post/model.ckpt-200000",
	use_domain_embeddings=False,
	do_adapter_layer=False,
	next_saved_step=19531,
	evaluate_print_step=1000,
	train_transformer_layer=[6, 7, 8, 9, 10, 11],
	do_train_bert=False,
)

EVAL_DOMAIN = EVAL_PARAMS.copy()
EVAL_DOMAIN.update(
	gpu_num = [0],
	eval_batch_size=100,
	root_dir="/mnt/raid5/taesun/bert/ubuntu/bert_evaluate_tensorboard/bert_domain_weighted_sum",
	init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot_domain/model.ckpt-200000",
	evaluate_print_step=1000,
	next_saved_step = 15625,
	use_domain_embeddings=True,
	do_adapter_layer=False,
)


EVAL_DOMAIN_ADAPTER = EVAL_PARAMS.copy()
EVAL_DOMAIN_ADAPTER.update(
	gpu_num=[0],
	eval_batch_size=250,
	root_dir="/mnt/raid5/taesun/bert/ubuntu/bert_evaluate_tensorboard/bert_domain_adapter",
	init_checkpoint="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_pretrain/ubuntu_pretrain_output_320_eot_domain/model.ckpt-200000",
	evaluate_print_step=200,
	next_saved_step = 31250,
	use_domain_embeddings=True,
	do_adapter_layer=True,
)

EVAL_ADVISING_BASE_PARAMS = EVAL_PARAMS.copy()
EVAL_ADVISING_BASE_PARAMS.update(
	gpu_num = [0],
	eval_batch_size=100,
	root_dir="/mnt/raid5/taesun/bert/advising/bert_evaluate_tensorboard/bert_base/",
	data_dir="/mnt/raid5/taesun/data/Advising/",
	recall_k_list=[1, 2, 5, 10, 50, 100],
	use_domain_embeddings=False,
	do_adapter_layer=False,
	next_saved_step=3125,
	evaluate_candidates_num=100,
	evaluate_print_step=100
)

EVAL_ADVISING_POST = EVAL_ADVISING_BASE_PARAMS.copy()
EVAL_ADVISING_POST.update(
	root_dir="/mnt/raid5/taesun/bert/advising/bert_evaluate_tensorboard/bert_post/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	#/mnt/raid5/taesun/bert/advising/conference/bert_post/20190509-105211/model.ckpt-18750
)

EVAL_ADVISING_POST_FT = EVAL_ADVISING_BASE_PARAMS.copy()
EVAL_ADVISING_POST_FT.update(
	root_dir="/mnt/raid5/taesun/bert/advising/bert_evaluate_tensorboard/bert_post_ft/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	#/mnt/raid5/taesun/bert/advising/conference/bert_post/20190509-105211/model.ckpt-18750
	train_transformer_layer=[12],
  do_train_bert = False,
)

EVAL_ADVISING_POST_DA = EVAL_ADVISING_BASE_PARAMS.copy()
EVAL_ADVISING_POST_DA.update(
	data_dir="/mnt/raid5/taesun/data/Advising/data_augmentation",
	root_dir="/mnt/raid5/taesun/bert/advising/bert_evaluate_tensorboard/bert_post_da/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	train_transformer_layer = [6, 7, 8, 9, 10, 11],
  do_train_bert = False,
	next_saved_step=6250,
	evaluate_candidates_num=100,
	evaluate_print_step=100
)


EVAL_ADVISING_POST_ADAPTER = EVAL_ADVISING_BASE_PARAMS.copy()
EVAL_ADVISING_POST_ADAPTER.update(
	root_dir="/mnt/raid5/taesun/bert/advising/bert_evaluate_tensorboard/bert_post_adapter/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	do_adapter_layer=True,
	use_domain_embeddings=False,
)

EVAL_ADVISING_DOMAIN = EVAL_ADVISING_BASE_PARAMS.copy()
EVAL_ADVISING_DOMAIN.update(
	gpu_num=[0],
	eval_batch_size=100,
	root_dir="/mnt/raid5/taesun/bert/advising/bert_evaluate_tensorboard/bert_domain_concat/",
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot/model.ckpt-100000",
	# /mnt/raid5/taesun/bert/advising/conference/bert_domain/20190512-104646/model.ckpt-3125
	use_domain_embeddings=True,
	do_adapter_layer=False,
	next_saved_step=6250,
	evaluate_candidates_num=100,
	evaluate_print_step=100,
	# train_transformer_layer=[4, 5, 6, 7, 8, 9, 10, 11],
	# do_train_bert=False,
)

EVAL_ADVISING_DOMAIN_ADAPTER = EVAL_ADVISING_BASE_PARAMS.copy()
EVAL_ADVISING_DOMAIN_ADAPTER.update(
	gpu_num=[0],
	eval_batch_size=100,
	init_checkpoint="/mnt/raid5/taesun/data/Advising/bert_pretrain/advising_pretrain_output_320_eot_domain_adapter/model.ckpt-100000",
	root_dir="/mnt/raid5/taesun/bert/advising/bert_evaluate_tensorboard/bert_domain_adapter/",
	original_init_checkpoint="/mnt/raid5/shared/bert/uncased_L-12_H-768_A-12/bert_model.ckpt",
	next_saved_step=3125,
	use_domain_embeddings=True,
	do_adapter_layer=True,
	train_transformer_layer=[6, 7, 8, 9, 10, 11],
	do_train_bert=False,
	evaluate_candidates_num=100,
	evaluate_print_step=100
)

EVAL_NEGATIVE_PARAMS = EVAL_PARAMS.copy()
EVAL_NEGATIVE_PARAMS.update(
	data_dir = "/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/multi_turn_negative_4"
)

EVAL_TRAIN_TEST = EVAL_PARAMS.copy()
EVAL_TRAIN_TEST.update(
	do_transformer_residual=True
)

EVAL_DIALOG_PARAMS = EVAL_PARAMS.copy()
EVAL_DIALOG_PARAMS.update(
	do_dialog_state_embedding=True,
	train_transformer_layer=[6, 7, 8, 9, 10, 11],
	graph=half_fine_tuning
)

EVAL_ADAPTER_PARAMS = EVAL_PARAMS.copy()
EVAL_ADAPTER_PARAMS.update(
	do_adapter_layer=True,
	train_transformer_layer=[],
)

EVAL_KNOWLEDGE_LSTM = EVAL_PARAMS.copy()
EVAL_KNOWLEDGE_LSTM.update(
	data_dir="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/bert_knowledge_top_5/",
	do_train_knowledge=True,
	graph=knowledge_lstm
)

EVAL_SIMILAR_DIALOG =EVAL_PARAMS.copy()
EVAL_SIMILAR_DIALOG.update(
	eval_batch_size=200,
	data_dir="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_top_3/",
	graph=similar_dialog_encoder,
	do_similar_dialog=True,
	rnn_hidden_dim=384,
	top_n=3,
)

EVAL_SIMILAR_DIALOG_V2 =EVAL_PARAMS.copy()
EVAL_SIMILAR_DIALOG_V2.update(
	eval_batch_size=200,
	data_dir="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_top_3/",
	graph=similar_dialog_encoder_v2,
	do_similar_dialog=True,
	rnn_hidden_dim=384,
	top_n=3,
)

EVAL_SIMILAR_TRANSFORMER =EVAL_PARAMS.copy()
EVAL_SIMILAR_TRANSFORMER.update(
	gpu_num=[0,1],
	eval_batch_size=200,
	data_dir="/mnt/raid5/taesun/data/ubuntu_corpus_v1/ubuntu_data/similar_dialog_top_3/",
	graph=similar_dialog_transformer,
	do_similar_dialog=True,
	rnn_hidden_dim=384,
	top_n=3,
)