import os
import argparse
import collections
import logging
from datetime import datetime
import json

from model import model_params
from train_model import Model

PARAMS_MAP = {
    "ubuntu_base": model_params.BASE_PARAMS,
    "ubuntu_post" : model_params.MODEL_POST_BASE,
    "ubuntu_post_ft" : model_params.MODEL_POST_FT,
    "ubuntu_post_da" : model_params.MODEL_POST_DA,
    "ubuntu_domain" : model_params.MODEL_DOMAIN,
    "ubuntu_domain_adapter" : model_params.MODEL_DOMAIN_ADAPTER,
    "ubuntu_negative_base" : model_params.MODEL_NEGATIVE_BASE,

    "eval_base": model_params.EVAL_PARAMS,
    "eval_post" : model_params.EVAL_POST,
    "eval_post_ft" : model_params.EVAL_POST_FT,
    "eval_post_da" : model_params.EVAL_POST_DA,
    "eval_domain" : model_params.EVAL_DOMAIN,
    "eval_domain_adapter" : model_params.EVAL_DOMAIN_ADAPTER,
    "eval_negative_base" : model_params.EVAL_NEGATIVE_PARAMS,

    "advising_base" : model_params.MODEL_ADVISING_BASE,
    "advising_post" : model_params.MODEL_ADVISING_POST,
    "advising_post_ft" : model_params.MODEL_ADVISING_POST_FT,
    "advising_post_da" : model_params.MODEL_ADVISING_POST_DA,
    "advising_post_adapter" : model_params.MODEL_ADVISING_POST_ADAPTER,
    "advising_domain" : model_params.MODEL_ADVISING_DOMAIN,
    "advising_domain_adapter" :model_params.MODEL_ADVISING_DOMAIN_ADAPTER,

    "eval_advising_base" : model_params.EVAL_ADVISING_BASE_PARAMS,
    "eval_advising_post" : model_params.EVAL_ADVISING_POST,
    "eval_advising_post_ft" : model_params.EVAL_ADVISING_POST_FT,
    "eval_advising_post_da" : model_params.EVAL_ADVISING_POST_DA,
    "eval_advising_post_adapter" : model_params.EVAL_ADVISING_POST_ADAPTER,
    "eval_advising_domain" : model_params.EVAL_ADVISING_DOMAIN,
    "eval_advising_domain_adapter" : model_params.EVAL_ADVISING_DOMAIN_ADAPTER,
}

def init_logger(path:str):
  if not os.path.exists(path):
      os.makedirs(path)
  logger = logging.getLogger()
  logger.handlers = []
  logger.setLevel(logging.DEBUG)
  debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
  debug_fh.setLevel(logging.DEBUG)

  info_fh = logging.FileHandler(os.path.join(path, "info.log"))
  info_fh.setLevel(logging.INFO)

  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
  debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

  ch.setFormatter(info_formatter)
  info_fh.setFormatter(info_formatter)
  debug_fh.setFormatter(debug_formatter)

  logger.addHandler(ch)
  logger.addHandler(debug_fh)
  logger.addHandler(info_fh)

  return logger

def train_model(args):
  hparams = PARAMS_MAP[args.model]
  timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  root_dir = os.path.join(hparams["root_dir"], "%s/" % timestamp)
  logger = init_logger(root_dir)
  logger.info("Hyper-parameters: %s" % str(hparams))
  hparams["root_dir"] = root_dir

  hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
  model = Model(hparams)
  model.train(pretrained_file=args.pretrained)

def evaluate_model(args):
  hparams = PARAMS_MAP[args.model]
  hparams["pickle_dir"] = args.eval_pickle
  hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
  model = Model(hparams)
  print(hparams.init_checkpoint)
  model.analysis_evaluate(saved_file=args.evaluate)

if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description="Bert / Run Classifier (Tensorflow)")
  arg_parser.add_argument("--model", dest="model", type=str, default=None,
                          help="Model Name")
  arg_parser.add_argument("--evaluate", dest="evaluate", type=str, default=None,
                          help="Path to the saved model.")
  arg_parser.add_argument("--eval_pickle", dest="eval_pickle", type=str, default=None,
                          help="Path to the pickle")
  arg_parser.add_argument("--pretrained", dest="pretrained", type=str, default=None,
                          help="Path to the saved model.")
  args = arg_parser.parse_args()

  if args.evaluate:
    evaluate_model(args)
  else:
    train_model(args)