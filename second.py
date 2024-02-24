#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division
import os, copy, sys
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "plt_cache")

from utils.eval_funcs import *
from utils.dataset_CIFAR import *
from utils.network_arch_resnet import *
from utils.trainval import *
from utils.plot_funcs import *
from utils.regularizers import *
from utils.datasets import *
from utils.models import *

import warnings  # ignore warnings

warnings.filterwarnings("ignore")


#####
# Input's parser
#####
import argparse

parser = argparse.ArgumentParser(description="Training a long tail recognition model.")
parser.add_argument("--project_name", type=str, default="second", help="Training stage")
parser.add_argument("--data_imb_type", type=str, default="exp")
parser.add_argument("--data_imb_factor", type=float, default=0.01)
parser.add_argument("--data_n_classes", type=int, default=100)
parser.add_argument("--data_batch_size", type=int, default=64)
parser.add_argument("--data_name", type=str, default="Cifar100", help="Dataset's name")

parser.add_argument("--model_is_pretrained", type=bool, default=True)
parser.add_argument("--model_load_only_representation", type=bool, default=False)
parser.add_argument(
    "--model_pretrained_path", type=str, default="./exp/Cifar10/ResNet34/first/wd_5e-3"
)
parser.add_argument("--model_name", type=str, default="ResNet34")
parser.add_argument("--model_use_etf", type=bool, default=False)
parser.add_argument("--model_train_etf_norm", type=bool, default=True)
parser.add_argument("--model_normalize_feature", type=bool, default=False)
parser.add_argument("--model_logit_adjustment_tau", type=float, default=0.0)
parser.add_argument("--model_logit_normalization", type=bool, default=False)

parser.add_argument("--optim_weight_decay", type=float, default=0)  # 1e-1
parser.add_argument("--optim_base_lr", type=float, default=0.00)  # 0.01
parser.add_argument("--optim_total_epoch_num", type=int, default=10)
parser.add_argument("--optim_loss_type", type=str, default="CE")
parser.add_argument("--optim_cb_type", type=str, default="softmax")
parser.add_argument("--optim_cb_beta", type=float, default=0.9999)
parser.add_argument("--optim_cb_gamma", type=float, default=2.0)
parser.add_argument("--optim_norm_type", type=str, default="L2")
parser.add_argument("--optim_norm_param", type=float, default=1 / 15.5)


parser.add_argument("--exp_name", type=str, default="WB")
parser.add_argument("--seeds", type=int, default=0)
parser.add_argument("--description", type=str, default="")
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--json_path", type=str, default="")
p_args = parser.parse_args()
do_retrain = p_args.retrain


args: BaseArgs = BaseArgs.load(p_args.json_path) if p_args.json_path else BaseArgs()
args.logger.info(sys.version)
args.logger.info(torch.__version__)


# ## Setup config parameters
#
# There are several things to setup, like which GPU to use, model name, hyper-parameters, etc. Please read the comments. By default, you should be able to run this script smoothly without changing anything.

#############
#  Set args
#############
args.seeds = p_args.seeds
set_seeds(args.seeds)
args = set_device(args)
set_cuda()

if p_args.json_path:
    pass
else:
    args.project_name = p_args.project_name
    args.data.imb_type = (
        p_args.data_imb_type
    )  # samling long-tailed training set with an exponetially-decaying function
    args.data.imb_factor = p_args.data_imb_factor  # imbalance factor = 100 = 1/0.01
    args.data.n_classes = (
        p_args.data_n_classes
    )  # number of classes in CIFAR100-LT with imbalance factor 100
    args.data.batch_size = p_args.data_batch_size  # batch size
    args.data.name = p_args.data_name

    args.model.is_pretrained = p_args.model_is_pretrained
    args.model.load_only_representation = p_args.model_load_only_representation
    args.model.pretrained_path = p_args.model_pretrained_path
    args.model.name = p_args.model_name
    args.model.use_etf = p_args.model_use_etf
    args.model.train_etf_norm = p_args.model_train_etf_norm
    args.model.normalize_feature = p_args.model_normalize_feature
    args.model.logit_adjustment_tau = p_args.model_logit_adjustment_tau
    args.model.logit_normalization = p_args.model_logit_normalization

    args.optim.weight_decay = p_args.optim_weight_decay  # set weight decay to 5e-3
    args.optim.weight_decay_target = [False, False, False]
    args.optim.base_lr = p_args.optim_base_lr
    args.optim.total_epoch_num = p_args.optim_total_epoch_num  # 320
    args.optim.loss_type = p_args.optim_loss_type
    args.optim.cb_type = p_args.optim_cb_type
    args.optim.cb_beta = p_args.optim_cb_beta
    args.optim.cb_gamma = p_args.optim_cb_gamma
    args.optim.norm_type = p_args.optim_norm_type
    args.optim.norm_param = p_args.optim_norm_param

    args.exp_name = p_args.exp_name
    args.description = p_args.description

    args.exp_name = p_args.exp_name
    args.description = p_args.description

if (
    do_retrain
    and args.optim.total_epoch_num <= 0
    or not do_retrain
    and args.optim.total_epoch_num > 0
):
    assert False, "Check do_retrain!"

args.logger.info(args.exp_name)
work_dir = get_work_dir_path(args)
args.save(work_dir)
log_result_path(args)
args, dataloaders_dict, new_label_list, label_names_list, img_num_per_cls = set_dataset(
    args
)


assert args.model.is_pretrained, "Model is not pretrained"
model, pgd_func, args = set_model(
    args, do_retrain, train_img_num_per_class=img_num_per_cls
)
models_dict = {}
models_dict[args.exp_name] = model

if do_retrain:
    track_records = do_train_cycle(
        args, model, dataloaders_dict, img_num_per_cls, new_label_list, pgd_func
    )
    check_model_statistic(
        args,
        models_dict,
        dataloaders_dict,
        new_label_list,
        label_names_list,
        img_num_per_cls,
        track_records,
    )
else:
    check_model_statistic(
        args,
        models_dict,
        dataloaders_dict,
        new_label_list,
        label_names_list,
        img_num_per_cls,
        reload_model=False,
    )
