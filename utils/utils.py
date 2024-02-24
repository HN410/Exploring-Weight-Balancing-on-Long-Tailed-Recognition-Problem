from datetime import datetime as dt
from types import SimpleNamespace
import os, sys, glob, json, math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
TRAIN_LOG_FILE_NAME = 'train.log'
BEST_MODEL_LOG_FILE_NAME = 'note_bestModel.log'
BEST_PARAM_FILE_NAME = 'best.paramOnly'
TRACK_RECORDS_FILE_NAME = "trackRecords.ckpt"
CHECK_POINT_FILE_NAME = "checkpoint.ckpt"
CONFMAT_FILE_NAME = "confMat.npy"
FEATURESMAT_FILE_NAME = "featuresMat.npy"
ACCURACY_FILE_NAME = "acc.json"
ARGS_FILE_NAME = "args.json"
DISCRIMINANT_RATIO_FILE_NAME = "fdr.json"
SPARSITY_FILE_NAME = "sparsity.json"
RANDOM_FDR_FILE_NAME = "random_fdr.json"
RANDOM_FDR_FILE_FORMAT = "random_fdr_{}.json"

EXP_OUTPUT_DIR = "exp_theory"

#########################
## args
#########################
from typing import ClassVar
import logging, copy
class Args():
    def __init__(self, vars_dict=None):
        super().__init__()
        if vars_dict:
            def _inner(vars_dict: dict):
                vars_dict = {k : Args(v) if isinstance(v, dict) else v for k, v in vars_dict.items()} 
                return vars_dict
            self.__dict__.update(**_inner(vars_dict))

    def __repr__(self):
        items_tuple = (f"{k}={self.__dict__[k]!r}" for k in self.__dict__)
        return "{}({})".format(type(self).__name__, ", ".join(items_tuple))
    
    @classmethod
    def __inner_to_vars(cls, args):
        vars_dict = vars(args)
        vars_dict = {k : cls.__inner_to_vars(vars_dict[k]) if isinstance(vars_dict[k], Args) else vars_dict[k] for k in vars_dict} 
        return vars_dict
    def to_vars(self):
        return self.__inner_to_vars(self)
    

class RootArgs(Args):
    TMP_ARGS_DICT: set = ("logger", "device", "TMP_ARGS_DICT")

    def __init__(self, vars_dict= None, add_logger = True):
        self.device: str
        self.logger: logging.Logger
        
        if vars_dict:
            super().__init__(vars_dict)
        self.setup_logger(add_logger)

            
    def setup_logger(self, add_logger):
        if add_logger:
            self.logger = get_logger()
        else:
            self.logger = SimpleNamespace()
            self.logger.info = lambda t: print(t, flush=True)
            
    # omit_tmps ... if True, tmp arguments are omitted to save as json
    def to_vars(self, omit_tmps= True, unify_children = False):
        if not omit_tmps:
            return super().to_vars()
        vars_dict = copy.copy(vars(self))
        for key in list(vars_dict.keys()):
            if key in self.TMP_ARGS_DICT:
                vars_dict.pop(key)
        vars_dict = {k: v.to_vars() if isinstance(v, Args) else v for k, v in vars_dict.items()}
        
        if unify_children:
            child_dict = {}
            for key, v in list(vars_dict.items()):
                if not isinstance(v, dict):
                    child_dict[key] = v
                    vars_dict.pop(key)
            vars_dict["others"] = child_dict
        return vars_dict
    
    def save(self, save_dir, file_name="args"):
        vars_dict = self.to_vars()
        with open(os.path.join(save_dir, file_name +".json"), "w") as f:
            json.dump(vars_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, json_path, add_logger = True):
        with open(json_path, "r") as f:
            vars_dict = json.load(f)
        return cls(vars_dict, add_logger)


### Training args
class DataArgs(Args):
    def __init__(self, vars_dict=None):
        self.name:str  = "Cifar100"
        self.imb_type: str = "exp" # samling long-tailed training set with an exponetially-decaying function
        self.imb_factor: str = 0.01  # imbalance factor = 100 = 1/0.01
        self.n_classes: int = 100
        self.batch_size: int = 64
        if vars_dict:
            super().__init__(vars_dict)

DEFAULT_BN_EPS = 1e-5
class ModelArgs(Args):
    def __init__(self, vars_dict=None):
        self.name: str = "ResNet34"
        self.is_pretrained: bool = False
        self.load_last_linear: bool = True
        self.load_only_representation: bool = False
        self.use_etf: bool = False
        self.use_dr_balancer: bool = False
        self.train_etf_norm: bool
        self.normalize_feature: bool = False
        self.pretrained_path: str
        self.etf_norm_rate: float = 1 # Head's norm( = 1) / Tail's norm
        self.logit_adjustment_tau: float = 0.0
        self.logit_adjustment_for_training: bool = False
        self.logit_normalization: bool = False
        self.use_additive_logit_adjustment: bool = True
        self.activation_type: str = "relu"
        self.activation_param: float = 0.03
        self.use_feature_bias: bool = False
        self.bn_weight_init_val: float = 1.0
        self.bn_eps:float = DEFAULT_BN_EPS
        self.last_relu: bool = False
        self.one_param_bn:bool = False
        self.dropout:float = 0
        self.use_bn: bool = True
        self.add_skip: bool = False
        self.ignore_restrict_load: bool = False
        
        
        self.modify_bn_weight: bool = False
        self.bn_weight_init_distribution: str
        self.pretrained_path: str
        self.pretrained_gamma: float
        self.pretrained_beta: float
        self.init_identity_bn: bool = True
        self.bn_weight_init_mean: float
        self.bn_weight_init_std: float
        self.bn_weight_init_min: float
        self.bn_weight_init_max: float
        
        self.mlp_hlayer_n: int = 3
        self.mlp_features_d: int = 1024
        
        self.identity_skip: bool = False
        if vars_dict:
            super().__init__(vars_dict)
    
class OptimArgs(Args):
    def __init__(self, vars_dict=None):
        self.base_lr: float
        self.total_epoch_num: int = 0
        self.loss_type: str = "none"
        self.weight_decay: float = 0.0
        self.weight_decay_target: list # [Many, Medium, Few] list of bool stating which class should be target of weight decay
        self.weight_decay_conv: bool = True
        self.weight_decay_bn: bool = True
        self.weight_decay_identity_bn: bool = True
        self.weight_decay_limit: bool = False # apply weight decay to some modules
        self.weight_decay_target_modules: list
        self.norm_type:str = "none"
        self.norm_param: float 
        self.cb_type: str
        self.cb_beta: float
        self.cb_gamma: float
        self.train_bn_bias: bool = True
        self.train_bn_weight: bool = True
        self.features_regularization: float = 0.0
        self.optimizer:str = "SGD"
        self.drw: int = 0 # Deferred Re-balancing Optimization Schedule, when to start rebalancing
        self.nc_lambda1: float = 0.0
        self.nc_lambda2: float = 0.0
        if vars_dict:
            super().__init__(vars_dict)


class BaseArgs(RootArgs):


    def __init__(self, vars_dict= None, add_logger = True):
        self.data: DataArgs
        self.model: ModelArgs
        self.optim: OptimArgs
        self.exp_name: str = ""
        self.project_name: str = "first"
        self.seeds: int = 0
        self.description: str = ""
        self.save_features: bool = False
        self.param_search: bool = False
        self.param_search_id: int = 0
        self.save_checkpoint: bool = False
        self.load_checkpoint: bool = False

        self.device: str
        self.logger: logging.Logger
        super().__init__(vars_dict, add_logger)
        
        if vars_dict:
            self.model = ModelArgs(vars(self.model))
            self.data = DataArgs(vars(self.data))
            self.optim = OptimArgs(vars(self.optim))
        else:
            self.data = DataArgs()
            self.model = ModelArgs()
            self.optim = OptimArgs()

### Experiments Args
class ExDataArgs(Args):
    def __init__(self, vars_dict=None):
        self.use_image: bool = False
        self.data_phase: str = "valid"
        self.batch_size: int = 2000
        if vars_dict:
            super().__init__(vars_dict)

class ExModelArgs(Args):
    def __init__(self, vars_dict = None):
        self.only_first_layer: bool = False
        self.args_pathes_list: list
        self.exp_names_list : list
        self.load_param_tf_list : list
        self.load_last_linear: bool = True
        self.shuffle_last_linear: bool = False
        self.dataset_name: str = "Cifar100"
        self.fine_grained_features: bool = False
        self.norm_features: bool = False
        self.relu_features: bool = False
        if vars_dict:
            super().__init__(vars_dict)
            
class ExShuffleArgs(Args):
    CONV_ALL_SHUFFLE = 0 # shuffle within all parameters
    CONV_OUTPUT_SHUFFLE = 1 # swap the order of outputs
    CONV_KERNEL_SHUFFLE = 2 # shuffle within each kernel
    def __init__(self, vars_dict=None):
        self.shuffle_param = False
        
        self.bn_shuffle = True
        self.conv_shuffle = True
        self.conv_shuffle_option = self.CONV_ALL_SHUFFLE
        
        if vars_dict:
            super().__init__(vars_dict)

class ExBaseArgs(RootArgs):
    def __init__(self, vars_dict= None, add_logger = True):
        self.per_modules: bool = False
        self.split_cos_type=False
        self.get_residual = False
        self.seeds: int = 0
        self.device: str
        self.exp_name: str
        self.comment: str = ""
        self.sub_name: str = "main"
        self.shuffle : ExShuffleArgs
        self.data : ExDataArgs
        self.model : ExModelArgs
        
        super().__init__(vars_dict, add_logger)
        
        if vars_dict:
            self.shuffle = ExShuffleArgs(vars(self.shuffle))
            self.data = ExDataArgs(vars(self.data))
            self.model = ExModelArgs(vars(self.model))
        else:
            self.shuffle = ExShuffleArgs()
            self.data = ExDataArgs()
            self.model = ExModelArgs()

class OneLayerExArgs(RootArgs):
    def __init__(self, vars_dict= None, add_logger = True):
        self.seeds: int = 0
        self.device: str
        self.exp_name: str
        self.comment: str = ""
        self.sub_name: str = "main"
        
        self.d_in: int = 512
        self.d_out: int = 512
        self.batch_size: int = 1024
        self.steps_n: int = 100
        
        self.linear_mean_list: list = [0.0]
        self.linear_std_list: list = [1.0]
        self.linear_ber_p: list = [1.0]
        self.bn_weights_mean_list: list = [1.0]
        self.bn_weights_std_list: list = [0.0]
        self.bn_bias_mean_list: list = [0.0]
        self.bn_bias_std_list: list = [0.0]
        self.normalize_list: list = [True]
        
        self.data_dist: list = ["gauss"]
        self.data_mu: list = [0.0]
        self.data_sigma: list = [1.0]
        self.ber_p : list = [1.0]
        self.relu_data: list = [True]
        self.labels_n: list = [1024]
        
        
        
        
        self.name_list: list = []
        
        
        super().__init__(vars_dict, add_logger)


class HypSearchArgs(RootArgs):
    def __init__(self, vars_dict= None, add_logger = True):
        self.seeds: int = 0
        self.json_path:str
        self.exp_name: str 

#########################
## work dir
#########################        
def get_work_dir_path(args: RootArgs):
    if type(args) == BaseArgs:
        work_dir=os.path.join(os.getcwd(), "exp", args.data.name, args.model.name,
            args.project_name, args.exp_name, str(args.param_search_id) if args.param_search else str(args.seeds) )
    elif type(args) == ExBaseArgs:
        work_dir = os.path.join(os.getcwd(), EXP_OUTPUT_DIR, "ce_model", args.model.dataset_name, args.exp_name, args.sub_name)
    elif type(args) == OneLayerExArgs:
        work_dir = os.path.join(os.getcwd(), EXP_OUTPUT_DIR, "ce_one_layer", args.exp_name, args.sub_name)
    else:
        assert False, "Invalid args type."
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    return work_dir


#########################
## other settings
#########################
# fix the random seed for fair comparison. 
# NOTE that we also removed "shuffle" lines in generating long-tailed CIFAR already (cf. util.dataset_CIFAR100LT.py)
def set_seeds(seed=0):
  torch.manual_seed(seed)
  np.random.seed(seed)
  
# Used to maintain consistency in the relationship between the magnitude of the gradient and that of the impact on Weight Decay
class copy_and_aggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, num_features: int):
        return x * torch.ones((num_features,), device = x.device)
    
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.mean(0,keepdim = True), None

# Mdofied version of the Pytorch's implementation
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d
class OneParamBN(torch.nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = torch.nn.Parameter(torch.empty(1, **factory_kwargs), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.empty(1, **factory_kwargs), requires_grad=True)
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            copy_and_aggregate.apply(self.weight, self.num_features),
            copy_and_aggregate.apply(self.bias, self.num_features),
            
            bn_training,
            exponential_average_factor,
            self.eps,
        )
    
def is_batchnorm(module: nn.Module):
    return isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, OneParamBN)

# set device, which gpu to use.
def set_device(args: BaseArgs):
    device ='cpu'
    if torch.cuda.is_available(): 
        device='cuda'
    args.device = device
    return args

def set_cuda():
    torch.cuda.device_count()
    torch.cuda.empty_cache()

#########################
## logger
#########################
def get_tmp_log_path():
    log_folder = "tmp/logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    tdatetime = dt.now()
    logs_n = len(glob.glob(os.path.join(log_folder, "*")))
    file_head = tdatetime.strftime('%Y%m%d%H%M%S%f')
    file_name = file_head + f"-{logs_n:03}"
    return os.path.join(log_folder, file_name)

def get_logger():
    logger = logging.getLogger("logger")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG) 
    logger.propagate = False

    sto_handler = logging.StreamHandler(stream=sys.stdout)
    sto_handler.setLevel(logging.INFO)
    sto_handler.setFormatter(logging.Formatter("%(message)s"))

    ste_handler = logging.StreamHandler(stream=sys.stderr)
    ste_handler.setLevel(logging.WARN)
    ste_handler.setFormatter(logging.Formatter("%(levelname)8s %(message)s"))

    file_handler = logging.FileHandler(filename=get_tmp_log_path()) 
    file_handler.setLevel(logging.DEBUG) 
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    logger.addHandler(sto_handler)
    logger.addHandler(ste_handler)
    logger.addHandler(file_handler)
    logger.debug("Logger: Loaded")

    return logger

def get_mean_std_statistics(json_file: str, args: BaseArgs):
    par_dir = os.path.dirname(get_work_dir_path(args))
    work_dir_list = glob.glob(os.path.join(par_dir, "*"))
    dict_list = []
    for work_dir in work_dir_list:
        json_path = os.path.join(work_dir, json_file)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                dict_list.append(json.load(f))
    mean_dict = {key: sum([dictionary[key] for dictionary in dict_list]) / len(dict_list) for key in dict_list[0].keys()}
    std_dict = {key: math.sqrt(sum([(dictionary[key] - mean_dict[key])**2 for dictionary in dict_list]) / len(dict_list)) for key in mean_dict.keys()}
    return mean_dict, std_dict

def get_mean_acc(args: BaseArgs, plus_std:bool = False):
    ans =  get_mean_std_statistics(ACCURACY_FILE_NAME, args)
    if plus_std:
        return ans
    else:
        return ans[0]

def get_mean_fdr(args: BaseArgs, plus_std:bool = False):
    ans = get_mean_std_statistics(DISCRIMINANT_RATIO_FILE_NAME, args)
    if plus_std:
        return ans
    else:
        return ans[0]


def log_result_path(args: RootArgs):
    result_log_path = "tmp/result_pathes.txt"
    with open(result_log_path, "a") as f:
        f.write(str(dt.now()) +"   "+ get_work_dir_path(args) + "\n")